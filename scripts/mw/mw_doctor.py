#!/usr/bin/env python3
"""mw-doctor — portable install/readiness checker for the Use Migrate Web toolkit.

Verifies THIS machine is really ready after install:
  (a) required tools are installed AND runnable (probe argv, not just file presence)
  (b) each image source does a REAL tiny smoke job (auth-pass alone is not enough)
  (c) the AI relay actually fires once

Config-driven and portable. Like relay-doctor but for the MW toolkit.

CLI::

    mw_doctor.py [--config PATH] [--json] [--section tools|images|relay|all]
                 [--skip-network] [--timeout N]

Exit codes:
  0  READY
  1  NOT_READY
  2  config / usage error

stdlib-only core · optional PyYAML · self-contained mini-YAML fallback · Python 3.9+
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

EXIT_READY = 0
EXIT_NOT_READY = 1
EXIT_ERR = 2

DEFAULT_CONFIG_REL = Path(".work") / "mw-doctor.yaml"
DEFAULT_TIMEOUT = 60.0

SECTIONS = frozenset({"tools", "images", "relay", "all"})

# Auth / credential failure markers (case-insensitive substring match)
_AUTH_MARKERS = (
    "auth",
    "unauthorized",
    "401",
    "403",
    "api key",
    "credential",
)

# Placeholder token in relay.smoke argv that means "write a temp prompt file"
_PING_TOKEN = "PING"

# Tests may force the fallback YAML loader even when PyYAML is installed.
_FORCE_MINI_YAML = False

try:
    import yaml as _yaml  # type: ignore
except ImportError:  # pragma: no cover
    _yaml = None


# ---------------------------------------------------------------------------
# minimal YAML subset loader (self-contained; mirror of menu_gate.py approach)
# ---------------------------------------------------------------------------

def _scalar(v: str) -> Any:
    """Parse a YAML scalar: quoted string, bool, null, int/float, or raw str."""
    v = v.strip()
    if not v:
        return None
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    # strip trailing unquoted comment
    if " #" in v:
        v = v.split(" #", 1)[0].strip()
    # simple inline list: [a, b, "c"]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        parts: List[str] = []
        buf = ""
        in_q: Optional[str] = None
        for ch in inner:
            if in_q:
                buf += ch
                if ch == in_q:
                    in_q = None
                continue
            if ch in "\"'":
                in_q = ch
                buf += ch
                continue
            if ch == ",":
                parts.append(buf.strip())
                buf = ""
                continue
            buf += ch
        if buf.strip():
            parts.append(buf.strip())
        return [_scalar(p) for p in parts]
    low = v.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


class MiniYamlError(ValueError):
    """Raised when the mini YAML loader hits an ambiguous / unrepresentable construct."""


def _mini_yaml(text: str) -> Any:
    """Minimal YAML-subset parser for mw-doctor config.

    Handles: nested mappings by 2-space indent, lists of mappings (`- key: val`
    and dash-on-its-own-line with mapping keys on following indented lines),
    quoted/unquoted scalars, booleans, null, inline `[a, b]` lists, `#` comments,
    and blank lines.  Enough for mw-doctor.yaml — not a full YAML 1.1 parser.

    On constructs that cannot be represented unambiguously, raises MiniYamlError
    instead of silently returning a partial/altered structure (fail closed).
    """
    lines = [
        l
        for l in (raw.rstrip() for raw in text.splitlines())
        if l.strip() and not l.strip().startswith("#")
    ]
    pos = 0

    def indent_of(l: str) -> int:
        return len(l) - len(l.lstrip(" "))

    def _is_list_item(s: str) -> bool:
        """True for ``- `` prefix or bare ``-`` (dash alone on the line)."""
        return s == "-" or s.startswith("- ")

    def _list_item_body(s: str) -> str:
        if s == "-":
            return ""
        return s[2:].strip()

    def _parse_mapping_continuation(d: Dict[str, Any], base_indent: int) -> None:
        """Read nested keys under a list-item mapping (indent > base_indent)."""
        nonlocal pos
        while pos < len(lines):
            nline = lines[pos]
            ni = indent_of(nline)
            if ni <= base_indent:
                break
            ns = nline.strip()
            if _is_list_item(ns):
                break
            if ":" not in ns:
                raise MiniYamlError(
                    f"unrepresentable YAML at line content {ns!r}: "
                    "expected key: value under list item"
                )
            nk, _, nv = ns.partition(":")
            pos += 1
            if nv.strip():
                d[nk.strip()] = _scalar(nv)
            elif pos < len(lines) and indent_of(lines[pos]) > ni:
                d[nk.strip()] = parse_block(indent_of(lines[pos]))
            else:
                d[nk.strip()] = None

    def parse_block(indent: int) -> Any:
        nonlocal pos
        result: Any = None
        while pos < len(lines):
            line = lines[pos]
            cur = indent_of(line)
            if cur < indent:
                break
            if cur > indent:
                raise MiniYamlError(
                    f"orphan deeper indent without a parent key near: {line.strip()!r}"
                )
            s = line.strip()
            if _is_list_item(s):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise MiniYamlError(
                        f"mixed mapping/list structure near: {s!r}"
                    )
                item = _list_item_body(s)
                pos += 1
                if item == "":
                    # Dash alone (or empty after dash): nested block on next lines
                    if pos < len(lines) and indent_of(lines[pos]) > indent:
                        nested_indent = indent_of(lines[pos])
                        nested_s = lines[pos].strip()
                        # Mapping keys on following indented lines (no leading dash)
                        if not _is_list_item(nested_s) and ":" in nested_s:
                            d: Dict[str, Any] = {}
                            _parse_mapping_continuation(d, indent)
                            result.append(d)
                        else:
                            result.append(parse_block(nested_indent))
                    else:
                        result.append(None)
                elif ":" in item and item[0] not in "\"'":
                    # mapping entry on the dash line: `- id: foo`
                    k, _, v = item.partition(":")
                    d = {k.strip(): _scalar(v) if v.strip() else None}
                    _parse_mapping_continuation(d, indent)
                    # dash-line sole key: with nested block as its value
                    if d and list(d.values()) == [None]:
                        only_k = next(iter(d))
                        if pos < len(lines) and indent_of(lines[pos]) > indent:
                            d[only_k] = parse_block(indent_of(lines[pos]))
                    result.append(d)
                else:
                    # plain scalar list item
                    result.append(_scalar(item))
            else:
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise MiniYamlError(
                        f"mixed list/mapping structure near: {s!r}"
                    )
                if ":" not in s:
                    raise MiniYamlError(
                        f"unrepresentable YAML near: {s!r} (expected key: value)"
                    )
                k, _, v = s.partition(":")
                pos += 1
                if v.strip():
                    result[k.strip()] = _scalar(v)
                elif pos < len(lines) and indent_of(lines[pos]) > cur:
                    result[k.strip()] = parse_block(indent_of(lines[pos]))
                else:
                    result[k.strip()] = None
        return result

    return parse_block(0) if lines else {}


def load_yaml_text(text: str, force_mini: bool = False) -> Any:
    """Load YAML text: prefer PyYAML unless forced to mini loader.

    MiniYamlError propagates so callers can exit 2 (fail closed) rather than
    accept a silently truncated structure.
    """
    use_mini = force_mini or _FORCE_MINI_YAML or _yaml is None
    if not use_mini:
        data = _yaml.safe_load(text)
        return data if data is not None else {}
    return _mini_yaml(text) or {}


def load_yaml_file(path: Path, force_mini: bool = False) -> Any:
    text = path.read_text(encoding="utf-8")
    return load_yaml_text(text, force_mini=force_mini)


# ---------------------------------------------------------------------------
# auth detection + subprocess helpers
# ---------------------------------------------------------------------------

def looks_like_auth_error(text: str) -> bool:
    """True if stdout/stderr looks like an auth/credential problem.

    Markers (case-insensitive): auth, unauthorized, 401, 403, api key, credential.
    """
    if not text:
        return False
    low = text.lower()
    return any(m in low for m in _AUTH_MARKERS)


def _as_argv(val: Any) -> Optional[List[str]]:
    """Normalize a probe/smoke value to an argv list of strings, or None if empty."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else None
    if isinstance(val, list):
        if not val:
            return None
        return [str(x) for x in val]
    return [str(val)]


@dataclass
class ProbeResult:
    """Outcome of one subprocess probe."""

    exit_code: Optional[int]  # None on missing / timeout / other launch error
    stdout: str = ""
    stderr: str = ""
    missing: bool = False
    timed_out: bool = False
    error: str = ""


def run_probe(argv: Sequence[str], timeout: float, cwd: Optional[Path] = None) -> ProbeResult:
    """Run argv with shell=False and a per-probe timeout.

    FileNotFoundError / exit 127 → missing. TimeoutExpired → timed_out.
    """
    if not argv:
        return ProbeResult(exit_code=None, error="empty argv", missing=True)
    try:
        completed = subprocess.run(
            list(argv),
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
        # Some systems report 127 for command-not-found when wrapped
        if completed.returncode == 127:
            return ProbeResult(
                exit_code=127,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                missing=True,
            )
        return ProbeResult(
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
    except FileNotFoundError:
        return ProbeResult(exit_code=None, missing=True, error="FileNotFoundError")
    except subprocess.TimeoutExpired as exc:
        out = ""
        err = ""
        if exc.stdout:
            out = exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(
                "utf-8", errors="replace"
            )
        if exc.stderr:
            err = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(
                "utf-8", errors="replace"
            )
        return ProbeResult(
            exit_code=None,
            stdout=out,
            stderr=err,
            timed_out=True,
            error=f"timeout after {timeout}s",
        )
    except OSError as exc:
        # e.g. permission denied, not a directory
        return ProbeResult(exit_code=None, error=str(exc), missing=True)


def extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the LAST JSON object printed on stdout (brace-balanced scan).

    Returns None if no valid object is found.
    """
    if not text or not text.strip():
        return None
    # Prefer full-line JSON objects first (common for CLI tools)
    candidates: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            candidates.append(s)
    # Also try the whole text if it looks like one object
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    # Brace-balanced scan from the rightmost '{'
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                ch = text[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[i : j + 1])
                        break
            break
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# check result model
# ---------------------------------------------------------------------------

# Status strings used in reports
ST_OK = "ok"
ST_MISSING = "missing"
ST_FAIL = "fail"
ST_AUTH_MISSING = "auth_missing"
ST_SKIPPED_NETWORK = "skipped(network)"


@dataclass
class ItemResult:
    name: str
    status: str
    detail: str = ""
    optional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "optional": self.optional,
        }
        return d


@dataclass
class RelayResult:
    status: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "detail": self.detail}


@dataclass
class DoctorReport:
    tools: List[ItemResult] = field(default_factory=list)
    images: List[ItemResult] = field(default_factory=list)
    relay: Optional[RelayResult] = None
    machine_note: Optional[str] = None
    section: str = "all"
    skip_network: bool = False

    def blocking(self) -> List[str]:
        """Human-readable blocking reasons (required items only)."""
        reasons: List[str] = []
        for t in self.tools:
            if t.optional:
                continue
            if t.status != ST_OK:
                reasons.append(f"{t.name} {t.status}")
        for img in self.images:
            if img.status != ST_OK:
                reasons.append(f"{img.name} {img.status}")
        if self.relay is not None and self.relay.status != ST_OK:
            reasons.append(f"relay {self.relay.status}")
        return reasons

    def is_ready(self) -> bool:
        return len(self.blocking()) == 0

    def to_json(self) -> Dict[str, Any]:
        sections: Dict[str, Any] = {}
        if self.section in ("tools", "all"):
            sections["tools"] = [t.to_dict() for t in self.tools]
        if self.section in ("images", "all"):
            sections["images"] = [i.to_dict() for i in self.images]
        if self.section in ("relay", "all"):
            if self.relay is not None:
                sections["relay"] = self.relay.to_dict()
            else:
                sections["relay"] = None
        out: Dict[str, Any] = {
            "section": self.section,
            "sections": sections,
            "ready": self.is_ready(),
            "blocking": self.blocking(),
        }
        if self.machine_note:
            out["machine_note"] = self.machine_note
        return out


# ---------------------------------------------------------------------------
# section evaluators
# ---------------------------------------------------------------------------

def check_tools(
    tools_cfg: Any,
    timeout: float,
    cwd: Optional[Path] = None,
) -> List[ItemResult]:
    """Run each tool probe. optional missing → missing but not blocking."""
    results: List[ItemResult] = []
    if tools_cfg is None:
        return results
    if not isinstance(tools_cfg, list):
        raise ValueError("tools must be a list")

    for entry in tools_cfg:
        if not isinstance(entry, dict):
            raise ValueError(f"tools entry must be a mapping, got {type(entry).__name__}")
        name = str(entry.get("name") or "").strip() or "(unnamed)"
        optional = bool(entry.get("optional", False))
        expect_exit = entry.get("expect_exit", 0)
        try:
            expect_exit = int(expect_exit)
        except (TypeError, ValueError):
            raise ValueError(f"tools[{name}].expect_exit must be an int") from None
        argv = _as_argv(entry.get("probe"))
        if not argv:
            results.append(
                ItemResult(
                    name=name,
                    status=ST_FAIL,
                    detail="no probe command configured",
                    optional=optional,
                )
            )
            continue

        pr = run_probe(argv, timeout=timeout, cwd=cwd)
        if pr.missing:
            results.append(
                ItemResult(
                    name=name,
                    status=ST_MISSING,
                    detail=pr.error or "command not found",
                    optional=optional,
                )
            )
            continue
        if pr.timed_out:
            results.append(
                ItemResult(
                    name=name,
                    status=ST_FAIL,
                    detail=pr.error or "timeout",
                    optional=optional,
                )
            )
            continue
        if pr.exit_code == expect_exit:
            results.append(
                ItemResult(name=name, status=ST_OK, detail="", optional=optional)
            )
        else:
            detail = f"exit {pr.exit_code} (want {expect_exit})"
            if pr.stderr.strip():
                detail += f": {pr.stderr.strip()[:200]}"
            results.append(
                ItemResult(
                    name=name,
                    status=ST_FAIL,
                    detail=detail,
                    optional=optional,
                )
            )
    return results


def check_image_sources(
    images_cfg: Any,
    timeout: float,
    skip_network: bool,
    cwd: Optional[Path] = None,
) -> List[ItemResult]:
    """Smoke-test each image source. No smoke command → config error (raise)."""
    results: List[ItemResult] = []
    if images_cfg is None:
        return results
    if not isinstance(images_cfg, list):
        raise ValueError("image_sources must be a list")

    for entry in images_cfg:
        if not isinstance(entry, dict):
            raise ValueError(
                f"image_sources entry must be a mapping, got {type(entry).__name__}"
            )
        name = str(entry.get("name") or "").strip() or "(unnamed)"
        # Reject auth-only: smoke is required
        if "smoke" not in entry or entry.get("smoke") is None:
            raise ConfigError(
                f"image source {name} has no smoke command — auth-only is not allowed"
            )
        argv = _as_argv(entry.get("smoke"))
        if not argv:
            raise ConfigError(
                f"image source {name} has no smoke command — auth-only is not allowed"
            )

        if skip_network:
            results.append(
                ItemResult(
                    name=name,
                    status=ST_SKIPPED_NETWORK,
                    detail="--skip-network",
                )
            )
            continue

        expect_exit = entry.get("expect_exit", 0)
        try:
            expect_exit = int(expect_exit)
        except (TypeError, ValueError):
            raise ValueError(f"image_sources[{name}].expect_exit must be an int") from None
        expect_contains = entry.get("expect_contains", None)
        if expect_contains is not None:
            expect_contains = str(expect_contains)

        pr = run_probe(argv, timeout=timeout, cwd=cwd)
        combined = f"{pr.stdout}\n{pr.stderr}"

        if pr.missing:
            # Missing binary for an image source — treat as fail (or auth if message says so)
            if looks_like_auth_error(combined):
                results.append(
                    ItemResult(
                        name=name,
                        status=ST_AUTH_MISSING,
                        detail="set credentials",
                    )
                )
            else:
                results.append(
                    ItemResult(
                        name=name,
                        status=ST_FAIL,
                        detail=pr.error or "command not found",
                    )
                )
            continue
        if pr.timed_out:
            results.append(
                ItemResult(name=name, status=ST_FAIL, detail=pr.error or "timeout")
            )
            continue

        # Auth scan ALWAYS runs, regardless of exit code (fail closed).
        if looks_like_auth_error(combined):
            results.append(
                ItemResult(
                    name=name,
                    status=ST_AUTH_MISSING,
                    detail="set credentials",
                )
            )
            continue

        if pr.exit_code != expect_exit:
            detail = f"exit {pr.exit_code} (want {expect_exit})"
            if pr.stderr.strip():
                detail += f": {pr.stderr.strip()[:200]}"
            results.append(ItemResult(name=name, status=ST_FAIL, detail=detail))
            continue

        # Prove a real tiny job: expect_contains hit, or non-empty stdout.
        if expect_contains is not None:
            if expect_contains not in pr.stdout:
                results.append(
                    ItemResult(
                        name=name,
                        status=ST_FAIL,
                        detail=f"stdout missing expected substring {expect_contains!r}",
                    )
                )
                continue
        elif not pr.stdout.strip():
            results.append(
                ItemResult(
                    name=name,
                    status=ST_FAIL,
                    detail=(
                        "smoke produced no result output "
                        "(cannot prove a real job)"
                    ),
                )
            )
            continue

        results.append(ItemResult(name=name, status=ST_OK, detail=""))
    return results


class ConfigError(ValueError):
    """Config/schema error that should exit 2."""


def _materialize_ping_argv(
    argv: List[str],
) -> Tuple[List[str], Optional[str]]:
    """If argv contains the PING placeholder, write a temp prompt file and substitute.

    Returns (new_argv, temp_path_or_None). Caller should delete the temp file.
    """
    if _PING_TOKEN not in argv:
        return list(argv), None
    # Create a tiny temp prompt file with a trivial prompt
    fd, path = tempfile.mkstemp(prefix="mw-doctor-ping-", suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("mw-doctor ping — reply with ok\n")
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    new_argv = [path if a == _PING_TOKEN else a for a in argv]
    return new_argv, path


def check_relay(
    relay_cfg: Any,
    timeout: float,
    skip_network: bool,
    cwd: Optional[Path] = None,
) -> Optional[RelayResult]:
    """Fire the relay once. Returns None if relay section is absent/empty."""
    if relay_cfg is None:
        return None
    if not isinstance(relay_cfg, dict):
        raise ValueError("relay must be a mapping")
    if not relay_cfg:
        return None

    argv = _as_argv(relay_cfg.get("smoke"))
    if not argv:
        raise ConfigError("relay has no smoke command")

    if skip_network:
        return RelayResult(status=ST_SKIPPED_NETWORK, detail="--skip-network")

    expect_status = str(relay_cfg.get("expect_status", "ok"))
    temp_path: Optional[str] = None
    try:
        run_argv, temp_path = _materialize_ping_argv(argv)
        pr = run_probe(run_argv, timeout=timeout, cwd=cwd)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    combined = f"{pr.stdout}\n{pr.stderr}"

    if pr.missing:
        if looks_like_auth_error(combined):
            return RelayResult(status=ST_AUTH_MISSING, detail="set credentials")
        return RelayResult(status=ST_FAIL, detail=pr.error or "command not found")
    if pr.timed_out:
        return RelayResult(status=ST_FAIL, detail=pr.error or "timeout")

    # Auth scan ALWAYS runs first, regardless of exit code (fail closed).
    if looks_like_auth_error(combined):
        return RelayResult(status=ST_AUTH_MISSING, detail="set credentials")

    # Relay is ok only if process exit==0 AND JSON status matches.
    if pr.exit_code != 0:
        detail = f"exit {pr.exit_code} (want 0)"
        if pr.stderr.strip():
            detail += f": {pr.stderr.strip()[:200]}"
        return RelayResult(status=ST_FAIL, detail=detail)

    obj = extract_last_json_object(pr.stdout)
    if obj is None:
        return RelayResult(status=ST_FAIL, detail="no JSON object on stdout")

    got = obj.get("status")
    if got is None:
        return RelayResult(status=ST_FAIL, detail="JSON missing 'status' field")
    if str(got) == expect_status:
        return RelayResult(status=ST_OK, detail="")
    # status like "auth" → auth_missing when it looks auth-ish
    if looks_like_auth_error(str(got)):
        return RelayResult(status=ST_AUTH_MISSING, detail=f"status={got!r}")
    return RelayResult(status=ST_FAIL, detail=f"status={got!r} (want {expect_status!r})")


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _status_label(status: str, optional: bool = False, detail: str = "") -> str:
    """Human label for a status."""
    if status == ST_OK:
        return "OK"
    if status == ST_MISSING:
        return "MISSING(optional)" if optional else "MISSING"
    if status == ST_AUTH_MISSING:
        hint = detail if detail else "set credentials"
        return f"AUTH_MISSING({hint})"
    if status == ST_SKIPPED_NETWORK:
        return "SKIPPED(network)"
    # fail
    if detail:
        return f"FAIL({detail[:80]})"
    return "FAIL"


def format_human(report: DoctorReport) -> str:
    """Sectioned human-readable report ending with READY / NOT_READY."""
    lines: List[str] = []

    if report.section in ("tools", "all") and report.tools:
        parts = [
            f"{t.name} {_status_label(t.status, t.optional, t.detail)}"
            for t in report.tools
        ]
        lines.append("tools: " + " · ".join(parts))
    elif report.section in ("tools", "all"):
        lines.append("tools: (none configured)")

    if report.section in ("images", "all") and report.images:
        parts = [
            f"{i.name} {_status_label(i.status, False, i.detail)}"
            for i in report.images
        ]
        lines.append("images: " + " · ".join(parts))
    elif report.section in ("images", "all"):
        lines.append("images: (none configured)")

    if report.section in ("relay", "all"):
        if report.relay is None:
            lines.append("relay: (not configured)")
        else:
            lines.append(
                "relay: "
                + _status_label(report.relay.status, False, report.relay.detail)
            )

    if report.is_ready():
        # Scoped passes must never read as full readiness (section != all).
        if report.section == "all":
            lines.append("mw-doctor: READY")
        else:
            lines.append(f"mw-doctor: READY (scope: {report.section})")
    else:
        blocking = report.blocking()
        # shorten skipped label for summary
        short = []
        for b in blocking:
            if ST_SKIPPED_NETWORK in b:
                # e.g. "freepik skipped(network)" → keep readable
                short.append(b.replace(ST_SKIPPED_NETWORK, "skipped"))
            else:
                short.append(b)
        lines.append("mw-doctor: NOT_READY (blocking: " + ", ".join(short) + ")")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# config resolution + main run
# ---------------------------------------------------------------------------

def resolve_config_path(explicit: Optional[Path], cwd: Optional[Path] = None) -> Path:
    """--config if given, else <cwd>/.work/mw-doctor.yaml. Raises FileNotFoundError."""
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"config not found: {p}")
        return p.resolve()
    base = cwd if cwd is not None else Path.cwd()
    default = (base / DEFAULT_CONFIG_REL).resolve()
    if not default.is_file():
        raise FileNotFoundError(
            f"config not found: {default} (pass --config or create {DEFAULT_CONFIG_REL})"
        )
    return default


def validate_config_for_section(config: Dict[str, Any], section: str) -> None:
    """Fail closed when the selected scope has nothing real to prove.

    empty/missing required sections must not report READY with zero checks.
    Raises ConfigError (exit 2).
    """
    if section not in SECTIONS:
        raise ConfigError(f"invalid section {section!r}; want one of {sorted(SECTIONS)}")

    tools = config.get("tools")
    images = config.get("image_sources")
    relay = config.get("relay")

    tools_ok = isinstance(tools, list) and len(tools) > 0
    images_ok = isinstance(images, list) and len(images) > 0
    relay_ok = (
        isinstance(relay, dict)
        and bool(relay)
        and _as_argv(relay.get("smoke")) is not None
    )

    if section == "all":
        if not (tools_ok and images_ok and relay_ok):
            raise ConfigError(
                "section all requires non-empty tools, image_sources, and a relay smoke"
            )
        return

    if section == "tools":
        if not tools_ok:
            raise ConfigError("section tools requires non-empty tools")
        return

    if section == "images":
        if not images_ok:
            raise ConfigError("section images requires non-empty image_sources")
        return

    if section == "relay":
        if not relay_ok:
            raise ConfigError("section relay requires a relay smoke")
        return


def run_doctor(
    config: Dict[str, Any],
    *,
    section: str = "all",
    skip_network: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    cwd: Optional[Path] = None,
) -> DoctorReport:
    """Evaluate config sections and return a DoctorReport (ready derived live)."""
    if section not in SECTIONS:
        raise ConfigError(f"invalid section {section!r}; want one of {sorted(SECTIONS)}")

    validate_config_for_section(config, section)

    report = DoctorReport(
        machine_note=str(config["machine_note"]) if config.get("machine_note") else None,
        section=section,
        skip_network=skip_network,
    )

    if section in ("tools", "all"):
        report.tools = check_tools(config.get("tools"), timeout=timeout, cwd=cwd)

    if section in ("images", "all"):
        report.images = check_image_sources(
            config.get("image_sources"),
            timeout=timeout,
            skip_network=skip_network,
            cwd=cwd,
        )

    if section in ("relay", "all"):
        report.relay = check_relay(
            config.get("relay"),
            timeout=timeout,
            skip_network=skip_network,
            cwd=cwd,
        )

    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mw_doctor",
        description=(
            "Portable install/readiness checker for the Use Migrate Web toolkit. "
            "Exit 0=READY, 1=NOT_READY, 2=config/usage error."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to mw-doctor YAML (default: <cwd>/{DEFAULT_CONFIG_REL})",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object to stdout",
    )
    p.add_argument(
        "--section",
        choices=sorted(SECTIONS),
        default="all",
        help="Which section(s) to evaluate (default: all)",
    )
    p.add_argument(
        "--skip-network",
        action="store_true",
        help="Do not run image-source or relay smoke calls (reported as skipped(network))",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="N",
        help=f"Per-probe timeout seconds (default {int(DEFAULT_TIMEOUT)})",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def run(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry. Returns process exit code."""
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        code = e.code
        if code is None:
            return EXIT_READY
        return int(code) if isinstance(code, int) else EXIT_ERR

    timeout = float(args.timeout)
    if timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return EXIT_ERR

    try:
        config_path = resolve_config_path(args.config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERR

    try:
        raw = load_yaml_file(config_path)
    except MiniYamlError as exc:
        print(
            f"error: config YAML not representable by mini parser "
            f"({config_path}): {exc}",
            file=sys.stderr,
        )
        return EXIT_ERR
    except Exception as exc:
        print(f"error: failed to load config {config_path}: {exc}", file=sys.stderr)
        return EXIT_ERR

    if not isinstance(raw, dict):
        print("error: config root must be a mapping", file=sys.stderr)
        return EXIT_ERR

    try:
        report = run_doctor(
            raw,
            section=args.section,
            skip_network=bool(args.skip_network),
            timeout=timeout,
            cwd=Path.cwd(),
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERR
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERR

    if args.json:
        print(json.dumps(report.to_json(), ensure_ascii=False, separators=(",", ":")))
    else:
        print(format_human(report))

    return EXIT_READY if report.is_ready() else EXIT_NOT_READY


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
