#!/usr/bin/env python3
"""mw-wow-report — before/after "wow" metrics aggregator.

Builds a machine-only improvement table from result files (Lighthouse JSON,
axe JSON, gate output, RTM report, file sizes). Values are NEVER typed by
AI/humans into config — every metric reads from a file + extractor.

  exit 0  — report built (default)
  exit 1  — --fail-on-regression and any REGRESSED, and/or
            --require-complete and any NO_DATA
  exit 2  — usage / config / parse error

stdlib-only core · optional PyYAML · self-contained mini-YAML fallback ·
Python 3.9+
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERR = 2

DEFAULT_CONFIG_REL = Path(".work") / "wow.yaml"
DEFAULT_TOLERANCE = 0.0

STATUS_IMPROVED = "IMPROVED"
STATUS_REGRESSED = "REGRESSED"
STATUS_UNCHANGED = "UNCHANGED"
STATUS_NO_DATA = "NO_DATA"

DIRECTIONS = frozenset({"higher_better", "lower_better"})
EXTRACT_KEYS = frozenset(
    {"json_path", "json_len", "regex_ratio", "regex", "file_size_kb"}
)

# Tests may force the fallback YAML loader even when PyYAML is installed.
_FORCE_MINI_YAML = False

try:
    import yaml as _yaml  # type: ignore
except ImportError:  # pragma: no cover
    _yaml = None


# ---------------------------------------------------------------------------
# minimal YAML subset loader (self-contained; mirrors menu_gate / page_check)
# ---------------------------------------------------------------------------


class MiniYamlError(ValueError):
    """Raised when the mini YAML loader hits an ambiguous construct."""


def _split_flow_items(inner: str) -> List[str]:
    """Split comma-separated items inside ``[]`` or ``{}``, respecting quotes."""
    parts: List[str] = []
    buf = ""
    in_q: Optional[str] = None
    depth = 0
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
        if ch in "[{":
            depth += 1
            buf += ch
            continue
        if ch in "]}":
            depth = max(0, depth - 1)
            buf += ch
            continue
        if ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
            continue
        buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _scalar(v: str) -> Any:
    """Parse a YAML scalar: quoted string, bool, null, int/float, flow list/map, or raw str."""
    v = v.strip()
    if not v:
        return None
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    if " #" in v:
        v = v.split(" #", 1)[0].strip()
    # flow mapping: {file: "x", extract: {json_path: "a.b"}}
    if v.startswith("{") and v.endswith("}"):
        inner = v[1:-1].strip()
        if not inner:
            return {}
        out: Dict[str, Any] = {}
        for part in _split_flow_items(inner):
            if ":" not in part:
                raise MiniYamlError(f"unrepresentable flow mapping entry: {part!r}")
            k, _, raw = part.partition(":")
            out[k.strip()] = _scalar(raw)
        return out
    # flow list: [a, b, "c"]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_scalar(p) for p in _split_flow_items(inner)]
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


def _mini_yaml(text: str) -> Any:
    """Minimal YAML-subset parser for wow.yaml.

    Handles nested mappings by 2-space indent, lists of mappings, quoted/
    unquoted scalars, booleans, null, inline ``[a, b]`` / ``{k: v}`` flows,
    ``#`` comments. Fail closed on unrepresentable constructs (MiniYamlError).
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
        return s == "-" or s.startswith("- ")

    def _list_item_body(s: str) -> str:
        if s == "-":
            return ""
        return s[2:].strip()

    def _parse_mapping_continuation(d: Dict[str, Any], base_indent: int) -> None:
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
                    raise MiniYamlError(f"mixed mapping/list structure near: {s!r}")
                item = _list_item_body(s)
                pos += 1
                if item == "":
                    if pos < len(lines) and indent_of(lines[pos]) > indent:
                        nested_indent = indent_of(lines[pos])
                        nested_s = lines[pos].strip()
                        if not _is_list_item(nested_s) and ":" in nested_s:
                            d: Dict[str, Any] = {}
                            _parse_mapping_continuation(d, indent)
                            result.append(d)
                        else:
                            result.append(parse_block(nested_indent))
                    else:
                        result.append(None)
                elif ":" in item and item[0] not in "\"'":
                    k, _, v = item.partition(":")
                    d = {k.strip(): _scalar(v) if v.strip() else None}
                    _parse_mapping_continuation(d, indent)
                    if d and list(d.values()) == [None]:
                        only_k = next(iter(d))
                        if pos < len(lines) and indent_of(lines[pos]) > indent:
                            d[only_k] = parse_block(indent_of(lines[pos]))
                    result.append(d)
                else:
                    result.append(_scalar(item))
            else:
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise MiniYamlError(f"mixed list/mapping structure near: {s!r}")
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
    """Load YAML text: prefer PyYAML unless forced to mini loader."""
    use_mini = force_mini or _FORCE_MINI_YAML or _yaml is None
    if not use_mini:
        try:
            data = _yaml.safe_load(text)
        except Exception as e:  # PyYAML ScannerError / ParserError etc.
            raise ValueError(f"YAML parse error: {e}") from e
        return data if data is not None else {}
    return _mini_yaml(text) or {}


def load_yaml_file(path: Path, force_mini: bool = False) -> Any:
    text = path.read_text(encoding="utf-8")
    return load_yaml_text(text, force_mini=force_mini)


# ---------------------------------------------------------------------------
# path helpers (root containment — fail closed)
# ---------------------------------------------------------------------------


def _realpath(p: Path) -> Path:
    return Path(os.path.realpath(str(p)))


def under_root(root: Path, rel: str) -> Path:
    """Join a project-relative path under root (no absolute escape)."""
    if rel is None:
        return root
    s = str(rel).replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    if s.startswith("/"):
        s = s.lstrip("/")
    return root / s


def path_escapes_root(root: Path, candidate: Path) -> bool:
    """True if candidate's real path is outside root's real path."""
    root_r = _realpath(root)
    cand_r = _realpath(candidate)
    try:
        cand_r.relative_to(root_r)
        return False
    except ValueError:
        return True


def contained_path(root: Path, rel: str) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve rel under root; fail closed if the real path escapes --root.

    Returns (path, None) on success, or (None, reason) on escape/error.
    """
    if rel is None or str(rel).strip() == "":
        return None, "empty path"
    joined = under_root(root, rel)
    norm = str(rel).replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    parts = Path(norm).parts
    if ".." in parts:
        return None, f"path escapes --root: {rel}"
    # For missing leaves, check existing ancestors stay under root.
    try:
        root_r = _realpath(root)
        probe = joined
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        probe_r = _realpath(probe)
        try:
            probe_r.relative_to(root_r)
        except ValueError:
            return None, f"path escapes --root: {rel}"
        if joined.exists() and path_escapes_root(root, joined):
            return None, f"path escapes --root: {rel}"
    except OSError:
        return None, f"path escapes --root: {rel}"
    return joined, None


# ---------------------------------------------------------------------------
# JSON path helpers
# ---------------------------------------------------------------------------


def _walk_json_path(data: Any, dotted: str) -> Any:
    """Walk a dotted path (e.g. categories.performance.score). Missing -> None."""
    if dotted is None or str(dotted).strip() == "":
        return None
    cur: Any = data
    for part in str(dotted).split("."):
        if part == "":
            return None
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def _as_number(val: Any) -> Optional[float]:
    """Coerce a value to a finite float; non-numeric / NaN / Inf -> None.

    Golden invariant: metric values are only finite numbers. Non-finite
    values (NaN, Infinity, overflow) must never become a status or delta.
    """
    if val is None or isinstance(val, bool):
        return None
    x: float
    if isinstance(val, (int, float)):
        x = float(val)
    elif isinstance(val, str):
        try:
            x = float(val.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(x):
        return None
    return x


def _require_finite_number(val: Any, label: str) -> float:
    """Coerce to finite float or raise ValueError (config / scale / tolerance)."""
    if val is None or isinstance(val, bool):
        raise ValueError(f"{label}: must be a finite number (got {val!r})")
    try:
        x = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label}: must be a finite number (got {val!r})") from e
    if not math.isfinite(x):
        raise ValueError(f"{label}: must be a finite number (got {val!r})")
    return x


# ---------------------------------------------------------------------------
# extractors — each returns (value|None, extra_dict, reason|None)
# ---------------------------------------------------------------------------


@dataclass
class ExtractResult:
    value: Optional[float]
    n: Optional[int] = None
    m: Optional[int] = None
    reason: Optional[str] = None


def extract_from_side(
    root: Path,
    side: Dict[str, Any],
    side_label: str,
) -> ExtractResult:
    """Run the configured extractor for one before/after side.

    Fail closed: missing file, bad parse, path escape, wrong type -> None.
    """
    file_rel = side.get("file")
    if file_rel is None or str(file_rel).strip() == "":
        # Config validation should catch this; belt-and-suspenders.
        return ExtractResult(None, reason=f"{side_label}: missing file")

    path, esc = contained_path(root, str(file_rel))
    if esc is not None or path is None:
        return ExtractResult(
            None, reason=f"{side_label}: {esc or 'path not contained'}"
        )

    extract = side.get("extract")
    if not isinstance(extract, dict) or not extract:
        return ExtractResult(None, reason=f"{side_label}: missing extract")

    # Prefer known extract keys; ignore unknown companions.
    known = [k for k in extract if k in EXTRACT_KEYS]
    # scale is a companion of json_path, not a primary extractor
    if not known:
        return ExtractResult(
            None, reason=f"{side_label}: no recognized extract key"
        )

    # Exactly one primary extractor; if several, prefer a stable order.
    primary = None
    for key in (
        "json_path",
        "json_len",
        "regex_ratio",
        "regex",
        "file_size_kb",
    ):
        if key in extract:
            primary = key
            break
    assert primary is not None

    if primary == "file_size_kb":
        return _extract_file_size_kb(path, side_label)

    if not path.is_file():
        return ExtractResult(
            None, reason=f"{side_label}: file not found: {file_rel}"
        )

    if primary == "json_path":
        return _extract_json_path(
            path, str(extract["json_path"]), extract.get("scale"), side_label
        )
    if primary == "json_len":
        return _extract_json_len(path, str(extract["json_len"]), side_label)
    if primary == "regex_ratio":
        return _extract_regex_ratio(
            path, str(extract["regex_ratio"]), side_label
        )
    if primary == "regex":
        return _extract_regex(path, str(extract["regex"]), side_label)

    return ExtractResult(None, reason=f"{side_label}: unknown extract")


def _extract_file_size_kb(path: Path, side_label: str) -> ExtractResult:
    if not path.is_file():
        return ExtractResult(
            None, reason=f"{side_label}: file not found: {path.name}"
        )
    try:
        size = os.path.getsize(str(path))
    except OSError as e:
        return ExtractResult(None, reason=f"{side_label}: cannot stat: {e}")
    return ExtractResult(round(size / 1024.0, 1))


def _load_json(path: Path, side_label: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"{side_label}: cannot read: {e}"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"{side_label}: bad json: {e}"


def _extract_json_path(
    path: Path,
    dotted: str,
    scale: Any,
    side_label: str,
) -> ExtractResult:
    data, err = _load_json(path, side_label)
    if err:
        return ExtractResult(None, reason=err)
    raw = _walk_json_path(data, dotted)
    num = _as_number(raw)
    if num is None:
        return ExtractResult(
            None,
            reason=f"{side_label}: json_path not numeric or missing: {dotted}",
        )
    if scale is not None and scale is not False:
        # Config error (exit 2): scale must be a finite number.
        snum = _require_finite_number(scale, f"{side_label}: scale")
        num = num * snum
        if not math.isfinite(num):
            return ExtractResult(
                None,
                reason=f"{side_label}: value not finite after scale",
            )
    return ExtractResult(num)


def _extract_json_len(path: Path, dotted: str, side_label: str) -> ExtractResult:
    data, err = _load_json(path, side_label)
    if err:
        return ExtractResult(None, reason=err)
    raw = _walk_json_path(data, dotted)
    if not isinstance(raw, list):
        return ExtractResult(
            None,
            reason=f"{side_label}: json_len target is not an array: {dotted}",
        )
    return ExtractResult(float(len(raw)))


def _extract_regex_ratio(
    path: Path, pattern: str, side_label: str
) -> ExtractResult:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return ExtractResult(None, reason=f"{side_label}: cannot read: {e}")
    try:
        cre = re.compile(pattern)
    except re.error as e:
        return ExtractResult(
            None, reason=f"{side_label}: invalid regex_ratio: {e}"
        )
    # Fail closed: require exactly one match (first-match fabricates numbers).
    matches = list(cre.finditer(text))
    if len(matches) == 0:
        return ExtractResult(
            None, reason=f"{side_label}: regex_ratio no match"
        )
    if len(matches) > 1:
        return ExtractResult(
            None,
            reason=(
                f"{side_label}: ambiguous: {len(matches)} matches "
                "— anchor the pattern"
            ),
        )
    m = matches[0]
    if m.lastindex is None or m.lastindex < 2:
        return ExtractResult(
            None, reason=f"{side_label}: regex_ratio no match"
        )
    try:
        n = int(m.group(1))
        denom = int(m.group(2))
    except (TypeError, ValueError):
        return ExtractResult(
            None, reason=f"{side_label}: regex_ratio non-integer captures"
        )
    if denom == 0:
        return ExtractResult(
            None,
            n=n,
            m=denom,
            reason=f"{side_label}: regex_ratio M==0",
        )
    ratio = float(n) / float(denom)
    if not math.isfinite(ratio):
        return ExtractResult(
            None,
            n=n,
            m=denom,
            reason=f"{side_label}: regex_ratio not finite",
        )
    return ExtractResult(ratio, n=n, m=denom)


def _extract_regex(path: Path, pattern: str, side_label: str) -> ExtractResult:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return ExtractResult(None, reason=f"{side_label}: cannot read: {e}")
    try:
        cre = re.compile(pattern)
    except re.error as e:
        return ExtractResult(None, reason=f"{side_label}: invalid regex: {e}")
    # Fail closed: require exactly one match (first-match fabricates numbers).
    matches = list(cre.finditer(text))
    if len(matches) == 0:
        return ExtractResult(None, reason=f"{side_label}: regex no match")
    if len(matches) > 1:
        return ExtractResult(
            None,
            reason=(
                f"{side_label}: ambiguous: {len(matches)} matches "
                "— anchor the pattern"
            ),
        )
    m = matches[0]
    if m.lastindex is None or m.lastindex < 1:
        return ExtractResult(
            None, reason=f"{side_label}: regex needs one capture group"
        )
    num = _as_number(m.group(1))
    if num is None:
        return ExtractResult(
            None, reason=f"{side_label}: regex capture not numeric"
        )
    return ExtractResult(num)


# ---------------------------------------------------------------------------
# metric compute
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    name: str
    direction: str
    unit: Optional[str]
    before: Optional[float]
    after: Optional[float]
    delta: Optional[float]
    pct_change: Optional[float]
    status: str
    n: Optional[int] = None
    m: Optional[int] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        def _json_num(v: Optional[float]) -> Optional[float]:
            if v is None:
                return None
            try:
                x = float(v)
            except (TypeError, ValueError):
                return None
            return x if math.isfinite(x) else None

        d: Dict[str, Any] = {
            "name": self.name,
            "direction": self.direction,
            "unit": self.unit,
            "before": _json_num(self.before),
            "after": _json_num(self.after),
            "delta": _json_num(self.delta),
            "pct_change": _json_num(self.pct_change),
            "status": self.status,
            "n": self.n,
            "reason": self.reason,
        }
        if self.m is not None:
            d["m"] = self.m
        return d


@dataclass
class WowReport:
    metrics: List[MetricResult] = field(default_factory=list)

    @property
    def improved(self) -> int:
        return sum(1 for m in self.metrics if m.status == STATUS_IMPROVED)

    @property
    def regressed(self) -> int:
        return sum(1 for m in self.metrics if m.status == STATUS_REGRESSED)

    @property
    def unchanged(self) -> int:
        return sum(1 for m in self.metrics if m.status == STATUS_UNCHANGED)

    @property
    def no_data(self) -> int:
        return sum(1 for m in self.metrics if m.status == STATUS_NO_DATA)

    @property
    def complete(self) -> bool:
        return self.no_data == 0

    @property
    def any_regression(self) -> bool:
        return self.regressed > 0

    def summary_dict(self) -> Dict[str, int]:
        return {
            "improved": self.improved,
            "regressed": self.regressed,
            "unchanged": self.unchanged,
            "no_data": self.no_data,
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": self.summary_dict(),
            "complete": self.complete,
            "any_regression": self.any_regression,
        }


def _pct_change(before: float, after: float) -> Optional[float]:
    if before == 0:
        return None
    return round(100.0 * (after - before) / abs(before), 1)


def compute_status(
    before: float,
    after: float,
    direction: str,
    tolerance: float,
) -> str:
    """Classify improved / regressed / unchanged using direction + tolerance.

    Tolerance is symmetric: movement within ±tolerance is UNCHANGED.
    Only clearing the band counts as IMPROVED or REGRESSED.
    """
    if direction == "higher_better":
        if after > before + tolerance:
            return STATUS_IMPROVED
        if after < before - tolerance:
            return STATUS_REGRESSED
        return STATUS_UNCHANGED
    # lower_better
    if after < before - tolerance:
        return STATUS_IMPROVED
    if after > before + tolerance:
        return STATUS_REGRESSED
    return STATUS_UNCHANGED


def evaluate_metric(
    root: Path,
    metric_cfg: Dict[str, Any],
    default_tolerance: float,
) -> MetricResult:
    """Evaluate one metric config against files under root."""
    name = str(metric_cfg.get("name") or "").strip()
    if not name:
        raise ValueError("metric missing name")

    direction = str(metric_cfg.get("direction") or "").strip()
    if direction not in DIRECTIONS:
        raise ValueError(
            f"metric {name}: direction must be higher_better|lower_better "
            f"(got {direction!r})"
        )

    unit = metric_cfg.get("unit")
    if unit is not None:
        unit = str(unit)

    tol_raw = metric_cfg.get("tolerance", default_tolerance)
    tolerance = _require_finite_number(
        tol_raw, f"metric {name}: tolerance"
    )
    if tolerance < 0:
        raise ValueError(f"metric {name}: tolerance must be >= 0")

    before_cfg = metric_cfg.get("before")
    after_cfg = metric_cfg.get("after")
    _require_file_side(name, "before", before_cfg)
    _require_file_side(name, "after", after_cfg)
    assert isinstance(before_cfg, dict) and isinstance(after_cfg, dict)

    before_ex = extract_from_side(root, before_cfg, "before")
    after_ex = extract_from_side(root, after_cfg, "after")

    # Prefer after's n/m for ratio display; fall back to before.
    n = after_ex.n if after_ex.n is not None else before_ex.n
    m = after_ex.m if after_ex.m is not None else before_ex.m

    # Belt-and-suspenders: never accept non-finite extracted values.
    before_v = _as_number(before_ex.value) if before_ex.value is not None else None
    after_v = _as_number(after_ex.value) if after_ex.value is not None else None

    if before_v is None or after_v is None:
        reasons: List[str] = []
        if before_v is None:
            reasons.append(before_ex.reason or "before: no data")
        if after_v is None:
            reasons.append(after_ex.reason or "after: no data")
        return MetricResult(
            name=name,
            direction=direction,
            unit=unit,
            before=before_v,
            after=after_v,
            delta=None,
            pct_change=None,
            status=STATUS_NO_DATA,
            n=n,
            m=m,
            reason="; ".join(reasons),
        )

    delta = after_v - before_v
    if not math.isfinite(delta):
        return MetricResult(
            name=name,
            direction=direction,
            unit=unit,
            before=before_v,
            after=after_v,
            delta=None,
            pct_change=None,
            status=STATUS_NO_DATA,
            n=n,
            m=m,
            reason="delta not finite",
        )
    status = compute_status(before_v, after_v, direction, tolerance)
    pct = _pct_change(before_v, after_v)
    if pct is not None and not math.isfinite(pct):
        pct = None
    return MetricResult(
        name=name,
        direction=direction,
        unit=unit,
        before=before_v,
        after=after_v,
        delta=delta,
        pct_change=pct,
        status=status,
        n=n,
        m=m,
        reason=None,
    )


def _require_file_side(metric_name: str, side_name: str, side: Any) -> None:
    """Config error if before/after is not a pure file-backed extract block.

    Numbers must come only from files. An inline ``value`` (even alongside
    ``file``) is rejected so nobody can smuggle a hand-typed number.
    """
    msg = (
        f"metric {metric_name} must read from a file, not an inline value"
    )
    if not isinstance(side, dict):
        raise ValueError(msg)
    file_rel = side.get("file")
    if file_rel is None or str(file_rel).strip() == "":
        raise ValueError(msg)
    extract = side.get("extract")
    if not isinstance(extract, dict) or not extract:
        raise ValueError(
            f"metric {metric_name} {side_name}: missing extract"
        )
    # Refuse any inline numeric override — value must be file-derived only.
    if "value" in side:
        raise ValueError(
            f"metric {metric_name} {side_name}: inline value not allowed "
            "— read from file"
        )


# ---------------------------------------------------------------------------
# config load / orchestration
# ---------------------------------------------------------------------------


def resolve_config_path(root: Path, config_arg: Optional[str]) -> Path:
    if config_arg:
        p = Path(config_arg)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p
    return root / DEFAULT_CONFIG_REL


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"config not found: {path}")
    try:
        data = load_yaml_file(path)
    except MiniYamlError as e:
        raise ValueError(f"config parse error {path}: {e}") from e
    except OSError as e:
        raise ValueError(f"cannot read config {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def build_report(root: Path, cfg: Dict[str, Any]) -> WowReport:
    """Build full wow report from config; counts derived live from metrics."""
    tol_raw = cfg.get("tolerance", DEFAULT_TOLERANCE)
    default_tolerance = _require_finite_number(tol_raw, "tolerance")
    if default_tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    metrics_cfg = cfg.get("metrics")
    if metrics_cfg is None:
        raise ValueError("config missing metrics list")
    if not isinstance(metrics_cfg, list):
        raise ValueError("metrics must be a list")
    if not metrics_cfg:
        raise ValueError("metrics list is empty")

    report = WowReport()
    for i, raw in enumerate(metrics_cfg):
        if not isinstance(raw, dict):
            raise ValueError(f"metrics[{i}] must be a mapping")
        report.metrics.append(
            evaluate_metric(root, raw, default_tolerance)
        )
    return report


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _fmt_num(v: Optional[float]) -> str:
    """Format a metric number for humans; never emit inf/nan (crash or lie)."""
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    if x == int(x) and abs(x) < 1e12:
        return str(int(x))
    # Prefer compact float display (safe: x is finite)
    return f"{x:.6g}"


def format_human(report: WowReport) -> str:
    lines: List[str] = []
    header = "metric | before | after | Δ | %change | status"
    lines.append(header)
    lines.append("-" * len(header))
    for m in report.metrics:
        delta_s = _fmt_num(m.delta)
        pct_s = "—" if m.pct_change is None else f"{m.pct_change}"
        lines.append(
            f"{m.name} | {_fmt_num(m.before)} | {_fmt_num(m.after)} | "
            f"{delta_s} | {pct_s} | {m.status}"
        )
    # Live summary counts (recomputed from metric statuses)
    lines.append(
        f"wow: {report.improved} improved · {report.regressed} regressed · "
        f"{report.unchanged} unchanged · {report.no_data} no-data"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wow_report.py",
        description=(
            "Before/after wow metrics table — values read only from result files "
            "(never inline)."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to wow.yaml (default: <root>/.work/wow.yaml)",
    )
    p.add_argument(
        "--root",
        default=None,
        help="Project root for resolving metric file paths (default: cwd)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report",
    )
    p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 if any metric REGRESSED beyond its tolerance",
    )
    p.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit 1 if any metric is NO_DATA (missing/unparseable source)",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def run(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: returns exit code 0 / 1 / 2."""
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        code = e.code
        if code is None or code == 0:
            return EXIT_OK
        return EXIT_ERR

    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return EXIT_ERR

    cfg_path = resolve_config_path(root, args.config)
    try:
        if not cfg_path.is_file():
            print(
                f"error: config not found (pass --config or create {cfg_path})",
                file=sys.stderr,
            )
            return EXIT_ERR
        cfg = load_config(cfg_path)
        report = build_report(root, cfg)
    except (ValueError, MiniYamlError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERR

    if args.json:
        print(json.dumps(report.to_json_dict(), indent=2, sort_keys=False))
    else:
        print(format_human(report))

    if args.fail_on_regression and report.any_regression:
        return EXIT_FAIL
    if args.require_complete and not report.complete:
        return EXIT_FAIL
    return EXIT_OK


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
