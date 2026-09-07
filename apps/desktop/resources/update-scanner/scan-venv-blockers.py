"""Candidate-owned target-root scanner for safe native-Windows updates.

GENERATED FILE -- do not edit. Regenerate with::

    python scripts/desktop-update/sync_scan_venv_blockers.py --write

Source: ``hermes_cli/_scan_venv_blockers.py``.  Parity is enforced by
``tests/hermes_cli/test_scan_venv_blockers_parity.py``.

This is ``hermes_cli/_scan_venv_blockers.py`` as shipped inside the Desktop
build (``extraResources``). The Desktop runs it with ``python -I <this file>
--root <install>`` under the target venv interpreter so the scan logic comes
from the candidate build, never from the mutable checkout being updated; it
therefore imports nothing from that checkout. Exactly six deliberate
substitutions separate it from the module copy:

* this docstring;
* the MCP argv classifier is inlined instead of imported from
  ``hermes_mcp_update_gate``;
* ``_validated_root``'s docstring drops the provenance wording;
* ``_validated_root`` does not require the scanner to live inside the target
  root (the carrier lives in the Desktop's resources);
* ``_updater_owned_backend_entry`` never consults the checkout's spawn ledger,
  so no serve/dashboard backend is deferred and ``deferred_backend_evidence``
  is always empty;
* ``venv_bin_dir`` is not imported from ``hermes_constants``; the Windows
  ``Scripts`` layout is spelled out instead.

Everything else is byte-identical to the module copy. In particular
``_redact_sensitive_cmdline`` is *not* substituted: it is self-contained in
both copies, so the packaged scanner reports the same diagnostic command line
the module does rather than a blanket ``"<redacted>"``.

The external interface always writes exactly one JSON document to stdout.
Valid clear and blocked scans exit zero. Invalid roots and probe failures exit
one and are fail-closed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Mapping, NoReturn, Sequence

MCP_MAIN_MODULE = "agent.transports.hermes_tools_mcp_server"


def is_exact_mcp_module_argv(argv: Sequence[str]) -> bool:
    """True only for an argument-free ``-m <MCP_MAIN_MODULE>`` launch."""
    parts = [str(part) for part in argv]
    indexes = [index for index, token in enumerate(parts) if token == "-m"]
    if not (
        parts
        and len(indexes) == 1
        and indexes[0] + 2 == len(parts)
        and parts[indexes[0] + 1] == MCP_MAIN_MODULE
    ):
        return False

    # ``-m`` must be the interpreter's operative action, not text after a
    # script name.  Accept Python's ordinary pre-module runtime switches, but
    # reject ``-c``, ``--``, unknown switches, and non-option operands.  This
    # is intentionally smaller than the complete Python CLI grammar: failure
    # to recognize an exotic launch only makes an updater refuse to kill it.
    before_module = parts[1 : indexes[0]]
    argument_options = {"-W", "-X", "--check-hash-based-pycs"}
    long_flags = {
        "--bytes-warning",
        "--debug",
        "--dont-write-bytecode",
        "--help-env",
        "--help-xoptions",
        "--ignore-environment",
        "--inspect",
        "--isolated",
        "--no-site",
        "--no-user-site",
        "--safe-path",
        "--unbuffered",
        "--verbose",
    }
    short_flag_chars = frozenset("bBdEIiOPqRsStuUvx")
    index = 0
    while index < len(before_module):
        token = before_module[index]
        if token in argument_options:
            index += 1
            if index >= len(before_module) or before_module[index].startswith("-"):
                return False
        elif token.startswith("-W") and token != "-W":
            pass
        elif token.startswith("-X") and token != "-X":
            pass
        elif token.startswith("--check-hash-based-pycs="):
            pass
        elif token in long_flags:
            pass
        elif token.startswith("-") and not token.startswith("--"):
            if not token[1:] or any(char not in short_flag_chars for char in token[1:]):
                return False
        else:
            return False
        index += 1
    return True

SCHEMA_VERSION = 2
_CREATE_TIME_TOLERANCE_SECONDS = 0.01

# Long CLI flags whose argument value must be redacted from the cmdline. Short flags (-t, -k, -p)
# are intentionally not redacted — ambiguous and useful diagnostics (toolset, port, profile).
_SENSITIVE_LONG_FLAGS: list[str] = [
    "--token", "--api-key", "--password", "--secret", "--authorization", "--access-key",
    "--private-key", "--session-key",
]

# Secret masking for the diagnostic command line, deliberately self-contained.
#
# Neither scanner copy may import the target checkout's ``agent.redact``: the
# carrier runs from the candidate build against a mutable install root, and
# reaching into that root for redaction code would defeat shipping the scanner
# as a resource at all. These rules are a small, auditable subset of
# ``agent/redact.py`` covering what a Windows command line can carry, and they
# are byte-identical in both copies so the packaged carrier reports the same
# command line the module does instead of a blanket ``<redacted>`` (#104687).
_SECRET_TOKEN_RE = re.compile(
    r"(?:sk-ant-|sk-|sk_live_|sk_test_|rk_live_|gh[opsur]_|github_pat_|glpat-|gsk_"
    r"|xox[abprs]-|xapp-|AIza|AKIA|SG\.|hf_|r8_|npm_|pypi-|xai-|tvly-|exa_|ntn_|gAAAA)"
    r"[A-Za-z0-9._-]{10,}"
)
# ``--api-key=V`` / ``AUTH_TOKEN=V`` / ``password: V``. The secret word must sit
# on an identifier boundary so ``monkey=`` and ``keyboard=`` do not match.
_SECRET_ASSIGN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"((?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|key|token|secret|password|passwd|credential|auth)"
    r"(?![A-Za-z0-9]))(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)
_URL_CREDENTIALS_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^\s/:@]+:[^\s/@]+@")
_AUTH_SCHEME_RE = re.compile(r"(?<![A-Za-z0-9])(Bearer|Basic)\s+\S+", re.IGNORECASE)

_PYTHON_PROCESS_NAMES = {"python.exe", "pythonw.exe", "python", "pythonw"}
_UPDATER_STOPPABLE_PURPOSES = ("serve", "dashboard")
# Process names the argv/cwd holder rungs may nominate. The exe-root rungs
# (an executable under venv\ or .hermes-runtime\) need no name gate; a
# process merely mentioning the venv path on its command line is a holder
# only when it is an interpreter or launcher that can have mapped the venv.
_HOLDER_PROCESS_NAME_PREFIXES = ("python", "pypy")
_HOLDER_PROCESS_NAMES = frozenset({"uv.exe", "uvx.exe", "hermes.exe", "uv", "uvx", "hermes"})
_HERMES_SHIM_BASENAMES = frozenset({"hermes", "hermes.exe"})


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


class _ProcessGenerationChanged(RuntimeError):
    """Raised when one PID names different processes during identity capture."""


@dataclass(frozen=True)
class _ProcessSnapshot:
    pid: int
    ppid: int
    name: str
    exe: str
    argv: tuple[str, ...]
    created_at: float
    process: Any


def _canonical(path: str | Path) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _within(path: str | Path, directory: str | Path) -> bool:
    try:
        return os.path.commonpath((_canonical(path), _canonical(directory))) == _canonical(
            directory
        )
    except (OSError, ValueError):
        return False


def _venv_dir(root: Path) -> Path:
    primary = root / "venv"
    if primary.exists() or not (root / ".venv").exists():
        return primary
    return root / ".venv"


def _validated_root(value: str | Path) -> tuple[Path, Path]:
    """Validate the explicit target root and target-venv provenance."""
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError("--root must be an absolute path")
    root = Path(_canonical(supplied))
    if not root.is_dir() or not (root / "hermes_cli").is_dir():
        raise ValueError("--root is not a Hermes installation")
    venv = _venv_dir(root)
    if not venv.is_dir():
        raise ValueError("target root has no venv")
    if _canonical(sys.prefix) != _canonical(venv):
        raise ValueError("scanner is not running from the target root venv")
    return root, venv


def _base_result(
    *,
    root: str = "",
    venv: str = "",
    ok: bool = False,
    blocked: bool = True,
    reason: str | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "scan",
        "ok": ok,
        "ready": bool(ok and not blocked),
        "blocked": blocked,
        "reason": reason,
        "root": root,
        "venv": venv,
        "processes": [],
        "mcp_bridges": [],
        "desktop_plugin_services": [],
        "pausable_gateways": 0,
        "pausable_gateway_processes": [],
        "deferred_backends": 0,
        "deferred_backend_evidence": [],
        "error": error,
    }


def _probe_fail_json(
    *, root: str = "", venv: str = "", code: str = "probe_failed", message: str = ""
) -> str:
    """Return the stable fail-closed probe envelope."""
    return json.dumps(
        _base_result(
            root=root,
            venv=venv,
            reason=code,
            error={"code": code, "message": message},
        )
    )


def _emit_probe_fail(
    diagnostic: str,
    *,
    root: str = "",
    venv: str = "",
    code: str = "probe_failed",
) -> NoReturn:
    print(_probe_fail_json(root=root, venv=venv, code=code, message=diagnostic))
    print(diagnostic, file=sys.stderr)
    raise SystemExit(1)


def _find_flag(text: str, flag: str) -> int:
    """Index of *flag* (case-insensitive) at string start or after a space; -1 if absent."""
    low, fl, pos = text.lower(), flag.lower(), 0
    while True:
        idx = low.find(fl, pos)
        if idx == -1 or idx == 0 or text[idx - 1] == " ":
            return idx
        pos = idx + 1


def _mask_secret_values(text: str) -> str:
    """Replace secret-shaped substrings of a command line with ``<redacted>``."""
    text = _URL_CREDENTIALS_RE.sub(lambda match: match.group(1) + "<redacted>@", text)
    text = _SECRET_ASSIGN_RE.sub(
        lambda match: match.group(1) + match.group(2) + "<redacted>", text
    )
    text = _AUTH_SCHEME_RE.sub(lambda match: match.group(1) + " <redacted>", text)
    return _SECRET_TOKEN_RE.sub("<redacted>", text)


def _redact_sensitive_cmdline(cmdline: str) -> str:
    """Mask secret-shaped values, then replace everything after a sensitive long flag.

    Self-contained on purpose (see ``_SECRET_TOKEN_RE``): this function is one
    of the regions the carrier does *not* substitute, so the packaged scanner
    and this module redact identically.
    """
    try:
        cmdline = _mask_secret_values(cmdline)
    except Exception:
        return "<redacted>"
    # --flag=value → preserve "--flag="; --flag value → preserve "--flag ".
    earliest = len(cmdline)
    for flag in _SENSITIVE_LONG_FLAGS:
        for suffix in ("=", " "):
            idx = _find_flag(cmdline, flag + suffix)
            if idx != -1 and idx + len(flag) + 1 < earliest:
                earliest = idx + len(flag) + 1
    if earliest < len(cmdline):
        return cmdline[:earliest] + "<redacted>"
    return cmdline


def _classify_local_preview_args(args: object) -> dict[str, object]:
    """Safe UI metadata for an exact ``python -m http.server`` argv; ``{}`` otherwise.

    The general holder detector truncates its diagnostic command line; reading argv separately
    preserves a useful directory label without exposing an unbounded command line to the renderer.
    """
    if not isinstance(args, (list, tuple)) or not all(isinstance(arg, str) for arg in args):
        return {}
    # ``-m`` must be the first argument after the executable: a later ``-m http.server`` can be
    # data passed to an unrelated script and must never authorize termination.
    if len(args) < 3 or args[1] != "-m" or args[2].lower() != "http.server":
        return {}
    port = 8000
    if len(args) > 3 and args[3].isdigit() and 0 < int(args[3]) <= 65535:
        port = int(args[3])
    label = ""
    try:
        directory_index = args.index("--directory")
        if directory_index + 1 < len(args):
            label = PureWindowsPath(args[directory_index + 1]).name
    except ValueError:
        pass
    metadata: dict[str, object] = {"kind": "local-preview", "safeToStop": True, "port": port}
    if label:
        metadata["label"] = label
    return metadata


def _local_preview_metadata(pid: int, name: str) -> dict[str, object]:
    if name.lower() not in _PYTHON_PROCESS_NAMES:
        return {}
    try:
        import psutil  # noqa: PLC0415

        process = psutil.Process(pid)
        metadata = _classify_local_preview_args(process.cmdline())
        if metadata:
            metadata["createTime"] = process.create_time()
        return metadata
    except Exception:
        return {}


def _executable_within(process: object, venv_dir: Path) -> bool:
    """True only when *process* runs an executable inside *venv_dir*.

    Unreadable identity is not containment: an ``AccessDenied`` executable
    belongs to something this scanner may not touch.
    """
    try:
        executable = process.exe()  # type: ignore[attr-defined]
    except Exception:
        return False
    return bool(executable) and _within(executable, venv_dir)


def _terminate_safe_preview(
    pid: int,
    expected_create_time: float,
    venv_dir: Path,
    *,
    psutil_module: object | None = None,
) -> tuple[bool, str | None]:
    """Terminate one verified local preview process inside *venv_dir*.

    A fresh ``psutil.Process`` identity check, exact argv classification, and a
    root-containment check on the executable all occur immediately before
    termination; psutil guards mutating Process methods against PID reuse (no
    taskkill stale-PID race).

    The child tree is included only for children that independently pass the
    same containment check. A preview server that spawned something outside the
    target venv is not authorization to kill that something (#104687 H10), so a
    foreign child downgrades this to a single-PID termination.
    """
    if int(pid) <= 0 or not math.isfinite(expected_create_time) or expected_create_time <= 0:
        return False, "invalid process identity"
    try:
        if psutil_module is None:
            import psutil as psutil_module  # type: ignore[no-redef]  # noqa: PLC0415

        process = psutil_module.Process(pid)  # type: ignore[attr-defined]
        if abs(process.create_time() - expected_create_time) > 0.001:
            return False, "process identity changed"
        if not _classify_local_preview_args(process.cmdline()):
            return False, "process is no longer a local preview"
        if not _executable_within(process, venv_dir):
            return False, "process is not running from the target root venv"
        try:
            children = list(process.children(recursive=True))
        except Exception:
            children = []
        contained = [child for child in children if _executable_within(child, venv_dir)]
        targets = (
            [*reversed(contained), process] if len(contained) == len(children) else [process]
        )
        for target in targets:
            target.terminate()
        _gone, alive = psutil_module.wait_procs(targets, timeout=3)  # type: ignore[attr-defined]
        for target in alive:
            target.kill()
        if alive:
            psutil_module.wait_procs(alive, timeout=2)  # type: ignore[attr-defined]
        return True, None
    except Exception as exc:
        return False, f"termination failed: {type(exc).__name__}"


def _tokens(argv: str | Sequence[str]) -> list[str]:
    if not isinstance(argv, str):
        return [str(value).strip('"') for value in argv]
    try:
        return [value.strip('"') for value in shlex.split(argv, posix=False)]
    except ValueError:
        return []


def _process_basename(value: object) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _hermes_cli_tail(argv: str | Sequence[str]) -> list[str] | None:
    """Return the operative ``hermes_cli.main`` argv without global options.

    Two launch shapes are recognized: ``python [runtime switches] -m
    hermes_cli.main <tail>`` and the console shim ``hermes[.exe] <tail>``
    (``gateway/status.py`` accepts the same shim shape for ``gateway run``).
    """
    parts = _tokens(argv)
    if parts and _process_basename(parts[0]).casefold() in _HERMES_SHIM_BASENAMES:
        module_index = -1
    else:
        module_indexes = [
            index
            for index in range(1, len(parts) - 1)
            if parts[index] == "-m" and parts[index + 1] == "hermes_cli.main"
        ]
        if len(module_indexes) != 1:
            return None
        module_index = module_indexes[0]
    # A non-option before ``-m`` is a script operand unless it is the value of
    # a recognized Python runtime switch. Keep this parser deliberately
    # conservative: an unrecognized launch is a hard blocker, never killable.
    prefix = parts[1:module_index] if module_index > 0 else []
    index = 0
    while index < len(prefix):
        token = prefix[index]
        if token in {"-X", "-W", "--check-hash-based-pycs"}:
            index += 1
            if index >= len(prefix) or prefix[index].startswith("-"):
                return None
        elif token.startswith(("-X", "-W")) and token not in {"-X", "-W"}:
            pass
        elif token.startswith("--check-hash-based-pycs="):
            pass
        elif token.startswith("-") and not token.startswith("--"):
            if not token[1:] or any(char not in "bBdEIiOPqRsStuUvx" for char in token[1:]):
                return None
        else:
            return None
        index += 1
    tail = parts[module_index + 2 :]
    index = 0
    while index < len(tail):
        token = tail[index]
        if token in {"--profile", "-p"}:
            if index + 1 >= len(tail) or tail[index + 1].startswith("-"):
                return None
            index += 2
            continue
        if token.startswith("--profile="):
            if token == "--profile=":
                return None
            index += 1
            continue
        break
    return tail[index:]


def _hermes_cli_command(argv: str | Sequence[str]) -> str | None:
    """Return the exact Hermes subcommand token, never a substring match."""
    tail = _hermes_cli_tail(argv)
    if not tail or tail[0].startswith("-"):
        return None
    return tail[0].lower()


def _gateway_subcommand(argv: str | Sequence[str]) -> str | None:
    """Hermes gateway lifecycle subcommand from a command line, or ``None``.

    A stdlib port of ``gateway.status._gateway_command_subcommand``, kept
    byte-identical in both scanner copies. It cannot simply delegate: the
    carrier runs with ``python -I`` under the target venv against the install
    root it is about to rewrite, so importing ``gateway.status`` out of that
    checkout is exactly what shipping the scanner as a resource exists to
    prevent — the same constraint that governs ``_redact_sensitive_cmdline``.
    Inlining keeps one implementation for both copies instead of a
    generator substitution that could drift silently;
    ``test_gateway_subcommand_matches_the_canonical_parser`` pins this port
    against the canonical parser over a shared corpus so a change to
    ``gateway/status.py`` fails here.

    No loose substring matching (``"gateway" in cmdline`` also matches
    ``gateway status`` and ``python -m tui_gateway``): a Hermes entrypoint
    plus the ``gateway`` subcommand, or a gateway-dedicated entrypoint.
    Tokenizing is quote-aware for Windows paths with spaces, and
    ``--profile``/``-p`` selectors are stripped anywhere in argv because
    ``_apply_profile_override`` removes them before argparse — a hand-rolled
    token scan regressed ``--profile gateway gateway run``, where the profile
    *value* shadowed the subcommand token.
    """
    if isinstance(argv, str):
        if not argv:
            return None
        try:
            raw_tokens = shlex.split(argv, posix=False)
        except ValueError:
            raw_tokens = argv.split()
    else:
        raw_tokens = [str(part) for part in argv]
    tokens = [token.strip("\"'").replace("\\", "/").lower() for token in raw_tokens]
    if not tokens:
        return None
    basenames = [token.rsplit("/", 1)[-1] for token in tokens]
    # Gateway-dedicated entrypoints carry no subcommand to inspect.
    if any(token == "gateway/run.py" or token.endswith("/gateway/run.py") for token in tokens):
        return "run"
    if any(base in ("hermes-gateway", "hermes-gateway.exe") for base in basenames):
        return "run"
    joined = " ".join(tokens)
    if (
        "hermes_cli.main" not in joined
        and "hermes_cli/main.py" not in joined
        and not any(base in ("hermes", "hermes.exe") for base in basenames)
    ):
        return None
    # Drop --profile X / -p X / --profile=X / -p=X (consumes a VALUE of "gateway" too).
    filtered: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
        elif token in ("--profile", "-p"):
            skip_next = True
        elif not token.startswith(("--profile=", "-p=")):
            filtered.append(token)
    for index, token in enumerate(filtered):
        if token == "gateway":
            # Bare ``hermes gateway`` defaults to ``run``.
            return filtered[index + 1] if index + 1 < len(filtered) else "run"
    return None


def _is_pausable_gateway(argv: str | Sequence[str]) -> bool:
    """True for a gateway runtime the updater's pause machinery stops.

    A running gateway shows up in the venv-holder scan as one or both halves
    of its launcher/worker chain (``venv\\Scripts\\python.exe -m
    hermes_cli.main gateway run`` and the uv-side interpreter re-running the
    same argv). Reporting those as blockers dead-ends the Desktop update:
    the preflight aborts with ``venv-blocked`` *before* spawning
    ``hermes-setup``, so the CLI updater's own
    ``_pause_windows_gateways_for_update()`` — which exists precisely to
    stop these processes (and is always active: ``hermes-setup`` invokes
    ``hermes update --yes --gateway``) — never gets the chance to run.

    This exemption must therefore cover exactly what that pause path stops,
    never less. Pause discovery is ``hermes_cli.gateway.find_gateway_pids``
    → ``_scan_gateway_pids(include_restart_managers=not
    supports_systemd_services())``, which on Windows is always ``True`` and
    accepts ``looks_like_gateway_runtime_command_line`` — the ``run`` OR
    ``restart`` shapes, including ``hermes-gateway.exe``, ``gateway/run.py``
    and bare ``hermes gateway``. A hand-rolled ``hermes_cli.main``-tail
    parser matched none of those, so a launcher-started gateway was reported
    as a hard blocker that the hand-off could not clear.
    ``update_cmd._classify_concurrent_instance`` already uses the same
    runtime matcher.

    Only gateway runtimes are exempted. Anything else running from the venv
    (an operator's REPL, a stray script, a ``serve`` backend that survived
    the desktop's own teardown, or a gateway *management* subcommand such as
    ``gateway status``/``gateway stop``) has no pause machinery downstream
    and must keep blocking the handoff.
    """
    return _gateway_subcommand(argv) in {"run", "restart"}


def _is_updater_owned_backend(pid: int, cmdline: str) -> bool:
    """True when *pid* is a Hermes backend the CLI updater can stop (positive ledger identity).

    The gateway exemption above keeps ``gateway run`` holders out of the blocker list because the updater's
    own pause machinery stops and resumes them. ``hermes serve`` / ``hermes dashboard`` backends had no such
    deferral, so a leaked serve child (or a Desktop-owned backend the teardown lost track of) dead-ended the
    hand-off with ``venv-blocked`` — or, worse, survived the hand-off and made the shim quarantine fail with
    ``os error 32`` (#98336) — even though the updater downstream owns exactly this case with its ledger
    rungs (`_ledger_reapable_backend_pids` reaps dead-spawner orphans; `_ledger_manual_serve_holders` stops
    manual serves and relaunches them on their recorded host/port).
    Positive identity only — never name/substring matching (#90778, and the 99558 identity-guard contract):
    """
    return _updater_owned_backend_entry(pid, cmdline) is not None


def _updater_owned_backend_entry(pid: int, cmdline: str) -> dict | None:
    """Candidate scan cannot trust a mutable target checkout's ownership ledger."""
    return None


def _deferred_backend_evidence(entries: list[dict]) -> list[dict]:
    """Sanitized evidence (pid, purpose, recorded port — never argv) for deferred backends.

    Structured ledger fields only — pid, purpose, recorded port — never the command line, which can carry
    tokens or private endpoints. Lets the scan result explain *why* a holder disappeared from ``processes``
    without echoing argv (#98350).
    """
    return [{"pid": entry.get("pid"), "purpose": entry.get("purpose"), "port": entry.get("port")}
            for entry in entries if isinstance(entry.get("pid"), int)]


def _spawner_is_this_handoff_desktop(entry: dict) -> bool:
    """True when the entry's live spawner is an ancestor of this scan.

    The scan is spawned by the Desktop app's update preflight, so the Desktop performing the
    hand-off is in our ancestor chain. Identity is ``(pid, create_time)`` — a recycled PID cannot
    forge the pair.
    """
    spawner_pid = entry.get("spawner_pid")
    if not isinstance(spawner_pid, int) or spawner_pid <= 0:
        return False
    try:
        import psutil  # noqa: PLC0415

        for ancestor in psutil.Process().parents():
            if ancestor.pid != spawner_pid:
                continue
            expected = entry.get("spawner_create")
            if expected is None:
                return True
            return abs(float(ancestor.create_time()) - float(expected)) < 2.0
    except Exception:
        return False
    return False


def _snapshot_for_pid(
    pid: int,
    *,
    parent_by_pid: Mapping[int, int] | None = None,
) -> _ProcessSnapshot | None:
    import psutil  # noqa: PLC0415

    try:
        process = psutil.Process(int(pid))
        created_at = float(process.create_time())
        argv_value = process.cmdline()
        exe = process.exe()
        ppid = int(
            process.ppid()
            if parent_by_pid is None
            else parent_by_pid.get(int(pid), 0)
        )
        name = str(process.name() or Path(str(exe)).name)
    except psutil.NoSuchProcess:
        return None
    if not isinstance(argv_value, (list, tuple)) or not isinstance(exe, str):
        raise RuntimeError(f"process {pid} returned invalid identity metadata")
    argv = tuple(str(value) for value in argv_value)
    if not exe or not argv or not math.isfinite(created_at) or created_at <= 0:
        raise RuntimeError(f"process {pid} returned incomplete identity metadata")
    try:
        same_generation = _process_generation_matches(pid, created_at)
    except psutil.NoSuchProcess:
        return None
    if not same_generation:
        raise _ProcessGenerationChanged(
            f"process {pid} changed generation during identity refresh"
        )
    return _ProcessSnapshot(
        pid=int(pid),
        ppid=ppid,
        name=name,
        exe=exe,
        argv=argv,
        created_at=created_at,
        process=process,
    )


def _process_generation_matches(pid: int, created_at: float) -> bool:
    """Compare against an uncached Process so PID reuse cannot splice metadata."""
    import psutil  # noqa: PLC0415

    live_created_at = float(psutil.Process(int(pid)).create_time())
    if not math.isfinite(live_created_at) or live_created_at <= 0:
        raise RuntimeError(f"process {pid} returned incomplete identity metadata")
    return live_created_at == created_at


# A loaded extension module or DLL is the strongest mutation-set evidence a
# scan can offer: the file is mapped, so Restart Manager attributes it to this
# process and the update is going to rewrite it. Prefer one over an ordinary
# data file when both are mapped.
_MUTATION_SET_RESOURCE_SUFFIXES = (".pyd", ".dll")


def _mapped_mutation_set_path(
    process: Any,
    *,
    venv: Path,
    runtime_dir: Path,
) -> str | None:
    """One file under *venv* this process currently maps, or ``None``.

    Desktop's force-release re-proves ownership of a named file before it
    terminates anything, and under the shared managed runtime that proof is
    *required*: ``.hermes-runtime`` is deliberately outside the update's
    mutation set because unrelated uv tool venvs borrow the managed
    interpreter, so "image under the install root" alone must not authorize a
    kill (``TERMINATION_SHARED_RUNTIME_WITHOUT_MUTATION_PROOF``). Restart
    Manager supplies that file when it can see the holder; a scanner-only
    holder had nothing to offer and was refused.

    The venv *is* the mutation set, so any mapped file under it is a valid
    claim, and a file under ``.hermes-runtime`` never is (the prefix test
    already excludes it; the exact re-proof below also covers a venv that
    junctions into the runtime). Never raises: the resource is optional
    evidence and an unreadable module list must not fail a scan.
    """
    venv_prefix = os.path.normcase(str(venv)).rstrip(os.sep) + os.sep
    runtime_prefix = os.path.normcase(str(runtime_dir)).rstrip(os.sep) + os.sep
    best: tuple[int, str, str] | None = None
    try:
        for entry in process.memory_maps():
            raw = getattr(entry, "path", None)
            if not isinstance(raw, str) or not raw:
                continue
            key = os.path.normcase(raw)
            if not key.startswith(venv_prefix) or key.startswith(runtime_prefix):
                continue
            rank = 0 if key.endswith(_MUTATION_SET_RESOURCE_SUFFIXES) else 1
            candidate = (rank, key, raw)
            if best is None or candidate < best:
                best = candidate
    except Exception:
        return None
    if best is None:
        return None
    resource = best[2]
    # One exact re-proof on the single winner, so a junction or a substituted
    # drive cannot smuggle a shared-runtime file in behind a venv-shaped path.
    if not _within(resource, venv) or _within(resource, runtime_dir):
        return None
    return resource


def _mcp_role(
    snapshot: _ProcessSnapshot,
    root: Path,
    wrappers: set[int],
    snapshots: dict[int, _ProcessSnapshot],
) -> tuple[str, int | None] | None:
    if not is_exact_mcp_module_argv(snapshot.argv):
        return None
    venv = _venv_dir(root)
    managed = root / ".hermes-runtime" / "python"
    if _within(snapshot.exe, venv):
        return "mcp_bridge_wrapper", snapshot.pid
    if _within(snapshot.exe, managed):
        parent = snapshot.ppid if snapshot.ppid in wrappers else None
        return "mcp_bridge_worker", parent

    # A uv/base-Python trampoline is actionable only while its relationship
    # to a verified target-venv wrapper remains live and visible.
    ancestor = snapshot.ppid
    seen: set[int] = set()
    while ancestor and ancestor not in seen:
        if ancestor in wrappers:
            return "mcp_bridge_worker", ancestor
        seen.add(ancestor)
        parent = snapshots.get(ancestor)
        if parent is None:
            break
        ancestor = parent.ppid
    return None


def _mcp_record(
    snapshot: _ProcessSnapshot,
    *,
    role: str,
    wrapper_pid: int | None,
    owner: str | None = None,
    parent_by_pid: Mapping[int, int] | None = None,
    resource: str | None = None,
) -> dict[str, Any]:
    if owner is None:
        owner = (
            _owner_from_ancestry(snapshot)
            if parent_by_pid is None
            else _owner_from_ancestry(
                snapshot,
                parent_by_pid=parent_by_pid,
            )
        )
    actionable = owner in {"codex", "claude"}
    record: dict[str, Any] = {
        "pid": snapshot.pid,
        "name": snapshot.name,
        "cmdline": _redact_sensitive_cmdline(" ".join(snapshot.argv))[:120],
        "created_at": snapshot.created_at,
        "owner": owner,
        "role": role,
        "actionable": actionable,
        "actionability": "exact_mcp_bridge" if actionable else "hard_block",
        "action": "terminate_exact_mcp" if actionable else "refuse",
    }
    if wrapper_pid is not None and wrapper_pid != snapshot.pid:
        record["wrapper_pid"] = wrapper_pid
    if resource:
        record["resource"] = resource
    return record


def _desktop_plugin_script(argv: Sequence[str], root: Path) -> Path | None:
    """Return a desktop-plugin ``.py`` entrypoint from an exact argv shape.

    Desktop plugins that need a persistent helper use the installation venv to
    execute their own absolute ``service.py`` directly.  Do not infer ownership
    from an arbitrary mention of a plugin path later in a command line: that
    would turn a user's one-off Python invocation into an updater kill target.
    """
    if len(argv) < 2:
        return None
    candidate = argv[1].strip('"')
    if not candidate or candidate.startswith("-"):
        return None
    script = Path(candidate)
    plugins_root = root.parent / "desktop-plugins"
    if (
        not script.is_absolute()
        or script.suffix.casefold() != ".py"
        or not _within(script, plugins_root)
    ):
        return None
    return Path(_canonical(script))


def _desktop_plugin_role(
    snapshot: _ProcessSnapshot,
    root: Path,
    wrappers: Mapping[int, Path],
    snapshots: Mapping[int, _ProcessSnapshot],
) -> tuple[str, int | None, Path] | None:
    """Classify an exact Desktop-plugin wrapper or managed-runtime child."""
    script = _desktop_plugin_script(snapshot.argv, root)
    if script is None:
        return None
    venv = _venv_dir(root)
    managed = root / ".hermes-runtime" / "python"
    if _within(snapshot.exe, venv):
        return "desktop_plugin_wrapper", snapshot.pid, script
    if not _within(snapshot.exe, managed):
        return None

    ancestor = snapshot.ppid
    seen: set[int] = set()
    while ancestor and ancestor not in seen:
        wrapper_script = wrappers.get(ancestor)
        if wrapper_script is not None and _canonical(wrapper_script) == _canonical(script):
            return "desktop_plugin_worker", ancestor, script
        seen.add(ancestor)
        parent = snapshots.get(ancestor)
        if parent is None:
            break
        ancestor = parent.ppid
    return None


def _desktop_plugin_service_host(
    wrapper: _ProcessSnapshot,
    script: Path,
) -> Any | None:
    """Prove the service's exact Windows Script Host supervisor is still live."""
    expected_host = script.parent / "service-host.vbs"
    try:
        parents = wrapper.process.parents()
    except Exception:
        return None
    descendant_created_at = wrapper.created_at
    for parent in parents:
        try:
            created_at = float(parent.create_time())
            name = str(parent.name() or "")
            exe = str(parent.exe() or "")
            argv = [str(value) for value in (parent.cmdline() or [])]
            if (
                not math.isfinite(created_at)
                or created_at <= 0
                or created_at > descendant_created_at
                or not _process_generation_matches(int(parent.pid), created_at)
            ):
                return None
        except Exception:
            return None
        descendant_created_at = created_at
        basenames = {Path(name).stem.casefold(), Path(exe).stem.casefold()}
        if not (basenames & {"wscript", "cscript"}):
            continue
        if any(
            _canonical(value.strip('"')) == _canonical(expected_host)
            for value in argv[1:]
            if value.strip('"')
        ):
            return parent
    return None


def _desktop_plugin_record(
    snapshot: _ProcessSnapshot,
    *,
    role: str,
    wrapper_pid: int | None,
    resource: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "pid": snapshot.pid,
        "name": snapshot.name,
        "cmdline": _redact_sensitive_cmdline(" ".join(snapshot.argv))[:120],
        "created_at": snapshot.created_at,
        "owner": "desktop",
        "role": role,
        "actionable": True,
        "actionability": "exact_desktop_plugin_service",
        "action": "terminate_desktop_plugin_service",
    }
    if wrapper_pid is not None and wrapper_pid != snapshot.pid:
        record["wrapper_pid"] = wrapper_pid
    if resource:
        record["resource"] = resource
    return record


def _owner_from_ancestry(
    snapshot: _ProcessSnapshot,
    *,
    parent_by_pid: Mapping[int, int] | None = None,
) -> str:
    """Attribute a bridge only when a live ancestor proves the owner."""
    if parent_by_pid is None:
        try:
            parents = []
            descendant_created_at = snapshot.created_at
            for parent in snapshot.process.parents():
                parent_created_at = float(parent.create_time())
                if (
                    not math.isfinite(parent_created_at)
                    or parent_created_at <= 0
                    or parent_created_at > descendant_created_at
                ):
                    return "unknown"
                parents.append((parent, parent_created_at))
                descendant_created_at = parent_created_at
        except Exception:
            return "unknown"
    else:
        import psutil  # noqa: PLC0415

        parents = []
        ancestor_pid = snapshot.ppid
        descendant_created_at = snapshot.created_at
        seen: set[int] = set()
        while ancestor_pid > 0 and ancestor_pid not in seen:
            seen.add(ancestor_pid)
            try:
                parent = psutil.Process(ancestor_pid)
                parent_created_at = float(parent.create_time())
            except Exception:
                break
            if (
                not math.isfinite(parent_created_at)
                or parent_created_at <= 0
                or parent_created_at > descendant_created_at
            ):
                break
            parents.append((parent, parent_created_at))
            descendant_created_at = parent_created_at
            try:
                ancestor_pid = int(parent_by_pid.get(ancestor_pid, 0))
            except (TypeError, ValueError):
                break
    for parent, expected_created_at in parents:
        try:
            name = str(parent.name() or "").lower()
            exe = str(parent.exe() or "").lower()
            argv = [str(value).lower() for value in (parent.cmdline() or [])]
        except Exception:
            try:
                if not _process_generation_matches(parent.pid, expected_created_at):
                    break
            except Exception:
                break
            continue
        try:
            if not _process_generation_matches(parent.pid, expected_created_at):
                break
        except Exception:
            break
        basenames = {
            Path(name).stem.lower(),
            Path(exe).stem.lower(),
            *(Path(value.strip('"')).stem.lower() for value in argv[:1]),
        }
        if basenames & {"codex", "codex-cli"}:
            return "codex"
        if basenames & {"claude", "claude-code"}:
            return "claude"
        joined = " ".join(argv)
        normalized_argv = joined.replace("\\", "/")
        if "/@anthropic-ai/claude-code/cli.js" in normalized_argv:
            return "claude"
        if "apps\\desktop" in joined or "apps/desktop" in joined:
            return "desktop"
    return "unknown"


def _generic_record(
    pid: int,
    name: str,
    cmdline: str,
    snapshot: _ProcessSnapshot | None,
    resource: str | None = None,
) -> dict[str, Any]:
    command = _hermes_cli_command(snapshot.argv if snapshot else cmdline)
    owner = "desktop" if command in {"serve", "dashboard"} else "unknown"
    role = "desktop_backend" if owner == "desktop" else "other"
    record: dict[str, Any] = {
        "pid": int(pid),
        "name": str(name),
        "cmdline": _redact_sensitive_cmdline(cmdline)[:120],
        "owner": owner,
        "role": role,
        "actionable": False,
        "actionability": "hard_block",
        "action": "refuse",
    }
    if snapshot is not None:
        record["created_at"] = snapshot.created_at
        if int(snapshot.ppid) > 0:
            record["parent_pid"] = int(snapshot.ppid)
    if resource:
        record["resource"] = resource
    # Safe-to-stop local previews (``python -m http.server``) still carry their
    # UI metadata. This is display/affordance data only — the record stays a
    # hard block for the updater; stopping one goes through the separate
    # ``--terminate-safe-preview`` path, which re-verifies identity, argv, and
    # target-root containment itself.
    record.update(_local_preview_metadata(int(pid), str(name)))
    return record


_UPDATE_SHIM_FLAG_OPTIONS = frozenset({
    "--accept-hooks", "--cli", "--dev", "--ignore-rules",
    "--ignore-user-config", "--no-restore-cwd", "--pass-session-id",
    "--safe-mode", "--tui", "--worktree", "--yolo", "-w",
})
_UPDATE_SHIM_VALUE_OPTIONS = frozenset({
    "--in", "--model", "--oneshot", "--profile", "--provider",
    "--reasoning", "--resume", "--skills", "--toolsets", "--usage-file",
    "-m", "-p", "-r", "-s", "-t", "-z",
})


def _is_current_update_shim_argv(argv: list[str]) -> bool:
    """True for ``hermes[.exe] [global options] update ...`` (the console shim running the updater)."""
    index = 1
    while index < len(argv):
        token = str(argv[index])
        if token.casefold() == "update":
            return True
        if token == "--":
            return index + 1 < len(argv) and str(argv[index + 1]).casefold() == "update"
        if token in _UPDATE_SHIM_FLAG_OPTIONS:
            index += 1
            continue
        if token in _UPDATE_SHIM_VALUE_OPTIONS:
            if index + 1 >= len(argv):
                return False
            index += 2
            continue
        if token.startswith("--") and "=" in token:
            option, value = token.split("=", 1)
            if option in _UPDATE_SHIM_VALUE_OPTIONS and value:
                index += 1
                continue
        return False
    return False


def _is_current_standalone_scanner_argv(argv: list[str], target_root: Path | str) -> bool:
    """True for this scanner's own launch line against *target_root*.

    The Desktop runs the carrier as ``python -I <this file> --root <root>``
    and the CLI form is ``python -m hermes_cli._scan_venv_blockers --root
    <root>``; under a uv venv both leave a ``venv\\Scripts\\python.exe``
    trampoline parent that must not be reported as a holder of its own scan.
    """
    if len(argv) != 5 or argv[3] != "--root":
        return False
    try:
        if argv[1] == "-I":
            own_scanner = _canonical(argv[2]) == _canonical(__file__)
        elif argv[1] == "-m":
            own_scanner = argv[2] == "hermes_cli._scan_venv_blockers"
        else:
            return False
        return own_scanner and _canonical(argv[4]) == _canonical(target_root)
    except (OSError, TypeError, ValueError):
        return False


def _holder_name_gate(name: str, exe: object) -> bool:
    """Whether a process name/exe may be nominated by the argv or cwd holder rungs."""
    candidates = {_process_basename(name).casefold(), _process_basename(exe or "").casefold()}
    candidates.discard("")
    return any(
        value.startswith(_HOLDER_PROCESS_NAME_PREFIXES) or value in _HOLDER_PROCESS_NAMES
        for value in candidates
    )


def _detect_target_venv_holders(
    root: Path | str,
    *,
    exclude_pids: set[int] | None = None,
    strict: bool = False,
    _parent_by_pid: Mapping[int, int] | None = None,
) -> list[tuple[int, str, str]]:
    """Find live holders of the install at *root*, optionally with strict identity proof.

    A holder is a process whose executable lives under the install venv,
    or an interpreter/launcher (``python*``, ``pypy*``, ``uv``, ``uvx``,
    ``hermes``) whose argv[0] names that venv, runs
    ``hermes_cli.main`` from the install, or is an exact MCP-bridge launch
    with the install as its working directory. The name gate matters: a
    shell that once activated the venv, an editor opened on ``pyenv.cfg``
    or a search over the checkout also carry the venv path on their argv and
    must never become force-release targets. Base-interpreter workers that
    re-run a target-venv trampoline's exact argv are added through the
    parent snapshot, so uv's launcher/worker pairs are reported together.

    ``strict`` raises instead of degrading on unreadable identity metadata,
    which the Desktop preflight turns into a fail-closed probe failure.
    """
    try:
        import psutil  # noqa: PLC0415
    except Exception as exc:
        if strict:
            raise RuntimeError(f"psutil is not available: {exc}") from exc
        return []

    target_root = Path(root)
    venv_dir = target_root / "venv"
    if not venv_dir.exists() and (target_root / ".venv").exists():
        venv_dir = target_root / ".venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep
    try:
        root_prefix = str(target_root.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        root_prefix = str(target_root).lower().rstrip(os.sep) + os.sep

    skip = set(exclude_pids or set())
    skip.add(os.getpid())
    # This scan's own launcher is never a holder of its own scan: the exact
    # console shim running ``hermes update``, the exact ``python -m
    # hermes_cli.main update`` trampoline, or this scanner's own trampoline.
    try:
        parent = psutil.Process(os.getpid()).parent()
        expected_hermes = venv_dir / "Scripts" / "hermes.exe"
        expected_python = venv_dir / "Scripts" / "python.exe"
        parent_argv = [str(value) for value in (parent.cmdline() or [])]
        parent_exe_key = os.path.normcase(os.path.realpath(str(parent.exe() or "")))
        parent_argv0_key = os.path.normcase(os.path.realpath(parent_argv[0])) if parent_argv else ""
        exact_hermes = parent_exe_key == os.path.normcase(os.path.realpath(expected_hermes)) and parent_argv0_key == parent_exe_key and _is_current_update_shim_argv(parent_argv)
        exact_python = False
        if parent_exe_key == os.path.normcase(os.path.realpath(expected_python)) and parent_argv0_key == parent_exe_key:
            exact_python = _hermes_cli_command(parent_argv) == "update" or _is_current_standalone_scanner_argv(parent_argv, target_root)
        if exact_hermes or exact_python:
            skip.add(int(parent.pid))
    except Exception:
        pass

    process_rows: list[dict[str, object]] = []
    matches: list[tuple[int, str, str]] = []
    try:
        proc_iter = psutil.process_iter(["pid", "exe", "name"])
    except Exception as exc:
        if strict:
            raise RuntimeError(f"process enumeration failed: {exc}") from exc
        return []
    for proc in proc_iter:
        try:
            info = proc.info
        except Exception as exc:
            if strict:
                raise RuntimeError("process identity enumeration was unreadable") from exc
            continue
        pid = info.get("pid")
        if pid is None:
            if strict:
                raise RuntimeError("process enumeration returned no PID")
            continue
        try:
            numeric_pid = int(pid)
        except (TypeError, ValueError) as exc:
            if strict:
                raise RuntimeError("process enumeration returned an invalid PID") from exc
            continue
        if numeric_pid in skip:
            continue
        exe = info.get("exe")
        exe_norm = ""
        if exe:
            try:
                exe_norm = str(Path(exe).resolve()).lower()
            except (OSError, ValueError):
                exe_norm = str(exe).lower()
        name = str(info.get("name") or "")
        is_holder = exe_norm.startswith(venv_prefix)
        if not is_holder and not _holder_name_gate(name, exe):
            continue
        try:
            raw_argv = info["cmdline"] if "cmdline" in info else proc.cmdline()
        except Exception:
            raw_argv = None
        raw_cwd = None
        if not is_holder:
            try:
                raw_cwd = info["cwd"] if "cwd" in info else proc.cwd()
            except Exception:
                raw_cwd = None
        argv = [str(value) for value in (raw_argv or [])]
        cmdline = " ".join(argv)
        cwd_low = str(raw_cwd or "").lower().rstrip(os.sep) + os.sep
        if strict and exe is None and raw_argv is None and raw_cwd is None and Path(name).name.casefold() in {"python.exe", "pythonw.exe", "hermes.exe"}:
            raise RuntimeError(f"process {numeric_pid} ({name}) identity metadata was unreadable")
        process_rows.append({"pid": numeric_pid, "ppid": info.get("ppid"), "exe": str(exe or ""), "exe_norm": exe_norm, "name": name, "argv": argv, "cmdline": cmdline})
        # The managed base interpreter can be shared with unrelated Python
        # users. Its location alone is not ownership of this target venv.
        if not is_holder and _holder_name_gate(name, exe):
            argv0 = argv[0].strip('"') if argv else ""
            if argv0 and Path(argv0).is_absolute() and _within(argv0, venv_dir):
                is_holder = True
            elif cwd_low.startswith(root_prefix) and _hermes_cli_tail(argv) is not None:
                is_holder = True
            elif cwd_low.startswith(root_prefix) and is_exact_mcp_module_argv(argv):
                is_holder = True
        if is_holder:
            matches.append((numeric_pid, name or (Path(exe).name if exe else "unreadable-process"), cmdline))

    matched_pids = {pid for pid, _name, _cmdline in matches}
    target_wrappers = {int(row["pid"]): row for row in process_rows if str(row["exe_norm"]).startswith(venv_prefix) and _process_basename(row["exe"]).casefold() in {"python.exe", "pythonw.exe"} and bool(row["argv"])}
    if not target_wrappers:
        return matches
    tails = {tuple(str(value) for value in row["argv"])[1:] for row in target_wrappers.values()}
    candidates = [row for row in process_rows if int(row["pid"]) not in matched_pids and _process_basename(row["exe"]).casefold() in {"python.exe", "pythonw.exe"} and tuple(str(value) for value in row["argv"])[1:] in tails]
    if not candidates:
        return matches
    rows_by_pid = {int(row["pid"]): row for row in process_rows}
    parent_by_pid = _parent_by_pid
    if parent_by_pid is None and any(row["ppid"] is None for row in candidates):
        try:
            parent_by_pid = {int(pid): int(ppid) for pid, ppid in psutil._ppid_map().items()}
        except Exception as exc:
            if strict:
                raise RuntimeError("parent process enumeration failed") from exc
            parent_by_pid = {}
    if parent_by_pid is not None:
        for row in process_rows:
            row["ppid"] = parent_by_pid.get(int(row["pid"]))

    def points_to_wrapper(row: dict[str, object]) -> bool:
        candidate_tail = tuple(str(value) for value in row["argv"])[1:]
        current = row
        seen: set[int] = set()
        while True:
            raw_parent = current["ppid"]
            if raw_parent is None:
                if strict:
                    raise RuntimeError(f"process {int(current['pid'])} was absent from the parent snapshot")
                return False
            try:
                ancestor = int(raw_parent)
            except (TypeError, ValueError):
                return False
            if ancestor <= 0 or ancestor in seen:
                return False
            seen.add(ancestor)
            wrapper = target_wrappers.get(ancestor)
            if wrapper is not None:
                return bool(candidate_tail and candidate_tail == tuple(str(value) for value in wrapper["argv"])[1:])
            parent = rows_by_pid.get(ancestor)
            if parent is None:
                return False
            current = parent

    for row in candidates:
        if points_to_wrapper(row):
            matches.append((int(row["pid"]), str(row["name"]) or _process_basename(row["exe"]), str(row["cmdline"])))
    return matches


def scan_venv_blockers(root: str | Path) -> dict[str, Any]:
    """Return a strict, typed blocker snapshot for a validated target root."""
    target_root, venv = _validated_root(root)
    try:
        import psutil  # noqa: PLC0415

        parent_by_pid: dict[int, int] | None = None
        ppid_map_fn = getattr(psutil, "_ppid_map", None)
        if callable(ppid_map_fn):
            parent_by_pid = {
                int(pid): int(ppid) for pid, ppid in ppid_map_fn().items()
            }
        detector_kwargs: dict[str, Any] = {}
        if parent_by_pid is not None:
            detector_kwargs["_parent_by_pid"] = parent_by_pid
        matches = _detect_target_venv_holders(
            target_root,
            strict=True,
            **detector_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"scan aborted: {exc}") from exc

    by_pid: dict[int, tuple[str, str]] = {
        int(pid): (str(name), str(cmdline)) for pid, name, cmdline in matches
    }
    snapshots: dict[int, _ProcessSnapshot] = {}
    unreadable: set[int] = set()
    for pid in by_pid:
        try:
            snapshot = (
                _snapshot_for_pid(pid)
                if parent_by_pid is None
                else _snapshot_for_pid(
                    pid,
                    parent_by_pid=parent_by_pid,
                )
            )
        except Exception:
            unreadable.add(pid)
            continue
        if snapshot is not None:
            snapshots[pid] = snapshot

    # Bind the refreshed identities to one fresh parent-table generation. The
    # discovery map can predate these exe/argv/create-time reads; reusing it
    # would splice a recycled PID's new identity onto its predecessor's parent
    # edge. Bracket one shared map with fresh create-time reads instead of
    # calling Process.ppid() (and rebuilding the full Windows map) per PID.
    if callable(ppid_map_fn) and snapshots:
        try:
            parent_by_pid = {int(pid): int(ppid) for pid, ppid in ppid_map_fn().items()}
        except Exception as exc:
            raise RuntimeError(
                "scan aborted: parent process enumeration failed"
            ) from exc
        stale_pids: set[int] = set()
        for pid, snapshot in snapshots.items():
            try:
                same_generation = _process_generation_matches(pid, snapshot.created_at)
            except psutil.NoSuchProcess:
                stale_pids.add(pid)
                continue
            except Exception:
                unreadable.add(pid)
                stale_pids.add(pid)
                continue
            if not same_generation:
                unreadable.add(pid)
                stale_pids.add(pid)
                continue
            snapshots[pid] = replace(
                snapshot,
                ppid=int(parent_by_pid.get(pid, 0)),
            )
        for pid in stale_pids:
            snapshots.pop(pid, None)

    wrappers = {
        pid
        for pid, snapshot in snapshots.items()
        if is_exact_mcp_module_argv(snapshot.argv)
        and _within(snapshot.exe, venv)
    }
    mcp_bridges: list[dict[str, Any]] = []
    desktop_plugin_services: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []
    gateways: list[dict[str, Any]] = []
    deferred_entries: list[dict] = []
    owner_by_anchor_generation: dict[tuple[int, float, int], str] = {}
    desktop_plugin_host_verified: dict[tuple[int, float, str], bool] = {}
    desktop_plugin_wrappers = {
        pid: script
        for pid, snapshot in snapshots.items()
        if _within(snapshot.exe, venv)
        and (script := _desktop_plugin_script(snapshot.argv, target_root)) is not None
    }
    # The managed runtime is shared with foreign uv venvs, so it is outside the
    # update's mutation set; a resource claim must never name a file under it.
    runtime_dir = target_root / ".hermes-runtime"
    for pid, (scanned_name, scanned_cmdline) in by_pid.items():
        snapshot = snapshots.get(pid)
        if snapshot is None and pid not in unreadable:
            continue  # exited between enumeration and identity read
        if snapshot is not None:
            # One file this holder maps inside the mutation set, so Desktop can
            # re-prove ownership before terminating a holder whose own image
            # lives under the shared runtime (#104687 H5).
            resource = _mapped_mutation_set_path(
                snapshot.process,
                venv=venv,
                runtime_dir=runtime_dir,
            )
            classified = _mcp_role(snapshot, target_root, wrappers, snapshots)
            if classified is not None:
                role, wrapper_pid = classified
                owner_anchor = (
                    snapshots.get(wrapper_pid, snapshot)
                    if wrapper_pid is not None
                    else snapshot
                )
                owner_key = (
                    owner_anchor.pid,
                    owner_anchor.created_at,
                    owner_anchor.ppid,
                )
                owner = owner_by_anchor_generation.get(owner_key)
                if owner is None:
                    owner = (
                        _owner_from_ancestry(owner_anchor)
                        if parent_by_pid is None
                        else _owner_from_ancestry(
                            owner_anchor,
                            parent_by_pid=parent_by_pid,
                        )
                    )
                    owner_by_anchor_generation[owner_key] = owner
                mcp_bridges.append(
                    _mcp_record(
                        snapshot,
                        role=role,
                        wrapper_pid=wrapper_pid,
                        owner=owner,
                        parent_by_pid=parent_by_pid,
                        resource=resource,
                    )
                )
                continue
            desktop_plugin = _desktop_plugin_role(
                snapshot,
                target_root,
                desktop_plugin_wrappers,
                snapshots,
            )
            if desktop_plugin is not None:
                role, wrapper_pid, script = desktop_plugin
                wrapper = snapshots.get(wrapper_pid) if wrapper_pid is not None else None
                host_key = (
                    wrapper.pid,
                    wrapper.created_at,
                    _canonical(script),
                ) if wrapper is not None else None
                host_verified = False
                if host_key is not None:
                    host_verified = desktop_plugin_host_verified.get(host_key)
                    if host_verified is None:
                        host_verified = _desktop_plugin_service_host(wrapper, script) is not None
                        desktop_plugin_host_verified[host_key] = host_verified
                if host_verified:
                    desktop_plugin_services.append(
                        _desktop_plugin_record(
                            snapshot,
                            role=role,
                            wrapper_pid=wrapper_pid,
                            resource=resource,
                        )
                    )
                    continue
            live_argv = snapshot.argv
            live_cmdline = " ".join(live_argv)
        else:
            live_cmdline = scanned_cmdline
            # A target-root candidate whose live exe/argv/create-time cannot
            # be re-read is never downstream-actionable. In particular, do
            # not exempt a captured ``gateway run`` string after AccessDenied:
            # without live identity it is an unverified hard blocker.
            processes.append(
                _generic_record(pid, scanned_name, live_cmdline, snapshot=None)
            )
            continue
        if _is_pausable_gateway(live_argv):
            gateway = _generic_record(pid, scanned_name, live_cmdline, snapshot)
            gateway.update(
                {
                    "owner": "gateway",
                    "role": "gateway_run",
                    "actionable": False,
                    "actionability": "downstream_drainable",
                    "action": "pause_downstream",
                }
            )
            gateways.append(gateway)
            continue
        deferred_entry = _updater_owned_backend_entry(pid, live_cmdline)
        if deferred_entry is not None:
            # Ledger-verified serve/dashboard backend the CLI updater's own
            # rungs stop (and relaunch) downstream — reporting it here would
            # dead-end the hand-off before that machinery can run (#98336).
            deferred_entries.append(deferred_entry)
            continue
        processes.append(
            _generic_record(pid, scanned_name, live_cmdline, snapshot, resource)
        )

    # Workers first: terminating a wrapper first can destroy the ancestry proof
    # required to authorize an external base-interpreter child.
    mcp_bridges.sort(
        key=lambda item: (item.get("role") != "mcp_bridge_worker", item["pid"])
    )
    desktop_plugin_services.sort(
        key=lambda item: (item.get("role") != "desktop_plugin_worker", item["pid"])
    )
    processes.sort(key=lambda item: item["pid"])
    gateways.sort(key=lambda item: item["pid"])
    blocked = bool(processes or mcp_bridges or desktop_plugin_services)
    result = _base_result(
        root=str(target_root),
        venv=str(venv),
        ok=True,
        blocked=blocked,
        reason="processes_running" if blocked else None,
    )
    result.update(
        {
            "processes": processes,
            "mcp_bridges": mcp_bridges,
            "desktop_plugin_services": desktop_plugin_services,
            "pausable_gateways": len(gateways),
            "pausable_gateway_processes": gateways,
            "deferred_backends": len(deferred_entries),
            # Diagnostic only: sanitized evidence (structured ledger identity,
            # never argv) explaining which holders the deferral consumed (#98350).
            "deferred_backend_evidence": _deferred_backend_evidence(deferred_entries),
        }
    )
    return result


def _live_mcp_bridge_process(
    root: Path,
    pid: int,
) -> tuple[Any, dict[str, Any]] | None:
    snapshot = _snapshot_for_pid(pid)
    if snapshot is None:
        return None
    snapshots = {snapshot.pid: snapshot}
    wrappers: set[int] = set()
    if is_exact_mcp_module_argv(snapshot.argv) and _within(snapshot.exe, _venv_dir(root)):
        wrappers.add(snapshot.pid)

    # Rebuild the live ancestry needed for owner authorization and for an
    # external uv/base worker's wrapper relationship. Managed-runtime location
    # proves the worker role without a live wrapper, but not who owns it.
    try:
        parents = snapshot.process.parents()
    except Exception:
        parents = []
    for parent_process in parents:
        try:
            parent = _snapshot_for_pid(int(parent_process.pid))
        except Exception:
            continue
        if parent is None:
            continue
        snapshots[parent.pid] = parent
        if is_exact_mcp_module_argv(parent.argv) and _within(
            parent.exe, _venv_dir(root)
        ):
            wrappers.add(parent.pid)
    classified = _mcp_role(snapshot, root, wrappers, snapshots)
    if classified is None:
        return None
    role, wrapper_pid = classified
    return snapshot.process, _mcp_record(
        snapshot, role=role, wrapper_pid=wrapper_pid
    )


def terminate_mcp_bridge(
    root: str | Path,
    *,
    pid: int,
    created_at: float,
) -> bool:
    """Kill exactly one still-identical MCP bridge PID; never a process tree."""
    target_root, _venv = _validated_root(root)
    try:
        expected_created_at = float(created_at)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(expected_created_at) or expected_created_at <= 0:
        return False
    try:
        live = _live_mcp_bridge_process(target_root, int(pid))
    except Exception:
        return False
    if live is None:
        return False
    process, record = live
    if (
        record.get("owner") not in {"codex", "claude"}
        or record.get("role")
        not in {"mcp_bridge_wrapper", "mcp_bridge_worker"}
        or record.get("actionable") is not True
        or record.get("action") != "terminate_exact_mcp"
    ):
        return False
    if abs(float(record["created_at"]) - expected_created_at) > _CREATE_TIME_TOLERANCE_SECONDS:
        return False
    try:
        process.kill()
    except Exception:
        return False
    return True


def _live_desktop_plugin_service_process(
    root: Path,
    pid: int,
) -> tuple[Any, dict[str, Any], Any | None] | None:
    """Re-prove one exact plugin service and its visible host before killing.

    Ancestors are collected nearest-first only to place the wrapper for role
    classification; the supervisor itself is proven separately by
    ``_desktop_plugin_service_host``. An ancestor that cannot be opened is
    skipped, exactly as the scan path does: after an update the Desktop
    relaunches the plugin host itself, so the chain reads wscript ->
    Hermes.exe -> WmiPrvSE.exe (session 0, OpenProcess denied). Treating
    that ancestor as "unit not provable" made every unit stop report
    ``not stopped`` and the update refuse with the force-release dialog
    (2026-09-06 18:06Z), on every update that followed a successful one.
    """
    snapshot = _snapshot_for_pid(pid)
    if snapshot is None:
        return None
    snapshots = {snapshot.pid: snapshot}
    try:
        parents = snapshot.process.parents()
    except Exception:
        return None
    for parent_process in parents:
        try:
            parent = _snapshot_for_pid(int(parent_process.pid))
        except _ProcessGenerationChanged:
            return None
        except Exception:
            continue
        if parent is not None:
            snapshots[parent.pid] = parent
    wrappers = {
        current_pid: script
        for current_pid, current in snapshots.items()
        if _within(current.exe, _venv_dir(root))
        and (script := _desktop_plugin_script(current.argv, root)) is not None
    }
    classified = _desktop_plugin_role(snapshot, root, wrappers, snapshots)
    if classified is None:
        return None
    role, wrapper_pid, script = classified
    wrapper = snapshots.get(wrapper_pid) if wrapper_pid is not None else None
    if wrapper is None:
        return None
    host = _desktop_plugin_service_host(wrapper, script)
    if host is None:
        return None
    return snapshot.process, _desktop_plugin_record(
        snapshot,
        role=role,
        wrapper_pid=wrapper_pid,
    ), host


def _desktop_plugin_service_unit(
    root: Path,
    pid: int,
) -> tuple[dict[str, Any], Any, _ProcessSnapshot, list[_ProcessSnapshot]] | None:
    """Re-prove one plugin service UNIT from either of its members.

    A Desktop plugin service is three processes that only make sense together:
    the Windows Script Host supervisor (``service-host.vbs``), the venv
    trampoline wrapper it launched, and the managed-runtime worker(s) that
    wrapper re-exec'd.  Returns ``(record, host, wrapper, workers)`` for the
    requested PID (worker or wrapper), or ``None`` when any member can no
    longer be proven live and identical.
    """
    live = _live_desktop_plugin_service_process(root, int(pid))
    if live is None:
        return None
    _process, record, host = live
    wrapper_pid = int(record.get("wrapper_pid", record["pid"]))
    wrapper = _snapshot_for_pid(wrapper_pid)
    if wrapper is None:
        return None
    script = _desktop_plugin_script(wrapper.argv, root)
    if script is None or not _within(wrapper.exe, _venv_dir(root)):
        return None
    try:
        children = list(wrapper.process.children(recursive=True))
    except Exception:
        return None
    wrappers = {wrapper.pid: script}
    snapshots: dict[int, _ProcessSnapshot] = {wrapper.pid: wrapper}
    for child in children:
        try:
            snapshot = _snapshot_for_pid(int(child.pid))
        except Exception:
            return None
        if snapshot is not None:
            snapshots[snapshot.pid] = snapshot
    workers: list[_ProcessSnapshot] = []
    for snapshot in snapshots.values():
        if snapshot.pid == wrapper.pid:
            continue
        classified = _desktop_plugin_role(snapshot, root, wrappers, snapshots)
        if classified is not None and classified[0] == "desktop_plugin_worker":
            workers.append(snapshot)
    return record, host, wrapper, workers


def _kill_proven_member(process: Any) -> bool:
    """Kill one identity-proven unit member; a member that already exited counts."""
    import psutil  # noqa: PLC0415

    try:
        process.kill()
    except psutil.NoSuchProcess:
        return True
    except Exception:
        return False
    return True


def _service_host_relaunch(host: Any) -> dict[str, Any] | None:
    """Describe the stopped supervisor so the Desktop can relaunch it later."""
    try:
        argv = [str(value) for value in (host.cmdline() or [])]
        created_at = float(host.create_time())
    except Exception:
        return None
    if not argv:
        return None
    try:
        cwd = host.cwd()
    except Exception:
        cwd = None
    return {
        "pid": int(host.pid),
        "created_at": created_at,
        "argv": argv,
        "cwd": str(cwd) if cwd else None,
    }


def terminate_desktop_plugin_service_unit(
    root: str | Path,
    *,
    pid: int,
    created_at: float,
) -> dict[str, Any]:
    """Stop one exact plugin service unit top-down; never an arbitrary tree.

    Order matters and is the whole point: the supervisor host dies FIRST so
    it cannot respawn anything, then the venv wrapper, then the managed-
    runtime worker(s).  The previous worker-first order was self-defeating:
    killing the worker made the uv trampoline wrapper exit on its own, the
    follow-up wrapper call then found nothing to prove, the host was never
    stopped, and ``service-host.vbs`` relaunched the service ten seconds
    later - inside the updater's lock-release window (2026-09-03/04
    ``venv-unlock-failed`` incidents).

    Every member is re-proven (PID + create time, exe root, script, host
    argv) in one snapshot before the first kill.  A member that exits
    between proof and kill counts as stopped.  Returns ``{"terminated":
    bool, "host": {...} | None}``; ``host`` describes the stopped supervisor
    so the Desktop can relaunch it after the update finishes or aborts.
    """
    target_root, _venv = _validated_root(root)
    try:
        expected_created_at = float(created_at)
    except (TypeError, ValueError):
        return {"terminated": False, "host": None}
    if not math.isfinite(expected_created_at) or expected_created_at <= 0:
        return {"terminated": False, "host": None}
    try:
        unit = _desktop_plugin_service_unit(target_root, int(pid))
    except Exception:
        return {"terminated": False, "host": None}
    if unit is None:
        return {"terminated": False, "host": None}
    record, host, wrapper, workers = unit
    if (
        record.get("owner") != "desktop"
        or record.get("role")
        not in {"desktop_plugin_wrapper", "desktop_plugin_worker"}
        or record.get("actionable") is not True
        or record.get("action") != "terminate_desktop_plugin_service"
        or abs(float(record["created_at"]) - expected_created_at)
        > _CREATE_TIME_TOLERANCE_SECONDS
    ):
        return {"terminated": False, "host": None}
    relaunch = _service_host_relaunch(host)
    stopped = _kill_proven_member(host)
    stopped = _kill_proven_member(wrapper.process) and stopped
    for worker in workers:
        stopped = _kill_proven_member(worker.process) and stopped
    return {"terminated": stopped, "host": relaunch if stopped else None}


def terminate_desktop_plugin_service(
    root: str | Path,
    *,
    pid: int,
    created_at: float,
) -> bool:
    """Boolean view of :func:`terminate_desktop_plugin_service_unit`."""
    return bool(
        terminate_desktop_plugin_service_unit(root, pid=pid, created_at=created_at)[
            "terminated"
        ]
    )

def terminate_venv_holder(
    root: str | Path,
    *,
    pid: int,
    created_at: float,
) -> bool:
    """Force-stop one freshly revalidated target-install holder.

    The Desktop's explicit Update action authorizes this force path.  This is
    intentionally broader than MCP/plugin ownership: every process returned by
    the target-root holder detector can be stopped.  PID creation-time matching
    only prevents a recycled PID from being targeted after the scan.
    """
    target_root, _venv = _validated_root(root)
    try:
        expected_created_at = float(created_at)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(expected_created_at) or expected_created_at <= 0:
        return False
    try:
        matches = _detect_target_venv_holders(target_root, strict=True)
        if int(pid) not in {int(found_pid) for found_pid, _name, _cmdline in matches}:
            return False
        snapshot = _snapshot_for_pid(int(pid))
    except Exception:
        return False
    if snapshot is None or abs(snapshot.created_at - expected_created_at) > _CREATE_TIME_TOLERANCE_SECONDS:
        return False
    try:
        snapshot.process.kill()
    except Exception:
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(add_help=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--terminate-mcp-bridge", type=int)
    parser.add_argument("--terminate-desktop-plugin-service", type=int)
    parser.add_argument("--terminate-venv-holder", type=int)
    parser.add_argument("--terminate-safe-preview", type=int)
    parser.add_argument("--created-at", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Print one JSON document; exit 0 for valid scan/action, 1 on failure."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _build_parser().parse_args(raw_argv)
    except (ValueError, argparse.ArgumentError) as exc:
        _emit_probe_fail(str(exc), code="invalid_arguments")

    root_text = str(args.root)
    mcp_terminate_requested = args.terminate_mcp_bridge is not None
    desktop_plugin_terminate_requested = (
        args.terminate_desktop_plugin_service is not None
    )
    venv_holder_terminate_requested = args.terminate_venv_holder is not None
    safe_preview_terminate_requested = args.terminate_safe_preview is not None
    terminate_requested = (
        mcp_terminate_requested
        or desktop_plugin_terminate_requested
        or venv_holder_terminate_requested
        or safe_preview_terminate_requested
    )
    created_requested = args.created_at is not None
    if terminate_requested != created_requested or (
        sum(
            (
                mcp_terminate_requested,
                desktop_plugin_terminate_requested,
                venv_holder_terminate_requested,
                safe_preview_terminate_requested,
            )
        )
        > 1
    ):
        _emit_probe_fail(
            "exactly one --terminate-* flag and --created-at must be supplied together",
            root=root_text,
            code="invalid_arguments",
        )

    if terminate_requested:
        try:
            target_root, venv = _validated_root(root_text)
            stopped_host: dict[str, Any] | None = None
            if mcp_terminate_requested:
                pid = args.terminate_mcp_bridge
                terminated = terminate_mcp_bridge(
                    target_root,
                    pid=pid,
                    created_at=args.created_at,
                )
                mode = "terminate_mcp_bridge"
            elif desktop_plugin_terminate_requested:
                pid = args.terminate_desktop_plugin_service
                unit_outcome = terminate_desktop_plugin_service_unit(
                    target_root,
                    pid=pid,
                    created_at=args.created_at,
                )
                terminated = bool(unit_outcome["terminated"])
                stopped_host = unit_outcome.get("host")
                mode = "terminate_desktop_plugin_service"
            elif venv_holder_terminate_requested:
                pid = args.terminate_venv_holder
                terminated = terminate_venv_holder(
                    target_root,
                    pid=pid,
                    created_at=args.created_at,
                )
                mode = "terminate_venv_holder"
            else:
                pid = args.terminate_safe_preview
                terminated, _error = _terminate_safe_preview(
                    int(pid), float(args.created_at), venv
                )
                mode = "terminate_safe_preview"
        except Exception as exc:
            _emit_probe_fail(str(exc), root=root_text, code="probe_failed")
        document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "ok": True,
            "terminated": terminated,
            "pid": pid,
            "created_at": args.created_at,
            "root": str(target_root),
            "venv": str(venv),
            "error": None,
        }
        if mode == "terminate_desktop_plugin_service":
            # The stopped supervisor's launch line: the Desktop relaunches it
            # once the update finishes or aborts, so stopping the plugin
            # service for the update does not strand it until next login.
            document["host"] = stopped_host
        print(json.dumps(document))
        raise SystemExit(0)

    try:
        import psutil  # noqa: PLC0415, F401
    except Exception as exc:
        # The scanner runs under the target venv interpreter and imports psutil
        # from the very site-packages the update is about to rewrite, so a
        # previous interrupted update can leave psutil half-written. Every
        # later update attempt then refuses identically, and the thing that
        # would repair psutil is the update itself (#104687 H9). The scan stays
        # fail-closed — a scanner that cannot enumerate must never report the
        # install free — but the refusal names its own repair instead of
        # surfacing as an anonymous probe failure.
        _emit_probe_fail(
            f"psutil is not available in the target venv: {exc}. "
            f'Repair it with: "{sys.executable}" -m pip install --force-reinstall psutil',
            root=root_text,
            code="scanner_dependency_unavailable",
        )
    try:
        data = scan_venv_blockers(root_text)
    except ValueError as exc:
        _emit_probe_fail(str(exc), root=root_text, code="invalid_root")
    except Exception as exc:
        _emit_probe_fail(str(exc), root=root_text, code="probe_failed")
    print(json.dumps(data))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
