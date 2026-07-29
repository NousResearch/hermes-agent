"""Gateway lifecycle guard for cron job creation (#30719).

An agent running inside a gateway can schedule a cron job that calls
``hermes gateway restart`` (or ``launchctl kickstart ai.hermes.gateway``
or ``systemctl restart hermes-gateway``).  When the cron fires, the
gateway dies, the supervisor (launchd KeepAlive / systemd Restart=)
revives it, auto-resume picks up the offending session, and the resumed
turn re-runs the same logic — a SIGTERM-respawn loop every ~10 seconds
until manually broken.

This module rejects cron job specs whose prompt or script contains a
direct shell-level gateway-lifecycle command.  It is enforced at
``cron.jobs.create_job`` so it fires on every job-creation path: the
``hermes cron create`` CLI subcommand AND the agent's ``cronjob`` model
tool (which calls ``create_job`` directly, bypassing the CLI layer).

The pattern is intentionally command-shaped: it anchors on a concrete
command identifier (``hermes gateway``, ``launchctl ... hermes-gateway``,
``systemctl ... hermes-gateway``, ``pkill`` against the gateway) so it
cannot fire on prose.  A cron ``prompt`` is fed to a future LLM, not a
shell, so an over-broad substring match on English ("Kong API gateway
autoscaling and restart behavior") would produce a high false-positive
rate without preventing the actual foot-gun, which requires a real
command shape.

This is a defence-in-depth layer.  ``tools/terminal_tool.py`` already
blocks these commands at *execution* time when ``_HERMES_GATEWAY=1``, and
``hermes gateway stop|restart`` refuse to self-target from inside the
gateway.  Blocking at *creation* time as well means the agent gets an
immediate, informative rejection instead of scheduling a job that will
only fail (silently) when it fires.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Optional


class GatewayLifecycleBlocked(ValueError):
    """Raised when a cron job spec contains a gateway-lifecycle command."""


# Shell-level command shapes that target the gateway lifecycle. Each branch
# is anchored on a concrete command identifier so a match can only fire on
# actual shell-command-shaped strings, not on prose.
_GATEWAY_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    # Branch A: `hermes gateway restart|stop` — the canonical foot-gun.
    # `start` is intentionally excluded: starting a gateway from inside a
    # gateway is benign (a no-op or "already running" error), and a
    # legitimate cron job might start a sibling profile's gateway.
    r"(?:hermes\s+gateway\s+(?:restart|stop))"
    # Branch B: launchctl ops on a hermes-gateway label. macOS launchd
    # labels look like `ai.hermes.gateway` / `hermes-gateway`. Requiring the
    # gateway identifier prevents blocking unrelated hermes services (e.g.
    # `launchctl unload ai.hermes.update-checker.plist`).
    r"|(?:launchctl\s+(?:kickstart|unload|load|stop|restart)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch C: systemctl ops on a hermes-gateway unit.
    r"|(?:systemctl\s+(?:-\S+\s+)*(?:restart|stop|start)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch D: pkill / kill targeting the hermes gateway process. Both
    # token orders because real reproductions show both.
    r"|(?:p?kill\b[^\n]*\bhermes\b[^\n]*\bgateway)"
    r"|(?:p?kill\b[^\n]*\bgateway\b[^\n]*\bhermes)"
)


_MAX_HELPER_SCAN_BYTES = 256 * 1024
_MAX_HELPER_SCAN_DEPTH = 3
_MAX_HELPERS_SCANNED = 16

# Terminal execution may hide the lifecycle command in a literal helper file.
# Keep extraction deliberately narrow: literal path commands
# (``./helper`` / ``scripts/helper`` / ``/tmp/helper``), shell source builtins,
# and literal script arguments to common shell interpreters. Shell control-flow
# keywords, variable assignments, and simple execution/privilege/environment
# wrappers are accepted before those command shapes. Quoted ``sh -c`` payloads
# are scanned recursively. Dynamic expansion and arbitrary PATH-only executable
# lookup remain outside this bounded inspection layer.
_SHELL_COMMAND_START = r"(?:^|[;&|]\s*|\$\(\s*|\{\s*)"
_SHELL_CONTROL_PREFIX = r"(?:(?:(?:if|then|elif|while|until|do|else)|!)\s+)*"
_SHELL_ASSIGNMENT = (
    r"[A-Za-z_]\w*=(?:[^\s;&|]+|\"[^\"]*\"|'[^']*')"
)
_SUDO_OPTION_WITH_ARG = (
    r"(?:-u|--user|-g|--group|-h|--host|-C|--chdir|-R|--chroot|"
    r"-r|--role|-t|--type|-T|--command-timeout)"
)
_SUDO_PREFIX = (
    rf"sudo(?:\s+(?:{_SUDO_OPTION_WITH_ARG}\s+\S+|-\S+))*"
)
_SHELL_LAUNCH_PREFIX = (
    r"(?:"
    + _SHELL_ASSIGNMENT
    + r"|(?:command|exec|builtin|nohup|time)"
    + r"|"
    + _SUDO_PREFIX
    + r"|env(?:\s+-\S+)*"
    + r")\s+"
)
_SHELL_LAUNCH_PREFIX = rf"(?:{_SHELL_LAUNCH_PREFIX})*"
_LITERAL_HELPER_TOKEN = r"[A-Za-z0-9_./:@%+=,~\-]+"
_LITERAL_COMMAND_PATH_PATTERN = re.compile(
    _SHELL_COMMAND_START
    + _SHELL_CONTROL_PREFIX
    + _SHELL_LAUNCH_PREFIX
    + r"[\"']?"
    r"((?:(?:~|/|\./|\../)[A-Za-z0-9_./:@%+=,\-]+|"
    r"[A-Za-z0-9_.:@%+=,~\-]+/[A-Za-z0-9_./:@%+=,~\-]+))"
    r"[\"']?",
    re.MULTILINE,
)
_SHELL_INTERPRETER_REF_PATTERN = re.compile(
    _SHELL_COMMAND_START
    + _SHELL_CONTROL_PREFIX
    + _SHELL_LAUNCH_PREFIX
    + r"(?:bash|sh|zsh|dash|ksh)\s+"
    r"(?:-[A-Za-z]+\s+)*"
    r"[\"']?([A-Za-z0-9_./:@%+=,~\-]+)[\"']?",
    re.IGNORECASE | re.MULTILINE,
)
_SHELL_SOURCE_REF_PATTERN = re.compile(
    _SHELL_COMMAND_START
    + _SHELL_CONTROL_PREFIX
    + _SHELL_LAUNCH_PREFIX
    + r"(?:source|\.)\s+(?:--\s+)?"
    + rf"[\"']?({_LITERAL_HELPER_TOKEN})[\"']?",
    re.IGNORECASE | re.MULTILINE,
)
_SHELL_COMMAND_STRING_PATTERN = re.compile(
    _SHELL_COMMAND_START
    + _SHELL_CONTROL_PREFIX
    + _SHELL_LAUNCH_PREFIX
    + r"(?:bash|sh|zsh|dash|ksh)\s+"
    r"(?:-\S+\s+)*?-[A-Za-z]*c[A-Za-z]*\s+"
    r"(?P<quote>[\"'])(?P<body>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)


def contains_gateway_lifecycle_command(text: str) -> bool:
    """Return True if *text* contains a gateway lifecycle command pattern."""
    if not text:
        return False
    return bool(_GATEWAY_LIFECYCLE_PATTERN.search(text))


def _resolve_terminal_script_path(script_path: str, cwd: Optional[Path | str]) -> Path:
    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        return raw
    base = Path(cwd).expanduser() if cwd else Path.cwd()
    return base / raw


def _read_file_for_scanning(path: Path) -> tuple[str, bool]:
    """Return ``(text, unsafe)`` without blocking on special or huge files.

    Missing literal paths are ignored because the shell will fail to execute
    them. Existing helpers fail closed when they cannot be resolved/read as a
    bounded regular file. Resolving first still permits normal symlinks to
    regular files while rejecting symlinks to FIFOs/devices/directories.
    """
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return "", False
    except (OSError, RuntimeError):
        return "", True

    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags)
        with os.fdopen(descriptor, "rb") as handle:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                return "", True
            if file_stat.st_size > _MAX_HELPER_SCAN_BYTES:
                return "", True
            data = handle.read(_MAX_HELPER_SCAN_BYTES + 1)
    except OSError:
        return "", True

    if len(data) > _MAX_HELPER_SCAN_BYTES:
        return "", True
    return data.decode("utf-8", errors="replace"), False


def _collect_literal_helper_refs(text: str) -> tuple[list[str], bool]:
    """Collect bounded helper refs, including quoted shell-command payloads."""
    seen_refs: set[str] = set()
    refs: list[str] = []
    pending: list[tuple[str, int]] = [(text or "", 0)]
    seen_payloads: set[str] = {text or ""}

    while pending:
        chunk, shell_depth = pending.pop(0)
        for pattern in (
            _LITERAL_COMMAND_PATH_PATTERN,
            _SHELL_INTERPRETER_REF_PATTERN,
            _SHELL_SOURCE_REF_PATTERN,
        ):
            for match in pattern.finditer(chunk):
                ref = match.group(1)
                if ref in seen_refs:
                    continue
                if len(refs) >= _MAX_HELPERS_SCANNED:
                    return refs, True
                seen_refs.add(ref)
                refs.append(ref)

        for match in _SHELL_COMMAND_STRING_PATTERN.finditer(chunk):
            payload = match.group("body")
            if payload in seen_payloads:
                continue
            if shell_depth >= _MAX_HELPER_SCAN_DEPTH:
                return refs, True
            if len(seen_payloads) >= _MAX_HELPERS_SCANNED:
                return refs, True
            seen_payloads.add(payload)
            pending.append((payload, shell_depth + 1))

    return refs, False


def _scan_literal_helpers(
    text: str,
    *,
    cwd: Optional[Path | str],
    depth: int,
    seen_paths: set[Path],
    files_scanned: list[int],
) -> bool:
    refs, reference_overflow = _collect_literal_helper_refs(text)
    if reference_overflow:
        return True

    for ref in refs:
        path = _resolve_terminal_script_path(ref, cwd)
        try:
            path_key = path.resolve(strict=False)
        except (OSError, RuntimeError):
            return True
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        files_scanned[0] += 1
        if files_scanned[0] > _MAX_HELPERS_SCANNED:
            return True

        helper_text, unsafe = _read_file_for_scanning(path)
        if unsafe:
            return True
        if not helper_text:
            continue
        if contains_gateway_lifecycle_command(helper_text):
            return True

        nested_refs, nested_reference_overflow = _collect_literal_helper_refs(helper_text)
        if nested_reference_overflow:
            return True
        if not nested_refs:
            continue
        if depth >= _MAX_HELPER_SCAN_DEPTH:
            return True
        if _scan_literal_helpers(
            helper_text,
            cwd=cwd,
            depth=depth + 1,
            seen_paths=seen_paths,
            files_scanned=files_scanned,
        ):
            return True

    return False


def contains_gateway_lifecycle_invocation(
    text: str,
    *,
    cwd: Optional[Path | str] = None,
    inspect_helpers: bool = True,
) -> bool:
    """Return True for a direct command or bounded literal-helper invocation.

    Terminal commands can invoke a helper whose contents contain the actual
    lifecycle operation. On a local backend this scans a bounded set of
    readable regular helper files as resolved from the exact execution cwd,
    including nested literal references. Existing helpers that are special,
    unreadable, too large, too deep, or too numerous fail closed rather than
    hanging or exhausting the gateway process.

    A non-local backend cannot truthfully inspect its helper bytes from the
    local gateway. Callers therefore pass ``inspect_helpers=False``; any
    literal helper invocation then fails closed instead of trusting an
    unrelated local path with the same spelling.
    """
    if contains_gateway_lifecycle_command(text):
        return True

    if not inspect_helpers:
        refs, reference_overflow = _collect_literal_helper_refs(text)
        return reference_overflow or bool(refs)

    return _scan_literal_helpers(
        text,
        cwd=cwd,
        depth=0,
        seen_paths=set(),
        files_scanned=[0],
    )


def _resolve_script_path(script_path: str) -> Path:
    """Resolve a cron ``script`` value the same way the scheduler does.

    The scheduler (``cron.scheduler``) resolves a bare/relative script path
    under ``<HERMES_HOME>/scripts/`` and only accepts absolute paths as-is.
    We MUST mirror that here so the guard scans the file that will actually
    run — otherwise a job whose script lives at the scheduler's real location
    (``~/.hermes/scripts/restart.sh``) but is passed as the bare name
    ``restart.sh`` would read as a nonexistent relative path and silently
    scan prompt-only content, letting the command through.
    """
    from hermes_constants import get_hermes_home

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        return raw
    return get_hermes_home() / "scripts" / raw


def _read_script_for_scanning(script_path: str) -> str:
    """Read a bounded regular cron script for lifecycle-pattern scanning.

    Decodes with ``errors="replace"`` so non-UTF-8 bytes cannot hide a plain
    command. Existing scripts that are special, unreadable, or oversized fail
    closed; missing scripts remain a downstream scheduler validation concern.
    """
    script_text, unsafe = _read_file_for_scanning(_resolve_script_path(script_path))
    if unsafe:
        raise GatewayLifecycleBlocked(
            "Blocked: cron script could not be safely inspected as a bounded "
            "regular file (#30719)."
        )
    return script_text


def check_gateway_lifecycle(
    prompt: Optional[str],
    script: Optional[str] = None,
    *,
    cwd: Optional[Path | str] = None,
) -> None:
    """Fail closed when an effective cron definition can invoke lifecycle.

    ``prompt`` and the top-level ``script`` bytes are scanned directly. Literal
    helpers referenced by either are inspected recursively from the effective
    execution cwd. When no explicit cwd is configured, script helpers resolve
    from the top-level script directory, matching scheduler execution.
    """
    prompt_text = prompt or ""
    script_text = ""
    script_cwd: Optional[Path | str] = cwd
    if script:
        script_path = _resolve_script_path(script)
        script_text = _read_script_for_scanning(script)
        if script_cwd is None:
            script_cwd = script_path.parent

    combined = prompt_text
    if script_text:
        combined = f"{combined}\n{script_text}"

    blocked = contains_gateway_lifecycle_command(combined)
    if not blocked and prompt_text:
        blocked = contains_gateway_lifecycle_invocation(
            prompt_text,
            cwd=cwd,
        )
    if not blocked and script_text:
        blocked = contains_gateway_lifecycle_invocation(
            script_text,
            cwd=script_cwd,
        )

    if blocked:
        raise GatewayLifecycleBlocked(
            "Blocked: cron job contains a gateway lifecycle command "
            "(restart/stop/kill), directly or through a literal helper. "
            "This is blocked to prevent agent-driven SIGTERM-respawn loops "
            "under launchd/systemd supervision (#30719). Run `hermes gateway "
            "restart` from a shell outside the running gateway instead."
        )
