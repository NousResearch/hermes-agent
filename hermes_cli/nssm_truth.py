"""Is the NSSM-registered Hermes gateway service *actually* running?

``nssm`` is an unsupported-but-real Windows deployment choice: the Windows
native guide points at Scheduled Tasks and says "if you genuinely want a
service, use ``nssm`` or ``sc create`` manually".  Operators who take that
route have no way to ask Hermes itself whether the service is up, so the
answer tends to come from ``nssm status`` alone -- service-manager bookkeeping
that reports ``SERVICE_PAUSED`` while the gateway process is alive and bound to
its port, and ``SERVICE_RUNNING`` for a process that has already died.

This module answers the question from the two sources that actually know, in
order:

1. The canonical gateway liveness probe, :func:`gateway.status.get_running_pid`.
   A record is trusted only when the runtime lock is held, the recorded start
   time still matches the live process (the PID-reuse guard), and the live
   command line still identifies as ``gateway run``/``gateway restart`` for the
   home being probed.  Both Windows launch shapes are recognised, because
   :func:`gateway.status.looks_like_gateway_runtime_command_line` tokenises
   argv properly: the dotted-module form
   ``pythonw.exe -m hermes_cli.main gateway run`` that
   ``hermes_cli/gateway_windows.py`` renders, and the script-path form
   ``...\\hermes_cli\\main.py gateway run`` that a manual or respawned launch
   records.  Nothing here re-implements that logic -- see
   :func:`looks_like_hermes_gateway_argv` for the reusable predicate.
2. ``nssm status <service>``, as a fallback and clearly labelled as the weaker
   of the two sources.

The module is import-safe everywhere: it never touches ``ctypes.WinDLL``, and
the NSSM half is skipped (rather than attempted and failed) off Windows.

Usage::

    from hermes_cli.nssm_truth import is_hermes_service_actually_running

    ok, detail = is_hermes_service_actually_running("HermesGateway")
    # (True, "live gateway process 19540 (home: C:\\Users\\me\\AppData\\Local\\hermes)")

Command line::

    python -m hermes_cli.nssm_truth [verbose] [--service NAME]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from gateway.status import get_running_pid, looks_like_gateway_runtime_command_line
from hermes_constants import get_hermes_home

DEFAULT_SERVICE_NAME = "HermesGateway"
NSSM_BINARY_NAME = "nssm"
#: Point this at an ``nssm.exe`` outside ``PATH`` (scoop/chocolatey/manual installs).
NSSM_EXECUTABLE_ENV = "HERMES_NSSM"

NSSM_QUERY_TIMEOUT_SECONDS = 10

# ``nssm status <service>`` prints exactly one of these (the QueryServiceStatus
# vocabulary), or a complaint on stdout when the name is not registered.
SERVICE_RUNNING = "SERVICE_RUNNING"
SERVICE_PAUSED = "SERVICE_PAUSED"
SERVICE_STOPPED = "SERVICE_STOPPED"
NOT_INSTALLED = "NOT_INSTALLED"

_NSSM_STATE_TOKENS: tuple[str, ...] = (
    SERVICE_RUNNING,
    SERVICE_PAUSED,
    SERVICE_STOPPED,
    "SERVICE_START_PENDING",
    "SERVICE_STOP_PENDING",
    "SERVICE_PAUSE_PENDING",
    "SERVICE_CONTINUE_PENDING",
)

# Observed from NSSM 2.24: a missing service exits 0 and prints
# "Can't open service!\nOpenService(): The specified service does not exist as
# an installed service." to stdout, so the exit code proves nothing here.
_NSSM_ABSENT_MARKERS: tuple[str, ...] = (
    "CAN'T OPEN SERVICE",
    "CANNOT OPEN SERVICE",
    "DOES NOT EXIST AS AN INSTALLED SERVICE",
)

# Conventional install roots, probed only after PATH so a user-installed copy
# in a custom directory can still be reached through HERMES_NSSM.
_NSSM_FALLBACK_BIN_DIRS: tuple[str, ...] = (
    r"{ProgramData}\chocolatey\bin",
    r"{ProgramFiles}\nssm",
    r"{USERPROFILE}\scoop\shims",
)


def looks_like_hermes_gateway_argv(argv: Optional[Sequence[object]]) -> bool:
    """True when an NSSM-configured ``Application``/``AppParameters`` argv launches the gateway.

    A thin, stable wrapper over the canonical
    :func:`gateway.status.looks_like_gateway_runtime_command_line` predicate so callers
    validating an NSSM service definition do not need their own argv heuristics: a
    hand-rolled ``"hermes_cli/main" in joined`` check rejects the launcher's real shape
    (``pythonw.exe -m hermes_cli.main gateway run``), which is the bug this wrapper exists
    to prevent.  Accepts the dotted-module form, the script-path form, the ``hermes``
    console entry point, and an optional ``--profile``/``-p`` selector anywhere in argv.
    """
    if not argv:
        return False
    return looks_like_gateway_runtime_command_line(" ".join(str(part) for part in argv))


def resolve_nssm_executable() -> Optional[str]:
    """Path to ``nssm.exe``: ``HERMES_NSSM`` override, then ``PATH``, then the usual roots.

    ``None`` when NSSM is not installed.  An explicitly-set ``HERMES_NSSM`` that does not
    exist also returns ``None`` rather than silently falling back to a different copy --
    an override is a statement about which binary to use.
    """
    override = os.environ.get(NSSM_EXECUTABLE_ENV, "").strip()
    if override:
        candidate = Path(override)
        return str(candidate) if candidate.is_file() else None

    found = shutil.which(NSSM_BINARY_NAME)
    if found:
        return found

    for template in _NSSM_FALLBACK_BIN_DIRS:
        directory = _expand_windows_dir(template)
        if not directory:
            continue
        for name in (f"{NSSM_BINARY_NAME}.exe", NSSM_BINARY_NAME):
            candidate = Path(directory) / name
            if candidate.is_file():
                return str(candidate)
    return None


def _expand_windows_dir(template: str) -> Optional[str]:
    """Expand a ``{VAR}\\...`` template from the environment, or None when VAR is unset."""
    key = template.split("}", 1)[0].lstrip("{")
    value = os.environ.get(key, "").strip()
    return template.replace("{" + key + "}", value) if value else None


def query_nssm_state(
    service_name: str = DEFAULT_SERVICE_NAME,
    *,
    executable: Optional[str] = None,
    runner: Optional[Callable[..., object]] = None,
    is_windows: Optional[bool] = None,
) -> Optional[str]:
    """``nssm status <service>`` as a state token, or ``None`` when NSSM cannot answer.

    Returns one of :data:`SERVICE_RUNNING`, :data:`SERVICE_PAUSED`,
    :data:`SERVICE_STOPPED`, a ``SERVICE_*_PENDING`` token, or :data:`NOT_INSTALLED`.
    ``None`` means "no usable answer" -- not Windows, NSSM absent, a timeout, or output
    this function does not recognise -- and callers must not read it as "stopped".
    """
    if is_windows is None:
        is_windows = sys.platform == "win32"
    if not is_windows:
        return None

    resolved = executable or resolve_nssm_executable()
    if not resolved:
        return None

    run = runner or subprocess.run
    try:
        result = run(
            [resolved, "status", service_name],
            capture_output=True,
            text=True,
            timeout=NSSM_QUERY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        # Missing binary, spawn refusal, or timeout: all "cannot answer".
        return None

    output = f"{getattr(result, 'stdout', '') or ''}{getattr(result, 'stderr', '') or ''}".upper()
    if any(marker in output for marker in _NSSM_ABSENT_MARKERS):
        return NOT_INSTALLED
    for token in _NSSM_STATE_TOKENS:
        if token in output:
            return token
    return None


def is_hermes_service_actually_running(
    service_name: str = DEFAULT_SERVICE_NAME,
    hermes_home: Optional[Path] = None,
    check_pid_file: bool = True,
    fallback_to_nssm: bool = True,
    *,
    cleanup_stale: bool = False,
    nssm_reader: Optional[Callable[[str], Optional[str]]] = None,
) -> tuple[bool, str]:
    """Return ``(is_running, detail)`` for the named NSSM service.

    Strategy, in order:

    1. The canonical gateway probe for the resolved home.  This is the source of truth:
       it requires a held runtime lock, a live PID whose recorded start time still
       matches (PID-reuse guard), and a live command line that still identifies as this
       profile's gateway.
    2. ``nssm status``, when no live gateway process was found.  This is service-manager
       bookkeeping only; the returned detail says so, because a service reporting
       ``SERVICE_RUNNING`` with no matching process is exactly the false positive that
       makes the two sources worth distinguishing.

    Args:
        service_name: NSSM service name (e.g. ``"HermesGateway"``).
        hermes_home: Home to probe.  ``None`` probes *this* process's home through the
            canonical unscoped call; an explicit path performs a scoped query against
            that home's ``gateway.pid``/``gateway.lock`` only (it never touches another
            profile's identity files).
        check_pid_file: Consult the canonical gateway probe first.
        fallback_to_nssm: Fall back to ``nssm status`` when no live process was found.
        cleanup_stale: Opt in to the canonical probe's stale-record housekeeping.  Off
            by default: this is a read-only truth query and must not delete identity
            files as a side effect.
        nssm_reader: Injection seam for the NSSM state lookup (defaults to
            :func:`query_nssm_state`).

    Returns:
        ``(True, detail)`` when a live gateway process was found, or when NSSM reports
        the service running.  ``(False, detail)`` otherwise.
    """
    if check_pid_file:
        pid = _live_gateway_pid(hermes_home, cleanup_stale=cleanup_stale)
        if pid is not None:
            return True, f"live gateway process {pid} (home: {_probed_home(hermes_home)})"

    if not fallback_to_nssm:
        return False, "no live gateway process matched the gateway identity files (NSSM check disabled)"

    state = (nssm_reader or query_nssm_state)(service_name)

    if state == SERVICE_RUNNING:
        return True, (
            f"nssm reports {SERVICE_RUNNING} for {service_name!r}, but no live gateway process "
            "matched the gateway identity files; trusting the service manager"
        )
    if state == NOT_INSTALLED:
        return False, (
            f"no live gateway process, and NSSM has no service named {service_name!r} installed"
        )
    if state is None:
        return False, (
            "no live gateway process, and NSSM could not report a state "
            f"for {service_name!r} (not installed, or {NSSM_BINARY_NAME} is not on PATH)"
        )
    return False, f"no live gateway process; nssm reports {state} for {service_name!r}"


def is_hermes_gateway_running(service_name: str = DEFAULT_SERVICE_NAME) -> bool:
    """Shorthand for callers that only want the boolean.  Swallows the detail string."""
    running, _detail = is_hermes_service_actually_running(service_name)
    return running


def _probed_home(hermes_home: Optional[Path]) -> Path:
    return Path(hermes_home) if hermes_home is not None else get_hermes_home()


def _live_gateway_pid(hermes_home: Optional[Path], *, cleanup_stale: bool) -> Optional[int]:
    """PID of the live gateway for the probed home, via the canonical ``gateway.status`` probe.

    The unscoped call is used for this process's own home because it also consults the
    runtime-status file; a scoped call is used when the caller names a home, where only
    that home's ``gateway.pid``/``gateway.lock`` are authoritative.
    """
    if hermes_home is None:
        return get_running_pid(cleanup_stale=cleanup_stale)
    return get_running_pid(Path(hermes_home) / "gateway.pid", cleanup_stale=cleanup_stale)


_USAGE = (
    "usage: python -m hermes_cli.nssm_truth [verbose] [--service NAME]\n"
    "\n"
    "Exit status is 0 when the gateway is running, 1 when it is not, 2 on a usage error.\n"
)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    probe: Optional[Callable[..., tuple[bool, str]]] = None,
    emit: Callable[[str], object] = print,
) -> int:
    """CLI entry point.  Returns the exit status instead of calling ``sys.exit``."""
    args = list(sys.argv[1:] if argv is None else argv)
    verbose = False
    service_name = DEFAULT_SERVICE_NAME

    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-h", "--help"):
            emit(_USAGE.rstrip())
            return 0
        if arg == "verbose":
            verbose = True
        elif arg == "--service":
            index += 1
            if index >= len(args):
                emit(f"--service needs a value\n{_USAGE.rstrip()}")
                return 2
            service_name = args[index]
        else:
            emit(f"unrecognised argument: {arg}\n{_USAGE.rstrip()}")
            return 2
        index += 1

    if verbose:
        emit(f"HERMES_HOME = {get_hermes_home()}")
        emit(f"{NSSM_EXECUTABLE_ENV} / PATH nssm = {resolve_nssm_executable() or 'not found'}")
        emit(f"nssm state for {service_name!r} = {query_nssm_state(service_name)}")

    running, detail = (probe or is_hermes_service_actually_running)(service_name)
    emit(f"is_hermes_service_actually_running({service_name!r}): {running} ({detail})")
    return 0 if running else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
