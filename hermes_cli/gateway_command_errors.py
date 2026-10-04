"""User-facing copy for ``hermes gateway install/start/stop/restart`` failures.

``hermes_cli/gateway.py`` is a facade; this sibling owns the small exception -> guidance table so
the most common service failures (``systemctl`` exited non-zero, there is no ``systemctl`` at all,
or a definition was refused because the code root it would run cannot resolve its dependencies) end
as a next step instead of a traceback.
"""

from __future__ import annotations

import subprocess


class SystemctlUnavailableError(RuntimeError):
    """``systemctl`` is not installed (Alpine, minimal containers, some WSL setups)."""

    def __init__(self) -> None:
        super().__init__("systemctl is not available on this system")


def _unviable_launcher_root_lines(exc: BaseException) -> list[str] | None:
    """The refusal message already carries the why and the repair; add only the outcome."""
    from hermes_cli.gateway_launchd import UnviableLauncherRootError

    if not isinstance(exc, UnviableLauncherRootError):
        return None
    return [*str(exc).splitlines(), "  The installed service definition was left unchanged."]


_JOURNAL_HINT = 'journalctl --user -u hermes-gateway --since "5 min ago"'

_SYSTEMCTL_FAILED_LINES = (
    "Could not {verb} the gateway service; systemd reported an error.",
    "See why with `hermes gateway status --deep` or `{journal}`.",
    "To reinstall the service run `hermes gateway install --force`.",
)

_NO_SYSTEMCTL_LINES = (
    "This system has no systemd, so Hermes cannot install a background service here.",
    "Run the gateway directly with `hermes gateway run` (keep it alive with tmux or screen).",
)


def _verb_for(exc: subprocess.CalledProcessError) -> str:
    cmd = exc.cmd if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd).split()
    for token in cmd:
        if token in ("start", "stop", "restart"):
            return token
    return "start"


def explain_service_failure(exc: BaseException) -> list[str] | None:
    """Lines to print for a service failure escaping the gateway command, or None when *exc* is not
    one this module knows how to explain (callers re-raise)."""
    if isinstance(exc, SystemctlUnavailableError):
        return list(_NO_SYSTEMCTL_LINES)
    if (refusal := _unviable_launcher_root_lines(exc)) is not None:
        return refusal
    if isinstance(exc, subprocess.CalledProcessError):
        lines = [line.format(verb=_verb_for(exc), journal=_JOURNAL_HINT) for line in _SYSTEMCTL_FAILED_LINES]
        lines.append(f"Details: {exc}")
        return lines
    return None
