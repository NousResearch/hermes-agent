"""systemctl invocation plumbing, moved verbatim from ``hermes_cli.gateway`` so that file keeps
going down toward its code-health line cap (moved code keeps its cap; see root AGENTS.md).
Callers import these names from :mod:`hermes_cli.gateway`, which re-exports them, so existing
import paths are unchanged.
"""

from __future__ import annotations

import subprocess
import sys

from hermes_cli.debug_redaction import redact_debug_support_text


def _run_systemctl(args: list[str], *, system: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run systemctl; raise RuntimeError (not raw FileNotFoundError) if missing, for callers bypassing
    ``supports_systemd_services()``."""
    from hermes_cli.gateway import _systemctl_cmd  # lazy: gateway owns the env bootstrap helpers
    try:
        return subprocess.run(_systemctl_cmd(system) + args, **kwargs)  # noqa: PLW1510 -- forwarding wrapper: callers pass subprocess kwargs via **kwargs (check may arrive through it); an explicit check=False would raise TypeError
    except FileNotFoundError:
        from hermes_cli.gateway_command_errors import SystemctlUnavailableError
        raise SystemctlUnavailableError() from None


def _service_scope_label(system: bool = False) -> str:
    return "system" if system else "user"


def print_systemctl_status(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(redact_debug_support_text(result.stdout), end="")
    if result.stderr:
        print(redact_debug_support_text(result.stderr), end="", file=sys.stderr)
