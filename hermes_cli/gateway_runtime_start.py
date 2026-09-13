"""One detached-process primitive; callers must first prove owner/service absence."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from hermes_cli._subprocess_compat import windows_detach_popen_kwargs, _WINDOWS_GATEWAY_BREAKAWAY_ENV
from hermes_cli.gateway_runtime_service import RuntimeStartError, remaining


def spawn_unmanaged_gateway(profile_home: Path, *, deadline: float) -> subprocess.Popen:
    """Request a daemon, not readiness. Refuse Windows no-breakaway fallback.

    No --replace, persistence, login changes, elevation, shell, or inherited stdio.
    The gateway runtime's own exclusive ownership fence arbitrates racing starts.
    """
    home = profile_home.resolve()
    root = Path(__file__).resolve().parent.parent
    # A root home is only pinned by an explicit selector: without one the child's
    # _apply_profile_override follows the sticky active_profile and boots the wrong
    # profile's daemon. A <root>/profiles/<name> home is already trusted as-is.
    selector = [] if home.parent.name == "profiles" else ["--profile", "default"]
    command = [sys.executable, "-m", "hermes_cli.main", *selector, "gateway", "run", "--quiet"]
    env = dict(os.environ)
    if sys.platform == "win32":
        from hermes_cli.gateway_windows import windowless_gateway_restart_spec
        command, _, overlay = windowless_gateway_restart_spec(command)
        if not overlay:
            raise RuntimeStartError("windows_interpreter_unavailable")
        env.update(overlay)
        env[_WINDOWS_GATEWAY_BREAKAWAY_ENV] = "1"
    env.update(HERMES_HOME=str(home), HERMES_GATEWAY_DETACHED="1", PYTHONIOENCODING="utf-8")
    # Profile selection is the explicit home, not the invoking client's display
    # name. Runtime policy comes from that profile, never a launcher's --yolo.
    env.pop("HERMES_PROFILE", None)
    env.pop("HERMES_YOLO_MODE", None)
    remaining(deadline)
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        with (logs / "gateway-stdio.log").open("ab", buffering=0) as output:
            remaining(deadline)
            return subprocess.Popen(command, cwd=root, env=env, close_fds=True,
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                    **windows_detach_popen_kwargs())
    except OSError as exc:
        reason = "windows_breakaway_unavailable" if sys.platform == "win32" else "spawn_failed"
        raise RuntimeStartError(reason) from exc
