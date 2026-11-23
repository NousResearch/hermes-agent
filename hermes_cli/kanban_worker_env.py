"""The environment the kanban dispatcher hands its worker client: an allowlist, not a scrub.

The client (``hermes_cli.kanban_worker_client``) only submits a claim to the assignee's gateway and
waits for the receipt. When that gateway is not running, the client starts it, and the daemon is
built from the client's own environment plus the assignee's secrets
(``gateway_runtime_start`` -> ``served_profile_child_env``). Anything else the client carries
therefore reaches the assignee's turns, and an overlay can replace a value but never remove one. A
scrub of the dispatcher's environment only catches the names it knows (provider keys, the
dispatcher's ``terminal.env_passthrough``); a token systemd or ``op run`` injected under any other
name went through. So the dispatcher's environment is reduced to host process plumbing and
Hermes' own location before the dispatcher adds the claim.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Mapping, Optional

# Keep a little wall-clock budget for the worker to observe a terminal timeout
# and make a terminal board call (kanban_block/kanban_complete/kanban_request_review)
# before max_runtime_seconds kills it.
KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS = 30

# Exact names (Windows matches case-insensitively). Host plumbing, never a profile's settings.
_HOST_NAMES = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "TZ", "LANG", "LANGUAGE",
    "TMPDIR", "TMP", "TEMP", "PYTHONPATH", "PYTHONUTF8", "PYTHONIOENCODING", "VIRTUAL_ENV", "CONDA_PREFIX",
    # The user systemd manager (a gateway started from a system unit still scopes its children).
    "DBUS_SESSION_BUS_ADDRESS", "SUDO_USER",
    # The login session's agent and display sockets (git over ssh, browser/computer-use tools).
    "SSH_AUTH_SOCK", "DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY",
    # TLS trust and proxies are the host's network, which a cold-started gateway needs to reach a provider.
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    # Windows: without these the CRT cannot open a socket or resolve cmd.exe.
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "OS", "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS", "PUBLIC", "ALLUSERSPROFILE", "PROGRAMDATA", "PROGRAMFILES",
    "PROGRAMFILES(X86)", "PROGRAMW6432", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "USERDOMAIN",
    "USERNAME", "HOMEDRIVE", "HOMEPATH", "COMPUTERNAME", "PROCESSOR_ARCHITEW6432",
    # Hermes' own location, install and host policy, read before any profile config loads: the
    # host-wide gateway lock directory (else a cold-started gateway takes a different "host" lock),
    # the no-install policy, and the shared board root and its lock's busy timeout.
    "HERMES_HOME", "HERMES_REAL_HOME", "HERMES_SCRATCH_DIR", "HERMES_RUNTIME_DIR", "HERMES_INSTALL_ROOT",
    "HERMES_PYTHON_SRC_ROOT", "HERMES_DATA_DIR_SUFFIX", "HERMES_HOME_MODE", "HERMES_MANAGED",
    "HERMES_CONTAINER", "HERMES_SKIP_CHMOD", "HERMES_GATEWAY_LOCK_DIR", "HERMES_DISABLE_LAZY_INSTALLS",
    "HERMES_KANBAN_HOME", "HERMES_KANBAN_BUSY_TIMEOUT_MS",
    # The administrator-managed scope directory (an IT bootstrap override, never in any profile's
    # ``.env``): without it a cold-started gateway loses the managed overlay stored elsewhere.
    "HERMES_MANAGED_DIR",
    # The XDG base directories by exact name (state: the host gateway lock; config: the package
    # index). Not the XDG_ prefix: XDG_ACTIVATION_TOKEN and any user-chosen XDG_* name can be a secret.
    "XDG_RUNTIME_DIR", "XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_CONFIG_DIRS", "XDG_DATA_HOME",
    "XDG_DATA_DIRS", "XDG_CACHE_HOME",
    # Termux detection (hermes_platform.host.runtime).
    "PREFIX", "TERMUX_VERSION",
})
_HOST_PREFIXES = ("LC_",)


def worker_client_env(env: Mapping[str, str], assignee_home: str | Path) -> dict[str, str]:
    """*env* reduced to what the worker client for *assignee_home*, and a gateway it starts, may take
    from the host. The launch profile's ``.env`` / source residue goes first: an allowlisted name it
    defines (an ``HTTPS_PROXY`` with credentials in the URL) is that profile's, not the host's."""
    from hermes_cli.config_backend import deployment_env_names
    from tools.environments.local import strip_launch_profile_env
    host = strip_launch_profile_env(dict(env), assignee_home)
    # The config backend's deployment (which backend, plane and instance) is the instance's, not a
    # profile's; without it the assignee's gateway reads its config from the file backend. Its
    # credential is not part of it: the assignee resolves its own and fails closed without it.
    names = _HOST_NAMES | {name.upper() for name in deployment_env_names()}
    return {k: v for k, v in host.items() if k.upper() in names or k.upper().startswith(_HOST_PREFIXES)}


def worker_terminal_timeout_env(
    max_runtime_seconds: Optional[int],
    current_timeout: Optional[str],
) -> Optional[str]:
    """Return a worker-scoped TERMINAL_TIMEOUT override, if needed.

    When ``max_runtime_seconds`` exceeds the terminal tool's default timeout,
    raise only the child's default so a long command isn't killed by the
    generic terminal default first.
    """
    if max_runtime_seconds is None:
        return None
    try:
        runtime = int(max_runtime_seconds)
    except (TypeError, ValueError):
        return None
    if runtime <= 0:
        return None

    desired = max(1, runtime - KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS)
    try:
        existing = int(str(current_timeout).strip()) if current_timeout else 0
    except (TypeError, ValueError):
        existing = 0
    if existing >= desired:
        return None
    return str(desired)


@contextlib.contextmanager
def worker_profile_scope(hermes_home: str):
    """Bind an assigned profile's whole runtime scope (home, secrets, terminal policy) for one
    dispatch-side read or spawn-env build.

    The dispatcher runs detached from any turn, so nothing binds a profile for it: ``load_config``,
    the toolset probes' ``get_secret`` reads and ``build_subprocess_env``'s passthrough resolution
    all fall back to the LAUNCH profile's ambient ``os.environ`` / ``TERMINAL_*``. Binding was
    previously conditional on ``is_multiplex_active()``, so on a single-profile host a worker for
    profile B was built entirely from the dispatcher's own environment.

    Home, secrets and terminal policy always move together: the spawned child only submits the
    claim to the assignee's own gateway, which runs the turn (and, if the child starts it, gets
    the assignee's own secrets from ``served_profile_child_env``).

    The secret mapping is never widened: a profile that is not this process's own home gets its own
    ``.env`` + external sources ONLY, while the launch home keeps its established
    env-over-``.env`` precedence (``launch_secret_scope``) so systemd / ``op run`` injection still
    resolves for a standalone dispatcher.
    """
    from agent.runtime_scope import bind_profile_scope
    with bind_profile_scope(hermes_home, hydrate_secrets=False):
        yield
