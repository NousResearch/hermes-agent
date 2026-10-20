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

from typing import Mapping

# Exact names (Windows matches case-insensitively). Host plumbing, never a profile's settings.
_HOST_NAMES = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "TZ", "LANG", "LANGUAGE",
    "TMPDIR", "TMP", "TEMP", "PYTHONPATH", "PYTHONUTF8", "PYTHONIOENCODING", "VIRTUAL_ENV", "CONDA_PREFIX",
    # The user systemd manager (a gateway started from a system unit still scopes its children).
    "DBUS_SESSION_BUS_ADDRESS", "SUDO_USER",
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
})
# XDG_*: the user's base directories (state: the host gateway lock; config: package index).
_HOST_PREFIXES = ("LC_", "XDG_")


def worker_client_env(env: Mapping[str, str]) -> dict[str, str]:
    """*env* reduced to what the worker client, and a gateway it starts, may take from the host."""
    return {k: v for k, v in env.items() if k.upper() in _HOST_NAMES or k.upper().startswith(_HOST_PREFIXES)}
