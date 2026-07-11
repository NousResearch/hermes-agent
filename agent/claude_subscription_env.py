"""Minimal subprocess environment for Claude subscription authentication.

The Claude Agent SDK spawns Claude Code. Passing ``os.environ`` would expose
every provider, gateway, and tool credential loaded by Hermes. This module
uses an allowlist. Claude Code receives the real ``HOME`` solely so its macOS
Keychain-backed Max login resolves; the SDK's strict filesystem sandbox denies
Bash access to that home outside the worker workspace.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from hermes_constants import get_host_user_home


_SAFE_INHERITED_ENV = frozenset(
    {
        "COLORTERM",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "NODE_EXTRA_CA_CERTS",
        "PATH",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "TZ",
        "USER",
    }
)


def build_claude_subscription_env(
    inherited: Mapping[str, str] | None = None,
    *,
    host_home: str | Path | None = None,
    profile_home: str | Path | None = None,
) -> dict[str, str]:
    """Build the complete environment passed to the Claude Code subprocess.

    The result is constructed from a safe allowlist rather than sanitized by
    deletion, so newly introduced secret names cannot leak by default.
    """

    source = inherited if inherited is not None else os.environ
    env = {
        key: str(source[key])
        for key in _SAFE_INHERITED_ENV
        if source.get(key) is not None
    }

    host_home_value = host_home if host_home is not None else get_host_user_home()
    if not host_home_value:
        raise RuntimeError("Claude subscription runtime could not resolve the host home")
    resolved_host_home = Path(host_home_value).expanduser()

    # Claude Max login is stored in the OS account's secure storage and is
    # keyed by the real HOME/USER identity. The SDK sandbox separately denies
    # Bash filesystem access outside the worker workspace.
    env["HOME"] = str(resolved_host_home)
    env["DISABLE_TELEMETRY"] = "1"
    env["DISABLE_ERROR_REPORTING"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


__all__ = ["build_claude_subscription_env"]
