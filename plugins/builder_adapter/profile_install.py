"""Governed deepseek-builder profile installation for isolated Hermes homes."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .errors import AdapterError


def install_isolated_profile(
    hermes_home: str | Path, policy: dict, *, isolation_marker: str
) -> Path:
    """Install the governed profile only into an explicitly isolated temp home."""
    root = Path(hermes_home).resolve()
    live = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).resolve()
    if isolation_marker != "HERMES_BUILDER_TEST_HOME" or root == live:
        raise AdapterError(
            "AUTHORIZATION_FAILED", "live Hermes profile installation is forbidden"
        )
    if (
        policy.get("profile") != "deepseek-builder"
        or policy.get("provider") != "deepseek"
        or policy.get("model") != "deepseek-v4-pro"
        or policy.get("fallback_chain") != []
    ):
        raise AdapterError("PROFILE_POLICY_MISMATCH", "profile policy is not canonical")
    profile_dir = root / "profiles" / "deepseek-builder"
    profile_dir.mkdir(parents=True, mode=0o700)
    config = {
        "model": {"provider": "deepseek", "default": "deepseek-v4-pro"},
        "fallback_providers": [],
        "platform_toolsets": {"cli": ["builder_adapter", "no_mcp"]},
        "plugins": {"enabled": ["builder_adapter"]},
        "builder_dispatch": {
            "confinement": {
                "kind": "application_tool_mediated",
                "os_sandbox": False,
                "terminal_tools": False,
                "process_tools": False,
            }
        },
    }
    target = profile_dir / "config.yaml"
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        data = json.dumps(config, sort_keys=True, indent=2).encode() + b"\n"
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target
