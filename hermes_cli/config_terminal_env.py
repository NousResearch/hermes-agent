"""The ``terminal.*`` config keys and the TERMINAL_* env vars tools.terminal_tool reads.

Split out of ``hermes_cli/config.py``.
"""

from __future__ import annotations

import json
from typing import Any, Optional


# ``terminal.<key>`` -> env var read by tools.terminal_tool. Every key maps to ``TERMINAL_<KEY>``
# except ``backend`` (historically ``TERMINAL_ENV``).
TERMINAL_CONFIG_ENV_MAP = {
    "backend": "TERMINAL_ENV",
    **{
        key: f"TERMINAL_{key.upper()}"
        for key in (
            "modal_mode", "degraded_mode", "cwd", "temp_dir", "timeout", "lifetime_seconds",
            "docker_image", "docker_forward_env", "singularity_image", "modal_image",
            "daytona_image", "vercel_runtime", "vercel_image", "ssh_host", "ssh_user", "ssh_port", "ssh_key",
            "container_cpu", "container_memory", "container_disk", "container_persistent",
            "docker_volumes", "docker_env", "docker_mount_cwd_to_workspace", "docker_network",
            "docker_extra_args", "docker_shm_size", "docker_run_as_host_user", "docker_snap_compat",
            "docker_persist_across_processes", "docker_shared_container_key",
            "docker_orphan_reaper", "bubblewrap_profile", "bubblewrap_binds",
            "bubblewrap_memory_mb", "bubblewrap_cpu_seconds", "bubblewrap_max_procs",
            "bubblewrap_home_allow", "bubblewrap_hide",
            "sandbox_dir", "persistent_shell")}}


def _terminal_env_value(value: Any) -> str:
    return json.dumps(value) if isinstance(value, (list, dict)) else str(value)


def _terminal_config_value_is_bridgeable(key: str, value: Any) -> bool:
    """Return whether a terminal config value owns its mirrored env var."""
    return not (key == "cwd" and str(value or "").strip() in {".", "auto", "cwd"})


def terminal_config_owned_env_vars(terminal_config: Any) -> set[str]:
    """Return env vars explicitly owned by a raw ``terminal`` config section."""
    if not isinstance(terminal_config, dict):
        return set()
    return {
        env_var
        for key, env_var in TERMINAL_CONFIG_ENV_MAP.items()
        if key in terminal_config
        and _terminal_config_value_is_bridgeable(key, terminal_config[key])}


def terminal_config_env_var_for_key(key: str) -> Optional[str]:
    """Return the env var mirrored by a ``terminal.*`` config key."""
    return TERMINAL_CONFIG_ENV_MAP.get(key[len("terminal."):]) if key.startswith("terminal.") else None
