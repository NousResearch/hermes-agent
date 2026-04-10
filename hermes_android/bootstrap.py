from __future__ import annotations

from typing import Any

from hermes_android.bundled_assets import configure_skill_env, sync_bundled_skills
from hermes_android.mobile_defaults import ensure_android_defaults, resolved_android_api_server_toolsets
from hermes_android.runtime_env import AndroidRuntimeEnv, prepare_runtime_env


def bootstrap_android_runtime(
    files_dir: str,
    *,
    api_server_port: int | None = None,
    api_server_key: str | None = None,
) -> dict[str, Any]:
    runtime = prepare_runtime_env(
        files_dir,
        api_server_port=api_server_port,
        api_server_key=api_server_key,
    )
    config = ensure_android_defaults(persist=True)
    skill_env = configure_skill_env()
    synced = sync_bundled_skills(quiet=True)
    return {
        "runtime": runtime.to_dict(),
        "toolsets": resolved_android_api_server_toolsets(config),
        "skill_env": skill_env,
        "skills_sync": synced,
    }


__all__ = ["AndroidRuntimeEnv", "bootstrap_android_runtime"]
