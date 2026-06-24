"""Managed runtime integration helpers for the KarinAI agent."""

from __future__ import annotations

import os
from typing import Mapping

from .config import ManagedRuntimeConfig, parse_bool
from .prompts import render_managed_system_prompt_from_variables


def is_managed_runtime(env: Mapping[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    return parse_bool(source.get("KARINAI_MANAGED_RUNTIME"), default=False)


def load_managed_runtime_config(
    env: Mapping[str, str] | None = None,
) -> ManagedRuntimeConfig:
    return ManagedRuntimeConfig.from_env(env or os.environ)


def render_managed_system_prompt(
    config: ManagedRuntimeConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    cfg = config or load_managed_runtime_config(env)
    return render_managed_system_prompt_from_variables(cfg.prompt_variables())


def compose_ephemeral_system_prompt(
    existing_prompt: str | None,
    config: ManagedRuntimeConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Append managed policy after request instructions so product policy wins."""
    managed_prompt = render_managed_system_prompt(config, env=env)
    existing = (existing_prompt or "").strip()
    if not existing:
        return managed_prompt
    return (
        "Client-provided runtime instructions:\n"
        f"{existing}\n\n"
        "KarinAI managed runtime instructions:\n"
        f"{managed_prompt}"
    ).strip()


def managed_agent_toolsets(
    config: ManagedRuntimeConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    cfg = config or load_managed_runtime_config(env)
    return list(cfg.enabled_toolsets), list(cfg.disabled_toolsets)


def apply_managed_startup_env(
    config: ManagedRuntimeConfig | None = None,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Apply derived upstream env vars before starting the gateway process."""
    target = env if env is not None else os.environ
    cfg = config or load_managed_runtime_config(target)
    applied = cfg.gateway_env()
    for key, value in applied.items():
        target[key] = value
    return applied


def prepare_managed_runtime_filesystem(
    config: ManagedRuntimeConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    """Create the managed workspace/state directories before gateway import."""
    cfg = config or load_managed_runtime_config(env)
    for path in (
        cfg.workspace_path,
        cfg.runtime_state_path,
        cfg.runtime_state_path / "home",
    ):
        path.mkdir(parents=True, exist_ok=True)
