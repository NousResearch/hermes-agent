"""Managed runtime integration helpers for the KarinAI agent."""

from __future__ import annotations

import os
from pathlib import Path
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


def _normalize_model_gateway_base_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    if text.endswith("/chat/completions"):
        text = text[: -len("/chat/completions")].rstrip("/")
    if not text.endswith("/v1"):
        text = f"{text}/v1"
    return text


def _load_yaml_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def write_managed_model_gateway_config(
    config: ManagedRuntimeConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Render managed model-provider config without persisting raw provider keys.

    Managed containers see exactly one model provider: the KarinAI model gateway.
    The gateway itself lives outside the user container and owns real provider
    credentials. The container stores only a key_env reference to the scoped
    runtime token that runtime-manager injected for this warm container.
    """

    cfg = config or load_managed_runtime_config(env)
    if not cfg.model_gateway_url:
        return None

    import yaml

    base_url = _normalize_model_gateway_base_url(cfg.model_gateway_url)
    config_path = cfg.runtime_state_path / "config.yaml"
    data = _load_yaml_mapping(config_path)
    original_data = dict(data)

    model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
    model_cfg = dict(model_cfg or {})
    model_cfg.update(
        {
            "default": cfg.model_gateway_model,
            "provider": "custom:karinai-model-gateway",
            "base_url": base_url,
            "api_mode": "chat_completions",
        }
    )
    data["model"] = model_cfg

    providers = data.get("providers") if isinstance(data.get("providers"), dict) else {}
    providers = dict(providers or {})
    providers["karinai-model-gateway"] = {
        "name": "KarinAI model gateway",
        "api": base_url,
        "key_env": "KARINAI_RUNTIME_TOKEN",
        "default_model": cfg.model_gateway_model,
        "transport": "chat_completions",
    }
    data["providers"] = providers

    if config_path.exists() and data == original_data:
        return config_path

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


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
