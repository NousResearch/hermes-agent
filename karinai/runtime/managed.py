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


def write_managed_gateway_config(
    config: ManagedRuntimeConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Render managed runtime config without persisting raw provider keys.

    Managed containers see exactly one model provider: the KarinAI model gateway,
    and exactly one image-generation backend when image generation is configured:
    the KarinAI image gateway. Both gateways live outside the user container and
    own real provider credentials. The container stores only backend/plugin
    selection plus ``key_env`` references to the scoped runtime token that
    runtime-manager injected for this warm container.

    Managed API-server runs have no interactive user approval channel today.
    Leaving upstream approvals.mode at its default manual value can therefore
    wedge a run on an approval request until the backend request times out. The
    product safety boundary for this mode is the KarinAI sandbox/tool policy and
    backend prompt filter, so managed runtime config explicitly disables
    interactive approvals while preserving upstream hardline blocks.
    """

    cfg = config or load_managed_runtime_config(env)

    import yaml

    config_path = cfg.runtime_state_path / "config.yaml"
    data = _load_yaml_mapping(config_path)
    original_data = dict(data)

    approvals_cfg = data.get("approvals") if isinstance(data.get("approvals"), dict) else {}
    approvals_cfg = dict(approvals_cfg or {})
    approvals_cfg["mode"] = "off"
    data["approvals"] = approvals_cfg

    if cfg.model_gateway_url:
        base_url = _normalize_model_gateway_base_url(cfg.model_gateway_url)
        data["model"] = {
            "default": cfg.model_gateway_model,
            "provider": "custom:karinai-model-gateway",
            "base_url": base_url,
            "api_mode": cfg.model_gateway_api_mode,
        }
        data["providers"] = {
            "karinai-model-gateway": {
                "name": "KarinAI model gateway",
                "api": base_url,
                "key_env": "KARINAI_RUNTIME_TOKEN",
                "default_model": cfg.model_gateway_model,
                "transport": cfg.model_gateway_api_mode,
            }
        }

    if cfg.image_gateway_url:
        image_cfg = {"provider": "karinai-image-gateway"}
        if cfg.image_gateway_model:
            image_cfg["model"] = cfg.image_gateway_model
        data["image_gen"] = image_cfg
    else:
        data.pop("image_gen", None)

    if config_path.exists() and data == original_data:
        return config_path

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def write_managed_model_gateway_config(
    config: ManagedRuntimeConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Backward-compatible alias for the broader managed gateway renderer."""
    return write_managed_gateway_config(config, env=env)


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
