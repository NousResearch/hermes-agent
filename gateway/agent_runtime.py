"""Shared gateway agent runtime/configuration helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home, parse_reasoning_effort


logger = logging.getLogger(__name__)
_hermes_home = get_hermes_home()


@dataclass(frozen=True)
class GatewayAgentRuntimeSpec:
    """Resolved runtime/config state for one gateway agent turn."""

    user_config: dict[str, Any]
    source: Any
    platform_key: str
    model: str
    runtime_kwargs: dict[str, Any]
    turn_route: dict[str, Any]
    provider_routing: dict[str, Any]
    fallback_model: list[Any] | dict[str, Any] | None
    reasoning_config: dict[str, Any] | None
    enabled_toolsets: list[str]
    combined_ephemeral: str | None
    loaded_skills: list[str]
    missing_skills: list[str]
    max_iterations: int


@dataclass(frozen=True)
class GatewayPreparedSyncTurnRuntime:
    """Resolved sync-turn runtime inputs shared by gateway foreground runs."""

    runtime_spec: GatewayAgentRuntimeSpec
    reasoning_config: dict[str, Any] | None
    max_iterations: int


def load_gateway_user_config() -> dict[str, Any]:
    """Load ~/.hermes/config.yaml, returning {} on failure."""
    try:
        config_path = _hermes_home / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        logger.debug("Could not load gateway config from %s", _hermes_home / "config.yaml")
    return {}


def reload_gateway_dotenv(
    env_path: Path,
    *,
    load_dotenv_fn: Callable[..., Any],
) -> None:
    """Reload the gateway .env file, tolerating legacy encodings."""

    try:
        load_dotenv_fn(env_path, override=True, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv_fn(env_path, override=True, encoding="latin-1")
    except Exception:
        pass


def resolve_gateway_model(config: dict[str, Any] | None = None) -> str:
    """Read model from config.yaml."""
    cfg = config if config is not None else load_gateway_user_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    if isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""


def resolve_runtime_agent_kwargs() -> dict[str, Any]:
    """Resolve provider credentials for gateway-created AIAgent instances."""
    from hermes_cli.runtime_provider import (
        format_runtime_provider_error,
        resolve_runtime_provider,
    )

    try:
        runtime = resolve_runtime_provider(
            requested=os.getenv("HERMES_INFERENCE_PROVIDER"),
        )
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
    }


def load_prefill_messages() -> list[dict[str, Any]]:
    """Load ephemeral prefill messages from config or env var."""
    file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
    if not file_path:
        try:
            cfg = load_gateway_user_config()
            file_path = cfg.get("prefill_messages_file", "")
        except Exception:
            pass
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = _hermes_home / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Prefill messages file must contain a JSON array: %s", path)
            return []
        return data
    except Exception as exc:
        logger.warning("Failed to load prefill messages from %s: %s", path, exc)
        return []


def load_ephemeral_system_prompt() -> str:
    """Load ephemeral system prompt from config or env var."""
    prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
    if prompt:
        return prompt
    try:
        cfg = load_gateway_user_config()
        return (cfg.get("agent", {}).get("system_prompt", "") or "").strip()
    except Exception:
        return ""


def load_reasoning_config() -> dict[str, Any] | None:
    """Load reasoning effort from config.yaml."""
    effort = ""
    try:
        cfg = load_gateway_user_config()
        effort = str(cfg.get("agent", {}).get("reasoning_effort", "") or "").strip()
    except Exception:
        pass
    result = parse_reasoning_effort(effort)
    if effort and result is None:
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result


def load_show_reasoning() -> bool:
    """Load show_reasoning toggle from config.yaml display section."""
    try:
        cfg = load_gateway_user_config()
        return bool(cfg.get("display", {}).get("show_reasoning", False))
    except Exception:
        return False


def load_provider_routing() -> dict[str, Any]:
    """Load OpenRouter provider routing preferences from config.yaml."""
    try:
        return load_gateway_user_config().get("provider_routing", {}) or {}
    except Exception:
        return {}


def load_fallback_model() -> list[Any] | dict[str, Any] | None:
    """Load fallback provider chain from config.yaml."""
    try:
        cfg = load_gateway_user_config()
        fb = cfg.get("fallback_providers") or cfg.get("fallback_model") or None
        if fb:
            return fb
    except Exception:
        pass
    return None


def load_smart_model_routing() -> dict[str, Any]:
    """Load optional smart cheap-vs-strong model routing config."""
    try:
        return load_gateway_user_config().get("smart_model_routing", {}) or {}
    except Exception:
        return {}


def platform_config_key(platform: Any) -> str:
    """Map a Platform enum to its config.yaml key (LOCAL→cli)."""
    value = getattr(platform, "value", str(platform))
    return "cli" if value == "local" else value


def resolve_turn_agent_config(
    user_message: str,
    model: str,
    runtime_kwargs: dict[str, Any],
    smart_model_routing: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve smart model route for the current turn."""
    from agent.smart_model_routing import resolve_turn_route

    primary = {
        "model": model,
        "api_key": runtime_kwargs.get("api_key"),
        "base_url": runtime_kwargs.get("base_url"),
        "provider": runtime_kwargs.get("provider"),
        "api_mode": runtime_kwargs.get("api_mode"),
        "command": runtime_kwargs.get("command"),
        "args": list(runtime_kwargs.get("args") or []),
        "credential_pool": runtime_kwargs.get("credential_pool"),
    }
    return resolve_turn_route(user_message, smart_model_routing or {}, primary)


def build_combined_ephemeral_prompt(
    *,
    context_prompt: str = "",
    gateway_prompt: str = "",
    platform_system_prompt: str = "",
    skill_prompt: str = "",
) -> str | None:
    """Build the combined per-turn ephemeral system prompt."""
    parts = [
        str(skill_prompt or "").strip(),
        str(context_prompt or "").strip(),
        str(gateway_prompt or "").strip(),
        str(platform_system_prompt or "").strip(),
    ]
    combined = "\n\n".join(part for part in parts if part)
    return combined or None


def build_gateway_agent_runtime(
    *,
    source: Any,
    user_message: str,
    context_prompt: str = "",
    gateway_ephemeral_system_prompt: str = "",
    provider_routing: dict[str, Any] | None = None,
    fallback_model: list[Any] | dict[str, Any] | None = None,
    smart_model_routing: dict[str, Any] | None = None,
    reasoning_config: dict[str, Any] | None = None,
    preloaded_skills: list[str] | None = None,
    skill_task_id: str | None = None,
    user_config: dict[str, Any] | None = None,
    model: str | None = None,
    runtime_kwargs: dict[str, Any] | None = None,
    enabled_toolsets: list[str] | None = None,
) -> GatewayAgentRuntimeSpec:
    """Resolve the shared runtime/configuration for one gateway turn."""
    from agent.skill_commands import build_preloaded_skills_prompt
    from hermes_cli.tools_config import _get_platform_tools

    resolved_user_config = dict(user_config or load_gateway_user_config())
    platform_key = platform_config_key(getattr(source, "platform", None))
    resolved_enabled_toolsets = (
        sorted(enabled_toolsets)
        if enabled_toolsets is not None
        else sorted(_get_platform_tools(resolved_user_config, platform_key))
    )
    resolved_model = str(model or resolve_gateway_model(resolved_user_config) or "")
    resolved_runtime_kwargs = dict(runtime_kwargs or resolve_runtime_agent_kwargs())
    if not resolved_runtime_kwargs.get("api_key"):
        raise RuntimeError("no provider credentials configured")

    skill_prompt = ""
    loaded_skills: list[str] = []
    missing_skills: list[str] = []
    if preloaded_skills:
        skill_prompt, loaded_skills, missing_skills = build_preloaded_skills_prompt(
            list(preloaded_skills or []),
            task_id=skill_task_id,
        )

    platform_system_prompt = ""
    try:
        platform_config = getattr(getattr(source, "platform", None), "value", None)
        raw_platform = (
            (resolved_user_config.get("gateway", {}) or {}).get("platforms", {}) or {}
        ).get(platform_config, {})
        extra = raw_platform.get("extra") if isinstance(raw_platform, dict) else None
        if isinstance(extra, dict):
            platform_system_prompt = str(extra.get("system_prompt") or "").strip()
    except Exception:
        platform_system_prompt = ""

    combined_ephemeral = build_combined_ephemeral_prompt(
        context_prompt=context_prompt,
        gateway_prompt=gateway_ephemeral_system_prompt,
        platform_system_prompt=platform_system_prompt,
        skill_prompt=skill_prompt,
    )
    resolved_reasoning = reasoning_config if reasoning_config is not None else load_reasoning_config()
    resolved_provider_routing = provider_routing if provider_routing is not None else load_provider_routing()
    resolved_fallback_model = fallback_model if fallback_model is not None else load_fallback_model()
    resolved_smart_routing = smart_model_routing if smart_model_routing is not None else load_smart_model_routing()
    turn_route = resolve_turn_agent_config(
        user_message,
        resolved_model,
        resolved_runtime_kwargs,
        resolved_smart_routing,
    )
    return GatewayAgentRuntimeSpec(
        user_config=resolved_user_config,
        source=source,
        platform_key=platform_key,
        model=resolved_model,
        runtime_kwargs=resolved_runtime_kwargs,
        turn_route=turn_route,
        provider_routing=resolved_provider_routing,
        fallback_model=resolved_fallback_model,
        reasoning_config=resolved_reasoning,
        enabled_toolsets=resolved_enabled_toolsets,
        combined_ephemeral=combined_ephemeral,
        loaded_skills=loaded_skills,
        missing_skills=missing_skills,
        max_iterations=int(os.getenv("HERMES_MAX_ITERATIONS", "90")),
    )


def prepare_gateway_sync_turn_runtime(
    *,
    env_path: Path,
    load_dotenv_fn: Callable[..., Any],
    resolve_runtime_agent_kwargs_fn: Callable[[], dict[str, Any]],
    load_reasoning_config_fn: Callable[[], dict[str, Any] | None],
    source: Any,
    user_message: str,
    context_prompt: str = "",
    gateway_ephemeral_system_prompt: str = "",
    provider_routing: dict[str, Any] | None = None,
    fallback_model: list[Any] | dict[str, Any] | None = None,
    smart_model_routing: dict[str, Any] | None = None,
    user_config: dict[str, Any] | None = None,
    model: str | None = None,
    enabled_toolsets: list[str] | None = None,
    preloaded_skills: list[str] | None = None,
    skill_task_id: str | None = None,
) -> GatewayPreparedSyncTurnRuntime:
    """Reload env/config and build the shared sync-turn runtime payload."""

    reload_gateway_dotenv(env_path, load_dotenv_fn=load_dotenv_fn)
    runtime_kwargs = resolve_runtime_agent_kwargs_fn()
    reasoning_config = load_reasoning_config_fn()
    runtime_spec = build_gateway_agent_runtime(
        source=source,
        user_message=user_message,
        context_prompt=context_prompt,
        gateway_ephemeral_system_prompt=gateway_ephemeral_system_prompt,
        provider_routing=provider_routing,
        fallback_model=fallback_model,
        smart_model_routing=smart_model_routing,
        reasoning_config=reasoning_config,
        preloaded_skills=preloaded_skills,
        skill_task_id=skill_task_id,
        user_config=user_config,
        model=model,
        runtime_kwargs=runtime_kwargs,
        enabled_toolsets=enabled_toolsets,
    )
    return GatewayPreparedSyncTurnRuntime(
        runtime_spec=runtime_spec,
        reasoning_config=reasoning_config,
        max_iterations=int(os.getenv("HERMES_MAX_ITERATIONS", "90")),
    )


def agent_config_signature(
    model: str,
    runtime: dict[str, Any],
    enabled_toolsets: list[str],
    ephemeral_prompt: str,
) -> str:
    """Compute a stable key from runtime values for agent cache reuse."""
    api_key = str(runtime.get("api_key", "") or "")
    api_key_fingerprint = hashlib.sha256(api_key.encode()).hexdigest() if api_key else ""
    blob = json.dumps(
        [
            model,
            api_key_fingerprint,
            runtime.get("base_url", ""),
            runtime.get("provider", ""),
            runtime.get("api_mode", ""),
            sorted(enabled_toolsets) if enabled_toolsets else [],
            ephemeral_prompt or "",
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
