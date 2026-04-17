"""Shared runtime helpers for gateway session hygiene auto-compression."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.model_metadata import (
    estimate_messages_tokens_rough,
    get_model_context_length,
)

_DEFAULT_HYGIENE_MODEL = "anthropic/claude-sonnet-4.6"
_HARD_MSG_LIMIT = 400
_HYGIENE_THRESHOLD_PCT = 0.85
_WARN_THRESHOLD_PCT = 0.95


@dataclass(slots=True)
class SessionHygieneRuntimeConfig:
    """Resolved runtime config for gateway session hygiene checks."""

    model: str = _DEFAULT_HYGIENE_MODEL
    compression_enabled: bool = True
    config_context_length: int | None = None
    provider: str = ""
    base_url: str = ""
    api_key: str = ""


def load_session_hygiene_runtime_config(
    *,
    hermes_home: Path,
    runtime_agent_kwargs_loader: Callable[[], dict[str, Any]],
) -> SessionHygieneRuntimeConfig:
    """Resolve the model/runtime settings used for gateway session hygiene."""

    config = SessionHygieneRuntimeConfig()
    config_data: dict[str, Any] = {}

    try:
        config_path = hermes_home / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path, encoding="utf-8") as handle:
                config_data = yaml.safe_load(handle) or {}

            model_cfg = config_data.get("model", {})
            if isinstance(model_cfg, str):
                config.model = model_cfg
            elif isinstance(model_cfg, dict):
                config.model = (
                    model_cfg.get("default")
                    or model_cfg.get("model")
                    or config.model
                )
                raw_ctx = model_cfg.get("context_length")
                if raw_ctx is not None:
                    try:
                        config.config_context_length = int(raw_ctx)
                    except (TypeError, ValueError):
                        pass
                config.provider = str(model_cfg.get("provider") or "")
                config.base_url = str(model_cfg.get("base_url") or "")

            compression_cfg = config_data.get("compression", {})
            if isinstance(compression_cfg, dict):
                config.compression_enabled = str(
                    compression_cfg.get("enabled", True)
                ).lower() in ("true", "1", "yes")
    except Exception:
        pass

    try:
        runtime = runtime_agent_kwargs_loader() or {}
    except Exception:
        runtime = {}

    if not config.provider:
        config.provider = str(runtime.get("provider") or "")
    if not config.base_url:
        config.base_url = str(runtime.get("base_url") or "")
    config.api_key = str(runtime.get("api_key") or "")

    if config.config_context_length is None and config.base_url:
        try:
            custom_providers = config_data.get("custom_providers")
            if isinstance(custom_providers, list):
                for provider_cfg in custom_providers:
                    if not isinstance(provider_cfg, dict):
                        continue
                    provider_url = str(provider_cfg.get("base_url") or "").rstrip("/")
                    if provider_url and provider_url == config.base_url.rstrip("/"):
                        model_map = provider_cfg.get("models", {})
                        if not isinstance(model_map, dict):
                            break
                        model_cfg = model_map.get(config.model, {})
                        if not isinstance(model_cfg, dict):
                            break
                        ctx_value = model_cfg.get("context_length")
                        if ctx_value is not None:
                            config.config_context_length = int(ctx_value)
                        break
        except (TypeError, ValueError):
            pass

    return config


def session_hygiene_token_snapshot(
    history: list[dict[str, Any]],
    *,
    stored_tokens: int,
) -> tuple[int, str]:
    """Return the best available prompt token estimate for hygiene checks."""

    if stored_tokens > 0:
        return stored_tokens, "actual"
    return estimate_messages_tokens_rough(history), "estimated"


def should_auto_compress_session_history(
    *,
    history: list[dict[str, Any]],
    approx_tokens: int,
    context_length: int,
    threshold_pct: float = _HYGIENE_THRESHOLD_PCT,
    hard_msg_limit: int = _HARD_MSG_LIMIT,
) -> bool:
    """Return True when gateway session hygiene should auto-compress history."""

    compress_token_threshold = int(context_length * threshold_pct)
    return (
        approx_tokens >= compress_token_threshold
        or len(history) >= hard_msg_limit
    )


def compression_candidate_messages(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter transcript entries down to the messages relevant for compression."""

    return [
        {"role": message.get("role"), "content": message.get("content")}
        for message in history
        if message.get("role") in ("user", "assistant") and message.get("content")
    ]


async def maybe_auto_compress_session_history(
    *,
    history: list[dict[str, Any]] | None,
    session_entry: Any,
    session_store: Any,
    hermes_home: Path,
    runtime_agent_kwargs_loader: Callable[[], dict[str, Any]],
    logger,
) -> list[dict[str, Any]]:
    """Silently auto-compress oversized gateway transcripts when needed."""

    if not history or len(history) < 4:
        return list(history or [])

    config = load_session_hygiene_runtime_config(
        hermes_home=hermes_home,
        runtime_agent_kwargs_loader=runtime_agent_kwargs_loader,
    )
    if not config.compression_enabled:
        return history

    context_length = get_model_context_length(
        config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        config_context_length=config.config_context_length,
        provider=config.provider,
    )
    compress_token_threshold = int(context_length * _HYGIENE_THRESHOLD_PCT)
    warn_token_threshold = int(context_length * _WARN_THRESHOLD_PCT)
    approx_tokens, token_source = session_hygiene_token_snapshot(
        history,
        stored_tokens=int(getattr(session_entry, "last_prompt_tokens", 0) or 0),
    )

    if not should_auto_compress_session_history(
        history=history,
        approx_tokens=approx_tokens,
        context_length=context_length,
    ):
        return history

    logger.info(
        "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
        "(threshold: %s%% of %s = %s tokens)",
        len(history),
        f"{approx_tokens:,}",
        token_source,
        int(_HYGIENE_THRESHOLD_PCT * 100),
        f"{context_length:,}",
        f"{compress_token_threshold:,}",
    )

    try:
        from run_agent import AIAgent

        runtime_kwargs = runtime_agent_kwargs_loader() or {}
        if not runtime_kwargs.get("api_key"):
            return history

        hygiene_messages = compression_candidate_messages(history)
        if len(hygiene_messages) < 4:
            return history

        hygiene_agent = AIAgent(
            **runtime_kwargs,
            model=config.model,
            max_iterations=4,
            quiet_mode=True,
            enabled_toolsets=["memory"],
            session_id=session_entry.session_id,
        )
        hygiene_agent._print_fn = lambda *args, **kwargs: None

        loop = asyncio.get_event_loop()
        compressed, _ = await loop.run_in_executor(
            None,
            lambda: hygiene_agent._compress_context(
                hygiene_messages,
                "",
                approx_tokens=approx_tokens,
            ),
        )

        new_session_id = hygiene_agent.session_id
        if new_session_id != session_entry.session_id:
            session_entry.session_id = new_session_id
            session_store._save()

        session_store.rewrite_transcript(session_entry.session_id, compressed)
        session_entry.last_prompt_tokens = 0

        new_tokens = estimate_messages_tokens_rough(compressed)
        logger.info(
            "Session hygiene: compressed %s → %s msgs, ~%s → ~%s tokens",
            len(history),
            len(compressed),
            f"{approx_tokens:,}",
            f"{new_tokens:,}",
        )
        if new_tokens >= warn_token_threshold:
            logger.warning(
                "Session hygiene: still ~%s tokens after compression",
                f"{new_tokens:,}",
            )
        return compressed
    except Exception as exc:
        logger.warning("Session hygiene auto-compress failed: %s", exc)
        return history
