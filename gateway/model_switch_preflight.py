"""Gateway preflight-compression warning after a model switch."""

from __future__ import annotations

from typing import Any, Optional

from models.metadata.context import MINIMUM_CONTEXT_LENGTH

from gateway.model_switch_display import resolve_display_context_length


def _append_warning(result, text: str) -> None:
    result.warning_message = (
        f"{result.warning_message} | {text}" if result.warning_message else text
    )


def _estimate_tokens(agent: Any, messages: Optional[list[dict]]) -> Optional[int]:
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return None
    if messages is not None:
        protect = (
            int(getattr(compressor, "protect_first_n", 3))
            + int(getattr(compressor, "protect_last_n", 20))
            + 1
        )
        if len(messages) > protect:
            try:
                from agent.model_metadata import estimate_request_tokens_rough

                return int(
                    estimate_request_tokens_rough(
                        messages,
                        system_prompt=getattr(agent, "_cached_system_prompt", None) or "",
                        tools=getattr(agent, "tools", None) or None,
                    )
                )
            except Exception:
                pass
    last = int(getattr(compressor, "last_prompt_tokens", 0) or 0)
    if last > 0:
        return last
    session = int(getattr(agent, "session_prompt_tokens", 0) or 0)
    return session if session > 0 else None


def _threshold(compressor, model: str, context_length: int, provider: str) -> int:
    preview = getattr(compressor, "preview_threshold_tokens", None)
    if callable(preview):
        return int(preview(model, context_length, provider))
    return max(
        int(context_length * float(getattr(compressor, "threshold_percent", 0.5))),
        MINIMUM_CONTEXT_LENGTH,
    )


def enrich_model_switch_warnings(
    result,
    runner,
    *,
    session_key: str,
    source,
    custom_providers: list | None,
    config: dict,
) -> None:
    lock = getattr(runner, "_agent_cache_lock", None)
    cache = getattr(runner, "_agent_cache", None)
    agent = None
    if lock is not None and cache is not None:
        with lock:
            entry = cache.get(session_key)
            if entry and entry[0] is not None:
                agent = entry[0]
    if agent is None or not getattr(agent, "compression_enabled", True):
        return
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    configured_ctx = model_cfg.get("context_length")
    try:
        configured_ctx = int(configured_ctx) if configured_ctx is not None else None
    except (TypeError, ValueError):
        configured_ctx = None
    new_ctx = resolve_display_context_length(
        result.new_model,
        result.target_provider,
        base_url=result.base_url or getattr(agent, "base_url", "") or "",
        api_key=result.api_key or getattr(agent, "api_key", "") or "",
        custom_providers=custom_providers or getattr(agent, "_custom_providers", None),
        config_context_length=configured_ctx,
        configured_model=model_cfg.get("default") or model_cfg.get("model"),
        configured_provider=model_cfg.get("provider"),
        configured_base_url=model_cfg.get("base_url"),
    )

    messages = None
    db = getattr(runner, "_session_db", None)
    store = getattr(runner, "session_store", None)
    if db is not None and store is not None:
        try:
            entry = store.get_or_create_session(source)
            messages = db.get_messages_as_conversation(entry.session_id)
        except Exception:
            pass
    estimate = _estimate_tokens(agent, messages)
    if estimate is None:
        return
    trigger = _threshold(compressor, result.new_model, new_ctx, result.target_provider)
    if estimate < trigger or int(
        getattr(compressor, "_ineffective_compression_count", 0) or 0
    ) >= 2:
        return
    old_ctx = int(getattr(compressor, "context_length", 0) or 0)
    prefix = f"Context window shrinks ({old_ctx:,} → {new_ctx:,}). " if old_ctx and new_ctx < old_ctx else ""
    _append_warning(
        result,
        (
            f"{prefix}Session is ~{estimate:,} tokens; {result.new_model} allows "
            f"{new_ctx:,} (auto-compress at ~{trigger:,}). Your next message will "
            "run preflight compression before the model replies."
        ),
    )


__all__ = ["enrich_model_switch_warnings"]
