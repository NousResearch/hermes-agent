"""Warn when an in-session model switch will trigger preflight compression on the next turn.

Addresses part of #23767 ("user-facing guardrail when switching from a
high-context provider to a substantially lower-context provider"). The other
proposed fixes from that issue (hard preflight token guard, metadata cache
invalidation on switch, compression safety invariant, oversized tool-output
handling) are tracked separately.

Mirrors the expensive-model guard pattern: merge into ``ModelSwitchResult.warning_message``
so Herm TUI, CLI, and gateway surfaces that already show switch warnings pick it up.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
from hermes_cli.model_switch import ModelSwitchResult, resolve_display_context_length


def _append_warning(result: ModelSwitchResult, text: str) -> None:
    if result.warning_message:
        result.warning_message = f"{result.warning_message} | {text}"
    else:
        result.warning_message = text


def _threshold_tokens(context_length: int, threshold_percent: float) -> int:
    return max(int(context_length * threshold_percent), MINIMUM_CONTEXT_LENGTH)


def _estimate_tokens(agent: Any, messages: Optional[List[dict]]) -> Optional[int]:
    cc = getattr(agent, "context_compressor", None)
    if cc is None:
        return None

    if messages is not None:
        protect = int(getattr(cc, "protect_first_n", 3)) + int(
            getattr(cc, "protect_last_n", 20)
        ) + 1
        if len(messages) <= protect:
            return None
        try:
            from agent.model_metadata import estimate_request_tokens_rough

            system_prompt = getattr(agent, "_cached_system_prompt", None) or ""
            tools = getattr(agent, "tools", None)
            return int(
                estimate_request_tokens_rough(
                    messages,
                    system_prompt=system_prompt,
                    tools=tools or None,
                )
            )
        except Exception:
            pass

    last = int(getattr(cc, "last_prompt_tokens", 0) or 0)
    if last > 0:
        return last
    session_prompt = int(getattr(agent, "session_prompt_tokens", 0) or 0)
    return session_prompt if session_prompt > 0 else None


def merge_preflight_compression_warning(
    result: ModelSwitchResult,
    *,
    agent: Any = None,
    messages: Optional[List[dict]] = None,
    custom_providers: list | None = None,
    config_context_length: int | None = None,
    configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None,
) -> None:
    """If the next user message will likely preflight-compress, append a warning."""
    if not result.success or agent is None:
        return
    if not getattr(agent, "compression_enabled", True):
        return

    cc = getattr(agent, "context_compressor", None)
    if cc is None:
        return

    # Classic CLI historically omitted custom_providers here while the /model
    # confirmation display threaded agent._custom_providers — so the shrink
    # warning fell through to the hardcoded catalog (e.g. "qwen" → 131072)
    # even when custom_providers[].models.<id>.context_length was 1M.
    if custom_providers is None:
        custom_providers = getattr(agent, "_custom_providers", None)

    old_ctx = int(getattr(cc, "context_length", 0) or 0)
    new_ctx = resolve_display_context_length(
        result.new_model,
        result.target_provider,
        base_url=result.base_url or getattr(agent, "base_url", "") or "",
        api_key=result.api_key or getattr(agent, "api_key", "") or "",
        model_info=result.model_info,
        custom_providers=custom_providers,
        config_context_length=config_context_length,
        configured_model=(
            configured_model
            if configured_model is not None
            else getattr(agent, "model", None)
        ),
        configured_provider=(
            configured_provider
            if configured_provider is not None
            else getattr(agent, "provider", None)
        ),
        configured_base_url=(
            configured_base_url
            if configured_base_url is not None
            else getattr(agent, "base_url", None)
        ),
    )
    if not new_ctx:
        return

    estimate = _estimate_tokens(agent, messages)
    if estimate is None:
        return

    pct = float(getattr(cc, "threshold_percent", 0.5))
    new_threshold = _threshold_tokens(new_ctx, pct)
    if estimate < new_threshold:
        return

    if int(getattr(cc, "_ineffective_compression_count", 0) or 0) >= 2:
        return

    parts: list[str] = []
    if old_ctx and new_ctx < old_ctx:
        parts.append(
            f"Context window shrinks ({old_ctx:,} → {new_ctx:,}). "
        )
    parts.append(
        f"Session is ~{estimate:,} tokens; "
        f"{result.new_model} allows {new_ctx:,} "
        f"(auto-compress at ~{new_threshold:,}). "
        f"Your next message will run preflight compression before the model replies."
    )
    _append_warning(result, "".join(parts))


def enrich_model_switch_warnings_for_gateway(
    result: ModelSwitchResult,
    runner: Any,
    *,
    session_key: str,
    source: Any,
    custom_providers: list | None = None,
    load_gateway_config: Callable[[], dict] | None = None,
) -> None:
    """Gateway helper: cached agent + session DB messages."""
    lock = getattr(runner, "_agent_cache_lock", None)
    cache = getattr(runner, "_agent_cache", None)
    agent = None
    if lock is not None and cache is not None:
        with lock:
            entry = cache.get(session_key)
            if entry and entry[0] is not None:
                agent = entry[0]
    if agent is None:
        return

    cfg_ctx = None
    configured_model = None
    configured_provider = None
    configured_base_url = None
    if load_gateway_config is not None:
        try:
            cfg = load_gateway_config()
            model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
            if isinstance(model_cfg, dict) and model_cfg.get("context_length") is not None:
                cfg_ctx = int(model_cfg["context_length"])
                configured_model = model_cfg.get("default") or model_cfg.get("model")
                configured_provider = model_cfg.get("provider")
                configured_base_url = model_cfg.get("base_url")
        except Exception:
            pass

    messages = None
    db = getattr(runner, "_session_db", None)
    store = getattr(runner, "session_store", None)
    if db is not None and store is not None:
        try:
            entry = store.get_or_create_session(source)
            messages = db.get_messages_as_conversation(entry.session_id)
        except Exception:
            pass

    merge_preflight_compression_warning(
        result,
        agent=agent,
        messages=messages,
        custom_providers=custom_providers,
        config_context_length=cfg_ctx,
        configured_model=configured_model,
        configured_provider=configured_provider,
        configured_base_url=configured_base_url,
    )
