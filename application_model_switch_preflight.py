"""Application preflight warning for a model change that triggers compression."""

from __future__ import annotations

from typing import Any, List, Optional

from models.metadata.context import MINIMUM_CONTEXT_LENGTH
from types import SimpleNamespace

from application_model_switch_persistence import route_changed
from models.metadata.context import get_model_context_length


def resolve_display_context_length(
    model: str, provider: str, base_url: str = "", api_key: Any = "",
    model_info: Any = None, custom_providers: list | None = None,
    config_context_length: int | None = None, configured_model: str | None = None,
    configured_provider: str | None = None, configured_base_url: str | None = None,
) -> Optional[int]:
    """Read canonical context metadata; only a matching persisted route keeps its pin."""
    pinned = config_context_length
    if pinned is not None and (configured_model or configured_provider or configured_base_url):
        old = {"default": configured_model, "provider": configured_provider,
               "base_url": configured_base_url}
        new = SimpleNamespace(new_model=model, target_provider=provider, base_url=base_url)
        if (configured_model and configured_model != model) or route_changed(old, new):
            pinned = None
    try:
        size = get_model_context_length(
            model, base_url=base_url or "", api_key=api_key or "",
            provider=provider or None, custom_providers=custom_providers,
            config_context_length=pinned,
        )
        if size:
            return int(size)
    except Exception:
        pass
    fallback = getattr(model_info, "context_window", None)
    return int(fallback) if fallback else None


def _append_warning(result: Any, text: str) -> None:
    if result.warning_message:
        result.warning_message = f"{result.warning_message} | {text}"
    else:
        result.warning_message = text


def _threshold_tokens(compressor: Any, model: str, context_length: int, provider: str = "") -> int:
    """The trigger the compressor WILL use after the switch (cap, model_thresholds and small-window
    floor included), so the warning quotes the real number; duck-typed engines keep the plain ratio."""
    preview = getattr(compressor, "preview_threshold_tokens", None)
    if callable(preview):
        return int(preview(model, context_length, provider))
    return max(int(context_length * float(getattr(compressor, "threshold_percent", 0.5))), MINIMUM_CONTEXT_LENGTH)


def _estimate_tokens(agent: Any, messages: Optional[List[dict]]) -> Optional[int]:
    cc = getattr(agent, "context_compressor", None)
    if cc is None:
        return None

    if messages is not None:
        protect = (
            int(getattr(cc, "protect_first_n", 3)) + int(getattr(cc, "protect_last_n", 20)) + 1)
        if len(messages) <= protect:
            return None
        try:
            from agent.model_metadata import estimate_request_tokens_rough

            system_prompt = getattr(agent, "_cached_system_prompt", None) or ""
            tools = getattr(agent, "tools", None)
            return int(
                estimate_request_tokens_rough(
                    messages, system_prompt=system_prompt, tools=tools or None))
        except Exception:
            pass

    last = int(getattr(cc, "last_prompt_tokens", 0) or 0)
    if last > 0:
        return last
    session_prompt = int(getattr(agent, "session_prompt_tokens", 0) or 0)
    return session_prompt if session_prompt > 0 else None


def merge_preflight_compression_warning(
    result: Any,
    *,
    agent: Any = None,
    messages: Optional[List[dict]] = None,
    custom_providers: list | None = None,
    config_context_length: int | None = None,
    configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None) -> None:
    """If the next user message will likely preflight-compress, append a warning."""
    if not result.success or agent is None:
        return
    if not getattr(agent, "compression_enabled", True):
        return

    cc = getattr(agent, "context_compressor", None)
    if cc is None:
        return

    # Fall back to the agent's custom providers: without them the shrink warning used the
    # hardcoded catalog (e.g. "qwen" → 131072) even when the provider declared 1M.
    if custom_providers is None:
        custom_providers = getattr(agent, "_custom_providers", None)

    def _or_agent(value, attr):
        return value if value is not None else getattr(agent, attr, None)

    old_ctx = int(getattr(cc, "context_length", 0) or 0)
    new_ctx = resolve_display_context_length(
        result.new_model,
        result.target_provider,
        base_url=result.base_url or getattr(agent, "base_url", "") or "",
        api_key=result.api_key or getattr(agent, "api_key", "") or "",
        model_info=result.model_info,
        custom_providers=custom_providers,
        config_context_length=config_context_length,
        configured_model=_or_agent(configured_model, "model"),
        configured_provider=_or_agent(configured_provider, "provider"),
        configured_base_url=_or_agent(configured_base_url, "base_url"))
    if not new_ctx:
        return

    estimate = _estimate_tokens(agent, messages)
    if estimate is None:
        return

    new_threshold = _threshold_tokens(cc, result.new_model, new_ctx, result.target_provider)
    if estimate < new_threshold:
        return

    if int(getattr(cc, "_ineffective_compression_count", 0) or 0) >= 2:
        return

    parts: list[str] = []
    if old_ctx and new_ctx < old_ctx:
        parts.append(f"Context window shrinks ({old_ctx:,} → {new_ctx:,}). ")
    parts.append(
        f"Session is ~{estimate:,} tokens; "
        f"{result.new_model} allows {new_ctx:,} "
        f"(auto-compress at ~{new_threshold:,}). "
        f"Your next message will run preflight compression before the model replies.")
    _append_warning(result, "".join(parts))
