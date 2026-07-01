"""User-visible notices for implicit model/provider changes.

These helpers intentionally stay small and dependency-light so they can be used
from gateway startup fallback, agent runtime fallback, and primary restoration
without creating import cycles.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


_TRUE_VALUES = {"1", "true", "yes", "y", "on", "allow", "allowed"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    return False


def _hostname(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    try:
        parsed = urlparse(text if "://" in text else f"https://{text}")
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def runtime_label(provider: Any, model: Any) -> str:
    """Human-readable provider/model label for notices."""
    provider_text = _clean(provider) or "unknown"
    model_text = _clean(model) or "unknown"
    if provider_text == "unknown" and model_text != "unknown":
        return model_text
    if model_text == "unknown":
        return provider_text
    return f"{provider_text}/{model_text}"


def reason_label(reason: Any) -> str:
    if reason is None:
        return ""
    value = getattr(reason, "value", reason)
    text = _clean(value).replace("_", " ")
    return text


def fallback_entry_targets_native_anthropic(entry: dict[str, Any] | None) -> bool:
    """Return True for native Anthropic API fallback entries.

    OpenRouter models like ``anthropic/claude-*`` are intentionally not blocked
    here; this guard is for direct Anthropic API usage and Anthropic-compatible
    routes that would spend Anthropic credentials without an explicit opt-in.
    """
    if not isinstance(entry, dict):
        return False
    provider = _clean(entry.get("provider")).lower()
    if provider == "anthropic":
        return True
    base_host = _hostname(entry.get("base_url"))
    if base_host == "api.anthropic.com" or base_host.endswith(".anthropic.com"):
        return True
    api_mode = _clean(entry.get("api_mode")).lower()
    return api_mode == "anthropic_messages" and provider not in {"openrouter", ""}


def anthropic_auto_fallback_allowed(entry: dict[str, Any] | None) -> bool:
    """Whether a native Anthropic fallback entry has an explicit auto-use opt-in."""
    if not isinstance(entry, dict):
        return False
    return any(
        _boolish(entry.get(key))
        for key in (
            "allow_auto_fallback",
            "allow_anthropic_fallback",
            "allow_paid_fallback",
        )
    )


def should_block_anthropic_auto_fallback(entry: dict[str, Any] | None) -> bool:
    return fallback_entry_targets_native_anthropic(entry) and not anthropic_auto_fallback_allowed(entry)


def build_anthropic_auto_fallback_blocked_notice(entry: dict[str, Any] | None) -> str:
    provider = entry.get("provider") if isinstance(entry, dict) else "anthropic"
    model = entry.get("model") if isinstance(entry, dict) else "unknown"
    target = runtime_label(provider, model)
    return (
        f"⚠️ Automatic fallback to Anthropic API blocked: {target}. "
        "Anthropic can incur direct API costs; set allow_auto_fallback: true "
        "on that fallback entry only if you intentionally want this."
    )


def build_fallback_model_change_notice(
    old_provider: Any,
    old_model: Any,
    new_provider: Any,
    new_model: Any,
    *,
    reason: Any = None,
    base_url: Any = None,
) -> str:
    old_label = runtime_label(old_provider, old_model)
    new_label = runtime_label(new_provider, new_model)
    message = f"⚠️ Model changed without your direction: {old_label} → {new_label}."
    reason_text = reason_label(reason)
    if reason_text:
        message += f" Reason: {reason_text}."

    new_provider_text = _clean(new_provider).lower()
    base_host = _hostname(base_url)
    if new_provider_text in {"custom", "ollama", "lmstudio", "llama-cpp"} or base_host in {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
    }:
        message += " This may behave differently from the primary model."
    return message


def build_primary_runtime_restored_notice(
    old_provider: Any,
    old_model: Any,
    new_provider: Any,
    new_model: Any,
) -> str:
    return (
        "✅ Primary model restored: "
        f"{runtime_label(old_provider, old_model)} → {runtime_label(new_provider, new_model)}."
    )


def emit_model_change_notice(agent: Any, message: str) -> None:
    """Emit a model-change notice to CLI and gateway without raising."""
    text = _clean(message)
    if not text:
        return
    try:
        vprint = getattr(agent, "_vprint", None)
        if callable(vprint):
            vprint(f"{getattr(agent, 'log_prefix', '')}{text}", force=True)
    except Exception:
        pass
    callback = getattr(agent, "status_callback", None)
    if callback:
        try:
            callback("model_change", text)
        except Exception:
            try:
                import logging

                logging.getLogger(__name__).debug(
                    "status_callback error in emit_model_change_notice",
                    exc_info=True,
                )
            except Exception:
                pass
