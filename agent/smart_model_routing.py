"""Complexity-based cheap-model routing for the main agent turn.

Activates the optional ``smart_model_routing`` config block: when a user
message looks "simple" (short, low-token, single-shot), the per-turn
runtime swaps to ``smart_model_routing.cheap_model`` for the duration of
that turn only. The original runtime is restored at the start of the
next turn.

This is **separate** from ``auxiliary.*.model`` (per-side-task overrides,
wired in agent/auxiliary_client.py) and from ``delegation.model``
(subagent global override). Smart routing is the only knob that touches
the main agent's own turn.

Design contract:

- Per-turn scoped: at most one swap per turn, restored before the next
  turn's first API call. Never leaks across turns.
- Cache-safe: when the active runtime changes, the cached system prompt's
  ``Model:``/``Provider:`` lines are rewritten in-place via
  ``rewrite_prompt_model_identity`` so the new backend's identity is
  correct. The persisted session-DB row keeps the primary's labels, so
  restoration yields a byte-identical prompt and the primary's prefix
  cache still matches.
- Fail-soft: any failure to build the cheap client is logged and skipped
  — the primary runtime continues untouched.
- Additive: the block is inert (no-op) when ``enabled=false`` or
  ``cheap_model`` is missing. Existed as a stub in setup.py before this
  implementation; now wired.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_DEFAULT_MAX_SIMPLE_CHARS = 200
_DEFAULT_MAX_SIMPLE_WORDS = 35


def _get_smart_routing_cfg(agent) -> Optional[Dict[str, Any]]:
    """Read ``smart_model_routing`` from the agent's loaded config.

    Returns the dict or ``None`` when the block is absent / disabled.
    """
    cfg = getattr(agent, "_smart_routing_cfg", None)
    if cfg is None:
        try:
            from hermes_cli.config import load_config
            cfg = (load_config() or {}).get("smart_model_routing") or {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("smart_model_routing: config load failed: %s", exc)
            cfg = {}
        agent._smart_routing_cfg = cfg
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return None
    return cfg


def _is_simple_message(text: str, max_chars: int, max_words: int) -> bool:
    """Cheap classifier: short + low-word-count = 'simple'.

    Designed to capture acknowledgements, status checks, single-line
    questions, confirmations. Will be wrong on edge cases ("hi" routed
    to M3 instead of cheap is the failure mode we're protecting against,
    not the inverse) — the classifier errs on the side of the primary
    model for anything unclear.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if len(text) > max_chars:
        return False
    word_count = len(re.findall(r"\S+", text))
    return word_count <= max_words


def _resolve_smart_target(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull provider/model/base_url/api_key hint from the config block.

    Returns a dict compatible with ``resolve_provider_client`` or ``None``
    when the block is malformed.
    """
    cheap = cfg.get("cheap_model")
    if not isinstance(cheap, dict):
        return None
    provider = str(cheap.get("provider") or "").strip()
    model = str(cheap.get("model") or "").strip()
    if not provider or not model:
        return None
    target = {"provider": provider, "model": model}
    base_url = cheap.get("base_url")
    if isinstance(base_url, str) and base_url.strip():
        target["base_url"] = base_url.strip()
    key_env = cheap.get("key_env") or cheap.get("api_key_env")
    if isinstance(key_env, str) and key_env.strip():
        target["key_env"] = key_env.strip()
    api_key = cheap.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        target["api_key"] = api_key.strip()
    return target


def _snapshot_primary_runtime(agent) -> Dict[str, Any]:
    """Capture the values ``apply_smart_routing`` will swap.

    Kept as a separate attribute (``_pre_smart_state``) so we don't
    disturb ``agent._primary_runtime`` — the existing fallback machinery
    continues to read/write that snapshot.
    """
    snap: Dict[str, Any] = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": dict(getattr(agent, "_client_kwargs", {}) or {}),
        "use_prompt_caching": getattr(agent, "_use_prompt_caching", False),
        "use_native_cache_layout": getattr(agent, "_use_native_cache_layout", False),
    }
    # Capture anthropic-specific fields so restore can rebuild the native
    # client if the primary is an Anthropic Messages backend.
    if getattr(agent, "api_mode", "") == "anthropic_messages":
        snap["anthropic_api_key"] = getattr(agent, "_anthropic_api_key", "")
        snap["anthropic_base_url"] = getattr(agent, "_anthropic_base_url", "")
        snap["is_anthropic_oauth"] = getattr(agent, "_is_anthropic_oauth", False)
    # Capture the context-compressor primary snapshot so restore re-instates
    # the primary's limits.
    cc = getattr(agent, "context_compressor", None)
    if cc is not None:
        snap["compressor_model"] = getattr(cc, "model", agent.model)
        snap["compressor_base_url"] = getattr(cc, "base_url", agent.base_url)
        snap["compressor_api_key"] = getattr(cc, "api_key", "")
        snap["compressor_provider"] = getattr(cc, "provider", agent.provider)
        snap["compressor_context_length"] = cc.context_length
    return snap


def apply_smart_routing(agent, user_message: str) -> bool:
    """If the message is simple and the block is enabled, swap runtime.

    Idempotent: if a swap is already in effect for this turn (state is
    non-empty) we no-op (the cheap runtime was already applied upstream).

    Returns True when a swap was applied this call.
    """
    if getattr(agent, "_smart_routed_active", False):
        # Already cheap for this turn — first call did the swap.
        return True
    if getattr(agent, "_fallback_activated", False):
        # Provider is in failover mid-session; the existing fallback chain
        # already picked a runtime. Don't override it.
        return False

    cfg = _get_smart_routing_cfg(agent)
    if not cfg:
        return False

    target = _resolve_smart_target(cfg)
    if not target:
        logger.debug("smart_model_routing: cheap_model missing provider/model")
        return False

    max_chars = int(cfg.get("max_simple_chars", _DEFAULT_MAX_SIMPLE_CHARS) or _DEFAULT_MAX_SIMPLE_CHARS)
    max_words = int(cfg.get("max_simple_words", _DEFAULT_MAX_SIMPLE_WORDS) or _DEFAULT_MAX_SIMPLE_WORDS)
    if not _is_simple_message(user_message, max_chars, max_words):
        return False

    provider = target["provider"]
    model = target["model"]

    try:
        from agent.auxiliary_client import resolve_provider_client
        fb_base_url = target.get("base_url") or ""
        fb_api_key = target.get("api_key") or ""
        if not fb_api_key and target.get("key_env"):
            import os
            fb_api_key = os.getenv(target["key_env"], "").strip() or ""
        client, resolved_model = resolve_provider_client(
            provider,
            model=model,
            raw_codex=True,
            explicit_base_url=fb_base_url or None,
            explicit_api_key=fb_api_key or None,
        )
    except Exception as exc:
        logger.warning(
            "smart_model_routing: provider resolve failed (%s/%s): %s",
            provider, model, exc,
        )
        return False
    if client is None:
        logger.debug(
            "smart_model_routing: provider not configured (%s/%s)",
            provider, model,
        )
        return False

    try:
        from hermes_cli.model_normalize import normalize_model_for_provider
        resolved_model = normalize_model_for_provider(resolved_model or model, provider)
    except Exception:
        resolved_model = resolved_model or model

    # Determine api_mode from base_url / provider.
    fb_base_url = str(client.base_url)
    api_mode = _detect_api_mode_for(agent, provider, fb_base_url)

    # Capture BEFORE mutating, so restore can put things back exactly.
    agent._pre_smart_state = _snapshot_primary_runtime(agent)

    agent.model = resolved_model
    agent.provider = provider
    agent.base_url = fb_base_url
    agent.api_mode = api_mode
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()

    # Mirror try_activate_fallback: the resolved client already carries the
    # right api_key + headers. Wire it up directly, and persist the resolved
    # auth/URL into agent.api_key + agent._client_kwargs so downstream
    # rebuilds (e.g. credential rotation) keep using the cheap provider.
    try:
        if api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
            effective_key = (
                getattr(client, "api_key", "") or resolve_anthropic_token() or ""
            ) if provider == "anthropic" else getattr(client, "api_key", "") or ""
            agent.api_key = effective_key
            agent._anthropic_api_key = effective_key
            agent._anthropic_base_url = fb_base_url
            _fb_timeout = _get_provider_timeout(agent, provider, resolved_model)
            agent._anthropic_client = build_anthropic_client(
                effective_key, agent._anthropic_base_url, timeout=_fb_timeout,
            )
            try:
                from agent.anthropic_adapter import _is_oauth_token
                agent._is_anthropic_oauth = (
                    _is_oauth_token(effective_key) if provider == "anthropic" else False
                )
            except Exception:
                agent._is_anthropic_oauth = False
            agent.client = None
            agent._client_kwargs = {}
        else:
            agent.api_key = getattr(client, "api_key", "") or ""
            agent.client = client
            fb_headers = getattr(client, "_custom_headers", None) or getattr(
                client, "default_headers", None
            )
            agent._client_kwargs = {
                "api_key": agent.api_key,
                "base_url": fb_base_url,
                **({"default_headers": dict(fb_headers)} if fb_headers else {}),
            }
            _fb_timeout = _get_provider_timeout(agent, provider, resolved_model)
            if _fb_timeout is not None:
                agent._client_kwargs["timeout"] = _fb_timeout
                try:
                    agent._replace_primary_openai_client(reason="smart_routing_timeout")
                except Exception:
                    pass
    except Exception as exc:
        # Fail-soft: log + revert so we don't leave a half-swapped state.
        logger.warning(
            "smart_model_routing: client swap failed (%s/%s): %s",
            provider, resolved_model, exc,
        )
        restore_smart_routing(agent)
        return False

    # Re-evaluate prompt caching for the new provider/model (mirrors failover).
    try:
        agent._use_prompt_caching, agent._use_native_cache_layout = (
            agent._anthropic_prompt_cache_policy(
                provider=provider,
                base_url=fb_base_url,
                api_mode=api_mode,
                model=resolved_model,
            )
        )
    except Exception:
        pass

    # Update context compressor limits for the cheap model. Without this,
    # compression decisions use the primary's context window — overflows
    # are possible on smaller-context cheap models.
    try:
        if getattr(agent, "context_compressor", None) is not None:
            from agent.model_metadata import get_model_context_length
            _ctx_api_key = agent.api_key if isinstance(agent.api_key, str) else ""
            _ctx = get_model_context_length(
                agent.model, base_url=agent.base_url,
                api_key=_ctx_api_key, provider=agent.provider,
                config_context_length=getattr(agent, "_config_context_length", None),
                custom_providers=getattr(agent, "_custom_providers", None),
            )
            agent.context_compressor.update_model(
                model=agent.model,
                context_length=_ctx,
                base_url=agent.base_url,
                api_key=agent.api_key,
                provider=agent.provider,
                api_mode=agent.api_mode,
            )
    except Exception:
        pass

    # Rewrite the cached system prompt's identity lines so the model
    # doesn't misreport itself.
    try:
        from agent.chat_completion_helpers import rewrite_prompt_model_identity
        rewrite_prompt_model_identity(agent, resolved_model, provider)
    except Exception:
        pass

    agent._smart_routed_active = True
    logger.info(
        "smart_model_routing: swapped to %s/%s for simple turn (msg=%d chars)",
        provider, resolved_model, len(user_message),
    )
    return True


def _get_provider_timeout(agent, provider: str, model: str):
    """Best-effort per-provider timeout; mirrors failover's lookup."""
    try:
        from hermes_cli.timeouts import get_provider_request_timeout
        return get_provider_request_timeout(provider, model)
    except Exception:
        return None


def restore_smart_routing(agent) -> bool:
    """Undo a smart-routing swap made earlier this session.

    Called at the start of every turn (before the smart-router decides
    again) so cheap-routing never leaks across turns. Idempotent — a
    no-op when no swap is active.
    """
    if not getattr(agent, "_smart_routed_active", False):
        # Also wipe stale state from a prior turn that completed without
        # restoration; defensive against crash recovery.
        agent._pre_smart_state = None
        return False
    state = getattr(agent, "_pre_smart_state", None)
    if not isinstance(state, dict):
        agent._smart_routed_active = False
        return False

    try:
        agent.model = state["model"]
        agent.provider = state["provider"]
        agent.base_url = state["base_url"]
        agent.api_mode = state["api_mode"]
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent.api_key = state.get("api_key", "") or ""
        agent._client_kwargs = dict(state.get("client_kwargs") or {})
        agent._use_prompt_caching = bool(state.get("use_prompt_caching", False))

        # Rebuild the primary's client from the snapshot's kwargs (mirrors
        # restore_primary_runtime, which is the existing canonical path).
        try:
            agent.client = agent._create_openai_client(
                dict(agent._client_kwargs),
                reason="smart_routing_restore",
                shared=False,
            )
        except Exception as exc:
            logger.warning("smart_model_routing: client restore failed: %s", exc)

        # Re-evaluate prompt caching for the primary model.
        try:
            agent._use_prompt_caching, agent._use_native_cache_layout = (
                agent._anthropic_prompt_cache_policy(
                    provider=agent.provider,
                    base_url=agent.base_url,
                    api_mode=agent.api_mode,
                    model=agent.model,
                )
            )
        except Exception:
            pass

        # Restore context compressor limits to primary.
        try:
            if getattr(agent, "context_compressor", None) is not None:
                from agent.model_metadata import get_model_context_length
                _ctx_api_key = agent.api_key if isinstance(agent.api_key, str) else ""
                _ctx = get_model_context_length(
                    agent.model, base_url=agent.base_url,
                    api_key=_ctx_api_key, provider=agent.provider,
                    config_context_length=getattr(agent, "_config_context_length", None),
                    custom_providers=getattr(agent, "_custom_providers", None),
                )
                agent.context_compressor.update_model(
                    model=agent.model,
                    context_length=_ctx,
                    base_url=agent.base_url,
                    api_key=agent.api_key,
                    provider=agent.provider,
                    api_mode=agent.api_mode,
                )
        except Exception:
            pass

        # Restore the cached system prompt's identity lines.
        try:
            from agent.chat_completion_helpers import rewrite_prompt_model_identity
            rewrite_prompt_model_identity(agent, agent.model, agent.provider)
        except Exception:
            pass

        logger.info(
            "smart_model_routing: restored primary %s/%s",
            agent.provider, agent.model,
        )
    finally:
        agent._pre_smart_state = None
        agent._smart_routed_active = False
    return True


def _detect_api_mode_for(agent, provider: str, base_url: str) -> str:
    """Pick the right api_mode for a cheap-model client.

    Mirrors the heuristic in try_activate_fallback.
    """
    fb_lower = (base_url or "").rstrip("/").lower()
    if (
        provider == "anthropic"
        or fb_lower.endswith("/anthropic")
        or base_url_hostname_simple(fb_lower) == "api.anthropic.com"
    ):
        return "anthropic_messages"
    if provider == "openai-codex":
        return "codex_responses"
    if provider == "bedrock":
        return "bedrock_converse"
    return "chat_completions"


def base_url_hostname_simple(url: str) -> str:
    """Host portion of a URL — cheap inline implementation.

    We don't import the project's utils here because of a possible
    circular import; this matches the simple case used for protocol
    decisions (https://api.anthropic.com → api.anthropic.com).
    """
    s = (url or "").strip()
    if not s:
        return ""
    if "://" in s:
        s = s.split("://", 1)[1]
    if "/" in s:
        s = s.split("/", 1)[0]
    if "@" in s:
        s = s.rsplit("@", 1)[1]
    if ":" in s:
        s = s.split(":", 1)[0]
    return s.lower()


__all__ = [
    "apply_smart_routing",
    "restore_smart_routing",
    "_is_simple_message",
    "_resolve_smart_target",
]
