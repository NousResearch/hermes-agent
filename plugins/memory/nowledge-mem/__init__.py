"""Nowledge Mem memory provider plugin for Hermes Agent."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from .provider import NowledgeMemProvider

logger = logging.getLogger(__name__)

_fallback_lock = threading.Lock()
_fallback_provider: Optional[NowledgeMemProvider] = None
_fallback_session_id = ""


def register(ctx) -> None:
    """Plugin entry point called by Hermes plugin loader.

    New Hermes memory discovery passes a provider collector with
    ``register_memory_provider``. Some Hermes releases also load user memory
    providers through the general plugin loader, whose ``PluginContext`` only
    exposes hooks. In that path we register a narrow ``post_llm_call`` fallback
    so one-shot ``hermes chat -q`` sessions still capture completed turns.
    """
    if hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(NowledgeMemProvider())
        return

    if hasattr(ctx, "register_hook"):
        ctx.register_hook("post_llm_call", _post_llm_call)
        logger.debug("Nowledge Mem registered Hermes post_llm_call fallback hook")


def _post_llm_call(**kwargs: Any) -> None:
    session_id = str(kwargs.get("session_id") or "").strip()
    user_message = str(kwargs.get("user_message") or "")
    assistant_response = str(kwargs.get("assistant_response") or "")
    if not session_id or (not user_message and not assistant_response):
        return

    provider = _get_fallback_provider(session_id=session_id, kwargs=kwargs)
    if provider is None:
        return
    provider.sync_turn(user_message, assistant_response, session_id=session_id)


def _get_fallback_provider(
    *,
    session_id: str,
    kwargs: dict[str, Any],
) -> Optional[NowledgeMemProvider]:
    global _fallback_provider, _fallback_session_id

    with _fallback_lock:
        if _fallback_provider is not None and _fallback_session_id == session_id:
            return _fallback_provider

        if _fallback_provider is not None:
            try:
                _fallback_provider.shutdown()
            except Exception:
                logger.debug("Nowledge Mem fallback provider shutdown failed", exc_info=True)

        provider = NowledgeMemProvider()
        try:
            provider.initialize(
                session_id,
                hermes_home=_resolve_hermes_home(kwargs),
                platform=str(kwargs.get("platform") or "cli"),
                agent_context="primary",
            )
        except Exception as error:
            logger.debug("Nowledge Mem fallback provider init failed: %s", error)
            _fallback_provider = None
            _fallback_session_id = ""
            return None

        _fallback_provider = provider
        _fallback_session_id = session_id
        return provider


def _resolve_hermes_home(kwargs: dict[str, Any]) -> str:
    raw = kwargs.get("hermes_home")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    try:
        from hermes_constants import get_hermes_home

        return str(get_hermes_home())
    except Exception:
        return ""
