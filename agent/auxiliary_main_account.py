"""Auxiliary 401 recovery: retry once with the main agent's live credential.

When an auxiliary call (compression in practice) still fails with an auth error after the
refresh and credential-pool rungs, and the main agent is on the same provider with a
DIFFERENT, live credential (e.g. adopted after a pool rotation), retry ONCE, immediately,
with that credential. A persistent failure is logged explicitly and the error propagates:
compression aborts as before and the history is left untouched.
"""

from __future__ import annotations

import logging
import weakref
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


def bind_live_main_credential(agent: Any) -> None:
    """Give the agent's compressor a reader for the main agent's LIVE credential.

    The reader returns the key the main loop currently uses, read at retry time, through a
    weak reference (no agent <-> compressor cycle). Any failure leaves native behaviour.
    """
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return
    try:
        ref = weakref.ref(agent)
    except TypeError:
        return

    def _live_main_api_key() -> str:
        live_agent = ref()
        if live_agent is None:
            return ""
        key = None
        if getattr(live_agent, "api_mode", "") == "anthropic_messages":
            key = getattr(live_agent, "_anthropic_api_key", None)
        key = key or getattr(live_agent, "api_key", "")
        return key if isinstance(key, str) else ""

    with suppress(Exception):
        compressor._live_api_key_fn = _live_main_api_key


def keep_live_reader(main_runtime: Any, normalized: dict[str, Any]) -> dict[str, Any]:
    """Carry a callable ``live_api_key`` reader through main-runtime normalization.

    Only a callable survives: never a frozen key snapshot.
    """
    live = main_runtime.get("live_api_key") if isinstance(main_runtime, dict) else None
    if callable(live) and not isinstance(live, str):
        normalized["live_api_key"] = live
    return normalized


def retry_call_kwargs(args: tuple, retry_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Kwargs for a ``retry_same_provider`` step; a third arg is an explicit API key."""
    call_kwargs = dict(retry_kwargs, resolved_provider=args[0], resolved_model=args[1])
    if len(args) > 2 and args[2]:
        call_kwargs["resolved_api_key"] = args[2]
        call_kwargs["resolved_base_url"] = None
    return call_kwargs


def short_auth_reason(err: Exception) -> str:
    text = str(err or "")
    return (text[:160] + "…") if len(text) > 160 else text


def main_account_retry_key(first_err: Exception, route: Any, auth_refresh_provider: Any) -> str:
    """Return the main agent's live key if it can rescue THIS auth failure, else ''.

    Conditions: auth error, Anthropic on both the aux route and the main runtime, a live
    reader supplied by the caller, and a non-empty Anthropic key that DIFFERS from the key
    that just failed.
    """
    from agent import auxiliary_client as aux

    if not aux._is_auth_error(first_err):
        return ""
    main_runtime = route.main_runtime if isinstance(route.main_runtime, dict) else {}
    live = main_runtime.get("live_api_key")
    if not callable(live):
        return ""
    aux_provider = aux._normalize_aux_provider(route.resolved_provider)
    if auth_refresh_provider != "anthropic" and aux_provider != "anthropic":
        return ""
    if str(main_runtime.get("provider") or "").strip().lower() != "anthropic":
        return ""
    try:
        key = live()
    except Exception:
        logger.debug("Auxiliary: main-account key reader failed", exc_info=True)
        return ""
    key = key.strip() if isinstance(key, str) else ""
    if not key.startswith("sk-ant-"):
        return ""
    failed = str(getattr(route.client, "api_key", "") or "").strip()
    return "" if key == failed else key  # same (dead) key: nothing new to try


def main_account_rung(first_err: Exception, route: Any, auth_refresh_provider: Any):
    """Ladder rung: one immediate retry with the main agent's live credential.

    Returns ``(response, None)`` or ``(None, err)``; re-raises errors the credential rungs
    do not accept.
    """
    from agent import auxiliary_client as aux

    key = main_account_retry_key(first_err, route, auth_refresh_provider)
    if not key:
        return None, first_err
    task, tag = route.task or "call", route.tag
    logger.info("Auxiliary %s%s: auth error persists (%s); retrying once with the main "
                "agent's credential", task, tag, short_auth_reason(first_err))
    model = route.resolved_model or route.final_model
    try:
        return (yield aux._LadderStep("retry_same_provider", ("anthropic", model, key))), None
    except Exception as retry_err:
        logger.warning("Auxiliary %s%s: retry with the main agent's credential failed too "
                       "(%s: %s); no result from this call, history is left untouched", task,
                       tag, type(retry_err).__name__, short_auth_reason(retry_err))
        if aux._credential_rung_accepts(retry_err) or aux._is_connection_error(retry_err):
            return None, retry_err
        raise
