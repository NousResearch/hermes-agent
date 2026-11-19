"""Ambient session-accounting context for auxiliary LLM calls.

Aux calls (vision, compression, title generation, web_extract, session_search, ...) go
through ``agent.auxiliary_client`` which has no session handle, so their usage was
historically discarded. The agent loop publishes its store, storage interface handle and session id here
(mirroring ``agent.portal_tags``) and the aux client records usage at its single
response-validation chokepoint. ContextVar semantics isolate concurrent agents, propagate
to worker threads via ``tools.thread_context`` and to asyncio tasks automatically.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# (session_db, interface handle or None, session_id) for the active agent turn, or None outside one.
_accounting: ContextVar[Optional[tuple]] = ContextVar("aux_accounting_context", default=None)

# MoA advisor/aggregator usage is already folded into conversation_loop's
# update_token_counts delta (tokens AND cost); recording it here would double-count.
_EXCLUDED_TASKS = frozenset({"moa_reference", "moa_aggregator"})


@dataclass(frozen=True)
class ServingCredential:
    """The credential an aux attempt that succeeded authenticated with, bound to the URL it went to.

    ``api_key`` is the client's static key, or None when a credential source (``key_cmd``, Entra ID)
    served it: pricing never mints a token of its own. Accounting-only; never persisted.
    """
    base_url: str
    api_key: Optional[str]

    @classmethod
    def of(cls, client: Any) -> Optional["ServingCredential"]:
        """The *client*'s credential, None when it exposes no URL or no readable key."""
        base_url = str(getattr(client, "base_url", "") or "")
        if not base_url:
            return None
        # The SDK parks a token source in ``_api_key_provider`` and leaves ``.api_key`` empty.
        if callable(getattr(client, "__dict__", {}).get("_api_key_provider")):
            return cls(base_url, None)
        api_key = getattr(client, "api_key", None)
        return cls(base_url, api_key.strip() or None) if isinstance(api_key, str) else None

    def serves(self, base_url: Optional[str]) -> bool:
        return bool(base_url) and self.base_url.rstrip("/") == str(base_url).rstrip("/")


def set_accounting_context(session_db: Any, session_id: Optional[str], *, persistence: Any = None):
    """Publish the active session's accounting handles; returns the token for ``reset_accounting_context``.

    ``None`` handles (no DB / no session id) clear the context. Usage records through the
    writer handle ``persistence`` when it is one beside ``session_db``, else on ``session_db``.
    """
    if session_db is None or not session_id:
        return _accounting.set(None)
    from agent.agent_init_session import writer_interface
    return _accounting.set((session_db, writer_interface(persistence, session_db), session_id))


def reset_accounting_context(token) -> None:
    """Restore the previous accounting context (pair with ``set_...``)."""
    try:
        _accounting.reset(token)
    except Exception:
        _accounting.set(None)


def _pricing_api_key(base_url: Optional[str], serving: Optional[ServingCredential]) -> Optional[str]:
    """Key for the pricing lookup of *base_url* (a keyed endpoint's ``/models`` answers 401 without one).

    The serving attempt's credential when it went to *base_url*: its static key, or none for a
    credential source. Otherwise the key the runtime sends to ``model.base_url`` — unless a live
    credential source owns the main endpoint, so no saved key goes beside it. A failed config reload
    must not discard an otherwise available cost estimate.
    """
    if serving is not None and serving.serves(base_url):
        return serving.api_key
    from hermes_cli.config_backend import ConfigBackendUnavailable
    try:
        from agent.auxiliary_client import _runtime_main_value
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider_custom import main_endpoint_key_for
        if callable(_runtime_main_value("api_key")):
            return None
        return main_endpoint_key_for(load_config().get("model"), base_url or "") or None
    except (Exception, ConfigBackendUnavailable):  # the backend's outage is a SystemExit
        logger.debug("Aux pricing credential lookup failed; estimating without a key", exc_info=True)
        return None


def record_aux_usage(
    response: Any, task: Optional[str], *, provider: Optional[str] = None,
    base_url: Optional[str] = None, serving: Optional[ServingCredential] = None,
) -> None:
    """Record an auxiliary response's token usage against the ambient session.

    Strictly best-effort (accounting must never break an aux call). No-ops outside an
    agent turn, for main-loop-accounted tasks (``_EXCLUDED_TASKS``), or without usage.
    The model is read from ``response.model`` (accurate after aux provider fallback);
    *provider*/*base_url* reflect the originally-resolved route. *serving* is the credential
    that served the call, for the pricing lookup only.
    """
    try:
        if not task or task in _EXCLUDED_TASKS:
            return
        ctx = _accounting.get()
        if ctx is None:
            return
        session_db, store, session_id = ctx
        raw_usage = getattr(response, "usage", None)
        if raw_usage is None:
            return

        from agent.usage_pricing import estimate_usage_cost, normalize_usage, with_served_service_tier

        usage = with_served_service_tier(normalize_usage(raw_usage, provider=provider), response)
        if not (
            usage.input_tokens or usage.output_tokens
            or usage.cache_read_tokens or usage.cache_write_tokens
            or usage.reasoning_tokens
        ):
            return
        model = str(getattr(response, "model", "") or "") or "unknown"
        estimated_cost = None
        try:
            cost = estimate_usage_cost(model, usage, provider=provider, base_url=base_url,
                                       api_key=_pricing_api_key(base_url, serving))
            if cost.amount_usd is not None:
                estimated_cost = float(cost.amount_usd)
        except Exception:
            logger.debug("Aux usage cost estimation failed", exc_info=True)
        # Inside a Relay-driven tool callback the thread runs Relay's loop; a worker store refuses
        # synchronous RPC there (#139029).
        from agent.runtime_session_store import call_off_event_loop
        if store is not None:
            from persistence.conversation.usage import TokenCounts
            call_off_event_loop(
                store.usage.record_auxiliary_usage,
                session_id, task, model=model, billing_provider=provider, billing_base_url=base_url,
                tokens=TokenCounts(
                    input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens, cache_write_tokens=usage.cache_write_tokens,
                    reasoning_tokens=usage.reasoning_tokens,
                ),
                estimated_cost_usd=estimated_cost,
            )
            return
        call_off_event_loop(
            session_db.record_auxiliary_usage,
            session_id, task, model=model, billing_provider=provider, billing_base_url=base_url,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens, cache_write_tokens=usage.cache_write_tokens,
            reasoning_tokens=usage.reasoning_tokens, estimated_cost_usd=estimated_cost,
        )
    except Exception:
        logger.debug("Aux usage recording failed (non-fatal)", exc_info=True)
