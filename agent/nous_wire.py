"""Nous Portal ``anthropic/*`` wire selection when ``nous.anthropic_wire`` is ``auto``.

Portal serves Claude two ways and Hermes cannot tell which from the request: an OpenRouter
passthrough (today, for every ``anthropic/*`` id) or GMI/Vertex (planned once GMI is back). The
native Messages wire is the better transport, but on the OpenRouter path it re-writes the previous
turn's prompt cache on 14-20% of consecutive calls in concurrent tool loops (measured 2026-09-06;
NousResearch/api#227), so the session must ride chat/completions there. On GMI that is untested,
and until it is measured ``auto`` never promotes to native.

The upstream IS visible in the first RESPONSE: OpenRouter stamps ``provider`` (chat wire) and
mints ``gen-<unix>-<rand>`` ids; GMI/Vertex responses carry neither. So ``auto`` starts every
session on chat (safe on both upstreams), reads the first response, and switches the session to
native only when the upstream is GMI and native has been cleared for GMI. One decision per
session, at call 1, before there is a cache to lose; later calls never flip.

``classify_upstream`` is pure and unit-tested; ``maybe_switch_wire_after_first_response`` is
the single hook, called from the usage recorder.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Flip to True only after the 20x6 concurrency probe (evals/postmortem/live_ab) is clean on a
# GMI-served anthropic/* id on the native wire. Until then ``auto`` is chat everywhere.
GMI_NATIVE_WIRE_CLEARED = False

_OPENROUTER_ID = re.compile(r"^gen-\d{9,}-[A-Za-z0-9_-]{8,}$")


def classify_upstream(response: Any) -> Optional[str]:
    """``"openrouter"`` / ``"gmi"`` / ``None`` (unknown) from a Portal response object.

    Works on both wires: the OpenAI SDK object exposes ``.provider`` (OpenRouter's upstream name,
    e.g. ``"Anthropic"``, ``"Amazon Bedrock"``) and an OpenRouter-minted ``.id``; the Anthropic SDK
    object has ``.id`` only. GMI/Vertex responses have Anthropic-native ``msg_…`` ids and no
    ``provider``. Anything else is unknown, and unknown never triggers a switch.
    """
    if response is None:
        return None
    if isinstance(getattr(response, "provider", None), str) and getattr(response, "provider"):
        return "openrouter"
    rid = getattr(response, "id", None)
    if isinstance(rid, str):
        if _OPENROUTER_ID.match(rid):
            return "openrouter"
        if rid.startswith("msg_"):
            return "gmi"
    return None


def wire_for_upstream(upstream: Optional[str]) -> str:
    """The api_mode ``auto`` wants once the upstream is known. Chat unless GMI and cleared."""
    if upstream == "gmi" and GMI_NATIVE_WIRE_CLEARED:
        return "anthropic_messages"
    return "chat_completions"


def maybe_switch_wire_after_first_response(agent: Any, response: Any, api_call_count: int) -> bool:
    """Decide the session's wire from its first response; the switch itself is applied at the
    start of the next iteration (``apply_pending_wire_switch``), never while a response is being
    consumed. Returns True when a switch was scheduled.

    Only for provider=nous, anthropic/* models, ``nous.anthropic_wire: auto``, and only on the
    session's first API call.
    """
    if api_call_count != 1 or getattr(agent, "_nous_wire_decided", False):
        return False
    if (getattr(agent, "provider", "") or "").lower() != "nous":
        return False
    model = str(getattr(agent, "model", "") or "")
    if not model.lower().startswith("anthropic/"):
        return False
    try:
        from hermes_cli.providers import _nous_anthropic_wire
        if _nous_anthropic_wire() != "auto":
            return False
    except Exception:
        return False
    agent._nous_wire_decided = True  # one decision per session, whatever it is
    upstream = classify_upstream(response)
    want = wire_for_upstream(upstream)
    if want == getattr(agent, "api_mode", None):
        logger.debug("nous wire auto: upstream=%s, staying on %s", upstream, want)
        return False
    agent._nous_wire_pending = (want, upstream)
    return True


def apply_pending_wire_switch(agent: Any) -> bool:
    """At iteration start, with no response in flight: perform the switch scheduled by
    ``maybe_switch_wire_after_first_response``. Reuses ``switch_model`` (same model/provider, new
    api_mode) so client rebuild, cache policy and ``_primary_runtime`` stay consistent. A failure
    is logged and the session stays on its current wire."""
    pending = getattr(agent, "_nous_wire_pending", None)
    if not pending:
        return False
    agent._nous_wire_pending = None
    want, upstream = pending
    try:
        from agent.agent_runtime_helpers import switch_model
        switch_model(agent, agent.model, "nous", api_key=getattr(agent, "api_key", "") or "",
                     base_url=getattr(agent, "base_url", "") or "", api_mode=want)
    except Exception as exc:  # never let wire selection break a turn
        logger.warning("nous wire auto: switch to %s failed (%s); staying on %s", want, exc, agent.api_mode)
        return False
    logger.info("nous wire auto: upstream=%s -> %s for the rest of session %s", upstream, want,
                getattr(agent, "session_id", "?"))
    return True
