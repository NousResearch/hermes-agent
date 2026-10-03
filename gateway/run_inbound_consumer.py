"""Post-admission consuming gateway plugin extension (issue #129958).

Minimal, profile-scoped, fail-closed consuming hook for new business messages.

Placement (owned by ``GatewayInboundMixin._handle_message`` in
``gateway/run_inbound.py``) is AFTER all core-owned gates:

- admission (profile routing, ignored channels, unserved profiles,
  authorization, bot admission),
- pause / maintenance-drain gates,
- pending-reply intercepts (update prompt, clarify, slash-confirm),
- running-session fast-path (steering, busy commands),
- idle slash-command dispatch (built-in, quick, plugin, skill),

and AFTER FIFO orphan rescue, INSIDE the claimed session slot (sentinel
held) so concurrent arrivals serialize through the busy path instead of
invoking the consumer twice.

Only idle, non-internal business messages reach the hook. Rejected,
ignored-channel, unserved-profile, bot-loop, internal/system, paused,
draining and control traffic never invoke it.

Outcomes (first decisive result wins, registration order):

- ``None`` / ``{"action": "pass"}`` (also ``"allow"``) → PASS, ordinary
  agent flow runs once.
- ``{"action": "consumed", "receipt": "<id>", "reply": "<optional>"}``
  → CONSUMED, agent flow suppressed, ``reply`` (when a non-empty string)
  returned through the normal core-owned delivery route. ``receipt`` is
  a required, non-empty plugin receipt identifier; a missing/invalid
  receipt is malformed → FAILED.
- ``{"action": "failed", "reply": "<optional>"}`` (also ``"fail"``,
  ``"error"``, ``"block"``, ``"deny"``) → FAILED, agent flow suppressed,
  short user-failure notice returned, sanitized operator diagnostic
  logged.

Fail-closed: unknown/malformed outcomes, callback exceptions and
timeouts suppress the agent turn (FAILED), never silently permit inline
execution. Plugin absence (no callbacks registered) is distinct — it is
PASS (ordinary flow). Cancellation propagates without starting an agent
turn.

Scope: invoked inside the routed profile's ``_async_profile_runtime_scope``
(the ``_handle_message`` call path already binds it), so
``get_plugin_manager()`` resolves the routed profile's own consumers.
Scope is restored by the ambient context manager, including on
exceptions. The hook receives no runner/adapter authority beyond the
documented kwargs.

Context: a versioned, immutable snapshot (``MappingProxyType``) with
canonical profile/platform/conversation, stable event identity, sender
identity and reply anchor. Inbound text is untrusted data. Direct
mutation of ``event.source`` routing fields is restored and logged —
destinations are never taken from plugin-returned fields.

Durable acceptance is a plugin responsibility (commit before CONSUMED,
reconcile by receipt on replay). Core records no cross-store
transaction and claims no exactly-once external delivery.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger("gateway.run")

POST_ADMISSION_HOOK = "post_gateway_admission"
POST_ADMISSION_VERSION = 1

# Short, sanitized user-facing notice for FAILED consumption. Deliberately
# not i18n-keyed in v1 to avoid locale-parity churn; plugins may supply
# their own short ``reply`` for the failed turn instead.
POST_ADMISSION_FAILURE_NOTICE = (
    "⚠️ A gateway plugin failed to handle this message, so it was not processed."
)

_MAX_REPLY_CHARS = 2000
_MAX_RECEIPT_CHARS = 256


def build_post_admission_context(
    event: Any,
    source: Any,
    session_key: str,
    *,
    reply_anchor: Optional[str],
) -> Mapping[str, Any]:
    """Versioned immutable snapshot presented to ``post_gateway_admission`` consumers."""
    platform = getattr(getattr(source, "platform", None), "value", None)
    platform = platform if isinstance(platform, str) else str(getattr(source, "platform", "") or "")
    ctx = {
        "version": POST_ADMISSION_VERSION,
        "profile": getattr(source, "profile", None),
        "platform": platform,
        "session_key": session_key,
        "conversation": MappingProxyType(
            {
                "chat_id": getattr(source, "chat_id", ""),
                "thread_id": getattr(source, "thread_id", None),
                "chat_type": getattr(source, "chat_type", "") or "",
            }
        ),
        "event_id": getattr(event, "message_id", None),
        "event_revision": getattr(event, "platform_update_id", None),
        "sender": MappingProxyType(
            {
                "user_id": getattr(source, "user_id", None),
                "user_name": getattr(source, "user_name", None),
            }
        ),
        "reply_anchor": reply_anchor,
        # Untrusted inbound data; never trusted for authorization or routing.
        "text": getattr(event, "text", "") or "",
        "reply_to_text": getattr(event, "reply_to_text", None),
    }
    return MappingProxyType(ctx)


def _action_of(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    action = result.get("action", result.get("decision", ""))
    return str(action or "").strip().lower()


def decide_post_admission(results: Any) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Interpret raw hook results → ``(decision, reply, receipt, diagnostic)``.

    ``decision`` is ``"pass"``, ``"consumed"`` or ``"failed"``. ``reply`` is
    the optional user-facing string (truncated), ``receipt`` the validated
    plugin receipt for CONSUMED, ``diagnostic`` a sanitized operator string.
    Empty/missing results → PASS. Any malformed entry with at least one
    registered consumer present → FAILED (fail-closed).
    """
    if not results:
        return "pass", None, None, None
    decisive: Optional[Tuple[str, Optional[str], Optional[str], Optional[str]]] = None
    decisive_count = 0
    for result in results:
        if result is None:
            continue
        if not isinstance(result, dict):
            decisive_count += 1
            if decisive is None:
                decisive = ("failed", None, None, "malformed non-dict result")
            continue
        action = _action_of(result)
        if action in ("", "pass", "allow", "continue"):
            continue
        if action in ("consumed", "consume", "consumes", "handled"):
            receipt = result.get("receipt", result.get("receipt_id", ""))
            reply = result.get("reply", result.get("response", result.get("message")))
            if not isinstance(receipt, str) or not receipt.strip():
                decisive_count += 1
                if decisive is None:
                    decisive = ("failed", None, None, "consumed without valid receipt")
                continue
            receipt = receipt.strip()[:_MAX_RECEIPT_CHARS]
            if reply is not None and not isinstance(reply, str):
                decisive_count += 1
                if decisive is None:
                    decisive = ("failed", None, receipt, "consumed with non-string reply")
                continue
            reply = (reply[:_MAX_REPLY_CHARS] if isinstance(reply, str) else None) or None
            decisive_count += 1
            if decisive is None:
                decisive = ("consumed", reply, receipt, None)
            continue
        if action in ("failed", "fail", "error", "block", "deny", "blocked"):
            reply = result.get("reply", result.get("response", result.get("message")))
            if reply is not None and not isinstance(reply, str):
                reply = None
            reply = (reply[:_MAX_REPLY_CHARS] if isinstance(reply, str) else None) or None
            reason = result.get("reason", result.get("error", ""))
            diagnostic = str(reason)[:200] if reason else "consumer reported failure"
            decisive_count += 1
            if decisive is None:
                decisive = ("failed", reply, None, diagnostic)
            continue
        # Unknown action → fail closed.
        decisive_count += 1
        if decisive is None:
            decisive = ("failed", None, None, f"unknown action {action!r}")
    if decisive is None:
        return "pass", None, None, None
    if decisive_count > 1:
        logger.warning(
            "post_gateway_admission: %d decisive consumer results; first wins", decisive_count
        )
    return decisive


def _snapshot_destination(source: Any) -> Dict[str, Any]:
    return {
        "platform": getattr(getattr(source, "platform", None), "value", None)
        if hasattr(getattr(source, "platform", None), "value")
        else getattr(source, "platform", None),
        "chat_id": getattr(source, "chat_id", None),
        "thread_id": getattr(source, "thread_id", None),
        "chat_type": getattr(source, "chat_type", None),
        "profile": getattr(source, "profile", None),
        "user_id": getattr(source, "user_id", None),
    }


def _restore_destination_if_mutated(event: Any, source: Any, snapshot: Dict[str, Any]) -> None:
    """Restore routing fields mutated by a consumer; destinations never come from plugins."""
    current = _snapshot_destination(source)
    if current == snapshot:
        return
    logger.warning(
        "post_gateway_admission consumer attempted to mutate destination metadata; restored"
    )
    try:
        for key, value in snapshot.items():
            if key == "platform":
                # The snapshot stores the enum's .value; coerce back so the
                # attribute lands on the enum, not a bare string.
                from gateway.config import Platform

                try:
                    setattr(source, key, Platform(value))
                except ValueError:
                    setattr(source, key, value)
                continue
            setattr(source, key, value)
        # Restore the event's source reference when it was swapped wholesale.
        if getattr(event, "source", None) is not source:
            event.source = source
    except Exception:
        logger.debug("post_gateway_admission destination restore failed", exc_info=True)


async def invoke_post_admission_hook(
    runner: Any, event: Any, source: Any, session_key: str
) -> Tuple[bool, Optional[str], Any, Any]:
    """Invoke the ``post_gateway_admission`` hook → ``(handled, reply, event, source)``.

    ``handled=False`` means PASS (run the ordinary agent flow). ``handled=True``
    means CONSUMED/FAILED (suppress the agent flow, return ``reply`` which may
    be ``None`` for a silent consume). ``event``/``source`` are returned with
    any destination mutation restored.
    """
    try:
        from hermes_cli.lifecycle import ainvoke_hook as _ainvoke_hook
    except Exception:
        return False, None, event, source

    snapshot = _snapshot_destination(source)
    try:
        reply_anchor = None
        try:
            from gateway.platforms.base import _reply_anchor_for_event as _anchor
            reply_anchor = _anchor(event)
        except Exception:
            reply_anchor = getattr(event, "message_id", None)
        context = build_post_admission_context(
            event, source, session_key, reply_anchor=reply_anchor
        )
    except Exception:
        logger.warning("post_gateway_admission context build failed; failing closed", exc_info=True)
        return True, POST_ADMISSION_FAILURE_NOTICE, event, source

    try:
        results = await _ainvoke_hook(
            POST_ADMISSION_HOOK,
            event=event,
            gateway=runner,
            session_store=getattr(runner, "session_store", None),
            source=source,
            session_key=session_key,
            context=context,
        )
    except asyncio.CancelledError:
        # Cancellation after a commit needs receipt lookup, not a new agent
        # turn. Propagate without starting inline execution.
        _restore_destination_if_mutated(event, source, snapshot)
        raise
    except Exception as exc:
        logger.warning("post_gateway_admission invocation failed: %s", exc)
        _restore_destination_if_mutated(event, source, snapshot)
        return True, POST_ADMISSION_FAILURE_NOTICE, event, source

    _restore_destination_if_mutated(event, source, snapshot)

    # The fail-closed timeout path contributes {"action": "block"}; decide maps it to FAILED.
    decision, reply, receipt, diagnostic = decide_post_admission(results)
    if decision == "pass":
        return False, None, event, source
    if decision == "consumed":
        logger.info(
            "post_gateway_admission consumed message (receipt=%s session=%s)",
            receipt or "unknown",
            session_key,
        )
        return True, reply, event, source
    # FAILED: sanitized operator diagnostic (identifiers only, never message text).
    try:
        platform = getattr(getattr(source, "platform", None), "value", "unknown")
    except Exception:
        platform = "unknown"
    logger.warning(
        "post_gateway_admission consumer failed (session=%s platform=%s diagnostic=%s)",
        session_key,
        platform,
        (diagnostic or "unknown")[:200],
    )
    return True, (reply or POST_ADMISSION_FAILURE_NOTICE), event, source
