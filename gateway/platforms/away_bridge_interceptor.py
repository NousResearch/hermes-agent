"""Telegram pre-dispatch interceptor for the away-mode Telegram bridge.

Design: planning/capabilities/AWAY-MODE-TELEGRAM-BRIDGE.md §4.4 (Rev 7).

Hook position (normative): top of ``BasePlatformAdapter.handle_message``
(gateway/platforms/base.py:5931), gated on ``platform == telegram`` —
after adapter-level owner authorization and text batching, before the
active-session guard and ``_pending_messages`` merge.

Behavior:
  - line-anchored, token-partitioned matcher: an ``A#<token>`` line starts a
    segment; the token line is stripped; the segment's remaining lines
    (including same-line remainder after the token) are the answer;
  - exact normalized return-intent lines (I'm back / Im back / חזרתי) anywhere
    in the batch disarm all armed sessions in this profile's broker;
  - owner-origin bridge-shaped batches are ALWAYS handled (claim, NACK, or
    transient-error suppression) — never dispatched to the ordinary DM agent;
  - mixed batches process claims then disarm regardless of line order;
  - all broker I/O via asyncio.to_thread with a short busy timeout (§4.4).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from tools.away_bridge_broker import (
    RETURN_INTENTS,
    load_away_bridge_config,
)

logger = logging.getLogger(__name__)

# ``A#`` + 15–16 alphanumeric chars (15 = the doc's own example token
# `A#K7QM4XT9P2WBN5F`; 16 = the §4.3 normative length we generate). At line
# start (after optional whitespace). The question text instructs the owner to
# start the reply line with the key, but stray text before the FIRST token
# line is tolerated and discarded (§4.4).
_TOKEN_LINE_RE = re.compile(r"^\s*(A#[A-Z0-9]{15,16})\s*(.*)$")

# The ordinary Telegram DM session key for a DM chat, used only for logging.
_DM_SESSION_KEY_PREFIX = "agent:main:telegram:dm:"


class InterceptResult:
    """Outcome of one interceptor run."""

    __slots__ = ("handled", "reply_texts")

    def __init__(self, handled: bool, reply_texts: Optional[List[str]] = None) -> None:
        self.handled = handled
        self.reply_texts = reply_texts or []

    @property
    def reply_text(self) -> Optional[str]:
        return "\n".join(self.reply_texts) if self.reply_texts else None


def partition_batch(text: str) -> Tuple[List[Tuple[str, str]], bool]:
    """Split batched text into (token, answer) segments + return-intent flag.

    Extraction boundary (normative, R5-3): the token line is STRIPPED; the
    segment's answer is its remaining lines, including the same-line remainder
    after the token. Content before the first token line is discarded.
    Mid-line mentions (``what does A#… mean?``) are NOT token lines.
    """
    segments: List[Tuple[str, str]] = []
    current_token: Optional[str] = None
    current_lines: List[str] = []
    return_intent = False
    for raw_line in (text or "").split("\n"):
        line = raw_line.rstrip()
        match = _TOKEN_LINE_RE.match(line)
        if match:
            if current_token is not None:
                segments.append((current_token, "\n".join(current_lines).strip()))
            current_token = match.group(1)
            remainder = match.group(2).strip()
            current_lines = [remainder] if remainder else []
            continue
        if current_token is not None:
            current_lines.append(line)
        normalized = line.strip().lower().rstrip(".!? \t").strip()
        if normalized in RETURN_INTENTS:
            return_intent = True
    if current_token is not None:
        segments.append((current_token, "\n".join(current_lines).strip()))
    return segments, return_intent


def _owner_matches(event: Any, owner_cfg: Dict[str, str]) -> bool:
    """Both chat ID and sender user ID must match (§4.8)."""
    source = getattr(event, "source", None)
    if source is None:
        return False
    owner_chat = str(owner_cfg.get("owner_chat_id") or "")
    owner_user = str(owner_cfg.get("owner_user_id") or "")
    if not owner_chat and not owner_user:
        return False
    chat_ok = bool(owner_chat) and str(getattr(source, "chat_id", "") or "") == owner_chat
    user_ok = bool(owner_user) and str(getattr(event, "user_id", "") or "") == owner_user
    return chat_ok and user_ok


def _resolve_token(broker, token: str):
    return broker.get_request_by_token(token)


def process_batch_sync(
    broker,
    event: Any,
    send_reply,  # callable(text) -> None
    cfg: Dict[str, Any],
) -> InterceptResult:
    """Synchronous matcher + broker work. Runs in a worker thread (§4.4)."""
    replies: List[str] = []
    text = str(getattr(event, "text", "") or "")
    segments, return_intent = partition_batch(text)

    if not segments and not return_intent:
        return InterceptResult(handled=False)

    # Return intent requires an armed session to mean anything (§4.4).
    if return_intent and not segments:
        try:
            armed = broker.has_armed()
        except Exception:
            # DB error on a return-intent batch: suppress + resend notice
            # (R5-4 extension of the DB-error norm, §4.4).
            _send_transient_notice(send_reply, replies)
            return InterceptResult(handled=True, reply_texts=replies)
        if not armed:
            return InterceptResult(handled=False)

    bridge_shaped = bool(segments) or (return_intent and armed_now(broker))

    # ---- claims first, then disarm (§4.4 mixed-batch rule) ----------------
    # A# rows in 'awaiting' OR 'claimed' state attempt the conditional claim
    # transaction (§4.2): 'claimed' rows lose and take the idempotent
    # duplicate-ack path. Unknown/prepared/cancelled rows NACK.
    claim_failed_db = False
    for token, answer in segments:
        try:
            row = broker.get_request_by_token(token)
        except Exception:
            claim_failed_db = True
            continue
        if row is None or row["state"] not in ("awaiting", "claimed"):
            replies.append(_nack_text(token))
            continue
        try:
            outcome = broker.claim(row["req_key"], "telegram", answer)
        except Exception:
            claim_failed_db = True
            continue
        if outcome["outcome"] == "won":
            replies.append(_accept_text(token))
        elif outcome["outcome"] == "lost":
            winner = outcome.get("winner") or "chat"
            if broker.ack_duplicate(row["req_key"], "telegram"):
                replies.append(_dup_ack_text(winner))
        else:
            # reject: unknown / prepared / cancelled between lookup and claim
            replies.append(_nack_text(token))

    # ---- disarm after claims (§4.6 Telegram global disarm) -----------------
    if return_intent:
        try:
            if broker.has_armed():
                result = broker.disarm_all()
                if result.get("disarmed_count", 0) > 0:
                    replies.append(
                        "You're back — all armed sessions in this profile are disarmed "
                        "(%d session(s)). Pending questions stay visible in their sessions."
                        % result["disarmed_count"]
                    )
        except Exception:
            # DB error while processing the return intent: suppress with a
            # transient-error notice; never fall through (§4.4, R5-4).
            _send_transient_notice(send_reply, replies)
            return InterceptResult(handled=True, reply_texts=replies)

    if claim_failed_db:
        _send_transient_notice(send_reply, replies)
        return InterceptResult(handled=True, reply_texts=replies)

    return InterceptResult(handled=True, reply_texts=replies)


def armed_now(broker) -> bool:
    try:
        return broker.has_armed()
    except Exception:
        return False


def _nack_text(token: str) -> str:
    return (
        "No active bridge request matches %s — if you meant to answer a pending "
        "question, reply with the key shown in the question." % token
    )


def _accept_text(token: str) -> str:
    return "Accepted for %s." % token


def _dup_ack_text(winner: str) -> str:
    label = "chat" if winner == "chat" else "Telegram"
    return "Already answered via %s — ignored." % label


def _send_transient_notice(send_reply, replies: List[str]) -> None:
    replies.append(
        "Bridge had a transient error — please resend your reply "
        "(the question is still waiting)."
    )


def run_sweep_sync(broker, cfg: Dict[str, Any], send_fn) -> Dict[str, Any]:
    """Startup sweep (§4.4): rate-limited re-asks + orphan-row GC.

    ``send_fn(text)`` is a synchronous send. Callers invoke this through
    ``asyncio.to_thread``. Fires ONLY on cold gateway start and 409-conflict
    recovery (both drop Telegram's pending queue) — never on watcher
    reconnects. Rate limit is transactional inside sweep_candidates; each
    send re-checks state='awaiting' immediately before sending (§4.4).
    """
    import time as _time

    sent: List[str] = []
    skipped = 0
    try:
        candidates = broker.sweep_candidates(float(cfg.get("sweep_window_hours", 1.0)))
    except Exception:
        logger.exception("[away-bridge] sweep candidate query failed")
        candidates = []
    for cand in candidates:
        try:
            row = broker.get_request(cand["req_key"])
        except Exception:
            row = None
        if row is None or row["state"] != "awaiting":
            skipped += 1
            continue
        token = row["token"] or cand.get("token") or ""
        text = "Still waiting for %s — %s" % (token, row["question"])
        try:
            send_fn(text)
            sent.append(cand["req_key"])
        except Exception:
            logger.exception("[away-bridge] sweep re-ask send failed for %s", cand["req_key"])
        # Tiny spacing so several re-asks do not hammer the Bot API in one
        # instant (rate limit is per-request, not per-sweep).
        _time.sleep(0.3)
    try:
        gc_result = broker.gc(float(cfg.get("retention_days", 30)))
    except Exception:
        logger.exception("[away-bridge] sweep GC failed")
        gc_result = {}
    return {"reask_sent": sent, "reask_skipped": skipped, "gc": gc_result}


async def run_sweep_async(hermes_home=None, send_fn=None) -> Dict[str, Any]:
    """Async wrapper used by the Telegram adapter's sweep triggers."""
    from tools.away_bridge_broker import (
        DEFAULT_INTERCEPTOR_BUSY_MS,
        AwayBridgeBroker,
    )

    cfg = load_away_bridge_config(hermes_home)
    if not cfg.get("enabled"):
        return {"skipped": "disabled"}
    broker = AwayBridgeBroker(hermes_home=hermes_home, busy_ms=DEFAULT_INTERCEPTOR_BUSY_MS)
    return await asyncio.to_thread(run_sweep_sync, broker, cfg, send_fn or (lambda _t: None))


async def away_bridge_sweep_guarded(adapter, send_fn) -> None:
    """Sweep wrapper for adapter call sites: never raises into the adapter.

    ``send_fn(text)`` is an async send bound to the owner chat. Exceptions are
    logged and swallowed — a sweep failure must not break connect() or the
    conflict-recovery ladder (§4.4).
    """
    try:
        result = await run_sweep_async(send_fn=send_fn)
        if result and result.get("reask_sent"):
            logger.info(
                "[away-bridge] startup sweep re-asked %d pending question(s)",
                len(result["reask_sent"]),
            )
    except Exception:
        logger.exception("[away-bridge] startup sweep failed")



def _is_authorized_owner_sync(event: Any, cfg: Dict[str, Any]) -> bool:
    return _owner_matches(event, cfg.get("telegram") or {})


def _platform_is_telegram(platform: Any) -> bool:
    """True for the Telegram platform, enum or string.

    ``gateway.session.Platform`` is a plain Enum whose ``str()`` renders as
    ``"Platform.TELEGRAM"`` — comparing ``str(source.platform) == "telegram"``
    silently never matches a real adapter event (mock-based tests with string
    platforms masked this). Compare on ``.value`` when present, else the raw
    string.
    """
    value = getattr(platform, "value", platform)
    return str(value or "") == "telegram"


async def intercept_message(event: Any, send_reply, hermes_home=None) -> InterceptResult:
    """Async entry point called from ``BasePlatformAdapter.handle_message``.

    ``send_reply`` is an async callable(str) that sends a Telegram message to
    the inbound event's chat. All broker I/O runs in a worker thread with a
    short busy timeout so the gateway event loop never blocks (§4.4).
    """
    try:
        cfg = load_away_bridge_config(hermes_home)
    except Exception:
        logger.warning("[away-bridge] config load failed; not intercepting", exc_info=True)
        return InterceptResult(handled=False)
    if not cfg.get("enabled"):
        logger.debug("[away-bridge] disabled; not intercepting")
        return InterceptResult(handled=False)
    source = getattr(event, "source", None)
    if source is None or not _platform_is_telegram(getattr(source, "platform", "")):
        logger.info(
            "[away-bridge] pass-through: source=%r platform=%r",
            type(source).__name__ if source is not None else None,
            getattr(getattr(source, "platform", None), "value", getattr(source, "platform", None)),
        )
        return InterceptResult(handled=False)
    if not _is_authorized_owner_sync(event, cfg):
        logger.info(
            "[away-bridge] pass-through: owner mismatch chat=%r user=%r (owner cfg %r)",
            getattr(source, "chat_id", None), getattr(event, "user_id", None),
            (cfg.get("telegram") or {}),
        )
        return InterceptResult(handled=False)

    from tools.away_bridge_broker import AwayBridgeBroker, DEFAULT_INTERCEPTOR_BUSY_MS

    broker = AwayBridgeBroker(hermes_home=hermes_home, busy_ms=DEFAULT_INTERCEPTOR_BUSY_MS)

    def _sync_work():
        return process_batch_sync(broker, event, None, cfg)

    try:
        result = await asyncio.to_thread(_sync_work)
    except Exception:
        logger.exception("[away-bridge] interceptor broker I/O failed; suppressing batch")
        try:
            await send_reply(
                "Bridge had a transient error — please resend your reply "
                "(the question is still waiting)."
            )
        except Exception:
            logger.exception("[away-bridge] failed to send transient-error notice")
        return InterceptResult(handled=True)
    for text in result.reply_texts:
        try:
            await send_reply(text)
        except Exception:
            logger.exception("[away-bridge] failed to send reply: %s", text[:80])
    return result
