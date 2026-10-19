"""Unbacked follow-up promise guard: a final reply that promises to come back later ("I'll check again in ~10
minutes", "I'll report back when the review lands") while nothing will wake the agent after the turn ends.

Between turns the agent is inert. It runs again only when the user writes, a background process started with
``notify_on_complete``/``watch_patterns`` reports, a background delegation completes, a ``/loop`` or ``/heartbeat``
fires, or a cron job runs. A promise with none of those armed is never kept, and the user finds out only by asking.
The guard re-prompts once (on the shared continuation budget) so the model either does the work now, arms a wake
source, or drops the promise.

Detection is deliberately narrow: only the reply's TAIL, only first-person deferred follow-ups, and never a
follow-up conditioned on the user acting first ("tell me when it's sent and I'll re-check").
"""

from __future__ import annotations

import logging
import re
from typing import Any

# A first-person promise of a LATER follow-up, in the last sentences of the reply.
_PROMISE_RE = re.compile(
    r"\b(?:i(?:['\u2019]ll| will| am going to|['\u2019]m going to)|we(?:['\u2019]ll| will))\s+"
    r"(?:(?:then|also|just|now)\s+)?"
    r"(?:report back|check back|circle back|get back to you|come back to you|follow up|ping you|update you"
    r"|let you know|tell you|check (?:again|in|on (?:it|them|that|this|progress|the \w+))|re-?check"
    r"|look again|poll)\b"
    r"[^.!?\n]{0,140}",
    re.IGNORECASE,
)
# "...in ~10 minutes", "...after the sweep", "...when it finishes", "...once both reviews return".
_DEFERRAL_RE = re.compile(
    r"\b(?:in (?:about |another |~|a few |a couple of )?\d*\s*(?:sec|second|min|minute|hour|h\b|m\b)"
    r"|in (?:a (?:bit|moment|while|few)|a couple)|later|shortly|after (?:the|it|that|this|each|both)"
    r"|when\b|once\b|as soon as|whenever)",
    re.IGNORECASE,
)
# The follow-up waits on the USER, who will be writing anyway: "tell me when it's sent and I'll re-check".
_USER_GATED_RE = re.compile(
    r"\b(?:tell me|let me know|ping me|say (?:the word|when|done|go)|once you(?:['\u2019]ve| have)?|when you"
    r"(?:['\u2019]ve| have)?|after you|if you|reply with|send me|paste)\b",
    re.IGNORECASE,
)
_TAIL_CHARS = 500
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def unbacked_promise_phrase(text: str) -> str:
    """The promise phrase if ``text`` ENDS on an unconditional deferred follow-up, else ``""``."""
    tail = (text or "").strip()[-_TAIL_CHARS:]
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(tail) if s.strip()]
    for sentence in sentences[-3:]:
        match = _PROMISE_RE.search(sentence)
        if not match or not _DEFERRAL_RE.search(sentence[match.start():]):
            continue
        if _USER_GATED_RE.search(sentence):
            continue
        return match.group(0).strip()
    return ""


def _turn_armed_cron(messages: list) -> bool:
    """A cron job was scheduled since the last user row (direct ``cronjob_manage`` or via ``tool_call``)."""
    for msg in reversed(messages or ()):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            return False
        for call in msg.get("tool_calls") or ():
            fn = (call.get("function") if isinstance(call, dict) else getattr(call, "function", None)) or {}
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
            args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
            if "cronjob" in str(name or "") or (name == "tool_call" and "cronjob" in str(args or "")):
                return True
    return False


def _has_pending_process_notice(agent: Any) -> bool:
    from tools.process_registry import process_registry
    return any(
        proc.notify_on_complete or proc.watch_patterns
        for owner in getattr(agent, "_process_owner_task_ids", ()) or ()
        for proc in process_registry.running_owned_by(owner))


def _has_live_delegation(agent: Any) -> bool:
    from tools.async_delegation import has_live_for_session
    return bool(agent.session_id) and has_live_for_session(parent_session_id=agent.session_id)


def _has_active_schedule(agent: Any) -> bool:
    from hermes_cli.heartbeat import HeartbeatManager
    from hermes_cli.loops import LoopManager
    sid = agent.session_id or ""
    return bool(sid) and (LoopManager(session_id=sid).is_active() or HeartbeatManager(session_id=sid).is_active())


def has_armed_wake_source(agent: Any, messages: list) -> bool:
    """Whether something will run the agent again after this turn without the user writing. Fails OPEN (True)
    on an unreadable source: a false "armed" costs one missed nudge, a false "unarmed" nags a correct reply."""
    if _turn_armed_cron(messages):
        return True
    for probe in (_has_pending_process_notice, _has_live_delegation, _has_active_schedule):
        try:
            if probe(agent):
                return True
        except Exception:
            logging.getLogger("agent.conversation_loop").debug("wake-source probe %s failed", probe.__name__,
                                                               exc_info=True)
            return True
    return False


def unbacked_followup_promise(agent: Any, text: str, messages: list) -> str:
    """The promise phrase when the reply promises a later follow-up and no wake source is armed, else ``""``.
    Only interactive sessions: a cron run or a delegated child reports to a caller, not to a waiting user."""
    if getattr(agent, "platform", None) == "cron" or getattr(agent, "_delegate_depth", 0):
        return ""
    phrase = unbacked_promise_phrase(text)
    if not phrase or has_armed_wake_source(agent, messages):
        return ""
    return phrase


UNBACKED_PROMISE_NUDGE = (
    "[System: Your reply ends by promising a later follow-up (\"{phrase}\"), but nothing will run you again "
    "after this turn ends. You only run again when the user writes, a background process started with "
    "notify_on_complete finishes, a background delegation completes, or a cron job, /loop or /heartbeat fires — "
    "and none of those is armed. Do one of these now: do the follow-up work in this turn (a foreground wait is "
    "fine for a few minutes), arm a wake source with a tool call, or send your reply again without the promise "
    "and tell the user to ask for the update.]"
)


def log_continuation(kind: str, attempt: int, *, text: str, tool_rows: int, promise: str) -> None:
    """One log line per re-prompt of the final-text continuation family (the counter is shared: max 2 per turn)."""
    logger = logging.getLogger("agent.conversation_loop")
    if kind == "stall":
        logger.info("Stall guard: turn ending on trailing continue-intent with no tool calls — re-prompting to act "
                    "(%d/2)", attempt)
    elif kind == "degenerate":
        logger.warning("Degenerate final: %d-char fragment %r ended the turn after %d tool result(s) — "
                       "re-prompting (%d/2)", len(text), text[:40], tool_rows, attempt)
    elif kind == "promise":
        logger.warning("Unbacked follow-up promise: reply promises %r but no wake source is armed — re-prompting "
                       "(%d/2)", promise[:80], attempt)
