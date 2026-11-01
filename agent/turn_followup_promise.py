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

import json
import logging
import re
from typing import Any

# A first-person promise of a LATER follow-up, in the reply's final sentence.
_PROMISE_RE = re.compile(
    r"\b(?:i(?:['\u2019]ll| will| am going to|['\u2019]m going to)|we(?:['\u2019]ll| will))\s+"
    r"(?:(?:then|also|just|now)\s+)?"
    r"(?:report back|check back|circle back|get back to you|come back to you|follow up|ping you|update you"
    r"|let you know|tell you|check (?:again|in|on (?:it|them|that|this|progress|the \w+))|re-?check"
    r"|look again|poll)\b"
    r"[^.!?\n]{0,140}",
    re.IGNORECASE,
)
_NUMBER_WORDS = r"(?:\d+|a|an|one|two|three|four|five|ten|fifteen|twenty|thirty|forty-five|sixty)"
# "...in ~10 minutes", "...in ten minutes", "...tomorrow", "...after the sweep", "...when it finishes".
_DEFERRAL_RE = re.compile(
    rf"\b(?:in (?:about |another |around |~|a few |a couple of )?(?:{_NUMBER_WORDS}\s*)?"
    r"(?:sec|second|min|minute|hour|h\b|m\b)"
    r"|in (?:a (?:bit|moment|while|few)|a couple)|later|shortly|tomorrow|tonight|overnight"
    r"|after (?:the|it|that|this|each|both)|when\b|once\b|as soon as|whenever)",
    re.IGNORECASE,
)
# Anything that could make the follow-up wait on the user, offer it as a choice, or ask the user for something. Broad on
# purpose: a missed nudge costs nothing, a nudge on a correct reply is the harm this guard must not do.
_USER_GATED_RE = re.compile(
    r"\b(?:tell me|let me know|ping me|say (?:the word|when|done|go)|once you|when you|after you|reply with|send me"
    r"|paste|if you|if that|should you|want me to|would you|do you want|shall i|unless you|first)\b",
    re.IGNORECASE,
)
# The only text allowed AFTER the promise: a short closing courtesy. Any other later sentence might report the outcome
# or add a condition, so the promise is treated as resolved (no nudge).
_SIGN_OFF_RE = re.compile(
    r"^\W*(?:thanks?(?: you)?|cheers|ty|let me know if you (?:need|have) anything(?: else)?|hope (?:that|this) helps"
    r"|more soon|talk soon|stay tuned|no action (?:is )?needed(?: from you)?(?: for now)?|nothing (?:else )?needed"
    r"(?: from you)?)\W*$",
    re.IGNORECASE,
)
# A promise introduced as an example or a quotation: "The example reply is: I will report back ...".
_EXAMPLE_INTRO_RE = re.compile(
    r"\b(?:example|e\.g\.|reply|response|message|text|output|log|line|says|said|wrote|reads|like|template|quote"
    r"|commitment|say)(?:\s+(?:is|was|would be|reads))?\s*:?\s*$",
    re.IGNORECASE,
)
_FENCED_RE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_QUOTED_RE = re.compile(r"`([^`\n]*)`|\"([^\"\n]*)\"|\u201c([^\u201d\n]*)\u201d")
_CLAUSE_BREAK_RE = re.compile(r"\s*([:;\u2014]|\s-\s)\s*")
_FUTURE_SELF_RE = re.compile(r"\b(?:i(?:['\u2019]ll| will)|we(?:['\u2019]ll| will))\b", re.IGNORECASE)
_TAIL_CHARS = 600
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _unquote(match: re.Match) -> str:
    """A quoted span that is itself a promise is someone else's words: drop it. Any other quoted span keeps its
    words, since a quoted command or a quoted "when it finishes" is still part of the model's own promise."""
    inner = next(g for g in match.groups() if g is not None)
    return " " if _PROMISE_RE.search(inner) else inner


def unbacked_promise_phrase(text: str) -> str:
    """The promise phrase if ``text`` ENDS on an unconditional deferred follow-up, else ``""``.

    Errs toward silence. The promise must be the reply's last statement: only closing courtesies ("Thanks!") may
    follow it, and its own sentence must end with it (no ": it finished", "; thanks for ..."). Any hint that the
    follow-up waits on the user, anywhere from the promise to the end, suppresses it. Fenced code, quoted promises
    and promises introduced as an example are not the model speaking."""
    cleaned = _QUOTED_RE.sub(_unquote, _FENCED_RE.sub(" ", text or ""))
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(cleaned.strip()[-_TAIL_CHARS:]) if s.strip()]
    while sentences and _SIGN_OFF_RE.match(sentences[-1]):
        sentences.pop()
    if not sentences:
        return ""
    sentence = sentences[-1]
    match = _PROMISE_RE.search(sentence)
    if not match or _EXAMPLE_INTRO_RE.search(sentence[:match.start()]):
        return ""
    promise, *aside = _CLAUSE_BREAK_RE.split(sentence[match.start():], maxsplit=1)
    # A colon after the promise introduces its outcome ("...: it finished"). A semicolon or dash may only continue
    # the plan ("; if it hasn't moved by then I'll steer it"); anything else there could be a result.
    if aside and (aside[0].startswith(":") or not _FUTURE_SELF_RE.search(aside[-1])):
        return ""
    if not _DEFERRAL_RE.search(promise) or _USER_GATED_RE.search(sentence):
        return ""
    return promise.strip().rstrip(".!")


# cronjob actions that can leave something scheduled to run; the tool result decides whether they did.
_CRON_ARMING_ACTIONS = frozenset({"create", "resume", "run"})


def _cron_result_arms(result: Any) -> bool:
    """Whether a successful cronjob result leaves a future run that will reach this conversation: an enabled job
    with a next run, a background run (its outcome re-enters the conversation), or a run handed to the gateway.
    A paused create, or an inline one-shot that already finished, arms nothing."""
    if not isinstance(result, dict) or result.get("success") is not True:
        return False
    if result.get("forwarded_to_gateway"):
        return True
    job = result.get("job") if isinstance(result.get("job"), dict) else {}
    if job.get("execution_mode") == "background":
        return True
    # Otherwise (including after an inline run, whose own outcome is already in this turn) the job must still have a
    # future run: an enabled, unpaused recurring job does; a finished one-shot or a paused job does not.
    return bool(job.get("enabled", True) and job.get("state") != "paused" and job.get("next_run_at"))


def _call_parts(call: Any) -> tuple[str, Any, str]:
    fn = (call.get("function") if isinstance(call, dict) else getattr(call, "function", None)) or {}
    get = fn.get if isinstance(fn, dict) else (lambda k: getattr(fn, k, None))
    call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
    return str(get("name") or ""), get("arguments"), str(call_id or "")


def _loads(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _cron_arming_call(name: str, raw_args: Any) -> bool:
    """A ``cronjob`` call (direct, or through the ``tool_call`` bridge) whose action arms a run."""
    args = _loads(raw_args)
    if name == "tool_call" and isinstance(args, dict):
        from tools.tool_search_validation import normalize_tool_call_entries
        entries, error = normalize_tool_call_entries(args)
        return not error and any(_cron_arming_call(e["name"], e["arguments"]) for e in entries)
    return ("cronjob" in name and isinstance(args, dict)
            and str(args.get("action") or "").strip().lower() in _CRON_ARMING_ACTIONS)


def _turn_armed_cron(messages: list) -> bool:
    """A cron job was armed since the last user row: an arming ``cronjob`` call whose result leaves a future run."""
    results: dict[str, Any] = {}
    for msg in reversed(messages or ()):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            return False
        if role == "tool":
            results[str(msg.get("tool_call_id") or "")] = _loads(msg.get("content"))
            continue
        for call in msg.get("tool_calls") or ():
            name, raw_args, call_id = _call_parts(call)
            result = results.get(call_id)
            if _cron_arming_call(name, raw_args) and _cron_result_arms(result):
                return True
    return False


def _has_pending_process_notice(agent: Any) -> bool:
    from tools.process_registry import process_registry
    return any(
        proc.notify_on_complete or proc.watch_patterns
        for owner in getattr(agent, "_process_owner_task_ids", ()) or ()
        for proc in process_registry.running_owned_by(owner))


def _has_live_delegation(agent: Any) -> bool:
    """A live background delegation dispatched from this conversation. A delegation records the session id
    at dispatch, and compression rotates ``agent.session_id`` afterwards, so match the whole compression
    lineage (never the conversation root: that crosses an explicit /new or branch boundary)."""
    from tools.async_delegation import has_live_for_session
    sid = agent.session_id or ""
    if not sid:
        return False
    db = getattr(agent, "_session_db", None)
    lineage = (db.get_compression_lineage(sid) if db is not None else None) or [sid]
    return any(has_live_for_session(parent_session_id=s) for s in {sid, *lineage})


def _has_active_schedule(agent: Any) -> bool:
    """An active /loop or /heartbeat for this session. Reads the stored state directly: the managers' loaders
    turn a read error into "no schedule", which would defeat the fail-open policy. A missing store or an
    unparseable row raises, so the caller counts it as armed."""
    from hermes_cli.goals import _get_session_db
    from hermes_cli.heartbeat import _META_PREFIX as heartbeat_prefix
    from hermes_cli.loops import _meta_key as loop_key
    sid = agent.session_id or ""
    if not sid:
        return False
    db = _get_session_db()
    if db is None:
        raise RuntimeError("session store unavailable")
    for key in (loop_key(sid), heartbeat_prefix + sid):
        raw = db.get_meta(key)
        if raw and json.loads(raw).get("status", "active") == "active":
            return True
    return False


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
    Cron runs and delegated children are exempt: they report to a caller, not to a waiting user."""
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
