"""Durable context-maintenance warnings, separate from compression's anti-thrash policy."""
from __future__ import annotations

import logging
from urllib.parse import quote

from agent.credits_tracker import AgentNotice

logger = logging.getLogger(__name__)


class _RevisionKey(str):
    """A normal key for one-argument clients; gateway also forwards its state order."""

    revision: int
    state_key: str

    def __new__(cls, key: str, revision: int = 0, state_key: str | None = None):
        value = super().__new__(cls, key)
        value.revision = revision
        value.state_key = state_key or key
        return value


def notice_revision_fields(key: str) -> dict:
    if isinstance(key, _RevisionKey):
        return {"state_revision": key.revision, "state_key": key.state_key}
    return {}  # Credits and other existing warning clients keep their wire shape.

# Typed terminal outcomes only: structural skips, explicit stops and generic
# compacted statuses are not evidence of a broken summary pipeline.
_FAILURES = {
    "summary_generation_aborted": "summary generation aborted",
    "summary_generation_failed": "summary generation failed; fallback used",
    "summary_auth_failure": "summary provider access or quota failure",
    "summary_network_failure": "summary provider network failure",
    "summary_truncated_failure": "incomplete summary",
    "summary_empty_content_failure": "empty summary",
    "aux_model_fallback": "summary provider fallback used",
    "session_split_failed": "session commit failed",
    "stall_interrupted": "summary attempt stalled or timed out",
}


def replay_context_notice(db, session_id: str, notice_callback, notice_clear_callback) -> None:
    """Snapshot replay also clears a warning resolved while the client was disconnected."""
    if not callable(getattr(type(db), "get_context_notice_state", None)):
        return  # Third-party stores without this optional state have no snapshot.
    state = db.get_context_notice_state(session_id)
    if not state:
        return
    notice = _notice(db, session_id, state)
    if notice is not None:
        notice_callback(notice)
    else:
        notice_clear_callback(_RevisionKey(_notice_key(db, session_id), state.get("revision", 0)))


def _notice_key(db, session_id: str) -> str:
    return f"context-maintenance:{quote(str(db.db_path), safe='')}:{quote(session_id, safe='')}"


def _notice(db, session_id: str, state: dict) -> AgentNotice | None:
    problems = []
    if state.get("failures", 0) >= 2:
        reason = _FAILURES.get(state.get("failure_class"), "summary attempt failed")
        problems.append(f"Summary-route or commit problems in {state['failures']} consecutive attempts (latest: {reason}).")
    if state.get("pressure", 0) >= 2:
        qualifier = "Estimated context" if state.get("pressure_source") != "provider" else "Context"
        reason = "candidate would grow the transcript" if state.get("pressure_source") == "rejected" else "insufficient reduction"
        problems.append(f"{qualifier} pressure remains high after {state['pressure']} compaction attempts (latest: {reason}).")
    if not problems:
        return None
    key = _RevisionKey(_notice_key(db, session_id), state.get("revision", 0))
    title = db.get_session_title(session_id)
    label = f"{title} ({session_id})" if title else session_id
    return AgentNotice(
        text=f"{label}: {' '.join(problems)}",
        level="warn", kind="sticky", key=key, id=key,
    )


def _record(agent, session_id: str, update_state) -> None:
    db = getattr(agent, "_session_db", None)
    if db is None or not session_id:
        return
    try:
        before, after = db.update_context_notice_state(session_id, update_state)
        old, new = _notice(db, session_id, before), _notice(db, session_id, after)
        if old == new:
            # Invisible state can still supersede an in-flight recovery TTL.
            # Publish only a clear watermark (never a warning or a recovery).
            if before and before != after and new is None and callable(
                callback := getattr(agent, "notice_clear_callback", None)
            ):
                callback(_RevisionKey(_notice_key(db, session_id), after.get("revision", 0)))
            return
        if new is not None and callable(callback := getattr(agent, "notice_callback", None)):
            callback(new)
        elif old is not None and callable(callback := getattr(agent, "notice_clear_callback", None)):
            key = _RevisionKey(_notice_key(db, session_id), after.get("revision", 0))
            callback(key)
            if callable(show := getattr(agent, "notice_callback", None)):
                recovered_key = _RevisionKey(f"{key}:recovered", key.revision, str(key))
                show(AgentNotice(
                    text=f"{session_id}: Context maintenance has recovered.", level="success", kind="ttl",
                    ttl_ms=5000, key=recovered_key, id=recovered_key,
                ))
    except Exception:
        logger.warning("Could not persist or deliver context notice for %s", session_id, exc_info=True)


def _pressure_verdict(state: dict, attempt_id: str, source: str) -> dict:
    seen = state.get("pressure_attempts", [])
    if not attempt_id or attempt_id in seen:
        return state
    return {**state, "pressure": state.get("pressure", 0) + 1,
            "pressure_attempts": [*seen, attempt_id], "pressure_source": source}


def record_compression_outcome(agent, payload: dict) -> None:
    """Consume completed host verdicts. First terminal outcome wins, even after restart.

    Keep attempt ids for the session lifetime so delayed/duplicate outcomes cannot
    resurrect a resolved warning. No counter here affects compression admission.
    """
    session_id, attempt_id = payload.get("session_id"), payload.get("attempt_id")
    committed = payload.get("commit_status") == "committed"
    if not attempt_id or payload.get("commit_status") not in {"committed", "aborted"}:
        return
    failure_class = payload.get("failure_class")
    failed = failure_class in _FAILURES
    rejected = failure_class == "would_grow"
    recovered = committed and not payload.get("fallback_used") and not failure_class
    if not (failed or rejected or recovered or (committed and payload.get("retry_of"))):
        return

    def update(state):
        seen = state.get("outcome_attempts", [])
        if attempt_id in seen:
            return state
        after = {**state, "outcome_attempts": [*seen, attempt_id]}
        if failed and payload.get("retry_of") not in seen:
            after["failures"] = state.get("failures", 0) + 1
            after["failure_class"] = failure_class
        if recovered:
            after["failures"] = 0
        if committed:
            after["pending_usage"] = attempt_id
        return _pressure_verdict(after, attempt_id, "rejected") if rejected else after

    _record(agent, session_id, update)


def record_insufficient_progress(agent) -> None:
    """An assembled-request estimate, not a semantic verdict or provider measurement."""
    attempt_id = getattr(agent, "_compression_attempt_id", None)

    def update(state):
        if not attempt_id or state.get("pending_usage") != attempt_id:
            return state  # No completed attempt: lock/cooldown/structural skips stay neutral.
        return _pressure_verdict(state, attempt_id, "estimated")

    _record(agent, getattr(agent, "session_id", ""), update)


def record_context_usage(agent, prompt_tokens: int = 0) -> None:
    """Only positive provider usage proves pressure relief; summary faults are independent."""
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", "")
    if db is None or not session_id:
        return
    try:
        state = db.get_context_notice_state(session_id)
    except Exception:
        logger.warning("Could not read context notice for %s", session_id, exc_info=True)
        return  # Unknown is not recovery; leave the visible warning alone.
    if not state.get("pending_usage") and not state.get("pressure"):
        return  # Ordinary responses retain the queued, non-blocking token-write path.
    threshold = getattr(agent.context_compressor, "threshold_tokens", 0)

    def update(state):
        if not state.get("pending_usage") and not state.get("pressure"):
            return state
        after = {**state, "pending_usage": None}
        if prompt_tokens > 0 and threshold > 0:
            if prompt_tokens < threshold:
                after["pressure"] = 0
            else:
                after = _pressure_verdict(after, state.get("pending_usage"), "provider")
        return after

    _record(agent, getattr(agent, "session_id", ""), update)
