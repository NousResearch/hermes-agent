"""A re-run of a turn that died before answering must not re-append its user message.

A failed turn still persists its user row (the crash-resilience turn-start persist), so
anything that re-sends the same input as a NEW process — the Bot Mode DM retry, a re-run
cron delivery, crash recovery — used to append a second identical copy. The recipient was
then shown the same message twice inside one prompt, and the transcript kept both forever.

The in-process guard (``_DB_PERSISTED_MARKER``) cannot cover this: the re-run is a fresh
process, so it rebuilds the user row as a brand-new object with no stamp.

The adoption is deliberately narrow — only an UNANSWERED tail is adopted. A completed turn
always ends with an assistant row, so deliberately sending the same message twice still
appends it twice.
"""

from __future__ import annotations

import shutil
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.turn_context import (
    RESUME_UNANSWERED_TURN_ENV,
    _resumes_unanswered_user_turn,
    build_turn_context,
)
from hermes_state import SessionDB
from run_agent import AIAgent

DM = "Message from 🤖 sender (@sender): please confirm receipt"


@pytest.fixture()
def resuming(monkeypatch):
    """What the retrying delivery runner sets for the process it re-runs."""
    monkeypatch.setenv(RESUME_UNANSWERED_TURN_ENV, "1")


@pytest.fixture()
def agent_db():
    tmp = tempfile.mkdtemp(prefix="resumed_unanswered_")
    db = SessionDB(Path(tmp) / "state.db")
    sid = "sess-resumed-unanswered"
    db.create_session(session_id=sid, source="desktop", model="test-model")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=db,
        session_id=sid,
    )
    agent._session_db_created = True
    agent._cached_system_prompt = "SYSTEM"
    agent._skip_mcp_refresh = True
    try:
        yield agent, db, sid
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message=DM,
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        return build_turn_context(**kwargs)


def _user_contents(ctx):
    return [m.get("content") for m in ctx.messages if isinstance(m, dict) and m.get("role") == "user"]


# ── the predicate ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tail,expected",
    [
        ([{"role": "user", "content": DM}], True),
        ([{"role": "user", "content": "something else"}], False),
        ([{"role": "user", "content": DM}, {"role": "assistant", "content": "ack"}], False),
        ([{"role": "tool", "content": DM}], False),
        ([], False),
    ],
    ids=["identical-unanswered", "different-user", "already-answered", "tool-tail", "empty"],
)
def test_predicate_adopts_only_an_identical_unanswered_tail(tail, expected, resuming):
    assert _resumes_unanswered_user_turn(tail, {"role": "user", "content": DM}) is expected


def test_predicate_never_adopts_an_empty_message(resuming):
    """Two blank turns in a row must not collapse into one."""
    assert _resumes_unanswered_user_turn([{"role": "user", "content": ""}], {"role": "user", "content": ""}) is False


# ── the real turn-setup path ─────────────────────────────────────────────────


def test_a_rerun_of_an_unanswered_turn_keeps_one_user_row(agent_db, resuming):
    """The regression, end to end: the first attempt persisted the row and died unanswered;
    the re-run must adopt it rather than write a second copy."""
    agent, db, sid = agent_db

    # What the attempt that died actually left behind: the turn-start persist wrote the row.
    agent._flush_messages_to_session_db([{"role": "user", "content": DM}])
    assert [r["content"] for r in db.get_messages_as_conversation(sid) if r["role"] == "user"] == [DM]

    # The re-run is a FRESH process: it reads history back from the DB, so nothing carries the
    # in-process ``_DB_PERSISTED_MARKER`` that dedups repeat flushes within one turn.
    ctx = _build(agent, conversation_history=db.get_messages_as_conversation(sid))

    assert _user_contents(ctx) == [DM], "the re-run must not append a second copy"
    rows = [r for r in db.get_messages_as_conversation(sid) if r["role"] == "user"]
    assert [r["content"] for r in rows] == [DM], "the transcript must still hold exactly one copy"
    # The turn still points at that row, so the persist override and reanchor stay correct.
    assert agent._persist_user_message_idx == len(ctx.messages) - 1


def test_a_repeat_after_a_completed_turn_still_appends(agent_db):
    """Sending the same message again on purpose is a real second turn."""
    agent, _db, _sid = agent_db

    ctx = _build(
        agent,
        conversation_history=[{"role": "user", "content": DM}, {"role": "assistant", "content": "ack"}],
    )

    assert _user_contents(ctx) == [DM, DM]


def test_a_first_turn_appends_normally(agent_db):
    agent, db, sid = agent_db

    ctx = _build(agent)

    assert _user_contents(ctx) == [DM]
    assert [r["content"] for r in db.get_messages_as_conversation(sid) if r["role"] == "user"] == [DM]


# ── the staged CLI dict must not outlive the turn ────────────────────────────
#
# The interactive CLI stages its own user dict, hands the turn
# ``conversation_history[:-1]``, and keeps a reference in
# ``agent._pending_cli_user_message``. On exit, ``_persist_active_session_before_close``
# appends that dict when it is not identity-present in ``messages``. Adopting the durable
# tail leaves the staged dict in no list at all, so without clearing it the close path
# writes the duplicate anyway — AFTER the answer, leaving the transcript ending on an
# unanswered user row (the #43849 replay state). Found by adversarial review of this fix.


def _close_path_would_append(agent, messages):
    """The identity test ``_persist_active_session_before_close`` applies."""
    pending = getattr(agent, "_pending_cli_user_message", None)
    return isinstance(pending, dict) and not any(m is pending for m in messages)


def test_adopting_the_tail_clears_the_staged_cli_dict(agent_db, resuming):
    """The regression the review caught: no orphan for the close path to resurrect."""
    agent, db, sid = agent_db
    agent._flush_messages_to_session_db([{"role": "user", "content": DM}])

    # Exactly what the CLI does: a NEW dict with the same text, kept on the agent, while the
    # turn is handed the history WITHOUT it (its tail is the previous, unanswered row).
    staged = {"role": "user", "content": DM}
    agent._pending_cli_user_message = staged

    ctx = _build(agent, conversation_history=db.get_messages_as_conversation(sid))

    assert agent._pending_cli_user_message is None, "the obsolete staged dict must be dropped"
    assert not _close_path_would_append(agent, ctx.messages), "close would re-append the duplicate"
    assert _user_contents(ctx) == [DM]
    assert [r["content"] for r in db.get_messages_as_conversation(sid) if r["role"] == "user"] == [DM]


def test_a_normal_turn_still_hands_the_staged_dict_to_the_transcript(agent_db):
    """Control: when the guard does NOT fire, the staged dict is the turn's row as before,
    so the close path finds it present and the existing handoff contract is untouched."""
    agent, _db, _sid = agent_db
    staged = {"role": "user", "content": DM}
    agent._pending_cli_user_message = staged

    ctx = _build(agent)

    assert any(m is staged for m in ctx.messages), "the staged dict must still BE the turn's row"
    assert not _close_path_would_append(agent, ctx.messages)


def _raise(*_a, **_k):
    raise RuntimeError("simulated persist failure")


def test_a_normal_turn_keeps_the_staged_dict_when_the_turn_start_persist_fails(agent_db, monkeypatch):
    """The pre-existing contract this fix must not weaken: an UNMARKED staged dict survives a
    failed turn-start flush precisely so the close path can still write the user's input
    (``_persist_under_lock`` only clears it once ``_db_persisted`` is stamped). Clearing it on
    the normal path would silently lose a message whenever the flush fails."""
    agent, _db, _sid = agent_db
    monkeypatch.setattr(agent, "_persist_session", _raise)
    staged = {"role": "user", "content": DM}
    agent._pending_cli_user_message = staged

    _build(agent)  # a first turn: the guard does not fire

    assert agent._pending_cli_user_message is staged, "a normal turn must keep it for the close retry"


def test_adopting_the_tail_drops_the_staged_dict_even_when_the_persist_fails(agent_db, monkeypatch, resuming):
    """The adopted row is already durable, so there is nothing for the close path to rescue —
    the obsolete dict goes regardless of whether this turn's own flush succeeded."""
    agent, _db, _sid = agent_db
    monkeypatch.setattr(agent, "_persist_session", _raise)
    staged = {"role": "user", "content": DM}
    agent._pending_cli_user_message = staged

    _build(agent, conversation_history=[{"role": "user", "content": DM}])

    assert agent._pending_cli_user_message is None


# ── the opt-in gate: the two ways adoption must refuse ───────────────────────


def test_without_the_opt_in_an_identical_tail_still_appends(agent_db):
    """The unsound inference this fix must NOT make. A person who re-sends the same word after a
    failed turn has sent a SECOND real message; with no caller claiming a re-run, it gets its own
    row. Found by adversarial review: inferring from the transcript alone swallowed it."""
    agent, db, sid = agent_db
    agent._flush_messages_to_session_db([{"role": "user", "content": DM}])

    ctx = _build(agent, conversation_history=db.get_messages_as_conversation(sid))

    assert _user_contents(ctx) == [DM, DM], "a real second send must not be adopted away"


def test_a_platform_delivery_is_never_adopted(agent_db, resuming):
    """Even under the opt-in, a row carrying a platform_message_id is a distinct delivery: the id
    is load-bearing for restart drain-window dedup (has_platform_message_id), so adopting the
    older row would drop it and let the message be re-delivered after a restart."""
    agent, db, sid = agent_db
    agent._flush_messages_to_session_db([{"role": "user", "content": DM}])

    ctx = _build(
        agent,
        conversation_history=db.get_messages_as_conversation(sid),
        persist_user_platform_id="discord-2",
    )

    assert _user_contents(ctx) == [DM, DM]
    assert ctx.messages[-1].get("platform_message_id") == "discord-2"


def test_the_env_flag_is_read_per_turn_not_cached(agent_db, monkeypatch):
    """A long-lived process must not stay in resume mode after one retried delivery."""
    agent, db, sid = agent_db
    agent._flush_messages_to_session_db([{"role": "user", "content": DM}])
    history = db.get_messages_as_conversation(sid)

    monkeypatch.setenv(RESUME_UNANSWERED_TURN_ENV, "1")
    assert _resumes_unanswered_user_turn(history, {"role": "user", "content": DM}) is True
    monkeypatch.delenv(RESUME_UNANSWERED_TURN_ENV)
    assert _resumes_unanswered_user_turn(history, {"role": "user", "content": DM}) is False
