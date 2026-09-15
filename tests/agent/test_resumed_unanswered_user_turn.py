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

from agent.turn_context import RESUME_UNANSWERED_TURN_ENV, build_turn_context
from hermes_state import SessionDB
from run_agent import AIAgent

DM = "Message from 🤖 sender (@sender): please confirm receipt"


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


def _close_path_would_append(agent, messages):
    """The identity test ``_persist_active_session_before_close`` applies."""
    pending = getattr(agent, "_pending_cli_user_message", None)
    return isinstance(pending, dict) and not any(m is pending for m in messages)


def _raise(*_a, **_k):
    raise RuntimeError("simulated persist failure")


USER = {"role": "user", "content": DM}


@pytest.mark.parametrize(
    ("opt_in", "left_behind", "platform_id", "expected"),
    [
        (True, [USER], None, [DM]),
        (False, [USER], None, [DM, DM]),  # no caller claims a re-run: a real second send
        (True, [USER], "discord-2", [DM, DM]),  # a platform delivery keeps its own dedup id
        (True, [USER, {"role": "assistant", "content": "ack"}], None, [DM, DM]),  # a real repeat
    ],
    ids=["opted-in-rerun", "no-opt-in", "platform-delivery", "answered-tail"],
)
def test_only_an_opted_in_rerun_adopts_the_row_its_failed_attempt_persisted(
    agent_db, monkeypatch, opt_in, left_behind, platform_id, expected,
):
    """The attempt that died persisted its user row; the re-run is a fresh process, so no
    in-process marker dedups it. Adoption stays narrow: an identical unanswered tail, and only
    when the caller says this is a re-run."""
    agent, db, sid = agent_db
    if opt_in:
        monkeypatch.setenv(RESUME_UNANSWERED_TURN_ENV, "1")
    agent._flush_messages_to_session_db([dict(m) for m in left_behind])

    ctx = _build(agent, conversation_history=db.get_messages_as_conversation(sid),
                 **({"persist_user_platform_id": platform_id} if platform_id else {}))

    assert _user_contents(ctx) == expected
    if expected == [DM]:
        assert [r["content"] for r in db.get_messages_as_conversation(sid) if r["role"] == "user"] == [DM]
        assert agent._persist_user_message_idx == len(ctx.messages) - 1


@pytest.mark.parametrize(
    ("adopting", "persist_fails"),
    [(True, False), (True, True), (False, True)],
    ids=["adopt", "adopt-persist-fails", "normal-persist-fails"],
)
def test_the_staged_cli_dict_is_dropped_only_when_the_tail_is_adopted(agent_db, monkeypatch, adopting, persist_fails):
    """The CLI stages its own user dict (``agent._pending_cli_user_message``) and the close path
    re-appends it when it is not in ``messages``. Adopting the durable tail leaves it in no list,
    so it must go, or close writes the duplicate after the answer (the #43849 replay state). A
    normal turn keeps it across a failed turn-start persist, so close can still write the input."""
    agent, db, sid = agent_db
    history = None
    if adopting:
        monkeypatch.setenv(RESUME_UNANSWERED_TURN_ENV, "1")
        agent._flush_messages_to_session_db([dict(USER)])
        history = db.get_messages_as_conversation(sid)
    if persist_fails:
        monkeypatch.setattr(agent, "_persist_session", _raise)
    staged = dict(USER)
    agent._pending_cli_user_message = staged

    ctx = _build(agent, conversation_history=history)

    assert _user_contents(ctx) == [DM]
    assert not _close_path_would_append(agent, ctx.messages)
    assert agent._pending_cli_user_message is (None if adopting else staged)
