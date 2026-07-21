"""Regression (#68454): /new, /resume, /branch must not re-append cold-resumed
transcript rows when flushing before session rotation.

Sibling of #68196 / #68205 (preflight compression path). Cold resume loads
plain dicts without ``_DB_PERSISTED_MARKER``. If the user rotates immediately
via /new, /resume, or /branch, the old-session flush must pass
``conversation_history=`` so identity skip treats the loaded prefix as durable.

These tests bind the real ``AIAgent._flush_messages_to_session_db`` methods onto
a lightweight stand-in so construction never hits network model-metadata
lookups (offline CI / hung OpenRouter).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hermes_state import SessionDB
from run_agent import AIAgent


def _make_flush_agent(db: SessionDB, session_id: str):
    """Minimal agent shell that owns the real flush implementation."""
    agent = SimpleNamespace(
        _session_db=db,
        _session_db_created=True,
        _persist_disabled=False,
        session_id=session_id,
        _session_persist_lock=None,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _pending_cli_user_message=None,
    )
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = (
        AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    )
    agent._flush_messages_to_session_db_unlocked = (
        AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    )
    return agent


def _contents(rows):
    return [r.get("content") for r in rows]


def test_rotation_flush_without_history_boundary_duplicates(tmp_path: Path) -> None:
    """Control: bare flush of unstamped cold-resume rows double-writes (#68454)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "COLD_ROTATE_DUP"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "persisted question")
    db.append_message(sid, "assistant", "persisted answer")

    loaded = db.get_messages_as_conversation(sid)
    agent = _make_flush_agent(db, sid)
    agent._flush_messages_to_session_db(loaded)  # missing conversation_history=

    rows = db.get_messages_as_conversation(sid, include_inactive=True)
    assert _contents(rows).count("persisted question") == 2


def test_rotation_flush_with_history_boundary_is_noop(tmp_path: Path) -> None:
    """Fix shape used by /new, /resume, /branch: pass conversation_history=self list."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "COLD_ROTATE_OK"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "persisted question")
    db.append_message(sid, "assistant", "persisted answer")

    loaded = db.get_messages_as_conversation(sid)
    agent = _make_flush_agent(db, sid)
    agent._flush_messages_to_session_db(
        loaded,
        conversation_history=loaded,
    )

    rows = db.get_messages_as_conversation(sid, include_inactive=True)
    assert _contents(rows) == ["persisted question", "persisted answer"]


def test_rotation_flush_writes_only_new_tail(tmp_path: Path) -> None:
    """If a turn added messages after cold resume, only the tail is written."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "COLD_ROTATE_TAIL"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "persisted question")
    db.append_message(sid, "assistant", "persisted answer")

    loaded = db.get_messages_as_conversation(sid)
    live = [*loaded, {"role": "user", "content": "new unpersisted turn"}]
    agent = _make_flush_agent(db, sid)
    agent._flush_messages_to_session_db(
        live,
        conversation_history=loaded,
    )

    rows = db.get_messages_as_conversation(sid, include_inactive=True)
    assert _contents(rows) == [
        "persisted question",
        "persisted answer",
        "new unpersisted turn",
    ]


def test_call_sites_pass_conversation_history_kwarg() -> None:
    """Source guard: the three rotation flushes named in #68454 stay fixed."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    cli = (root / "cli.py").read_text()
    mixin = (root / "hermes_cli" / "cli_commands_mixin.py").read_text()

    # /new site uses conversation_history=self.conversation_history
    assert "conversation_history=self.conversation_history" in cli
    # /resume and /branch sites in the mixin
    assert mixin.count("conversation_history=self.conversation_history") >= 2
