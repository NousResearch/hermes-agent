"""v16 delegate-tag migration must not hide ordinary short tool-using roots.

The v16 backfill tags pre-marker subagent rows with ``_delegate_from`` so
pickers stay clean. Its parentless "__orphaned__" heuristic (untitled,
``message_count <= 25``, has a tool message, no children) also matches every
short CLI or cron run that called a tool, so upgrading a pre-v16 store hid
those real conversations from every session picker. Only rows that are
provably delegates — ephemeral children of a live parent, or rows already
carrying an explicit marker — may be tagged and hidden.
"""

from __future__ import annotations

import json

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION

_PRE_DELEGATE_TAG_VERSION = 15


def _tool_turn(db: SessionDB, sid: str) -> None:
    db.append_message(sid, "user", "check disk usage")
    db.append_message(sid, "assistant", None, tool_calls=[
        {"id": "call-1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
    db.append_message(sid, "tool", "42G free", tool_call_id="call-1", tool_name="terminal")
    db.append_message(sid, "assistant", "42G free.")


def _delegate_from(session: dict | None):
    assert session is not None
    raw = session.get("model_config")
    cfg = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
    return cfg.get("_delegate_from")


@pytest.fixture
def migrated(tmp_path):
    db_path = tmp_path / "state.db"
    old = SessionDB(db_path=db_path)
    # Ordinary untitled, short, tool-using roots: the shape of a quick CLI run or cron job.
    old.create_session("cli-root", source="cli")
    _tool_turn(old, "cli-root")
    old.create_session("cron-root", source="cron")
    _tool_turn(old, "cron-root")
    # Pre-marker subagent run under a live parent (no branch/compression/reset edge).
    old.create_session("parent", source="cli")
    old.append_message("parent", "user", "delegate the audit")
    old.create_session("eph-kid", source="cli", parent_session_id="parent")
    _tool_turn(old, "eph-kid")
    # Real delegate whose parent was deleted: already explicitly tagged.
    old.create_session("tagged-kid", source="cli", model_config={"_delegate_from": "gone-parent"})
    _tool_turn(old, "tagged-kid")
    old._conn.execute("UPDATE schema_version SET version = ?", (_PRE_DELEGATE_TAG_VERSION,))
    old._conn.commit()
    old.close()

    db = SessionDB(db_path=db_path)
    yield db
    db.close()


def _rich_ids(db: SessionDB, **kwargs) -> set[str]:
    return {row["id"] for row in db.list_sessions_rich(limit=100, **kwargs)}


def _recent_ids(db: SessionDB) -> set[str]:
    return {row["id"] for row in db.list_recent_sessions_bounded(limit=100)}


def test_short_tool_using_cli_and_cron_roots_stay_visible_after_v16(migrated: SessionDB):
    assert migrated._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    for sid in ("cli-root", "cron-root"):
        session = migrated.get_session(sid)
        assert _delegate_from(session) is None, sid
        assert migrated.resolve_resume_session_id(sid) == sid
    default_ids = _rich_ids(migrated)
    assert {"cli-root", "cron-root"} <= default_ids
    assert "cron-root" in _rich_ids(migrated, source="cron")
    assert "cli-root" in _rich_ids(migrated, source="cli")
    assert {"cli-root", "cron-root"} <= _recent_ids(migrated)


def test_ephemeral_child_is_still_tagged_with_its_parent_and_hidden(migrated: SessionDB):
    assert _delegate_from(migrated.get_session("eph-kid")) == "parent"
    assert "eph-kid" not in _rich_ids(migrated)
    assert "eph-kid" not in _recent_ids(migrated)
    # A delegate child never hijacks the parent's resume target.
    assert migrated.resolve_resume_session_id("parent") == "parent"


def test_explicitly_tagged_delegate_keeps_its_marker_and_stays_hidden(migrated: SessionDB):
    assert _delegate_from(migrated.get_session("tagged-kid")) == "gone-parent"
    assert "tagged-kid" not in _rich_ids(migrated)
    assert "tagged-kid" not in _rich_ids(migrated, source="cli")
    assert "tagged-kid" not in _recent_ids(migrated)
