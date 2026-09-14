"""Creation evidence must not be manufactured by legacy reopen-time inference."""
import json
import sqlite3
import time
from contextlib import closing

import pytest

from gateway.session_recovery import SessionRecoveryMixin
from hermes_state import SessionDB
from hermes_state_common import _RESET_END_REASONS


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    store = SessionDB(tmp_path / "state.db")
    try:
        yield store
    finally:
        store.close()


@pytest.mark.parametrize("reason", _RESET_END_REASONS)
def test_reopen_never_promotes_preboundary_or_conflicting_children(db, reason):
    key = "agent:main:telegram:dm:reset-test"
    start = time.time() - 1000
    db.create_session("parent", source="telegram", session_key=key)
    db.end_session("parent", "branched")
    db.create_session("branch", source="telegram", parent_session_id="parent", session_key=key)
    db.append_message("branch", role="user", content="Keep this branch's history")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'parent'", (start + 50,))
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'branch'", (start + 100,))
    db._conn.commit()
    db.reopen_session("parent")
    db.end_session("parent", reason)
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'parent'", (start + 200,))
    db._conn.commit()

    # These post-boundary controls must not turn a source/marker conflict or
    # malformed metadata into a durable reset. JSON null is marker PRESENCE too.
    controls = {
        "marked-branch": '{"_branched_from":"parent"}',
        "empty-branch": '{"_branched_from":""}',
        "null-branch": '{"_branched_from":null}',
        "marked-delegate": '{"_delegate_from":"parent"}',
        "null-delegate": '{"_delegate_from":null}',
        "null-reset": '{"_reset_from":null}',
        "creation-only": '{"_reset_created_from":"parent"}',
        "malformed": '{"broken":',
        "array": '[]',
        "scalar": 'null',
        "deep": '{"nested":' + '[' * 5000 + '0' + ']' * 5000 + '}',
    }
    for sid, raw in controls.items():
        db.create_session(sid, source="telegram", parent_session_id="parent", session_key=key)
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, model_config = ? WHERE id = ?",
            (start + 300, raw, sid),
        )
        db._conn.commit()
    for sid, timestamp, source, child_key in (
        ("infinite-child", float("inf"), "telegram", key),
        ("invalid-child", "not-a-time", "telegram", key),
        ("raw-subagent", start + 300, "subagent", key),
        ("raw-tool", start + 300, "tool", key),
        ("wrong-key", start + 300, "telegram", key + ":other"),
        ("empty-key", start + 300, "telegram", ""),
        ("blank-key", start + 300, "telegram", " "),
    ):
        db.create_session(sid, source=source, parent_session_id="parent", session_key=child_key)
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (timestamp, sid))
        db._conn.commit()
        controls[sid] = None
    for i, (ended, parent_key) in enumerate((
        (None, key), (float("inf"), key), (float("-inf"), key), ("not-a-time", key),
        (start + 200, ""), (start + 200, " "),
    )):
        sid, parent = f"bad-boundary-{i}", f"bad-parent-{i}"
        db.create_session(parent, source="telegram", session_key=parent_key)
        db.end_session(parent, reason)
        db.create_session(sid, source="telegram", parent_session_id=parent, session_key=parent_key)
        db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (ended, parent))
        db._conn.commit()
        db.reopen_session(parent)
        controls[sid] = None
    # Ordered, genuinely markerless legacy reset children remain compatible.
    for sid, ts in (("legacy-equal", start + 200), ("legacy-after", start + 300)):
        db.create_session(sid, source="telegram", parent_session_id="parent", session_key=key)
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (ts, sid))
        db._conn.commit()
    db.reopen_session("parent")
    db.reopen_session("parent")  # The now-cleared parent cannot change the verdict.

    rows = {row[0]: row[1] for row in db._conn.execute("SELECT id, model_config FROM sessions")}
    assert rows["branch"] is None
    for sid, raw in controls.items():
        assert rows[sid] == raw, sid
    for sid in ("legacy-equal", "legacy-after"):
        assert json.loads(rows[sid]) == {"_reset_from": "parent"}
    assert db.get_session("parent")["ended_at"] is None
    assert db.get_session("parent")["end_reason"] is None
    assert db._conn.execute(
        "SELECT content FROM messages WHERE session_id = 'branch'"
    ).fetchall()[0][0] == "Keep this branch's history"


@pytest.mark.parametrize("created", (True, False), ids=("creation", "legacy"))
def test_reset_creation_evidence_survives_enrichment_without_upgrading_legacy(db, created):
    key = "agent:main:telegram:dm:reset-test"
    db.create_session("parent", source="telegram", session_key=key)
    db.end_session("parent", "session_reset")
    # Compose the actual gateway creation writer with the real SessionDB, rather
    # than hand-seeding the new marker that this regression is meant to require.
    kwargs = SessionRecoveryMixin._session_create_kwargs(
        session_id="reset", session_key=key, origin=None, source_value="telegram",
        display_name=None, parent_session_id="parent",
    )
    if not created:
        kwargs["model_config"] = {"_reset_from": "parent"}
    db.create_session(**kwargs)
    stored = json.loads(db.get_session("reset")["model_config"])
    assert stored.get("_reset_created_from") == ("parent" if created else None)

    # A later model-settings upsert may neither erase/replace creation evidence
    # nor introduce it into a legacy identity row that never had it.
    db.ensure_session("reset", source="telegram", model="model-one", model_config={
        "temperature": 0.25, "_reset_from": "wrong-parent", "_reset_created_from": "wrong-parent",
    })
    stored = json.loads(db.get_session("reset")["model_config"])
    assert stored["temperature"] == 0.25
    assert stored["_reset_from"] == "parent"
    assert stored.get("_reset_created_from") == ("parent" if created else None)
    db.patch_session_model_config("reset", {"extra_setting": True})
    db.update_session_model("reset", "model-two", provider="test-provider")
    db.reopen_session("parent")
    db.end_session("parent", "session_switch")
    db.reopen_session("parent")
    path = db.db_path
    db.close()
    with closing(sqlite3.connect(path)) as reader:
        row = reader.execute(
            "SELECT parent_session_id, model_config FROM sessions WHERE id = 'reset'"
        ).fetchone()
        assert row[0] == "parent"
        stored = json.loads(row[1])
        assert stored["_reset_from"] == "parent"
        assert stored.get("_reset_created_from") == ("parent" if created else None)
        assert stored["temperature"] == 0.25
        assert stored["extra_setting"] is True
        assert stored["provider"] == "test-provider"
