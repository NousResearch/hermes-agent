"""v33 re-tags subagent rows that lost ``_delegate_from`` before their first turn.

A reserved child row whose model_config was patched before the constructor
create kept the patch and dropped the marker. The parent's resume tip walk then
followed the child as a continuation, so opening the parent showed the subagent.
"""

from __future__ import annotations

import json

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema_ddl import SCHEMA_VERSION
from persistence.sqlite.conversation._store.compression import get_compression_tip
from persistence.sqlite.conversation._store.messages import append_message, resolve_resume_session_id
from persistence.sqlite.conversation._store.sessions import create_session, end_session


def _config(db: StoreHolder, session_id: str) -> dict:
    raw = db._conn.execute("SELECT model_config FROM sessions WHERE id = ?", (session_id,)).fetchone()[0]
    return json.loads(raw) if raw else {}


def test_v33_retags_unmarked_subagent_children_and_parent_resumes_itself(tmp_path):
    db_path = tmp_path / "state.db"
    old = StoreHolder(db_path=db_path)
    create_session(old, "parent", source="desktop")
    create_session(old, "kid", source="subagent", parent_session_id="parent")
    # The shape the regression left behind: the early patch survived, the marker did not.
    old._conn.execute("UPDATE sessions SET model_config = ? WHERE id = 'kid'",
                      (json.dumps({"prompt_refresh_epoch": 17}),))
    # A continuation that is not a subagent must stay a continuation.
    create_session(old, "cont-root", source="cli")
    create_session(old, "cont", source="cli", parent_session_id="cont-root")
    append_message(old, "parent", "user", "parent turn")
    append_message(old, "kid", "assistant", "child report")
    append_message(old, "cont", "user", "continued turn")
    old._conn.commit()
    assert resolve_resume_session_id(old, "parent") == "kid"
    old._conn.execute("UPDATE schema_version SET version = 32")
    old._conn.commit()
    old.close()

    migrated = StoreHolder(db_path=db_path)
    try:
        assert _config(migrated, "kid") == {"prompt_refresh_epoch": 17, "_delegate_from": "parent"}
        assert "_delegate_from" not in _config(migrated, "cont")
        assert resolve_resume_session_id(migrated, "parent") == "parent"
        assert resolve_resume_session_id(migrated, "cont-root") == "cont"
        assert migrated._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        migrated.close()


def _downgrade_to_v32(db: StoreHolder) -> None:
    db._conn.execute("UPDATE schema_version SET version = 32")
    db._conn.commit()
    db.close()


def test_v33_skips_compression_continuation_of_an_unmarked_child(tmp_path):
    # While the bug was live, Desktop resumed the child as the chat on an agent without the marker. If
    # that conversation compressed, the continuation was written source='subagent' with no marker.
    db_path = tmp_path / "state.db"
    old = StoreHolder(db_path=db_path)
    create_session(old, "parent", source="desktop")
    create_session(old, "kid", source="subagent", parent_session_id="parent")
    old._conn.execute("UPDATE sessions SET model_config = ? WHERE id = 'kid'",
                      (json.dumps({"prompt_refresh_epoch": 17}),))
    append_message(old, "parent", "user", "parent turn")
    append_message(old, "kid", "assistant", "child report")
    end_session(old, "kid", "compression")
    create_session(old, "kid-cont", source="subagent", parent_session_id="kid")
    append_message(old, "kid-cont", "user", "typed after compression")
    old._conn.commit()
    assert get_compression_tip(old, "kid") == "kid-cont"
    _downgrade_to_v32(old)

    migrated = StoreHolder(db_path=db_path)
    try:
        assert _config(migrated, "kid")["_delegate_from"] == "parent"
        assert "_delegate_from" not in _config(migrated, "kid-cont")
        assert get_compression_tip(migrated, "kid") == "kid-cont"
        assert resolve_resume_session_id(migrated, "kid") == "kid-cont"
        assert resolve_resume_session_id(migrated, "parent") == "parent"
    finally:
        migrated.close()


def test_v33_heals_the_rest_of_the_store_past_a_malformed_model_config(tmp_path):
    db_path = tmp_path / "state.db"
    old = StoreHolder(db_path=db_path)
    create_session(old, "parent", source="desktop")
    create_session(old, "bad", source="subagent", parent_session_id="parent")
    create_session(old, "kid", source="subagent", parent_session_id="parent")
    old._conn.execute("UPDATE sessions SET model_config = '{not json' WHERE id = 'bad'")
    old._conn.execute("UPDATE sessions SET model_config = ? WHERE id = 'kid'",
                      (json.dumps({"prompt_refresh_epoch": 17}),))
    append_message(old, "parent", "user", "parent turn")
    append_message(old, "kid", "assistant", "child report")
    old._conn.commit()
    assert resolve_resume_session_id(old, "parent") == "kid"
    _downgrade_to_v32(old)

    migrated = StoreHolder(db_path=db_path)
    try:
        assert _config(migrated, "kid") == {"prompt_refresh_epoch": 17, "_delegate_from": "parent"}
        raw_bad = migrated._conn.execute("SELECT model_config FROM sessions WHERE id = 'bad'").fetchone()[0]
        assert raw_bad == "{not json"
        assert resolve_resume_session_id(migrated, "parent") == "parent"
    finally:
        migrated.close()
