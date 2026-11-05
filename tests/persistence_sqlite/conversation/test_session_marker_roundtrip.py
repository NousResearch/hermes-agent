"""Stored fork markers on SQLite remain available beside the typed projection."""

import json
import sqlite3

from persistence.conversation.sessions import NewSession, ReplaceSessionConfig
from tests.persistence_sqlite.conftest import store_for


def _stored_config(scope, session_id):
    with sqlite3.connect(store_for(scope)) as connection:
        return connection.execute("SELECT model_config FROM sessions WHERE id = ?", (session_id,)).fetchone()[0]


def test_create_and_read_preserve_every_lineage_marker(backend, scope_a):
    config = {"before": 1, "_branched_from": " branch ", "_reset_from": " reset ", "after": 2}
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession("parent", "cli"))
    sessions.create(NewSession("child", "cli", parent_session_id="parent", model_config=config))
    assert json.loads(_stored_config(scope_a, "child")) == config
    record = sessions.get("child")
    assert dict(record.model_config) == config
    assert record.lineage_origin_id == " branch "
    assert sessions.get_model_config_value("child", "_branched_from") == " branch "
    sessions.replace_session_config("child", ReplaceSessionConfig(model_config=config))
    assert _stored_config(scope_a, "child") == json.dumps(config)
