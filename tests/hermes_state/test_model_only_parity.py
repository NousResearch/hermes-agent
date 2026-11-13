"""Dialect parity for the model-only display-marker predicate (RFC #23717 seam).

The shared marker test (``hermes_state_common._sql_model_only_truthy``, embedded in
``DISPLAY_VISIBLE_SQL`` and the compaction-display probe) must agree with the Python
twin (``SessionDB._is_model_only_row``) on every value shape, on both dialects:
the probe below runs the shared expression through the store's own connection, so
on PostgreSQL it exercises the converter plus the shim functions exactly as the
display reads do.
"""

import json
import os

import pytest

from agent.context_compressor import MODEL_ONLY_DISPLAY_METADATA_KEY
from hermes_state import SessionDB
from hermes_state_common import _sql_model_only_truthy

_KEY = MODEL_ONLY_DISPLAY_METADATA_KEY

# (display_metadata raw JSON text, Python-twin expectation: is the row model-only?)
# Twin truthiness: JSON true, any non-empty string, any non-zero number, and a
# non-empty object/array value hide the row; false / 0 / 0.0 / -0.0, JSON null,
# a missing key, and invalid JSON all read as absent.
SHAPES = [
    (json.dumps({_KEY: True}), True),
    (json.dumps({_KEY: False}), False),
    (json.dumps({_KEY: "true"}), True),
    (json.dumps({_KEY: "false"}), True),
    (json.dumps({_KEY: "0"}), True),
    (json.dumps({_KEY: 0}), False),
    (json.dumps({_KEY: 0.0}), False),
    (json.dumps({_KEY: 1}), True),
    (json.dumps({_KEY: -0.0}), False),
    (json.dumps({_KEY: None}), False),
    (json.dumps({"other_key": 1}), False),
    (json.dumps({_KEY: {}}), False),
    (json.dumps({_KEY: []}), False),
    (json.dumps({_KEY: [1]}), True),
    (json.dumps({_KEY: ""}), False),
    ("not json at all", False),
]


@pytest.fixture(params=["sqlite", "postgres"])
def state_db(request, tmp_path):
    if request.param == "postgres":
        # The ``postgres_db`` fixture exists only in the stage-B delta (and the
        # environment flag only ever coexists with it there); skip on the stage-A
        # tree before fixture resolution instead of erroring on a missing fixture.
        if os.environ.get("HERMES_TEST_POSTGRES") != "1":
            pytest.skip("HERMES_TEST_POSTGRES not set; skipping the PostgreSQL dialect lane")
        yield request.getfixturevalue("postgres_db")
        return
    with SessionDB(tmp_path / "state.db") as database:
        yield database


def test_predicate_matches_python_twin_on_both_dialects(state_db):
    """The shared marker predicate, executed through the store's own connection
    (PostgreSQL: converter + shims), matches the Python twin on every shape."""
    state_db.create_session("model-only-parity", "cli")

    def seed(connection):
        connection.execute("CREATE TABLE model_only_parity_probe (seq INTEGER, raw TEXT)")
        for seq, (raw, _expected) in enumerate(SHAPES):
            connection.execute(
                "INSERT INTO model_only_parity_probe (seq, raw) VALUES (?, ?)", (seq, raw))
        connection.execute("DELETE FROM model_only_parity_probe WHERE seq < 0")

    state_db._execute_write(seed)
    rows = state_db._read_all(
        f"SELECT seq, raw, {_sql_model_only_truthy('raw', _KEY)} AS model_only "
        "FROM model_only_parity_probe ORDER BY seq")
    assert [row["seq"] for row in rows] == list(range(len(SHAPES)))
    assert [bool(row["model_only"]) for row in rows] == [expected for _raw, expected in SHAPES]
    state_db._execute_write(lambda conn: conn.execute("DROP TABLE model_only_parity_probe"))


def test_display_projection_hides_model_only_rows(state_db):
    """End-to-end: the display projection (DISPLAY_VISIBLE_SQL) hides JSON-true
    marker rows and keeps false / absent marker rows, on both dialects."""
    session_id = "model-only-display"
    state_db.create_session(session_id, "cli")
    hidden = state_db.append_message(session_id, "assistant", "hidden row",
                                    display_metadata={_KEY: True})
    kept = [
        state_db.append_message(session_id, "assistant", "false marker",
                               display_metadata={_KEY: False}),
        state_db.append_message(session_id, "assistant", "absent marker",
                               display_metadata={"other_key": 1}),
    ]
    ids = [message["id"] for message in state_db.get_messages(session_id, include_compacted=True)]
    assert hidden not in ids
    assert set(kept) <= set(ids)
