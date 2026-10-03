"""Issue #109450: one torn-UTF-8 system_prompts cell must not kill session list/load.

A mid-write process death truncated the 3-byte U+2713 (e2 9c 93) to its first two bytes
(e2 9c) inside a TEXT cell, and every session-list/load query that materializes
COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved aborted with
sqlite3.OperationalError "Could not decode to UTF-8 column '_system_prompt_resolved'"
— one corrupt row made the whole session panel unable to list or open ANY session.

This file deliberately uses only SessionDB's public API (plus sqlite3 for seeding):
it must stay collectible against unmodified main so the red/green proof is a real
test failure (OperationalError), not an import error.
"""

import json
import sqlite3

import pytest

from hermes_state import SessionDB

# "…mark complete with ✓" — the exact torn tail from the issue: U+2713's first 2 bytes.
TORN_TAIL = b"\xe2\x9c"
GOOD_PROMPT = "You are Hermes Agent. Steps: write code, review, run tests, mark complete with \u2713"


def _seed_torn_prompt(db):
    db.create_session("s-torn", "cli", system_prompt="You are Hermes Agent. Steps: seed placeholder")
    row = db._read_one("SELECT hash FROM system_prompts WHERE prompt LIKE '%seed placeholder%'")
    torn = (b"You are Hermes Agent. Steps: write code, review, run tests, mark complete with "
            + TORN_TAIL)
    # CAST keeps the damaged value in TEXT storage — the issue's shape: Python's sqlite3
    # eagerly decodes TEXT cells, and the torn multi-byte tail is undecodable.
    db._execute_write(lambda conn: conn.execute(
        "UPDATE system_prompts SET prompt = CAST(? AS TEXT) WHERE hash = ?",
        (torn, row["hash"])))
    stored = db._conn.execute(
        "SELECT typeof(sp.prompt) FROM system_prompts sp"
        " JOIN sessions s ON s.system_prompt_hash = sp.hash WHERE s.id = 's-torn'"
    ).fetchone()[0]
    assert stored == "text"


def test_torn_system_prompt_degrades_on_every_session_surface(tmp_path):
    """get_session and list_sessions_rich must return the row with U+FFFD in the one
    torn cell instead of aborting the whole query (the #109450 crash)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s-good", "cli", system_prompt=GOOD_PROMPT)
        _seed_torn_prompt(db)

        loaded = db.get_session("s-torn")
        listed = {row["id"]: row for row in db.list_sessions_rich()}

        assert loaded["system_prompt"].endswith("\ufffd")
        assert "\u2713" not in loaded["system_prompt"]  # only the torn cell degraded
        assert set(listed) == {"s-good", "s-torn"}  # no session lost from the panel
        assert listed["s-good"]["system_prompt"] == GOOD_PROMPT  # valid multi-byte survives
        assert listed["s-torn"]["system_prompt"] == loaded["system_prompt"]
        assert json.dumps(loaded) and json.dumps(listed["s-torn"])  # serializable, never bytes
    finally:
        db.close()


def test_torn_model_config_cell_fails_the_mutation_seam_closed(tmp_path):
    """The read side degrades; the read-modify-write side fails closed: a torn
    model_config cell aborts patch_session_model_config with OperationalError and
    stays byte-identical, instead of degrading to U+FFFD soup that parses away to {}
    and lets the patch overwrite the stored field."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        raw = b'{"keep": "yes"}\xe2\x9c'
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET model_config = CAST(? AS TEXT) WHERE id = 's'", (raw,)))

        with pytest.raises(sqlite3.OperationalError):
            db.patch_session_model_config("s", {"new": 1})

        stored = db._read_one("SELECT CAST(model_config AS BLOB) FROM sessions WHERE id = 's'")[0]
        assert bytes(stored) == raw  # fail closed: nothing was rewritten
        # The session still lists and loads while the cell stays damaged.
        assert db.get_session("s")["id"] == "s"
        assert [row["id"] for row in db.list_sessions_rich()] == ["s"]
    finally:
        db.close()
