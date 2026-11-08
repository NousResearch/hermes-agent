"""Gateway-minted messaging sessions must carry a model (#146723).

``record_gateway_session_peer`` self-heals a missing row with an INSERT that omitted
``model``, so every messaging session minted by the gateway kept ``sessions.model = NULL``
forever and the TUI ``/sessions`` switcher died on ``null.replace``. The insert now
carries the profile's configured model, and an existing value (a ``/model`` override the
row already recorded) is never overwritten.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import config
from hermes_state import SessionDB


def _model_of(db_path, session_id):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT model FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def test_gateway_peer_insert_carries_the_configured_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config, "load_config_readonly", lambda: {"model": {"default": "test-model"}}
    )
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.record_gateway_session_peer(
        "sess-gw-1",
        source="telegram",
        user_id="u1",
        session_key="telegram:dm:u1",
        chat_id="c1",
    )
    assert _model_of(db_path, "sess-gw-1") == "test-model"


def test_gateway_peer_insert_accepts_a_top_level_model_string(tmp_path, monkeypatch):
    """``model: <name>`` is the other accepted config shape."""
    monkeypatch.setattr(
        config, "load_config_readonly", lambda: {"model": "flat-model"}
    )
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.record_gateway_session_peer(
        "sess-gw-2", source="discord", session_key="discord:dm:u2", chat_id="c2"
    )
    assert _model_of(db_path, "sess-gw-2") == "flat-model"


def test_gateway_peer_keeps_an_existing_model(tmp_path, monkeypatch):
    """A row that already recorded a model (e.g. a /model override) keeps it."""
    monkeypatch.setattr(
        config, "load_config_readonly", lambda: {"model": {"default": "configured-model"}}
    )
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db._insert_session_row("sess-gw-3", source="cli", model="override-model")
    db.record_gateway_session_peer(
        "sess-gw-3", source="telegram", session_key="telegram:dm:u3", chat_id="c3"
    )
    assert _model_of(db_path, "sess-gw-3") == "override-model"


def test_normal_insert_still_carries_the_model(tmp_path):
    """Control: the non-gateway insert path already stored the model."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db._insert_session_row("sess-local-1", source="cli", model="cli-model")
    assert _model_of(db_path, "sess-local-1") == "cli-model"
