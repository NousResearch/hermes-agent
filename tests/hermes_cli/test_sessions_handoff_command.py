"""Programmatic ``hermes sessions handoff`` coverage."""

import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_state import SessionDB
from hermes_cli.sessions_cmd import _cmd_handoff


def _db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.ensure_session("20260913_120000_abcdef", "slack")
    db.set_session_title("20260913_120000_abcdef", "handoff source")
    return db


def test_command_queues_explicit_target_without_waiting(tmp_path, monkeypatch, capsys):
    db = _db(tmp_path, monkeypatch)
    message_file = tmp_path / "handoff.md"
    message_file.write_text("Continue this exact task", encoding="utf-8")
    try:
        result = _cmd_handoff(db, Namespace(
            session_id="handoff source",
            to="SLACK:C012MixedCase",
            scope="T_WORKSPACE",
            chat_type="group",
            message_file=message_file,
            require_thread=True,
            wait=False,
            json=True,
        ))
        row = db.get_handoff_state("20260913_120000_abcdef")
        session = db.get_session("20260913_120000_abcdef")
    finally:
        db.close()

    assert result == 0
    assert row is not None
    assert row["state"] == "pending"
    assert row["platform"] == "slack"
    assert row["target_ref"] == "C012MixedCase"
    assert row["scope_id"] == "T_WORKSPACE"
    assert row["chat_type"] == "group"
    assert row["require_thread"] is True
    assert session is not None
    assert session["handoff_kickoff_text"] == "Continue this exact task"
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "state": "pending",
        "session_id": "20260913_120000_abcdef",
        "target": "slack:C012MixedCase",
    }


def test_command_rejects_existing_thread_target(tmp_path, monkeypatch, capsys):
    db = _db(tmp_path, monkeypatch)
    try:
        result = _cmd_handoff(db, Namespace(
            session_id="20260913_120000_abcdef",
            to="slack:C1:1700000000.000100",
            message_file=None,
            require_thread=False,
            wait=False,
            json=False,
        ))
    finally:
        db.close()

    assert result == 2
    assert "fresh thread" in capsys.readouterr().out


def test_json_not_found_is_one_object(tmp_path, monkeypatch, capsys):
    db = _db(tmp_path, monkeypatch)
    try:
        result = _cmd_handoff(db, Namespace(
            session_id="missing", to="slack:C1", message_file=None,
            require_thread=False, wait=False, json=True,
        ))
    finally:
        db.close()

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "not found" in payload["error"]


def test_json_pending_timeout_is_one_object(tmp_path, monkeypatch, capsys):
    db = _db(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.sessions_cmd.time",
        SimpleNamespace(
            monotonic=MagicMock(side_effect=[0.0, 61.0]),
            sleep=lambda _seconds: None,
        ),
    )
    try:
        result = _cmd_handoff(db, Namespace(
            session_id="handoff source", to="slack:C1", message_file=None,
            require_thread=False, wait=True, json=True,
        ))
    finally:
        db.close()

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "failed"
    assert "timed out" in payload["error"]


def test_json_running_timeout_is_one_object(tmp_path, monkeypatch, capsys):
    db = _db(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.sessions_cmd.time",
        SimpleNamespace(
            monotonic=MagicMock(side_effect=[0.0, 0.0, 901.0]),
            sleep=lambda _seconds: None,
        ),
    )
    original_state = db.get_handoff_state

    def running_state(session_id):
        row = original_state(session_id)
        return {**row, "state": "running"} if row else row

    monkeypatch.setattr(db, "get_handoff_state", running_state)
    try:
        result = _cmd_handoff(db, Namespace(
            session_id="handoff source", to="slack:C1", message_file=None,
            require_thread=False, wait=True, json=True,
        ))
    finally:
        db.close()

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "running"
    assert "still running" in payload["error"]


def test_command_refuses_active_source_turn(tmp_path, monkeypatch, capsys):
    db = _db(tmp_path, monkeypatch)
    session_id = "20260913_120000_abcdef"
    assert db.try_acquire_session_turn_lease(session_id, "source-turn") is True
    try:
        result = _cmd_handoff(db, Namespace(
            session_id=session_id, to="slack:C1", message_file=None,
            require_thread=False, wait=False, json=True,
        ))
    finally:
        db.release_session_turn_lease(session_id, "source-turn")
        db.close()

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "active turn" in payload["error"]