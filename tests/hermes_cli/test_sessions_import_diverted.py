"""Replay diverted JSONL via ``hermes sessions import --from diverted``."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_state import SessionDB


def _write_diverted(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_import_diverted_appends_jsonl_into_session(capsys):
    home = get_hermes_home()
    session_id = "sess-diverted"
    db = SessionDB()
    try:
        db.create_session(session_id, "cli")
        db.append_message(session_id, "user", "already-in-store")
    finally:
        db.close()

    jsonl = _write_diverted(
        home / "sessions" / f"{session_id}.jsonl",
        [
            {"role": "user", "content": "hello-from-divert"},
            {"role": "assistant", "content": "reply-from-divert"},
        ],
    )

    from hermes_cli.foreign_sessions import run_sessions_import

    inspect_args = Namespace(
        from_source="diverted",
        path=None,
        session_id=session_id,
        inspect_only=True,
    )
    assert run_sessions_import(inspect_args) is not None
    inspect_out = capsys.readouterr().out
    assert str(jsonl) in inspect_out
    assert "2" in inspect_out

    db = SessionDB()
    try:
        contents = [m.get("content") for m in db.get_messages(session_id)]
        assert "hello-from-divert" not in contents
        assert "reply-from-divert" not in contents
    finally:
        db.close()

    apply_args = Namespace(
        from_source="diverted",
        path=None,
        session_id=session_id,
        inspect_only=False,
    )
    assert run_sessions_import(apply_args) == session_id

    db = SessionDB()
    try:
        contents = [m.get("content") for m in db.get_messages(session_id)]
        assert "already-in-store" in contents
        assert "hello-from-divert" in contents
        assert "reply-from-divert" in contents
        before = len(contents)
    finally:
        db.close()

    assert run_sessions_import(apply_args) == session_id
    db = SessionDB()
    try:
        assert len(db.get_messages(session_id)) == before
    finally:
        db.close()
