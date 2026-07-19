"""Atomic SessionDB mutation regressions for plugin-owned session metadata."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB


def _config(row):
    return json.loads(row["model_config"] or "{}")


def test_mutate_session_atomically_creates_reopens_and_ends(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", "plugin-test")

    created = db.mutate_session(
        "plugin-session",
        lambda current: {
            "model_config": {
                "schema_version": 0,
                "plugin": {"status": "active", "sequence": []},
            },
            "ended_at": None,
            "end_reason": None,
        },
        create={"source": "plugin-test", "parent_session_id": "parent"},
    )
    assert created["source"] == "plugin-test"
    assert created["ended_at"] is None
    assert _config(created)["schema_version"] == 0

    ended = db.mutate_session(
        "plugin-session",
        lambda current: {
            "model_config": {
                **_config(current),
                "plugin": {"status": "completed", "sequence": []},
            },
            "ended_at": 123.0,
            "end_reason": "plugin_completed",
        },
    )
    assert ended["ended_at"] == 123.0
    assert ended["end_reason"] == "plugin_completed"

    reopened = db.mutate_session(
        "plugin-session",
        lambda current: {
            "model_config": {
                **_config(current),
                "plugin": {"status": "active", "sequence": []},
            },
            "ended_at": None,
            "end_reason": None,
        },
    )
    assert reopened["ended_at"] is None
    assert reopened["end_reason"] is None
    assert _config(reopened)["plugin"]["sequence"] == []
    db.close()


def test_mutate_session_concurrent_read_modify_write_loses_no_siblings(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "shared-session",
        "plugin-test",
        model_config={"schema_version": 0, "siblings": {}},
    )
    count = 50

    def add(index: int):
        def mutate(current):
            config = _config(current)
            siblings = dict(config.get("siblings") or {})
            siblings[str(index)] = {"status": "completed"}
            config["siblings"] = siblings
            return {"model_config": config}

        db.mutate_session("shared-session", mutate)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(add, range(count)))

    final = db.get_session("shared-session")
    assert final is not None
    config = _config(final)
    assert config["schema_version"] == 0
    assert set(config["siblings"]) == {str(i) for i in range(count)}
    db.close()


def test_mutate_session_rolls_back_callback_exception_and_requires_create_defaults(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("existing", "plugin-test", model_config={"keep": True})

    def fail(_current):
        raise RuntimeError("mutator failed")

    with pytest.raises(RuntimeError, match="mutator failed"):
        db.mutate_session("existing", fail)
    assert _config(db.get_session("existing")) == {"keep": True}

    with pytest.raises(KeyError, match="missing"):
        db.mutate_session("missing", lambda _current: {"model_config": {}})
    assert db.get_session("missing") is None
    db.close()


def test_mutate_session_rejects_unknown_columns(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("existing", "plugin-test")

    with pytest.raises(ValueError, match="Unsupported mutable session field"):
        db.mutate_session("existing", lambda _current: {"message_count": 99})

    row = db.get_session("existing")
    assert row is not None
    assert row["message_count"] == 0
    db.close()
