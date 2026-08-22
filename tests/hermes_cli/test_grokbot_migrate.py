"""Tests for the Grok Bot migration importer (hermes migrate grokbot)."""

from __future__ import annotations

import json
import time

import pytest

from hermes_cli.grokbot_import import (
    EXPORT_SCHEMA_VERSION,
    ExportValidationError,
    load_export,
    plan_import,
    run_import,
    slugify,
)


def make_export() -> dict:
    """A valid schema-1 export with two bots and three conversations."""
    now = time.time()
    return {
        "schema": EXPORT_SCHEMA_VERSION,
        "exported_at": "2026-08-21T13:00:00-0400",
        "app_version": "0.24.0",
        "account": {},
        "bots": [
            {
                "id": "roster-0",
                "name": "Email Assistant Boi",
                "title": "Inbox triage",
                "description": "Reads and drafts my email.",
                "instructions": "Never send without approval.",
                "model": "",
                "memories": ["Dana signs annual deals only."],
                "tools": ["gmail"],
                "plugins": [],
            },
            {
                "id": "roster-1",
                "name": "Sales & Ops",
                "title": "",
                "description": "",
                "instructions": "",
                "model": "grok-4",
                "memories": [],
                "tools": [],
                "plugins": [],
            },
        ],
        "conversations": [
            {
                "bot_id": "roster-0",
                "thread_id": "roster-0",
                "title": "Chat with Email Assistant Boi",
                "messages": [
                    {"role": "user", "text": "hey", "ts": now - 60},
                    {"role": "assistant", "text": "Hey, what's up?", "ts": now - 30},
                ],
            },
            {
                "bot_id": "roster-0",
                "thread_id": "roster-0-b",
                "title": "Follow-up",
                "messages": [
                    {"role": "user", "text": "draft it", "ts": now - 10},
                    {"role": "assistant", "text": "Drafted.", "ts": now - 5},
                ],
            },
            {
                "bot_id": "roster-1",
                "thread_id": "roster-1",
                "title": "Pipeline review",
                "messages": [
                    {"role": "user", "text": "status?", "ts": now - 20},
                    {"role": "assistant", "text": "On track.", "ts": now - 15},
                ],
            },
        ],
        "files": {},
        "provenance": {"layers": ["witness"], "sandboxes": [], "warnings": []},
    }


def write_export(tmp_path, export: dict) -> str:
    path = tmp_path / "grokbot-export.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Email Assistant Boi!", "email-assistant-boi"),
        ("Sales  &  Ops", "sales-ops"),
        ("--Weird __ NAME__--", "weird-name"),
        ("   ", "grok-bot"),
        ("", "grok-bot"),
    ],
)
def test_slugify(name, expected):
    assert slugify(name) == expected


# ---------------------------------------------------------------------------
# load_export validation
# ---------------------------------------------------------------------------


def test_load_export_accepts_valid(tmp_path):
    export = make_export()
    loaded = load_export(write_export(tmp_path, export))
    assert loaded["schema"] == EXPORT_SCHEMA_VERSION
    assert len(loaded["bots"]) == 2
    assert len(loaded["conversations"]) == 3


def test_load_export_rejects_missing_file(tmp_path):
    with pytest.raises(ExportValidationError):
        load_export(tmp_path / "nope.json")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda e: e.update({"schema": 2}),
        lambda e: e.update({"access_token": "eyJhbGciOiJIUzI1NiJ9"}),
        lambda e: e["bots"].append({"id": "roster-0", "name": "Dup"}),
        lambda e: e["conversations"][0].update({"bot_id": "ghost"}),
        lambda e: e["conversations"][0]["messages"].append(
            {"role": "tool", "text": "x", "ts": 1}
        ),
        lambda e: e["conversations"][0]["messages"].append(
            {"role": "assistant", "text": 42, "ts": 1}
        ),
    ],
)
def test_load_export_rejects_invalid(tmp_path, mutator):
    export = make_export()
    mutator(export)
    with pytest.raises(ExportValidationError):
        load_export(write_export(tmp_path, export))


def test_load_export_rejects_non_dict_root(tmp_path):
    path = tmp_path / "grokbot-export.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ExportValidationError):
        load_export(path)


# ---------------------------------------------------------------------------
# plan_import
# ---------------------------------------------------------------------------


def test_plan_import_maps_names_and_dedupes(tmp_path):
    export = make_export()
    # Two bots that slug to the same base must not collide.
    export["bots"][1]["name"] = "Email Assistant Boi"
    records, by_bot = plan_import(export)
    names = [r.profile_name for r in records]
    assert names == ["email-assistant-boi", "email-assistant-boi-2"]
    assert sorted(by_bot) == ["roster-0", "roster-1"]


def test_plan_import_filters_by_bot(tmp_path):
    export = make_export()
    records, _ = plan_import(export, target_bots=["roster-1"])
    assert [r.name for r in records] == ["Sales & Ops"]


# ---------------------------------------------------------------------------
# run_import (integration against a temp HERMES_HOME)
# ---------------------------------------------------------------------------


def test_import_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    export_path = write_export(tmp_path, make_export())
    rc = run_import(export_path, dry_run=True)
    assert rc == 0
    profile_root = tmp_path / "hermes" / "profiles"
    assert not profile_root.exists() or not any(profile_root.iterdir())


def test_import_creates_profile_soul_memory_and_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    export_path = write_export(tmp_path, make_export())
    rc = run_import(export_path)
    assert rc == 0

    from hermes_cli.profiles import get_profile_dir
    from hermes_state import SessionDB

    profile_dir = get_profile_dir("email-assistant-boi")
    assert profile_dir.is_dir()

    soul = (profile_dir / "SOUL.md").read_text(encoding="utf-8")
    assert "Email Assistant Boi" in soul
    assert "Never send without approval." in soul

    memory = (profile_dir / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    assert "Dana signs annual deals only." in memory

    db = SessionDB(profile_dir / "state.db")
    sessions = db.search_sessions(source="grokbot-import", limit=10)
    assert len(sessions) == 2
    by_id = {s["id"]: s for s in sessions}
    first = by_id["grokbot-import-0"]
    assert first["pinned"] == 1
    assert first["message_count"] == 2
    msgs = db.get_messages("grokbot-import-0")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hey"
    db.close()

    # The second bot landed in its own profile.
    ops_dir = get_profile_dir("sales-ops")
    assert ops_dir.is_dir()
    db2 = SessionDB(ops_dir / "state.db")
    assert len(db2.search_sessions(source="grokbot-import", limit=10)) == 1
    db2.close()


def test_import_conflict_without_force_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from hermes_cli.profiles import create_profile

    create_profile("email-assistant-boi")
    export_path = write_export(tmp_path, make_export())
    rc = run_import(export_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "pass --force" in out


def test_import_failure_rolls_back_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    import hermes_state

    class Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("disk full")

    monkeypatch.setattr(hermes_state, "SessionDB", Boom)
    export_path = write_export(tmp_path, make_export())
    rc = run_import(export_path)
    assert rc == 1

    from hermes_cli.profiles import get_profile_dir

    assert not get_profile_dir("email-assistant-boi").exists()
    assert not get_profile_dir("sales-ops").exists()


def test_import_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    export_path = write_export(tmp_path, make_export())
    assert run_import(export_path) == 0
    # Second run without --force: marker file present → merge, no conflict.
    assert run_import(export_path) == 0

    from hermes_cli.profiles import get_profile_dir
    from hermes_state import SessionDB

    db = SessionDB(get_profile_dir("email-assistant-boi") / "state.db")
    assert len(db.search_sessions(source="grokbot-import", limit=10)) == 2
    db.close()
