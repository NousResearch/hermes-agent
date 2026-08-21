"""JSON export stitches compression lineages only when lineage=logical.

Covers the salvage of lineage-aware JSON/JSONL export:
- export_session stays a single physical row
- export_session_lineage stitches root->tip from any member
- branch/delegate/tool children are not folded into the chain
- export_all(lineage='logical') folds continuations; default/single keeps fragments
- CLI JSON/JSONL and console export honor --lineage
"""

from __future__ import annotations

import json

from hermes_state import SessionDB


def _build_chain(db: SessionDB) -> None:
    db.create_session("root", source="cli")
    db.append_message("root", "user", "u1")
    db.append_message("root", "assistant", "a1")
    db.end_session("root", "compression")

    db.create_session("mid", source="cli", parent_session_id="root")
    db.append_message("mid", "user", "u2")
    db.append_message("mid", "assistant", "a2")
    db.end_session("mid", "compression")

    db.create_session("tip", source="cli", parent_session_id="mid")
    db.append_message("tip", "user", "u3")
    db.append_message("tip", "assistant", "a3")


def _contents(exported) -> list:
    return [message.get("content") for message in (exported.get("messages") or [])]


def test_export_session_stays_single_row(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _build_chain(db)
        exported = db.export_session("mid")
        assert exported is not None
        assert exported["id"] == "mid"
        assert _contents(exported) == ["u2", "a2"]
        assert "lineage_session_ids" not in exported
        assert "segments" not in exported
    finally:
        db.close()


def test_export_session_lineage_stitches_from_any_member(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _build_chain(db)
        for member in ("root", "mid", "tip"):
            exported = db.export_session_lineage(member)
            assert exported is not None, member
            assert exported["lineage_session_ids"] == ["root", "mid", "tip"], member
            assert _contents(exported) == ["u1", "a1", "u2", "a2", "u3", "a3"], member
    finally:
        db.close()


def test_export_session_lineage_excludes_branch_delegate_tool(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _build_chain(db)
        db.create_session(
            "branch",
            source="cli",
            parent_session_id="root",
            model_config={"_branched_from": "root"},
        )
        db.append_message("branch", "user", "b1")
        db.create_session(
            "delegate",
            source="delegate",
            parent_session_id="mid",
            model_config={"_delegate_from": "mid"},
        )
        db.append_message("delegate", "user", "d1")
        db.create_session("tool", source="tool", parent_session_id="mid")
        db.append_message("tool", "user", "t1")

        for member in ("root", "mid", "tip"):
            exported = db.export_session_lineage(member)
            assert exported["lineage_session_ids"] == ["root", "mid", "tip"]
            assert _contents(exported) == ["u1", "a1", "u2", "a2", "u3", "a3"]

        for child, content in (
            ("branch", "b1"),
            ("delegate", "d1"),
            ("tool", "t1"),
        ):
            exported = db.export_session_lineage(child)
            assert exported["lineage_session_ids"] == [child]
            assert _contents(exported) == [content]
    finally:
        db.close()


def test_export_all_logical_folds_continuations(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _build_chain(db)
        db.create_session("solo", source="cli")
        db.append_message("solo", "user", "s1")

        rows = db.export_all(lineage="logical")
        conversation = [
            row
            for row in rows
            if row.get("lineage_session_ids") == ["root", "mid", "tip"]
        ]
        assert len(conversation) == 1
        assert _contents(conversation[0]) == ["u1", "a1", "u2", "a2", "u3", "a3"]
        ids = {row["id"] for row in rows}
        assert "solo" in ids
        assert "mid" not in ids
        assert "root" not in ids
    finally:
        db.close()


def test_export_all_single_keeps_fragments(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _build_chain(db)
        db.create_session("solo", source="cli")
        db.append_message("solo", "user", "s1")

        for rows in (db.export_all(), db.export_all(lineage="single")):
            ids = sorted(row["id"] for row in rows)
            assert ids == ["mid", "root", "solo", "tip"]
            mid = next(row for row in rows if row["id"] == "mid")
            assert _contents(mid) == ["u2", "a2"]
            assert "lineage_session_ids" not in mid
    finally:
        db.close()


def test_export_all_logical_keeps_branch_delegate_tool(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="cli")
        db.append_message("parent", "user", "p1")
        db.end_session("parent", "compression")
        db.create_session("cont", source="cli", parent_session_id="parent")
        db.append_message("cont", "user", "c1")
        db.create_session(
            "branch",
            source="cli",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )
        db.append_message("branch", "user", "b1")
        db.create_session(
            "delegate",
            source="delegate",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        db.append_message("delegate", "user", "d1")
        db.create_session("tool", source="tool", parent_session_id="parent")
        db.append_message("tool", "user", "t1")

        rows = db.export_all(lineage="logical")
        ids = {row["id"] for row in rows}
        assert "branch" in ids
        assert "delegate" in ids
        assert "tool" in ids
        conversation = next(
            row
            for row in rows
            if row.get("lineage_session_ids") == ["parent", "cont"]
        )
        assert _contents(conversation) == ["p1", "c1"]
        assert "parent" not in ids
    finally:
        db.close()


def _jsonl_export_args(**overrides):
    from argparse import Namespace

    defaults = dict(
        sessions_action="export",
        format="jsonl",
        output="-",
        session_id=None,
        lineage="single",
        redact=False,
        only=None,
        dry_run=False,
        yes=False,
        source=None,
        older_than=None,
        newer_than=None,
        before=None,
        after=None,
        title=None,
        end_reason=None,
        cwd=None,
        min_messages=None,
        max_messages=None,
        model=None,
        provider=None,
        user=None,
        chat_id=None,
        chat_type=None,
        branch=None,
        min_tokens=None,
        max_tokens=None,
        min_cost=None,
        max_cost=None,
        min_tool_calls=None,
        max_tool_calls=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_cli_jsonl_honors_lineage(capsys):
    from hermes_cli.sessions_cmd import cmd_sessions

    db = SessionDB()
    try:
        _build_chain(db)
    finally:
        db.close()

    cmd_sessions(_jsonl_export_args(session_id="mid"))
    single = json.loads(capsys.readouterr().out.strip())
    assert single["id"] == "mid"
    assert _contents(single) == ["u2", "a2"]
    assert "lineage_session_ids" not in single

    cmd_sessions(_jsonl_export_args(session_id="mid", lineage="logical"))
    logical = json.loads(capsys.readouterr().out.strip())
    assert logical["lineage_session_ids"] == ["root", "mid", "tip"]
    assert _contents(logical) == ["u1", "a1", "u2", "a2", "u3", "a3"]


def test_cli_jsonl_export_all_folds_when_logical(capsys):
    from hermes_cli.sessions_cmd import cmd_sessions

    db = SessionDB()
    try:
        _build_chain(db)
        db.create_session("solo", source="cli")
        db.append_message("solo", "user", "s1")
    finally:
        db.close()

    cmd_sessions(_jsonl_export_args())
    single_rows = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert sorted(row["id"] for row in single_rows) == ["mid", "root", "solo", "tip"]

    cmd_sessions(_jsonl_export_args(lineage="logical"))
    logical_rows = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    ids = {row["id"] for row in logical_rows}
    assert "solo" in ids
    assert "mid" not in ids
    conversation = next(
        row
        for row in logical_rows
        if row.get("lineage_session_ids") == ["root", "mid", "tip"]
    )
    assert _contents(conversation) == ["u1", "a1", "u2", "a2", "u3", "a3"]


def test_console_json_honors_lineage():
    from hermes_cli.console_engine import HermesConsoleEngine

    db = SessionDB()
    try:
        _build_chain(db)
    finally:
        db.close()

    engine = HermesConsoleEngine()
    single = engine.execute(
        "sessions export - --session-id mid",
        confirmed=True,
    )
    assert single.status == "ok"
    single_row = json.loads(single.output.strip())
    assert single_row["id"] == "mid"
    assert _contents(single_row) == ["u2", "a2"]

    logical = engine.execute(
        "sessions export - --session-id mid --lineage logical",
        confirmed=True,
    )
    assert logical.status == "ok"
    logical_row = json.loads(logical.output.strip())
    assert logical_row["lineage_session_ids"] == ["root", "mid", "tip"]
    assert _contents(logical_row) == ["u1", "a1", "u2", "a2", "u3", "a3"]
