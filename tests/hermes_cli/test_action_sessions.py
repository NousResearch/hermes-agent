"""Read-only cross-profile Action Session aggregation."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import hermes_cli.sessions_cmd as sessions_cmd
from hermes_cli.action_sessions import (
    _continuation_lineage,
    _last_live_line,
    collect_active_actions,
    render_action_cards,
)
from hermes_cli.subcommands.sessions import build_sessions_parser


def _state_db(
    home: Path,
    rows: list[dict],
    leases: Sequence[tuple] = (),
    delegations: Sequence[dict] = (),
    messages: Sequence[tuple] = (),
) -> None:
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "state.db")
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT, model_config TEXT,
            parent_session_id TEXT, started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
            title TEXT, session_key TEXT, chat_id TEXT, chat_type TEXT, thread_id TEXT,
            origin_json TEXT, cwd TEXT, git_branch TEXT, git_repo_root TEXT,
            profile_name TEXT, last_activity_at REAL, last_activity_description TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, timestamp REAL NOT NULL
        );
        CREATE TABLE session_turn_leases (
            conversation_id TEXT PRIMARY KEY, holder TEXT NOT NULL,
            acquired_at REAL NOT NULL, expires_at REAL NOT NULL
        );
        CREATE TABLE async_delegations (
            delegation_id TEXT PRIMARY KEY, parent_session_id TEXT, state TEXT NOT NULL,
            dispatched_at REAL NOT NULL, updated_at REAL NOT NULL, completed_at REAL,
            task_json TEXT, result_json TEXT
        );
        """
    )
    columns = [row[1] for row in conn.execute("PRAGMA table_info(sessions)")]
    for row in rows:
        values = [row.get(column) for column in columns]
        conn.execute(
            f"INSERT INTO sessions ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            values,
        )
    conn.executemany("INSERT INTO messages (session_id, timestamp) VALUES (?,?)", messages)
    conn.executemany("INSERT INTO session_turn_leases VALUES (?,?,?,?)", leases)
    for row in delegations:
        columns = list(row)
        conn.execute(
            f"INSERT INTO async_delegations ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            [row[column] for column in columns],
        )
    conn.commit()
    conn.close()


def _kanban_db(path: Path, tasks: list[dict], runs: Sequence[dict] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, assignee TEXT, status TEXT NOT NULL,
            created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER,
            workspace_kind TEXT, workspace_path TEXT, branch_name TEXT, session_id TEXT,
            model_override TEXT, provider_override TEXT, current_step_key TEXT,
            block_kind TEXT, last_failure_error TEXT, current_run_id INTEGER
        );
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY, task_id TEXT NOT NULL, profile TEXT, status TEXT NOT NULL,
            started_at INTEGER NOT NULL, ended_at INTEGER, last_heartbeat_at INTEGER,
            summary TEXT, metadata TEXT, error TEXT
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY, task_id TEXT NOT NULL, run_id INTEGER,
            kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL
        );
        """
    )
    for table, rows in (("tasks", tasks), ("task_runs", runs)):
        for row in rows:
            columns = list(row)
            conn.execute(
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                [row[column] for column in columns],
            )
    conn.commit()
    conn.close()


def test_collects_only_registered_active_roots_and_their_children(tmp_path):
    default = tmp_path / "default"
    carver = tmp_path / "profiles" / "carver"
    _state_db(
        default,
        [
            {"id": "slack-root", "source": "slack", "model": "gpt-5", "started_at": 900,
             "last_activity_at": 995, "last_activity_description": "tool running: terminal",
             "session_key": "agent:main:slack:group:T:C:123", "chat_id": "C", "thread_id": "123"},
            {"id": "child", "source": "subagent", "model": "gpt-5", "parent_session_id": "slack-root",
             "started_at": 980, "last_activity_at": 998, "last_activity_description": "reading source"},
            {"id": "cron-noise", "source": "cron", "model": "gpt-5", "started_at": 990,
             "last_activity_at": 999},
            {"id": "oneshot-noise", "source": "oneshot", "model": "gpt-5", "started_at": 990,
             "last_activity_at": 999},
            {"id": "unregistered", "source": "slack", "model": "gpt-5", "started_at": 990,
             "last_activity_at": 999},
        ],
        leases=[("slack-root", "pid=7:platform=slack", 990, 1100)],
    )
    _state_db(
        carver,
        [{"id": "desktop-root", "source": "desktop", "model": "sonnet", "profile_name": "carver",
          "started_at": 800, "last_activity_at": 850, "cwd": "/repo", "git_branch": "feat/x"}],
    )
    kanban = tmp_path / "kanban.db"
    _kanban_db(
        kanban,
        [
            {"id": "T1", "title": "Slack action", "assignee": "default", "status": "running",
             "created_at": 800, "started_at": 900, "workspace_kind": "worktree",
             "workspace_path": "/work", "branch_name": "feat/slack", "session_id": "slack-root",
             "current_run_id": 1},
            {"id": "T2", "title": "Desktop action", "assignee": "carver", "status": "blocked",
             "created_at": 700, "started_at": 800, "workspace_kind": "dir",
             "workspace_path": "/repo", "branch_name": "feat/x", "session_id": "desktop-root",
             "block_kind": "needs_input"},
            {"id": "T3", "title": "Finished", "assignee": "default", "status": "done",
             "created_at": 700, "started_at": 800, "completed_at": 900, "session_id": "unregistered"},
            {"id": "T4", "title": "Cron noise", "status": "running", "created_at": 990,
             "started_at": 990, "session_id": "cron-noise"},
            {"id": "T5", "title": "Oneshot noise", "status": "running", "created_at": 990,
             "started_at": 990, "session_id": "oneshot-noise"},
        ],
        runs=[{"id": 1, "task_id": "T1", "profile": "default", "status": "running",
               "started_at": 910, "last_heartbeat_at": 999, "summary": "executing"}],
    )

    cards = collect_active_actions(
        profile_homes={"default": default, "carver": carver},
        kanban_paths={"default": kanban},
        now=1000,
        stale_after_seconds=100,
    )

    assert [card["action_id"] for card in cards] == ["T1", "T2"]
    assert [child["session_id"] for child in cards[0]["children"]] == ["child"]
    assert cards[0]["lease_state"] == "leased"
    assert cards[0]["owner_kind"] == "gateway"
    assert cards[0]["owner_profile"] == "default"
    assert cards[0]["task_status"] == "running"
    assert cards[0]["run_status"] == "running"
    assert cards[0]["run_last_heartbeat_at"] == 999
    assert cards[0]["run_summary"] == "executing"
    assert cards[0]["owner_route"] == {
        "platform": "slack", "session_key": "agent:main:slack:group:T:C:123",
        "chat_id": "C", "thread_id": "123",
    }
    assert cards[1]["lease_state"] == "stale"
    assert cards[1]["owner_kind"] == "desktop"
    assert cards[1]["blocker"] == "needs_input"


def test_duplicate_session_ids_resolve_by_profile_evidence(tmp_path):
    default = tmp_path / "default"
    carver = tmp_path / "profiles" / "carver"
    _state_db(default, [{"id": "same", "source": "cli", "started_at": 900, "last_activity_at": 990}])
    _state_db(carver, [{"id": "same", "source": "acp", "profile_name": "carver",
                        "started_at": 900, "last_activity_at": 990}])
    kanban = tmp_path / "kanban.db"
    _kanban_db(kanban, [{"id": "T1", "title": "Action", "assignee": "carver", "status": "running",
                         "created_at": 900, "started_at": 900, "session_id": "same"}])

    card = collect_active_actions(
        profile_homes={"default": default, "carver": carver},
        kanban_paths={"default": kanban},
        now=1000,
    )[0]

    assert card["owner_profile"] == "carver"


def test_active_lease_wins_over_ended_or_stale_markers(tmp_path):
    home = tmp_path / "default"
    _state_db(
        home,
        [
            {"id": "root", "source": "slack", "started_at": 100, "ended_at": 200,
             "end_reason": "compression", "last_activity_at": 200},
            {"id": "continued", "source": "slack", "parent_session_id": "root",
             "started_at": 200, "ended_at": 300, "last_activity_at": 200,
             "last_activity_description": "old activity"},
        ],
        leases=[("root", "holder", 950, 1050)],
    )
    kanban = tmp_path / "kanban.db"
    _kanban_db(kanban, [{"id": "T1", "title": "Action", "status": "review", "created_at": 100,
                         "started_at": 100, "session_id": "continued"}])

    card = collect_active_actions(
        profile_homes={"default": home}, kanban_paths={"default": kanban}, now=1000,
        stale_after_seconds=60,
    )[0]

    assert card["lease_state"] == "leased"
    assert card["session_ended"] is True
    assert card["phase"] == "awaiting_review"


def test_compression_lineage_matches_canonical_cycle_bound_and_falsey_rules():
    root = {"id": "root", "end_reason": "compression"}
    zero = {"id": "zero", "started_at": 100, "_lineage_activity_at": 0}
    one = {"id": "one", "started_at": 0, "_lineage_activity_at": 1}
    assert [row["id"] for row in _continuation_lineage(
        "root", {"root": root, "zero": zero, "one": one}, {"root": [zero, one]},
    )] == ["root", "one"]

    loop = {"id": "loop", "end_reason": "compression", "_lineage_activity_at": 2}
    fallback = {"id": "fallback", "_lineage_activity_at": 1}
    assert [row["id"] for row in _continuation_lineage(
        "root", {"root": root, "loop": loop, "fallback": fallback},
        {"root": [loop], "loop": [root, fallback]},
    )] == ["root", "loop"]

    chain = {"root": root}
    links: dict[str, list[dict]] = {}
    parent = "root"
    for index in range(1, 102):
        session_id = f"s{index}"
        row = {"id": session_id, "end_reason": "compression", "_lineage_activity_at": index}
        chain[session_id] = row
        links[parent] = [row]
        parent = session_id
    bounded = _continuation_lineage("root", chain, links)
    assert len(bounded) == 101
    assert bounded[-1]["id"] == "s100"

    falsey = {"id": "", "_lineage_activity_at": 5}
    assert [row["id"] for row in _continuation_lineage("root", {"root": root}, {"root": [falsey]})] == ["root"]


def test_delegation_from_compression_continuation_is_attached(tmp_path):
    home = tmp_path / "default"
    _state_db(
        home,
        [
            {"id": "root", "source": "slack", "started_at": 100, "ended_at": 200,
             "end_reason": "compression", "last_activity_at": 200},
            {"id": "continued", "source": "slack", "parent_session_id": "root",
             "started_at": 190, "last_activity_at": 990,
             "last_activity_description": "last-activity candidate"},
            {"id": "message-winner", "source": "slack", "parent_session_id": "root",
             "started_at": 180, "last_activity_at": 980,
             "last_activity_description": "message-fresh continuation"},
            {"id": "future-start", "source": "slack", "parent_session_id": "root",
             "started_at": 1000, "last_activity_at": 981,
             "last_activity_description": "started_at must not override activity"},
            {"id": "marked-fork", "source": "slack", "parent_session_id": "root",
             "model_config": json.dumps({"_branched_from": "legacy-id"}),
             "started_at": 220, "last_activity_at": 1000,
             "last_activity_description": "fork must not win"},
            {"id": "stale", "source": "slack", "parent_session_id": "root",
             "started_at": 210, "ended_at": 220, "end_reason": "ws_orphan_reap",
             "last_activity_at": 995, "last_activity_description": "stale sibling"},
            {"id": "stale-child", "source": "subagent", "parent_session_id": "stale",
             "started_at": 215, "last_activity_at": 996},
        ],
        delegations=[
            {"delegation_id": "deleg_continued", "parent_session_id": "continued", "state": "running",
             "dispatched_at": 950, "updated_at": 999, "task_json": json.dumps({"model": "sonnet"})},
            {"delegation_id": "deleg_winner", "parent_session_id": "message-winner", "state": "running",
             "dispatched_at": 940, "updated_at": 997, "task_json": json.dumps({"model": "sonnet"})},
            {"delegation_id": "deleg_fork", "parent_session_id": "marked-fork", "state": "running",
             "dispatched_at": 970, "updated_at": 1000, "task_json": json.dumps({"model": "wrong"})},
            {"delegation_id": "deleg_stale", "parent_session_id": "stale", "state": "completed",
             "dispatched_at": 960, "updated_at": 998, "task_json": json.dumps({"model": "old"})},
        ],
        messages=[("message-winner", 999)],
    )
    kanban = tmp_path / "kanban.db"
    _kanban_db(kanban, [{"id": "T1", "title": "Action", "status": "running", "created_at": 100,
                         "started_at": 100, "session_id": "root"}])

    card = collect_active_actions(
        profile_homes={"default": home}, kanban_paths={"default": kanban}, now=1000,
    )[0]

    delegation_ids = {child.get("delegation_id") for child in card["children"] if child["kind"] == "delegation"}
    session_ids = {child.get("session_id") for child in card["children"] if child["kind"] == "session"}
    assert delegation_ids == {"deleg_winner"}
    assert "stale-child" not in session_ids
    assert card["session_ended"] is False
    assert card["current_activity"] == "message-fresh continuation"


def test_delegation_state_and_live_log_activity_are_attached(tmp_path):
    home = tmp_path / "default"
    log_path = home / "cache" / "delegation" / "live" / "deleg_1" / "task-0.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("header\n12:00:00 tool     | -> web_search(query)\n", encoding="utf-8")
    _state_db(
        home,
        [
            {"id": "root", "source": "tui", "started_at": 900, "last_activity_at": 990},
            {"id": "child", "source": "subagent", "parent_session_id": "root",
             "started_at": 950, "last_activity_at": 995},
        ],
        delegations=[{
            "delegation_id": "deleg_1", "parent_session_id": "root", "state": "running",
            "dispatched_at": 950, "updated_at": 999,
            "task_json": json.dumps({"task_transcripts": {"0": str(log_path)}, "model": "sonnet"}),
        }],
    )
    kanban = tmp_path / "kanban.db"
    _kanban_db(kanban, [{"id": "T1", "title": "Action", "status": "running", "created_at": 900,
                         "started_at": 900, "session_id": "root"}])

    card = collect_active_actions(
        profile_homes={"default": home}, kanban_paths={"default": kanban}, now=1000,
    )[0]

    delegation = next(child for child in card["children"] if child["kind"] == "delegation")
    assert delegation["delegation_id"] == "deleg_1"
    assert delegation["state"] == "running"
    assert delegation["current_activity"] == "tool | -> web_search(query)"
    assert card["owner_kind"] == "terminal"


def test_delegation_log_path_cannot_escape_profile_cache(tmp_path):
    home = tmp_path / "default"
    secret_dir = tmp_path / "outside"
    secret_dir.mkdir()
    secret = secret_dir / "task-secret.log"
    secret.write_text("12:00:00 secret   | DO_NOT_EXPOSE\n", encoding="utf-8")
    _state_db(
        home,
        [{"id": "root", "source": "tui", "started_at": 900, "last_activity_at": 990}],
        delegations=[{
            "delegation_id": str(secret_dir), "parent_session_id": "root", "state": "running",
            "dispatched_at": 950, "updated_at": 999,
            "task_json": json.dumps({}),
        }],
    )
    kanban = tmp_path / "kanban.db"
    _kanban_db(kanban, [{"id": "T1", "title": "Action", "status": "running", "created_at": 900,
                         "started_at": 900, "session_id": "root"}])

    card = collect_active_actions(profile_homes={"default": home}, kanban_paths={"default": kanban}, now=1000)[0]

    assert "DO_NOT_EXPOSE" not in json.dumps(card)


def test_live_log_reader_rejects_dotdot_and_symlink_escape(tmp_path):
    live_root = tmp_path / "live"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "task-0.log").write_text("12:00:00 secret | DO_NOT_EXPOSE\n", encoding="utf-8")
    live_root.mkdir()
    (live_root / "escape").symlink_to(outside, target_is_directory=True)

    assert _last_live_line("..", live_root) == ""
    assert _last_live_line("escape", live_root) == ""


def test_renderer_has_required_operator_fields(tmp_path):
    home = tmp_path / "default"
    _state_db(home, [{"id": "root", "source": "cli", "model": "m", "started_at": 900,
                      "last_activity_at": 995, "last_activity_description": "testing"}])
    kanban = tmp_path / "kanban.db"
    _kanban_db(kanban, [{"id": "T1", "title": "Action", "assignee": "carmen", "status": "running",
                         "created_at": 900, "started_at": 900, "workspace_kind": "worktree",
                         "workspace_path": "/work", "branch_name": "feat/x", "session_id": "root"}])
    cards = collect_active_actions(profile_homes={"default": home}, kanban_paths={"default": kanban}, now=1000)

    rendered = render_action_cards(cards, now=1000)

    for expected in (
        "T1", "execution", "carmen", "m", "testing", "/work", "feat/x", "1m 40s",
        "blocker: —", "session: live", "terminal:default", '"session_id":"root"',
    ):
        assert expected in rendered

    cards[0]["title"] = "safe\nFORGED\x1b[2J\x9bCSI\x9dOSC\x9cST\u202eRTL\u2066ISO"
    cards[0]["current_activity"] = "line\r\nINJECT"
    hostile = render_action_cards(cards)
    for unsafe in ("\x1b", "\x9b", "\x9d", "\x9c", "\u202e", "\u2066"):
        assert unsafe not in hostile
    assert "safe FORGED [2J" in hostile
    assert "line INJECT" in hostile


def test_sessions_parser_registers_read_only_active_view():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_sessions_parser(subparsers, cmd_sessions=lambda _args, **_kwargs: None)

    args = parser.parse_args(["sessions", "active", "--json", "--stale-after", "120"])

    assert args.sessions_action == "active"
    assert args.json is True
    assert args.stale_after == 120


def test_active_command_emits_json_without_opening_single_profile_sessiondb(monkeypatch, capsys):
    expected = [{"action_id": "T1", "children": []}]
    monkeypatch.setattr("hermes_cli.action_sessions.collect_active_actions", lambda **_kwargs: expected)

    result = sessions_cmd.cmd_sessions(
        argparse.Namespace(sessions_action="active", json=True, stale_after=120)
    )

    assert result is None
    assert json.loads(capsys.readouterr().out) == expected
