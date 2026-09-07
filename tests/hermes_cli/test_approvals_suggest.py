"""Tests for ``hermes approvals suggest`` (hermes_cli/approvals_suggest.py).

Approval history in Hermes is implied, not ledgered: the session DB
(state.db) stores every assistant ``terminal`` tool call plus its paired
``role='tool'`` result. A dangerous-classified command whose result is not a
BLOCKED/denied/pending marker ran with user consent. These tests build
synthetic state.db fixtures and verify scanning, ranking, safety exclusions,
and the --apply merge path.
"""

from __future__ import annotations

import json
import sqlite3
import time
from argparse import Namespace

import pytest

import tools.approval as approval_module
from hermes_cli.approvals_suggest import (
    Proposal,
    apply_proposals,
    build_proposals,
    derive_glob,
    is_unsafe_class,
    normalize_command,
    parse_apply_indices,
    scan_approval_history,
    suggest_command,
)


# ---------------------------------------------------------------------------
# Fixture helpers: synthetic session DB
# ---------------------------------------------------------------------------

def _make_db(path):
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            timestamp REAL NOT NULL
        );
        """
    )
    con.commit()
    return con


_ID_COUNTER = [0]


def _add_terminal_call(con, command, result="ok: done", ts=None):
    """Insert an assistant terminal tool call + its paired tool result."""
    _ID_COUNTER[0] += 1
    call_id = f"call_{_ID_COUNTER[0]}"
    ts = ts if ts is not None else time.time()
    tool_calls = json.dumps(
        [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ]
    )
    con.execute(
        "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) "
        "VALUES ('s1', 'assistant', '', ?, ?)",
        (tool_calls, ts),
    )
    con.execute(
        "INSERT INTO messages (session_id, role, content, tool_call_id, timestamp) "
        "VALUES ('s1', 'tool', ?, ?, ?)",
        (result, call_id, ts + 1),
    )
    con.commit()


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "state.db"
    con = _make_db(path)
    yield path, con
    con.close()


@pytest.fixture
def isolated_allowlist(monkeypatch):
    """Fake config-backed allowlist store so no real config.yaml is touched."""
    store = {"patterns": set(), "saves": 0}

    def fake_load():
        return set(store["patterns"])

    def fake_save(patterns):
        store["patterns"] = set(patterns)
        store["saves"] += 1

    monkeypatch.setattr(approval_module, "load_permanent_allowlist", fake_load)
    monkeypatch.setattr(approval_module, "save_permanent_allowlist", fake_save)
    # apply_proposals also syncs into the in-process set; keep it isolated.
    saved = approval_module._permanent_approved.copy()
    approval_module._permanent_approved.clear()
    yield store
    approval_module._permanent_approved.clear()
    approval_module._permanent_approved.update(saved)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

class TestScan:
    def test_scan_finds_approved_dangerous_commands(self, db_path):
        path, con = db_path
        for _ in range(3):
            _add_terminal_call(con, "git push --force origin main")
        # A benign command never enters approval history.
        _add_terminal_call(con, "ls -la")
        records = scan_approval_history(path, days=0)
        commands = [c for c, _ in records]
        assert len(commands) == 3
        assert all("git push" in c for c in commands)


    def test_days_window_filters_old_history(self, db_path):
        path, con = db_path
        old_ts = time.time() - 200 * 86400
        _add_terminal_call(con, "git push --force origin main", ts=old_ts)
        _add_terminal_call(con, "git push --force origin main")
        assert len(scan_approval_history(path, days=90)) == 1
        assert len(scan_approval_history(path, days=0)) == 2



# ---------------------------------------------------------------------------
# Normalize / glob derivation
# ---------------------------------------------------------------------------

class TestNormalizeAndGlob:
    def test_normalize_folds_home_prefix(self):
        import os

        home = os.path.expanduser("~")
        normalized = normalize_command(f"git checkout -- {home}/project/file.txt")
        assert "~/project/file.txt" in normalized
        assert home not in normalized

    def test_derive_glob_uses_first_two_tokens(self):
        assert derive_glob("git push --force origin main") == "git push *"
        assert derive_glob("docker restart web") == "docker restart *"


    def test_derive_glob_never_anchors_unsafe_binaries(self):
        assert derive_glob("rm -rf ./build") is None
        assert derive_glob("sudo apt install foo") is None
        assert derive_glob("/bin/rm -rf ./build") is None
        assert derive_glob("mkfs.ext4 /dev/sdb1") is None


# ---------------------------------------------------------------------------
# Ranking + safety exclusion
# ---------------------------------------------------------------------------

class TestRankingAndSafety:
    def test_ranking_orders_by_frequency(self, db_path):
        path, con = db_path
        for _ in range(5):
            _add_terminal_call(con, "docker restart web")
        for _ in range(2):
            _add_terminal_call(con, "git push --force origin main")
        records = scan_approval_history(path, days=0)
        proposals = build_proposals(records, min_count=2)
        assert [p.pattern for p in proposals] == ["docker restart *", "git push *"]
        assert proposals[0].count == 5
        assert proposals[1].count == 2

    def test_min_count_threshold(self, db_path):
        path, con = db_path
        _add_terminal_call(con, "git push --force origin main")
        records = scan_approval_history(path, days=0)
        assert build_proposals(records, min_count=2) == []
        assert len(build_proposals(records, min_count=1)) == 1






# ---------------------------------------------------------------------------
# --apply / dry-run
# ---------------------------------------------------------------------------

def _args(db, **kw):
    base = dict(
        db=str(db), days=0, min_count=1, limit=20,
        apply_indices=None, json=False,
    )
    base.update(kw)
    return Namespace(**base)


class TestApply:

    def test_apply_merges_and_persists(self, isolated_allowlist):
        isolated_allowlist["patterns"] = {"podman *"}
        proposals = [
            Proposal(pattern="git push *", kind="glob", count=5),
            Proposal(pattern="docker restart *", kind="glob", count=3),
        ]
        merged = apply_proposals(proposals, [0])
        assert merged == {"podman *", "git push *"}
        assert isolated_allowlist["patterns"] == {"podman *", "git push *"}
        assert isolated_allowlist["saves"] == 1
        # In-process allowlist synced too (long-lived process consistency).
        assert "git push *" in approval_module._permanent_approved

    def test_suggest_apply_end_to_end(self, db_path, isolated_allowlist, capsys):
        path, con = db_path
        for _ in range(4):
            _add_terminal_call(con, "git push --force origin main")
        for _ in range(3):
            _add_terminal_call(con, "docker restart web")
        rc = suggest_command(_args(path, apply_indices="1,2"))
        assert rc == 0
        assert isolated_allowlist["patterns"] == {"git push *", "docker restart *"}
        out = capsys.readouterr().out
        assert "git push *" in out and "docker restart *" in out



class TestJsonOutput:
    def test_json_proposal_output(self, db_path, isolated_allowlist, capsys):
        path, con = db_path
        for _ in range(4):
            _add_terminal_call(con, "git push --force origin main")
        rc = suggest_command(_args(path, json=True))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["proposals"][0]["pattern"] == "git push *"
        assert payload["proposals"][0]["count"] == 4
        assert payload["proposals"][0]["n"] == 1
        assert isolated_allowlist["saves"] == 0


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------

class TestParserWiring:
    def test_build_approvals_parser_wires_suggest(self):
        import argparse

        from hermes_cli.subcommands.approvals import build_approvals_parser

        sentinel = object()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_approvals_parser(subparsers, cmd_approvals=lambda a: sentinel)

        args = parser.parse_args(
            ["approvals", "suggest", "--apply", "1,3", "--json",
             "--days", "30", "--min-count", "5", "--limit", "10"]
        )
        assert args.approvals_command == "suggest"
        assert args.apply_indices == "1,3"
        assert args.json is True
        assert args.days == 30
        assert args.min_count == 5
        assert args.limit == 10
        assert args.func(args) is sentinel

    def test_suggest_defaults_are_dry_and_bounded(self):
        import argparse

        from hermes_cli.subcommands.approvals import build_approvals_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_approvals_parser(subparsers, cmd_approvals=lambda a: None)
        args = parser.parse_args(["approvals", "suggest"])
        assert args.apply_indices is None
        assert args.json is False
        assert args.days == 90
        assert args.min_count == 2
