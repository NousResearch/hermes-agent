"""Tests for ``hermes kanban rebind-contract`` (operator repoint of completion_contract).

Covers the gap: ``completion_contract`` is pinned at claim time and there was no
operator path to move it (``edit_task`` can't touch it; ``complete_task(force=True)``
only overrides the live-claim fence, never acceptance). ``rebind_task`` must:

* only accept a new contract that is a PR in the SAME repo as the current
  contract, or ``local-only`` (cross-repo, and local-only -> a repo, are refused);
* append an auditable ``contract_rebound`` event carrying old/new/reason/actor;
* leave acceptance itself untouched -- the next ``complete_task`` on the rebound
  contract still runs real GitHub check-run evidence collection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Same-repo enforcement + audit event (pure db-layer, no network)
# ---------------------------------------------------------------------------


def test_rebind_same_repo_pr_to_pr_succeeds_with_event(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="https://github.com/acme/repo/pull/39")
        ok = kb.rebind_contract(
            conn, tid, new_contract="https://github.com/acme/repo/pull/46",
            reason="#39 predates CI", actor="anika",
        )
        assert ok is True
        assert kb.get_task(conn, tid).completion_contract == "https://github.com/acme/repo/pull/46"
        events = [e for e in kb.list_events(conn, tid) if e.kind == "contract_rebound"]
        assert len(events) == 1
        payload = events[-1].payload
        assert payload == {
            "old_contract": "https://github.com/acme/repo/pull/39",
            "new_contract": "https://github.com/acme/repo/pull/46",
            "reason": "#39 predates CI",
            "actor": "anika",
        }


def test_rebind_cross_repo_pr_refused(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="https://github.com/acme/repo/pull/39")
        with pytest.raises(ValueError, match="must match"):
            kb.rebind_contract(
                conn, tid, new_contract="https://github.com/other/repo/pull/1",
                reason="trying to escape", actor="anika",
            )
        # Refused: contract untouched, no audit event.
        assert kb.get_task(conn, tid).completion_contract == "https://github.com/acme/repo/pull/39"
        assert not [e for e in kb.list_events(conn, tid) if e.kind == "contract_rebound"]


def test_rebind_owner_repo_shorthand_enforces_same_repo(kanban_home: Path) -> None:
    """Contract stored as bare ``OWNER/REPO`` (not a PR URL) is a repo too."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="acme/repo")
        with pytest.raises(ValueError, match="must match"):
            kb.rebind_contract(conn, tid, new_contract="https://github.com/other/repo/pull/1",
                                reason="x", actor="a")
        ok = kb.rebind_contract(conn, tid, new_contract="https://github.com/acme/repo/pull/9",
                                 reason="pin to a real PR", actor="a")
        assert ok is True
        assert kb.get_task(conn, tid).completion_contract == "https://github.com/acme/repo/pull/9"


def test_rebind_to_local_only_allowed_but_not_back_out(kanban_home: Path) -> None:
    """A PR contract can be released to local-only, but local-only can't be
    rebound onto a repo -- there is no existing repo commitment to "move"."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="https://github.com/acme/repo/pull/39")
        assert kb.rebind_contract(conn, tid, new_contract="local-only", reason="drop CI gate", actor="a")
        assert kb.get_task(conn, tid).completion_contract == "local-only"
        with pytest.raises(ValueError, match="must match"):
            kb.rebind_contract(conn, tid, new_contract="https://github.com/acme/repo/pull/40",
                                reason="x", actor="a")


def test_rebind_noop_returns_false_no_event(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="https://github.com/acme/repo/pull/39")
        assert kb.rebind_contract(conn, tid, new_contract="https://github.com/acme/repo/pull/39",
                                   reason="x", actor="a") is False
        assert not [e for e in kb.list_events(conn, tid) if e.kind == "contract_rebound"]


def test_rebind_unknown_task_returns_false(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        assert kb.rebind_contract(conn, "t_doesnotexist", new_contract="local-only",
                                   reason="x", actor="a") is False


def test_rebind_invalid_new_contract_rejected(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="https://github.com/acme/repo/pull/39")
        with pytest.raises(ValueError):
            kb.rebind_contract(conn, tid, new_contract="not a contract",
                                reason="x", actor="a")


def test_rebind_contract_denied_for_delegated_child(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delegate_task child session is fenced off from this operator verb,
    same as ``edit``/``block``/``complete`` -- it's a mutating board action."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="https://github.com/acme/repo/pull/39")
    monkeypatch.setattr("agent.delegation_context.kanban_path_is_fenced", lambda *_a, **_kw: True)
    args = argparse.Namespace(kanban_action="rebind-contract", task_id=tid,
                               new_contract="https://github.com/acme/repo/pull/46",
                               reason="predates CI", board=None)
    rc = kanban_cli.kanban_command(args)
    assert rc != 0
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).completion_contract == "https://github.com/acme/repo/pull/39"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_argv_parses_rebind_contract(kanban_home: Path) -> None:
    """End-to-end argv parsing + dispatch: ``hermes kanban rebind-contract <id> <new> --reason ...``."""
    from hermes_cli import kanban_parser

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="https://github.com/acme/repo/pull/39")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kanban_parser.build_parser(parser.add_subparsers(dest="command"))
    ns = parser.parse_args(["kanban", "rebind-contract", tid,
                             "https://github.com/acme/repo/pull/46", "--reason", "predates CI"])
    assert kanban_cli.kanban_command(ns) == 0
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).completion_contract == "https://github.com/acme/repo/pull/46"


def test_cli_rebind_contract_requires_reason(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="https://github.com/acme/repo/pull/39")
    args = argparse.Namespace(task_id=tid, new_contract="https://github.com/acme/repo/pull/46", reason=None)
    assert kanban_cli._cmd_rebind_contract(args) != 0


def test_cli_rebind_contract_success_and_cross_repo_error(
    kanban_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", completion_contract="https://github.com/acme/repo/pull/39")

    args = argparse.Namespace(task_id=tid, new_contract="https://github.com/acme/repo/pull/46",
                               reason="#39 predates CI")
    assert kanban_cli._cmd_rebind_contract(args) == 0
    assert f"Rebound {tid} completion_contract to https://github.com/acme/repo/pull/46" in capsys.readouterr().out

    bad_args = argparse.Namespace(task_id=tid, new_contract="https://github.com/other/repo/pull/1",
                                   reason="nope")
    rc = kanban_cli._cmd_rebind_contract(bad_args)
    assert rc != 0
    assert "must match" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Acceptance is UNTOUCHED: rebinding to a green sibling PR still requires the
# real GitHub check-run evidence to say "success" before complete_task closes.
# ---------------------------------------------------------------------------


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = [[]]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha, "app": {"id": 1},
                       "status": "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                value = [{"total_count": 1, "check_runs": [run]}]
            elif "/statuses" in self.path:
                value = [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "print(urllib.request.urlopen(u).read().decode())\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.linux_only
def test_rebind_does_not_bypass_acceptance(github) -> None:
    with connect() as conn:
        tid = kb.create_task(conn, title="rebind-then-complete", completion_contract="acme/repo")
        # Rebind onto a "green sibling" PR, same repo -- allowed.
        assert kb.rebind_contract(conn, tid, new_contract="https://github.com/acme/repo/pull/46",
                                   reason="#39 predates CI", actor="anika")
        rebound = [e for e in kb.list_events(conn, tid) if e.kind == "contract_rebound"]
        assert rebound and rebound[-1].payload["new_contract"] == "https://github.com/acme/repo/pull/46"

        # A failing check-run on the rebound PR still blocks completion --
        # the rebind changed WHICH PR is checked, never whether one is required.
        github["conclusion"] = "failure"
        assert not kb.complete_task(conn, tid, result="done",
            metadata={"published_pr": "https://github.com/acme/repo/pull/46"})
        assert kb.get_task(conn, tid).status != "done"
        receipts = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
        assert receipts and receipts[-1]["pr_url"] == "https://github.com/acme/repo/pull/46"
        assert receipts[-1]["ok"] is False

        # Only once the SAME rebound contract's evidence turns green does it close.
        github["conclusion"] = "success"
        assert kb.complete_task(conn, tid, result="done",
            metadata={"published_pr": "https://github.com/acme/repo/pull/46"})
        assert kb.get_task(conn, tid).status == "done"
