"""Explicit MCP acceptance mode (interface v1): scoped stdio client + lifecycle.

Every server here is the SYNTHETIC stdio fixture under tests/hermes_cli/fixtures/
(protocol proof only — no network, no GitHub, no credentials). Profiles A and B are
disposable fixture homes with their own config.yaml/.env; the scoped client must use
each named profile's OWN `github_acceptance` server and env scope, never the default
scope, the ambient process env, gh, or any fallback.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "kanban_acceptance_mcp_fixture.py"
PR_URL = "https://github.com/acme/repo/pull/7"
AMBIENT_RESIDUE = {"GITHUB_TOKEN": "ambient-must-not-leak", "GH_TOKEN": "ambient-must-not-leak"}


def _evidence(app_id, head="a" * 40, conclusion="success", check_status="completed",
              state="OPEN", *, complete=None):
    return {
        "schema_version": 1,
        "repository": "acme/repo",
        "pull_number": 7,
        "identity": {"kind": "github_app", "app_id": app_id, "installation_id": 1000 + app_id},
        "pr_before": {"head_sha": head, "base_ref": "main", "state": state},
        "classic_required": [{"context": "required", "app_id": app_id}],
        "ruleset_required": [],
        "check_runs": [{"id": 42, "name": "required", "head_sha": head, "app_id": app_id,
                        "status": check_status, "conclusion": conclusion,
                        "url": f"https://github.com/acme/repo/actions/runs/{app_id}"}],
        "statuses": [],
        "pr_after": {"head_sha": head, "base_ref": "main", "state": state},
        "complete": complete or {"classic": True, "rulesets": True,
                                 "check_runs": True, "statuses": True},
        "counts": {"check_runs": 1, "statuses": 0},
    }


def _write_profile(home, app_id, *, transport="mcp", server=True, mode="ok", gate=None):
    """A disposable fixture profile home with its OWN config and .env secret scope."""
    home.mkdir(parents=True, exist_ok=True)
    evidence_path = home / "evidence.json"
    evidence_path.write_text(json.dumps(_evidence(app_id)), encoding="utf-8")
    lines = ["kanban:", "  pr_acceptance:", f"    transport: {transport}"]
    if server:
        env = {"FIXTURE_MODE": mode, "FIXTURE_EVIDENCE": str(evidence_path),
               "FIXTURE_TRACE": str(home / "trace.jsonl"),
               "GITHUB_APP_ID": "${APP_ID}", "GITHUB_APP_INSTALLATION_ID": "${INSTALL_ID}",
               "GITHUB_APP_PRIVATE_KEY": "${APP_KEY}"}
        if gate:
            env["FIXTURE_GATE"] = str(gate)
        lines += ["mcp_servers:", "  github_acceptance:",
                  f"    command: {json.dumps(sys.executable)}",
                  f"    args: [{json.dumps(str(FIXTURE_SERVER))}]", "    env:"]
        lines += [f"      {key}: {json.dumps(value)}" for key, value in env.items()]
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (home / ".env").write_text(
        f"APP_ID={app_id}\nINSTALL_ID={1000 + app_id}\nAPP_KEY=fake-scope-{app_id}-key\n",
        encoding="utf-8")
    return evidence_path


def _traces(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _receipts(conn, tid):
    return [json.loads(row[0]) for row in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]


def test_pr_acceptance_transport_is_a_registered_gh_default_setting():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["kanban"]["pr_acceptance"]["transport"] == "gh"


def test_read_transport_resolves_the_own_profile_config(tmp_path, monkeypatch):
    from hermes_cli.kanban_pr_acceptance_mcp import _McpConfigError, read_transport
    home = tmp_path / "home"
    a = _write_profile(home / "profiles" / "a", 11, transport="mcp")
    assert isinstance(a, Path)
    assert read_transport(str(home / "profiles" / "a")) == "mcp"
    b = _write_profile(home / "profiles" / "b", 22, transport="gh")
    assert read_transport(str(home / "profiles" / "b")) == "gh"
    # An unset key (or a profile without the section) keeps the native transport.
    (home / "profiles" / "b" / "config.yaml").write_text("model: x\n", encoding="utf-8")
    assert read_transport(str(home / "profiles" / "b")) is None
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "kanban:\n  pr_acceptance:\n    transport: mcp\n", encoding="utf-8")
    assert read_transport(None) == "mcp"
    # An unknown transport value fails closed; it never silently degrades to gh.
    ghost = home / "profiles" / "ghost"
    ghost.mkdir()
    (ghost / "config.yaml").write_text(
        "kanban:\n  pr_acceptance:\n    transport: rest\n", encoding="utf-8")
    with pytest.raises(_McpConfigError):
        read_transport(str(ghost))


def test_mcp_mode_completes_with_v1_evidence_and_scoped_env(tmp_path, monkeypatch):
    from hermes_cli.kanban_pr_acceptance_mcp import read_transport
    home = tmp_path / "home"
    _write_profile(home / "profiles" / "a", 11)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key, value in AMBIENT_RESIDUE.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin") + os.pathsep + os.environ["PATH"])
    kb.init_db()
    assert read_transport(str(home / "profiles" / "a")) == "mcp"
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo", assignee="a")
        ok = kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR_URL})
        assert ok and kb.get_task(conn, tid).status == "done"
        receipt = _receipts(conn, tid)[-1]
    assert receipt["ok"] and receipt["classification"] == "success"
    assert receipt["head_sha"] == "a" * 40 and receipt["pr_url"] == PR_URL
    assert receipt["required"] == [{"context": "required", "app_id": 11}]
    assert receipt["checks"][0]["id"] == 42 and receipt["checks"][0]["conclusion"] == "success"
    # The synthetic server saw exactly the three narrow arguments and the profile's OWN
    # App env scope; the ambient launch credentials never reached the child.
    trace = _traces(home / "profiles" / "a" / "trace.jsonl")[-1]
    assert trace["arguments"] == {"owner": "acme", "repo": "repo", "pullNumber": 7}
    assert trace["has_app_credentials"] is True
    assert "GITHUB_TOKEN" not in trace["env_key_names"]
    assert "GH_TOKEN" not in trace["env_key_names"]


def test_mcp_only_never_falls_back_to_gh(tmp_path, monkeypatch):
    home = tmp_path / "home"
    profile = home / "profiles" / "a"
    _write_profile(profile, 11, server=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key, value in AMBIENT_RESIDUE.items():
        monkeypatch.setenv(key, value)
    gh_marker = tmp_path / "gh_invoked"
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "gh").write_text(
        f"#!{sys.executable}\nPath({str(gh_marker)!r}).write_text('x')\n"
        "print(json.dumps({}))\n", encoding="utf-8")
    (shim / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="mcp-only", completion_contract="acme/repo", assignee="a")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR_URL})
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipts(conn, tid)[-1]
    assert receipt["classification"] == "auth"
    assert "github_acceptance" in receipt["detail"]
    assert not gh_marker.exists()


def test_named_profile_cannot_borrow_the_default_scope_server(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_profile(home, 99)  # default scope owns a configured server…
    _write_profile(home / "profiles" / "b", 22, server=False)  # …profile b does not.
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key, value in AMBIENT_RESIDUE.items():
        monkeypatch.setenv(key, value)
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR_URL})
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipts(conn, tid)[-1]
    assert receipt["classification"] == "auth"
    assert "github_acceptance" in receipt["detail"]
    assert not (home / "trace.jsonl").exists()  # the default scope's server never ran
    # The default scope may still use its own server for its own unassigned card.
    local = kb.create_task(conn, title="home-scope", completion_contract="acme/repo")
    assert kb.complete_task(conn, local, result="done", metadata={"published_pr": PR_URL})
    home_receipt = _receipts(conn, local)[-1]
    assert home_receipt["ok"] and home_receipt["required"] == [
        {"context": "required", "app_id": 99}]


def test_a_b_a_profile_switch_uses_each_profile_own_server(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_profile(home / "profiles" / "a", 11)
    _write_profile(home / "profiles" / "b", 22)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key, value in AMBIENT_RESIDUE.items():
        monkeypatch.setenv(key, value)
    kb.init_db()
    with connect() as conn:
        seen = []
        for assignee, app_id in (("a", 11), ("b", 22), ("a", 11)):
            tid = kb.create_task(conn, title=f"as-{assignee}",
                                 completion_contract="acme/repo", assignee=assignee)
            assert kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR_URL})
            receipt = _receipts(conn, tid)[-1]
            assert receipt["ok"] and receipt["required"] == [
                {"context": "required", "app_id": app_id}]
            seen.append(_traces(home / "profiles" / assignee / "trace.jsonl")[-1]["arguments"])
    assert seen == [{"owner": "acme", "repo": "repo", "pullNumber": 7}] * 3


@pytest.mark.parametrize("mode", ["tool_error", "mismatch", "extra_text", "no_structured"])
def test_hostile_fixture_output_fails_closed_without_raw_text(tmp_path, monkeypatch, mode):
    from hermes_cli.kanban_pr_acceptance_mcp import MCP_EVIDENCE_INVALID_DETAIL
    home = tmp_path / "home"
    _write_profile(home / "profiles" / "a", 11, mode=mode)
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title=f"hostile-{mode}", completion_contract="acme/repo",
                             assignee="a")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR_URL})
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipts(conn, tid)[-1]
    assert receipt["classification"] == "infra"
    assert receipt["detail"] == MCP_EVIDENCE_INVALID_DETAIL
    serialized = json.dumps(receipt)
    assert "tampered" not in serialized and "fixture_tool_error" not in serialized
    assert "schema_version" not in serialized  # no raw MCP block material is persisted


def test_reclaim_during_mcp_collection_leaves_no_receipt_on_the_new_run(tmp_path, monkeypatch):
    home = tmp_path / "home"
    gate = tmp_path / "gate"
    _write_profile(home / "profiles" / "a", 11, gate=str(gate))
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    trace = home / "profiles" / "a" / "trace.jsonl"
    with connect() as conn:
        tid = kb.create_task(conn, title="race", completion_contract="acme/repo", assignee="a")
        run_id = kb.claim_task(conn, tid).current_run_id
        outcome = {}

        def complete():
            # SQLite objects are thread-bound: the completing thread owns its connection.
            with connect() as worker_conn:
                outcome["ok"] = kb.complete_task(worker_conn, tid, result="done",
                                                 expected_run_id=run_id,
                                                 metadata={"published_pr": PR_URL})

        worker = threading.Thread(target=complete)
        worker.start()
        deadline = time.monotonic() + 30
        while not trace.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert trace.exists(), "fixture server never received the call"
        with connect() as rival:
            assert kb.block_task(rival, tid, reason="Reassigned during acceptance")
            assert kb.unblock_task(rival, tid)
            replacement = kb.claim_task(rival, tid).current_run_id
        gate.write_text("go", encoding="utf-8")
        worker.join(30)
        assert not worker.is_alive()
        assert outcome["ok"] is False
        assert kb.get_task(conn, tid).current_run_id == replacement != run_id
        assert kb.get_task(conn, tid).status != "done"
        assert conn.execute(
            "SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,)).fetchone()[0] == 0