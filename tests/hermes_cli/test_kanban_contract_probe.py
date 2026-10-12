"""A card is never created with a PR completion contract that cannot pass.

A PR-form contract is gated on repository-required checks, so a repo that
requires none (or a host with no ``gh`` to read them) makes the contract
unsatisfiable: the gate refuses completion forever. Creation settles that up
front instead. The probe is stubbed at the ``gh api`` seam (argv -> JSON).
"""
import json
import os
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_acceptance as pa
from hermes_cli.kanban_db_connect import connect

_GRAPHQL = "graphql"
_RULES = "repos/acme/repo/rules/branches/main?per_page=100"


def _graphql_payload(required=()):
    return {"data": {"repository": {"defaultBranchRef": {
        "name": "main",
        "branchProtectionRule": {"requiredStatusChecks": [{"context": c} for c in required]},
    }}}}


def _rules_payload(required=()):
    checks = [{"context": c} for c in required]
    rule = {"type": "required_status_checks", "parameters": {"required_status_checks": checks}}
    return [[rule]] if checks else [[]]


def _stub_gh(tmp_path, monkeypatch, payloads, *, present=True):
    """Install a ``gh`` whose ``api <endpoint>`` prints a canned payload, and log
    every endpoint it was asked for. ``present=False`` models a host with no gh."""
    shim = tmp_path / "bin"
    shim.mkdir()
    log = tmp_path / "gh_calls.txt"
    (shim / "gh").write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"payloads = json.loads({json.dumps(payloads)!r})\n"
        f"open({str(log)!r}, 'a').write(sys.argv[2] + chr(10))\n"
        "print(json.dumps(payloads.get(sys.argv[2], {})))\n")
    (shim / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", (str(shim) + os.pathsep + os.environ["PATH"]) if present else str(shim))
    if not present:
        monkeypatch.setenv("PATH", "/nonexistent-bin")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    pa._probe_cache.clear()
    kb.init_db()
    return log


def _created_event(conn, tid):
    rows = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='created'",
                        (tid,)).fetchall()
    assert len(rows) == 1
    return json.loads(rows[0][0])


@pytest.mark.platforms("linux")
def test_no_required_checks_downgrades_at_creation(tmp_path, monkeypatch):
    _stub_gh(tmp_path, monkeypatch, {_GRAPHQL: _graphql_payload(), _RULES: _rules_payload()})
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        task = kb.get_task(conn, tid)
        assert task.completion_contract == "local-only"
        note = _created_event(conn, tid)["contract_downgrade"]
        assert note["from"] == "acme/repo" and note["to"] == "local-only"
        assert "no required status checks" in note["reason"]


@pytest.mark.platforms("linux")
def test_required_checks_keep_the_contract(tmp_path, monkeypatch):
    payloads = {_GRAPHQL: _graphql_payload(["ci"]), _RULES: _rules_payload()}
    _stub_gh(tmp_path, monkeypatch, payloads)
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.get_task(conn, tid).completion_contract == "acme/repo"
        assert "contract_downgrade" not in _created_event(conn, tid)


@pytest.mark.platforms("linux")
def test_ruleset_required_checks_count_too(tmp_path, monkeypatch):
    payloads = {_GRAPHQL: _graphql_payload(), _RULES: _rules_payload(["ci"])}
    _stub_gh(tmp_path, monkeypatch, payloads)
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.get_task(conn, tid).completion_contract == "acme/repo"


@pytest.mark.platforms("linux")
def test_missing_gh_downgrades_at_creation(tmp_path, monkeypatch):
    _stub_gh(tmp_path, monkeypatch, {_GRAPHQL: _graphql_payload(), _RULES: _rules_payload()},
             present=False)
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.get_task(conn, tid).completion_contract == "local-only"
        assert "gh" in _created_event(conn, tid)["contract_downgrade"]["reason"]


@pytest.mark.platforms("linux")
def test_an_unreadable_probe_never_rewrites_the_contract(tmp_path, monkeypatch):
    # A login that cannot read the repo resolves to null; a shape we do not
    # understand proves nothing, so the creator's declaration stands.
    unreadable = {"data": {"repository": None}}
    _stub_gh(tmp_path, monkeypatch, {_GRAPHQL: unreadable, _RULES: _rules_payload()})
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.get_task(conn, tid).completion_contract == "acme/repo"
        assert "contract_downgrade" not in _created_event(conn, tid)


@pytest.mark.platforms("linux")
def test_the_probe_is_cached_per_repo(tmp_path, monkeypatch):
    log = _stub_gh(tmp_path, monkeypatch,
                   {_GRAPHQL: _graphql_payload(["ci"]), _RULES: _rules_payload()})
    with connect() as conn:
        kb.create_task(conn, title="one", completion_contract="acme/repo")
        kb.create_task(conn, title="two", completion_contract="acme/repo")
        assert log.read_text().split() == [_GRAPHQL, _RULES]
