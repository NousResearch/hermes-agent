"""Typed, secret-free Kanban PR-acceptance failure evidence.

Regression class: the PR completion-contract verifier collapsed every
``gh`` subprocess failure — unauthenticated gh, HTTP 401/403/404, rate
limit, timeout, unknown exits — into one generic ``infra`` receipt with
``head_sha=null``, so delivered green PRs were uncompletable with no
actionable evidence. These tests pin the corrections: each materially
distinct failure is typed (``capability`` / ``auth_unavailable`` /
``rate_limited`` / ``network`` / ``provider_error``, with generic
``infra`` kept only as the final net), and no ``gh`` stderr substring —
denylisted prefix or bare credential — ever survives into the persisted
receipt.

GitHub *authentication scoping* (profile-scoped child env / secret-surface
GH_TOKEN resolution) is intentionally NOT covered here: that boundary is
owned by upstream PR #112767. These tests keep the verifier's credential
behavior exactly as on base — whatever ``gh`` resolves on its own — and
pin only the failure-evidence contract.
"""
import json
import os
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

_PR_URL = "https://github.com/acme/repo/pull/7"
_INFRA_DETAIL = ("GitHub acceptance evidence unavailable or incomplete; "
                 "check gh authentication/API access and retry.")


def _stub_run(monkeypatch, *, stderr="", exc_factory=None, returns=None):
    """Stub ``subprocess.run`` in the module under test; records each call."""
    import hermes_cli.kanban_pr_acceptance as acc
    calls = []

    def record(command, **kwargs):
        calls.append((command, kwargs))
        if exc_factory is not None:
            raise exc_factory(command)
        if returns is not None:
            return returns
        raise subprocess.CalledProcessError(returncode=1, cmd=command, output="", stderr=stderr)

    monkeypatch.setattr(acc.subprocess, "run", record)
    return calls


def test_missing_gh_binary_is_a_typed_capability_result(tmp_path, monkeypatch):
    """No gh CLI => explicit capability classification, never generic null-head infra."""
    import hermes_cli.kanban_pr_acceptance as acc
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    receipt = acc.collect_acceptance("acme/repo", _PR_URL)
    assert receipt["classification"] == "capability"
    assert receipt["head_sha"] is None and receipt["checks"] == []
    assert "gh CLI" in receipt["detail"]
    assert receipt["pr_url"] == _PR_URL


@pytest.mark.parametrize("stderr,expected", [
    ("To get started with GitHub CLI, please run: gh auth login", "auth_unavailable"),
    ("gh: To use GitHub in a terminal, run: gh auth login", "auth_unavailable"),
    ("gh: Bad credentials for api.github.com (HTTP 401)", "auth_unavailable"),
    ("gh: HTTP 403: API rate limit exceeded for 1.2.3.4", "rate_limited"),
    ("gh: Not Found (HTTP 404)", "provider_error"),
    ("gh: unexpected exit code 42 with no known signature", "provider_error"),
], ids=["login-banner", "login-alt", "http-401", "rate-limit", "http-404", "unknown-exit"])
def test_gh_stderr_signatures_classify_without_persisting_stderr(monkeypatch, stderr, expected):
    """Each materially distinct gh failure maps to its typed classification, and
    the persisted detail is the fixed secret-free sentence — no stderr substring."""
    import hermes_cli.kanban_pr_acceptance as acc
    token = "ghp_" + "q" * 36
    calls = _stub_run(
        monkeypatch, stderr=f"{stderr}; GH_TOKEN={token} credential leaked?")
    receipt = acc.collect_acceptance("acme/repo", _PR_URL)
    assert receipt["classification"] == expected
    blob = json.dumps(receipt)
    assert token not in blob
    assert stderr not in blob
    assert "gh: " not in blob
    # Credential behavior unchanged from base: the verifier passes no custom
    # environment to gh (authentication scoping is owned upstream, #112767).
    assert calls and "env" not in calls[0][1]


def test_gh_timeout_is_a_typed_network_failure(monkeypatch):
    """A hung gh call becomes network evidence with recovery advice, not infra."""
    import hermes_cli.kanban_pr_acceptance as acc
    _stub_run(monkeypatch, exc_factory=lambda cmd: subprocess.TimeoutExpired(cmd=cmd, timeout=30))
    receipt = acc.collect_acceptance("acme/repo", _PR_URL)
    assert receipt["classification"] == "network"
    assert "timed out" in receipt["detail"]
    assert "network" in receipt["recovery"] or "retry" in receipt["recovery"]


def test_graphql_bad_credentials_map_to_auth_unavailable(monkeypatch):
    """GraphQL errors[] bodies carrying Bad credentials become typed auth failures."""
    import hermes_cli.kanban_pr_acceptance as acc
    monkeypatch.setattr(acc.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(
                            command, 0,
                            stdout=json.dumps({"data": None, "errors": [{"message": "Bad credentials"}]}),
                            stderr=""))
    receipt = acc.collect_acceptance("acme/repo", _PR_URL)
    assert receipt["classification"] == "auth_unavailable"
    assert receipt["head_sha"] is None
    assert "credential" in receipt["detail"]


@pytest.mark.linux_only
def test_persisted_acceptance_evidence_never_embeds_gh_stderr(tmp_path, monkeypatch):
    """Through the real completion boundary, a bare legacy 40-hex credential in gh
    stderr reaches neither the persisted pr_acceptance event nor last_failure_error."""
    import hermes_cli.kanban_pr_acceptance as acc
    legacy = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"

    def legacy_credential(command, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=command, output="",
            stderr=f"gh: credential {legacy} rejected for api.github.com")

    monkeypatch.setattr(acc.subprocess, "run", legacy_credential)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": _PR_URL})
        task = kb.get_task(conn, tid)
        assert task.status != "done"
        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,)).fetchall()
        assert rows, "refusal must persist durable acceptance evidence"
        blob = json.dumps([json.loads(r[0]) for r in rows]) + str(task.last_failure_error)
        assert legacy not in blob
        assert "api.github.com" not in blob
        assert "gh: " not in blob
        receipt = json.loads(rows[-1][0])
        assert receipt["classification"] == "provider_error"
        assert receipt["detail"] == "GitHub API call failed (rc=1)."


@pytest.mark.linux_only
def test_generic_infra_stays_the_final_net(tmp_path, monkeypatch):
    """A non-gh exception (e.g. malformed evidence) still fails closed with the
    generic infra classification — typing never widens acceptance."""
    import hermes_cli.kanban_pr_acceptance as acc
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nprint('not json at all')\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    receipt = acc.collect_acceptance("acme/repo", _PR_URL)
    assert receipt["classification"] == "infra"
    assert receipt["detail"] == _INFRA_DETAIL
    assert receipt["ok"] is False
