"""Regression coverage for native Gitea Kanban PR acceptance."""

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_acceptance as acceptance
from hermes_cli.kanban_db_connect import connect


SHA = "a" * 40


def _responses(*, state="open", merged=False, status="success", status_sha=SHA,
               reread_sha=SHA):
    pr = {
        "state": state,
        "merged": merged,
        "head": {"sha": SHA},
        "base": {"ref": "main"},
    }
    return [
        pr,
        {"protected": True, "required_status_checks": True,
         "status_check_contexts": ["ci/test"]},
        {"sha": status_sha, "total_count": 1, "statuses": [
            {"id": 7, "context": "ci/test", "status": status,
             "target_url": "http://gitea:3000/flug/webbird/actions/runs/7"}
        ]},
        {**pr, "head": {"sha": reread_sha}},
    ]


def _install_api(monkeypatch, responses):
    calls = []
    values = iter(responses)

    def fake_api(ref, endpoint, *, token, query=None):
        calls.append((ref.base_url, endpoint, token, query))
        return next(values)

    monkeypatch.setattr(acceptance, "_gitea_api", fake_api)
    monkeypatch.setattr(acceptance, "_gitea_token", lambda profile_home: "test-token")
    return calls


def test_validate_contract_accepts_exact_gitea_urls_and_rejects_untrusted_shapes():
    assert acceptance.validate_contract("http://gitea:3000/flug/webbird/pulls/12") == (
        "http://gitea:3000/flug/webbird/pulls/12"
    )
    assert acceptance.validate_contract("https://code.example/flug/webbird/pulls/12") == (
        "https://code.example/flug/webbird/pulls/12"
    )
    assert acceptance.validate_contract("gitea:http://gitea:3000/flug/webbird") == (
        "gitea:http://gitea:3000/flug/webbird"
    )
    for value in (
        "ftp://gitea/flug/webbird/pulls/12",
        "http://gitea/flug/webbird/issues/12",
        "http://forge.example/flug/webbird/pulls/12",
        "http://user:secret@gitea/flug/webbird/pulls/12",
        "http://gitea/flug/webbird/pulls/12?token=secret",
        "https://example.com/arbitrary/path",
    ):
        with pytest.raises(ValueError, match="Gitea"):
            acceptance.validate_contract(value)


@pytest.mark.parametrize("state,merged", [("open", False), ("closed", True)])
def test_gitea_acceptance_allows_open_or_merged_pr_with_exact_head_checks(
    monkeypatch, state, merged
):
    calls = _install_api(monkeypatch, _responses(state=state, merged=merged))
    receipt = acceptance.collect_acceptance(
        "http://gitea:3000/flug/webbird/pulls/12", None
    )
    assert receipt["ok"] is True, receipt
    assert receipt["classification"] == "success"
    assert receipt["head_sha"] == SHA
    assert receipt["checks"] == [{
        "name": "ci/test", "id": 7,
        "url": "http://gitea:3000/flug/webbird/actions/runs/7",
        "head_sha": SHA, "classification": "success", "conclusion": "success",
    }]
    assert all(token == "test-token" for _base, _path, token, _query in calls)


def test_gitea_acceptance_rejects_closed_unmerged_pr(monkeypatch):
    _install_api(monkeypatch, _responses(state="closed", merged=False))
    receipt = acceptance.collect_acceptance(
        "http://gitea:3000/flug/webbird/pulls/12", None
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "failure"
    assert "closed without merge" in receipt["detail"]


@pytest.mark.parametrize(
    "status,classification", [("pending", "pending"), ("running", "pending"),
                               ("failure", "failure"), ("error", "infra")]
)
def test_gitea_required_status_classification(monkeypatch, status, classification):
    _install_api(monkeypatch, _responses(status=status))
    receipt = acceptance.collect_acceptance(
        "http://gitea:3000/flug/webbird/pulls/12", None
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == classification
    assert receipt["checks"][0]["classification"] == classification


def test_gitea_head_change_is_stale(monkeypatch):
    _install_api(monkeypatch, _responses(reread_sha="b" * 40))
    receipt = acceptance.collect_acceptance(
        "http://gitea:3000/flug/webbird/pulls/12", None
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "stale"
    assert "head/base changed" in receipt["detail"]


def test_gitea_statuses_must_belong_to_exact_head(monkeypatch):
    _install_api(monkeypatch, _responses(status_sha="b" * 40))
    receipt = acceptance.collect_acceptance(
        "http://gitea:3000/flug/webbird/pulls/12", None
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "infra"


def test_gitea_pr_identity_mismatch_is_infra(monkeypatch):
    responses = _responses()
    responses[0] = {**responses[0], "number": 13}
    _install_api(monkeypatch, responses)
    receipt = acceptance.collect_acceptance(
        "http://gitea:3000/flug/webbird/pulls/12", None
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "infra"


@pytest.mark.parametrize("code", [401, 403])
def test_gitea_auth_failures_are_stable_and_never_expose_tokens(monkeypatch, code):
    secret = "gitea-secret-that-must-not-leak"
    monkeypatch.setattr(acceptance, "_gitea_token", lambda profile_home: secret)

    def denied(*args, **kwargs):
        raise acceptance._GateAuthError(f"HTTP {code} on /api/v1/repos/flug/webbird/pulls/12")

    monkeypatch.setattr(acceptance, "_gitea_api", denied)
    receipt = acceptance.collect_acceptance(
        "http://gitea:3000/flug/webbird/pulls/12", None
    )
    assert receipt["classification"] == "auth"
    assert f"HTTP {code}" in receipt["detail"]
    assert secret not in json.dumps(receipt)


def test_private_gitea_without_token_reports_configuration(monkeypatch):
    monkeypatch.setattr(acceptance, "_gitea_token", lambda profile_home: None)

    def denied(*args, **kwargs):
        raise acceptance._GateAuthError("HTTP 401 on /api/v1/repos/flug/webbird/pulls/12")

    monkeypatch.setattr(acceptance, "_gitea_api", denied)
    receipt = acceptance.collect_acceptance(
        "http://gitea:3000/flug/webbird/pulls/12", None
    )
    assert receipt["classification"] == "auth"
    assert "GITEA_TOKEN" in receipt["detail"]


def test_gitea_token_comes_from_assignee_profile(tmp_path, monkeypatch):
    launch_home = tmp_path / "home"
    launch_home.mkdir()
    assignee_home = launch_home / "profiles" / "worker"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GITEA_TOKEN=worker-token\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("GITEA_TOKEN", "ambient-token")
    assert acceptance._gitea_token(str(assignee_home)) == "worker-token"


def test_gitea_repo_contract_binds_once_to_matching_published_pr(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    _install_api(monkeypatch, _responses())
    with connect() as conn:
        task_id = kb.create_task(
            conn,
            title="publish",
            completion_contract="gitea:http://gitea:3000/flug/webbird",
        )
        assert kb.complete_task(
            conn,
            task_id,
            result="done",
            metadata={"published_pr": "http://gitea:3000/flug/webbird/pulls/12"},
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "done"
        assert task.completion_contract == "http://gitea:3000/flug/webbird/pulls/12"


def test_gitea_repo_contract_refuses_sibling_repo_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not run for a mismatched binding")

    monkeypatch.setattr(acceptance, "_gitea_api", unexpected, raising=False)
    with connect() as conn:
        task_id = kb.create_task(
            conn,
            title="publish",
            completion_contract="gitea:http://gitea:3000/flug/webbird",
        )
        assert not kb.complete_task(
            conn,
            task_id,
            result="done",
            metadata={"published_pr": "http://gitea:3000/flug/other/pulls/12"},
        )
        assert kb.get_task(conn, task_id).status != "done"
    assert called is False
