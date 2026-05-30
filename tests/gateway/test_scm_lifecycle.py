import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.dev_control.scm_lifecycle import (
    DevSCMLifecycleStore,
    build_code_review_prompt,
    compose_merge_readiness,
    execute_merge,
    fetch_pr_state,
    parse_code_review_result,
    request_merge_approval,
)
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _green_pr_state():
    return {
        "repo": "Felippen/Oryn",
        "pr_number": 42,
        "branch": "codex/scm-lifecycle",
        "pr_url": "https://github.com/Felippen/Oryn/pull/42",
        "head_sha": "abc123",
        "ci_state": "success",
        "ci_status": {"state": "success", "warnings": []},
        "review_state": "open",
        "mergeable": True,
        "merge_state": "clean",
        "warnings": [],
        "raw": {},
    }


def _green_readiness(**overrides):
    pr_state = {**_green_pr_state(), **overrides.pop("pr_state", {})}
    return compose_merge_readiness(
        repo="Felippen/Oryn",
        pr_number=42,
        pr_state=pr_state,
        draft_status=overrides.pop("draft_status", "approved_for_launch"),
        verification=overrides.pop("verification", {"verdict": "verified"}),
        code_review=overrides.pop("code_review", {"verdict": "approved"}),
        plan_id="plan-1",
        task_id="task-1",
    )


def test_pr_state_fail_opens_to_unknown_on_api_error():
    def failing_urlopen(request, timeout):
        raise RuntimeError("network down")

    state = fetch_pr_state(repo="Felippen/Oryn", pr_number=42, opener=failing_urlopen)

    assert state["object"] == "hermes.dev_pr_state"
    assert state["ci_state"] == "unknown"
    assert state["mergeable"] is None
    assert state["warnings"]


def test_pr_state_reads_github_and_ci_head_sha():
    def fake_urlopen(request, timeout):
        return FakeResponse({
            "html_url": "https://github.com/Felippen/Oryn/pull/42",
            "head": {"ref": "codex/scm-lifecycle", "sha": "abc123"},
            "mergeable": True,
            "mergeable_state": "clean",
            "state": "open",
            "draft": False,
        })

    state = fetch_pr_state(
        repo="Felippen/Oryn",
        pr_number=42,
        opener=fake_urlopen,
        ci_status_fetcher=lambda repo, ref: {"state": "success", "warnings": [], "failing": []},
    )

    assert state["head_sha"] == "abc123"
    assert state["ci_state"] == "success"
    assert state["mergeable"] is True


def test_code_review_result_parser_normalizes_structured_worker_output():
    parsed = parse_code_review_result(
        """
        ```json
        DEV_CODE_REVIEW_RESULT
        {
          "object": "hermes.dev_code_review_result",
          "verdict": "changes_requested",
          "findings": [{"severity": "high", "file": "Sources/App.swift", "line": 12, "note": "Bug"}],
          "summary": "One blocker.",
          "evidence_refs": ["diff:Sources/App.swift:12"]
        }
        ```
        """
    )

    assert parsed["verdict"] == "changes_requested"
    assert parsed["findings"][0]["file"] == "Sources/App.swift"
    assert parsed["evidence_refs"] == ["diff:Sources/App.swift:12"]


def test_code_review_prompt_forces_direct_diff_review_only():
    prompt = build_code_review_prompt(
        plan={
            "title": "Docs plan",
            "tasks": [{"acceptance_criteria": ["Docs note exists."]}],
        },
        pr_state={
            "repo": "Felippen/hermes-agent",
            "pr_number": 26,
            "pr_url": "https://github.com/Felippen/hermes-agent/pull/26",
            "head_sha": "abc123",
        },
    )

    assert "Perform a direct PR diff review only" in prompt
    assert "gh pr view 26 --repo Felippen/hermes-agent" in prompt
    assert "gh pr diff 26 --repo Felippen/hermes-agent --name-only" in prompt
    assert "gh pr diff 26 --repo Felippen/hermes-agent --patch" in prompt
    assert "Do not run slash commands such as /review" in prompt
    assert "Do not run CodeRabbit, coderabbit" in prompt
    assert "Do not start background tools" in prompt


def test_code_review_result_parser_recovers_unfenced_worker_json():
    parsed = parse_code_review_result(
        """
        Reviewed the PR. No findings.

        {
          "object": "hermes.dev_code_review_result",
          "verdict": "approved",
          "findings": [],
          "summary": "Review approved.",
          "evidence_refs": ["diff:docs/lab.md:1"]
        }
        """
    )

    assert parsed["verdict"] == "approved"
    assert parsed["summary"] == "Review approved."
    assert parsed["evidence_refs"] == ["diff:docs/lab.md:1"]
    assert "recovered review JSON" in " ".join(parsed["warnings"])


def test_code_review_result_parser_recovers_wrapped_unfenced_worker_json():
    parsed = parse_code_review_result(
        '''
        {
          "object": "hermes.dev_code_review_result",
          "verdict": "approved",
          "findings": [],
          "summary": "Using the draft PR diff as the evidence source, PR #23 satisfies
          the docs criterion.",
          "evidence_refs": [
            "Felippen/hermes-agent#23",
            "PR diff: docs/lab-dogfood.md:1"
          ]
        }
        '''
    )

    assert parsed["verdict"] == "approved"
    assert parsed["summary"].startswith("Using the draft PR diff")
    assert parsed["evidence_refs"] == ["Felippen/hermes-agent#23", "PR diff: docs/lab-dogfood.md:1"]


def test_code_review_result_parser_recovers_marker_key_wrapped_json():
    parsed = parse_code_review_result(
        """
        {
          "hermes.dev_code_review_result": {
            "verdict": "approved",
            "findings": [],
            "summary": "PR #30 was inspected using only gh pr view/diff.",
            "evidence_refs": [
              {"source": "gh pr diff 30 --repo Felippen/hermes-agent --patch"}
            ]
          }
        }
        """
    )

    assert parsed["verdict"] == "approved"
    assert parsed["summary"].startswith("PR #30")
    assert parsed["evidence_refs"][0]["source"].startswith("gh pr diff 30")


@pytest.mark.parametrize(
    ("name", "kwargs", "gate"),
    [
        ("draft", {"draft_status": "draft"}, "draft"),
        ("ci", {"pr_state": {"ci_state": "failure"}}, "ci"),
        ("verification", {"verification": {"verdict": "needs_review"}}, "verification"),
        ("review", {"code_review": {"verdict": "changes_requested"}}, "code_review"),
        ("mergeable", {"pr_state": {"mergeable": False}}, "mergeable"),
    ],
)
def test_merge_readiness_blocks_each_gate(name, kwargs, gate):
    readiness = _green_readiness(**kwargs)

    assert readiness["ready"] is False, name
    assert any(item["gate"] == gate for item in readiness["blocked_by"])


def test_merge_readiness_ready_only_when_all_gates_pass():
    readiness = _green_readiness()

    assert readiness["ready"] is True
    assert readiness["blocked_by"] == []


def test_merge_approval_is_single_use_and_head_sha_bound(tmp_path):
    store = DevSCMLifecycleStore(db_path=tmp_path / "state.db")
    readiness = _green_readiness()
    approval = request_merge_approval(store=store, readiness=readiness)["approval"]
    approved = store.approve_merge_approval(approval["approval_id"], approved_by="felipe")

    assert approved["status"] == "approved"
    assert approved["head_sha"] == "abc123"

    calls = []
    result = execute_merge(
        store=store,
        approval_id=approved["approval_id"],
        live_readiness=readiness,
        executor=lambda **kwargs: calls.append(kwargs) or {"ok": True},
        executor_enabled=True,
    )

    assert result["merged"] is True
    assert calls[0]["head_sha"] == "abc123"
    reused = execute_merge(
        store=store,
        approval_id=approved["approval_id"],
        live_readiness=readiness,
        executor=lambda **kwargs: {"ok": True},
        executor_enabled=True,
    )
    assert reused["merged"] is False
    assert "not approved" in reused["reason"]


def test_execute_merge_refuses_and_invalidates_when_head_sha_changed(tmp_path):
    store = DevSCMLifecycleStore(db_path=tmp_path / "state.db")
    readiness = _green_readiness()
    approval = request_merge_approval(store=store, readiness=readiness)["approval"]
    approved = store.approve_merge_approval(approval["approval_id"], approved_by="felipe")

    result = execute_merge(
        store=store,
        approval_id=approved["approval_id"],
        live_readiness=_green_readiness(pr_state={"head_sha": "newsha"}),
        executor=lambda **kwargs: pytest.fail("executor must not run"),
        executor_enabled=True,
    )

    assert result["merged"] is False
    assert result["approval"]["status"] == "invalidated"
    assert "head_sha changed" in result["reason"]


def test_execute_merge_refuses_and_invalidates_when_gate_regressed(tmp_path):
    store = DevSCMLifecycleStore(db_path=tmp_path / "state.db")
    readiness = _green_readiness()
    approval = request_merge_approval(store=store, readiness=readiness)["approval"]
    approved = store.approve_merge_approval(approval["approval_id"], approved_by="felipe")

    result = execute_merge(
        store=store,
        approval_id=approved["approval_id"],
        live_readiness=_green_readiness(verification={"verdict": "failed"}),
        executor=lambda **kwargs: pytest.fail("executor must not run"),
        executor_enabled=True,
    )

    assert result["merged"] is False
    assert result["approval"]["status"] == "invalidated"
    assert "regressed" in result["reason"]


def test_execute_merge_disabled_until_branch_protection_confirmed(tmp_path):
    store = DevSCMLifecycleStore(db_path=tmp_path / "state.db")
    readiness = _green_readiness()
    approval = request_merge_approval(store=store, readiness=readiness)["approval"]
    approved = store.approve_merge_approval(approval["approval_id"], approved_by="felipe")

    result = execute_merge(
        store=store,
        approval_id=approved["approval_id"],
        live_readiness=readiness,
        executor=lambda **kwargs: pytest.fail("executor must not run"),
        executor_enabled=False,
    )

    assert result["merged"] is False
    assert result["approval"]["status"] == "approved"
    assert "branch protection" in result["reason"]


@pytest.mark.asyncio
async def test_scm_lifecycle_api_routes_with_fake_readiness(tmp_path, monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(adapter, "_ensure_dev_execution_store", lambda: type("Store", (), {"db_path": db_path})())
    readiness = _green_readiness()
    monkeypatch.setattr(adapter, "_dev_merge_readiness_snapshot", lambda **kwargs: readiness)

    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app.router.add_get("/v1/dev/merge-readiness", adapter._handle_dev_merge_readiness)
    app.router.add_post("/v1/dev/merge-approvals", adapter._handle_dev_merge_approvals)
    app.router.add_post("/v1/dev/merge-approvals/{approval_id}/approve", adapter._handle_dev_merge_approval_approve)

    async with TestClient(TestServer(app)) as cli:
        ready_response = await cli.get(
            "/v1/dev/merge-readiness?repo=Felippen/Oryn&pr_number=42&plan_id=plan-1&task_id=task-1",
            headers={"Authorization": "Bearer sk-secret"},
        )
        approval_response = await cli.post(
            "/v1/dev/merge-approvals",
            headers={"Authorization": "Bearer sk-secret"},
            json={"readiness": readiness},
        )
        approval_body = await approval_response.json()
        approval_id = approval_body["approval"]["approval_id"]
        approve_response = await cli.post(
            f"/v1/dev/merge-approvals/{approval_id}/approve",
            headers={"Authorization": "Bearer sk-secret"},
            json={"approved_by": "felipe"},
        )
        ready_body = await ready_response.json()
        approve_body = await approve_response.json()

    assert ready_response.status == 200
    assert ready_body["ready"] is True
    assert approval_response.status == 200
    assert approve_response.status == 200
    assert approve_body["approval"]["status"] == "approved"
