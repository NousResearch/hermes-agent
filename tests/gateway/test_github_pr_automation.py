import hashlib
import hmac
import json

from gateway.dev_control.github_pr_automation import (
    AUTO_FIX_LABEL,
    AUTO_MERGE_LABEL,
    AUTO_RELEASE_LABEL,
    DevGitHubPRAutomationStore,
    build_pr_fix_prompt,
    next_patch_version,
    process_github_webhook,
    reconcile_github_pr_automation,
    verify_github_signature,
)
from gateway.dev_control.scm_lifecycle import DevSCMLifecycleStore, compose_merge_readiness


def _signature(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _pr_payload(*, labels=None, action="opened", merged=False):
    return {
        "action": action,
        "repository": {"full_name": "Felippen/Oryn"},
        "pull_request": {
            "number": 72,
            "state": "closed" if action == "closed" else "open",
            "merged": merged,
            "draft": False,
            "head": {"ref": "codex/example", "sha": "abc123"},
            "labels": [{"name": label} for label in (labels or [])],
        },
        "sender": {"login": "felipe"},
    }


def _pr_state(**overrides):
    return {
        "repo": "Felippen/Oryn",
        "pr_number": 72,
        "branch": "codex/example",
        "head_sha": "abc123",
        "ci_state": "success",
        "ci_status": {"state": "success", "warnings": []},
        "review_state": "open",
        "mergeable": True,
        "merge_state": "clean",
        "warnings": [],
        "raw": {},
        **overrides,
    }


def test_github_webhook_signature_validation():
    body = b'{"ok":true}'
    secret = "webhook-secret"

    assert verify_github_signature(body=body, signature_header=_signature(body, secret), secret=secret)
    assert not verify_github_signature(body=body, signature_header=_signature(b"{}", secret), secret=secret)
    assert not verify_github_signature(body=body, signature_header="", secret=secret)


def test_duplicate_delivery_is_idempotent(tmp_path, monkeypatch):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")
    monkeypatch.setattr(
        "gateway.dev_control.github_pr_automation.fetch_github_review_gate",
        lambda repo, pr_number: {"verdict": "approved"},
    )

    result = process_github_webhook(
        store=store,
        scm_store=scm_store,
        delivery_id="delivery-1",
        event_type="pull_request",
        payload=_pr_payload(),
        pr_state_fetcher=lambda **kwargs: _pr_state(),
        command_runner=lambda command, timeout=60: {"exit_code": 0, "output": "ok"},
    )
    duplicate = process_github_webhook(
        store=store,
        scm_store=scm_store,
        delivery_id="delivery-1",
        event_type="pull_request",
        payload=_pr_payload(),
        pr_state_fetcher=lambda **kwargs: _pr_state(),
    )

    assert result["event"]["delivery_id"] == "delivery-1"
    assert duplicate["duplicate"] is True


def test_unlabeled_pr_observes_and_requests_copilot_only(tmp_path, monkeypatch):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")
    monkeypatch.setattr(
        "gateway.dev_control.github_pr_automation.fetch_github_review_gate",
        lambda repo, pr_number: {"verdict": "approved"},
    )

    result = process_github_webhook(
        store=store,
        scm_store=scm_store,
        delivery_id="delivery-2",
        event_type="pull_request",
        payload=_pr_payload(labels=[]),
        pr_state_fetcher=lambda **kwargs: _pr_state(),
        command_runner=lambda command, timeout=60: {"exit_code": 0, "output": "ok"},
    )

    assert [action["action"] for action in result["actions"]] == ["request_copilot_review"]
    assert result["readiness"]["ready"] is True


def test_auto_fix_requires_label_and_launches_ao(tmp_path, monkeypatch):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")
    spawned = []

    class Router:
        def spawn(self, runtime, **kwargs):
            spawned.append({"runtime": runtime, "kwargs": kwargs})
            return type("Session", (), {"id": "session-1"})()

    monkeypatch.setattr(
        "gateway.dev_control.github_pr_automation.fetch_github_review_gate",
        lambda repo, pr_number: {"verdict": "approved"},
    )

    result = process_github_webhook(
        store=store,
        scm_store=scm_store,
        delivery_id="delivery-3",
        event_type="pull_request_review_comment",
        payload=_pr_payload(labels=[AUTO_FIX_LABEL]),
        pr_state_fetcher=lambda **kwargs: _pr_state(),
        command_runner=lambda command, timeout=60: {
            "exit_code": 0,
            "output": "apps/oryn-workspace/Sources/App.swift\napps/oryn-workspace/Tests/AppTests.swift\n",
        },
        router=Router(),
    )

    assert any(action["action"] == "fix_review_comments" for action in result["actions"])
    assert spawned
    assert spawned[0]["kwargs"]["branch"] == "codex/example"
    assert "Do not merge, release, publish" in spawned[0]["kwargs"]["prompt"]


def test_auto_fix_attempt_limit_goes_to_needs_human(tmp_path, monkeypatch):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")
    for _ in range(2):
        store.record_run({
            "repo": "Felippen/Oryn",
            "pr_number": 72,
            "head_sha": "abc123",
            "action": "fix_ci",
            "status": "launched",
            "labels": [AUTO_FIX_LABEL],
            "result": {},
        })
    monkeypatch.setattr(
        "gateway.dev_control.github_pr_automation.fetch_github_review_gate",
        lambda repo, pr_number: {"verdict": "approved"},
    )

    result = process_github_webhook(
        store=store,
        scm_store=scm_store,
        delivery_id="delivery-4",
        event_type="pull_request_review_comment",
        payload=_pr_payload(labels=[AUTO_FIX_LABEL]),
        pr_state_fetcher=lambda **kwargs: _pr_state(),
    )

    fix = [action for action in result["actions"] if action["action"] == "fix_ci"][0]
    assert fix["status"] == "needs_human"
    assert "Fix attempt limit" in fix["reason"]


def test_merge_readiness_without_linked_plan_does_not_block_on_draft():
    readiness = compose_merge_readiness(
        repo="Felippen/Oryn",
        pr_number=72,
        pr_state=_pr_state(),
        draft_status=None,
        verification={"verdict": "verified"},
        code_review={"verdict": "approved"},
    )

    assert readiness["gates"]["draft"]["state"] == "not_applicable"
    assert readiness["ready"] is True


def test_auto_merge_refuses_when_branch_protection_env_missing(tmp_path, monkeypatch):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")
    monkeypatch.delenv("HERMES_DEV_MERGE_EXECUTOR_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_DEV_BRANCH_PROTECTION_CONFIRMED", raising=False)
    monkeypatch.setattr(
        "gateway.dev_control.github_pr_automation.fetch_github_review_gate",
        lambda repo, pr_number: {"verdict": "approved"},
    )

    result = process_github_webhook(
        store=store,
        scm_store=scm_store,
        delivery_id="delivery-5",
        event_type="pull_request",
        payload=_pr_payload(labels=[AUTO_MERGE_LABEL]),
        pr_state_fetcher=lambda **kwargs: _pr_state(),
        command_runner=lambda command, timeout=60: {"exit_code": 0, "output": "ok"},
    )

    merge = [action for action in result["actions"] if action["action"] == "merge"][0]
    assert merge["status"] == "needs_human"
    assert "disabled" in " ".join(merge["warnings"])


def test_auto_release_only_for_oryn_merged_pr_with_label(tmp_path):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")

    result = process_github_webhook(
        store=store,
        scm_store=scm_store,
        delivery_id="delivery-6",
        event_type="pull_request",
        payload=_pr_payload(labels=[AUTO_RELEASE_LABEL], action="closed", merged=True),
        pr_state_fetcher=lambda **kwargs: _pr_state(),
        release_dispatcher=lambda **kwargs: {"ok": True, "version": "0.5.35", "build": 74},
    )

    release = [action for action in result["actions"] if action["action"] == "release"][0]
    assert release["status"] == "completed"
    assert release["result"]["version"] == "0.5.35"


def test_next_patch_version():
    assert next_patch_version("0.5.34") == "0.5.35"


def test_fix_prompt_contains_allowed_scope():
    prompt = build_pr_fix_prompt(
        repo="Felippen/Oryn",
        pr_number=72,
        pr_state=_pr_state(),
        reason="CI failed.",
        allowed_files=["apps/oryn-workspace/Sources/App.swift"],
    )

    assert "apps/oryn-workspace/Sources/App.swift" in prompt
    assert "If a correct fix requires any other file, stop" in prompt


def test_polling_reconciler_observes_unlabeled_pr_without_mutation(tmp_path, monkeypatch):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")
    commands = []

    def runner(command, timeout=60):
        commands.append(command)
        if command[:3] == ["gh", "pr", "list"] and "--state" in command:
            state = command[command.index("--state") + 1]
            if state == "open":
                return {
                    "exit_code": 0,
                    "output": json.dumps([{
                        "number": 72,
                        "isDraft": False,
                        "labels": [],
                        "headRefName": "codex/example",
                        "headRefOid": "abc123",
                        "statusCheckRollup": [],
                    }]),
                }
            return {"exit_code": 0, "output": "[]"}
        if command[:3] == ["gh", "pr", "edit"]:
            return {"exit_code": 0, "output": "review requested"}
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        "gateway.dev_control.github_pr_automation.fetch_github_review_gate",
        lambda repo, pr_number: {"verdict": "unknown"},
    )

    result = reconcile_github_pr_automation(
        store=store,
        scm_store=scm_store,
        repos=["Felippen/Oryn"],
        command_runner=runner,
        pr_state_fetcher=lambda **kwargs: _pr_state(),
    )

    assert result["action_count"] == 0
    assert not any(command[:3] == ["gh", "pr", "edit"] for command in commands)


def test_polling_reconciler_requests_copilot_once_for_trusted_label(tmp_path, monkeypatch):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")
    commands = []

    def runner(command, timeout=60):
        commands.append(command)
        if command[:3] == ["gh", "pr", "list"] and "--state" in command:
            state = command[command.index("--state") + 1]
            if state == "open":
                return {
                    "exit_code": 0,
                    "output": json.dumps([{
                        "number": 72,
                        "isDraft": False,
                        "labels": [{"name": AUTO_MERGE_LABEL}],
                        "headRefName": "codex/example",
                        "headRefOid": "abc123",
                        "statusCheckRollup": [],
                    }]),
                }
            return {"exit_code": 0, "output": "[]"}
        if command[:3] == ["gh", "pr", "edit"]:
            return {"exit_code": 0, "output": "review requested"}
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        "gateway.dev_control.github_pr_automation.fetch_github_review_gate",
        lambda repo, pr_number: {"verdict": "unknown"},
    )

    first = reconcile_github_pr_automation(
        store=store,
        scm_store=scm_store,
        repos=["Felippen/Oryn"],
        command_runner=runner,
        pr_state_fetcher=lambda **kwargs: _pr_state(),
    )
    second = reconcile_github_pr_automation(
        store=store,
        scm_store=scm_store,
        repos=["Felippen/Oryn"],
        command_runner=runner,
        pr_state_fetcher=lambda **kwargs: _pr_state(),
    )

    assert first["action_count"] == 1
    assert first["repos"][0]["actions"][0]["action"] == "request_copilot_review"
    assert second["action_count"] == 0
    assert sum(1 for command in commands if command[:3] == ["gh", "pr", "edit"]) == 1


def test_polling_reconciler_auto_fix_uses_label_and_failed_checks(tmp_path, monkeypatch):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")
    spawned = []

    class Router:
        def spawn(self, runtime, **kwargs):
            spawned.append({"runtime": runtime, "kwargs": kwargs})
            return type("Session", (), {"id": "session-poll"})()

    def runner(command, timeout=60):
        if command[:3] == ["gh", "pr", "list"] and "--state" in command:
            state = command[command.index("--state") + 1]
            if state == "open":
                return {
                    "exit_code": 0,
                    "output": json.dumps([{
                        "number": 72,
                        "isDraft": False,
                        "labels": [{"name": AUTO_FIX_LABEL}],
                        "headRefName": "codex/example",
                        "headRefOid": "abc123",
                        "statusCheckRollup": [{"name": "test", "conclusion": "FAILURE"}],
                    }]),
                }
            return {"exit_code": 0, "output": "[]"}
        if command[:4] == ["gh", "pr", "diff", "72"]:
            return {"exit_code": 0, "output": "apps/oryn-workspace/Sources/App.swift\n"}
        if command[:3] == ["gh", "pr", "edit"]:
            return {"exit_code": 0, "output": "review requested"}
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        "gateway.dev_control.github_pr_automation.fetch_github_review_gate",
        lambda repo, pr_number: {"verdict": "approved"},
    )

    result = reconcile_github_pr_automation(
        store=store,
        scm_store=scm_store,
        repos=["Felippen/Oryn"],
        command_runner=runner,
        pr_state_fetcher=lambda **kwargs: _pr_state(),
        router=Router(),
    )

    actions = result["repos"][0]["actions"]
    assert [action["action"] for action in actions] == ["fix_ci"]
    assert spawned
    assert spawned[0]["kwargs"]["branch"] == "codex/example"


def test_polling_reconciler_releases_merged_oryn_pr_once(tmp_path):
    store = DevGitHubPRAutomationStore(tmp_path / "state.db")
    scm_store = DevSCMLifecycleStore(tmp_path / "state.db")
    releases = []

    def runner(command, timeout=60):
        if command[:3] == ["gh", "pr", "list"] and "--state" in command:
            state = command[command.index("--state") + 1]
            if state == "merged":
                return {
                    "exit_code": 0,
                    "output": json.dumps([{
                        "number": 72,
                        "labels": [{"name": AUTO_RELEASE_LABEL}],
                        "mergedAt": "2026-05-30T12:00:00Z",
                    }]),
                }
            return {"exit_code": 0, "output": "[]"}
        raise AssertionError(f"unexpected command: {command}")

    def release_dispatcher(**kwargs):
        releases.append(kwargs)
        return {"ok": True, "version": "0.5.35", "build": 74}

    first = reconcile_github_pr_automation(
        store=store,
        scm_store=scm_store,
        repos=["Felippen/Oryn"],
        command_runner=runner,
        release_dispatcher=release_dispatcher,
    )
    second = reconcile_github_pr_automation(
        store=store,
        scm_store=scm_store,
        repos=["Felippen/Oryn"],
        command_runner=runner,
        release_dispatcher=release_dispatcher,
    )

    assert first["action_count"] == 1
    assert first["repos"][0]["actions"][0]["action"] == "release"
    assert second["action_count"] == 0
    assert len(releases) == 1
