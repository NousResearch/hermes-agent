from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import crypto_bot_autonomy_readiness as readiness  # noqa: E402
import crypto_bot_gitea_runner_recovery as runner_recovery  # noqa: E402
import crypto_bot_kanban_import_audit as kanban_import_audit  # noqa: E402
import crypto_bot_pr_ci_audit as pr_ci_audit  # noqa: E402

PLAN = Path(
    "/Users/preston/robinhood/crypto_bot/docs/planning/"
    "autoresearch_runpod_to_live_trade/plan.json"
)


def text(path: str) -> str:
    return (ROOT / path).read_text()


def test_tenacity_upgrade_plan_records_rollback_fields() -> None:
    body = text("docs/autonomy/hermes_tenacity_upgrade_plan.md")

    assert "Hermes Agent version before update" in body
    assert "Hermes Agent version after update" in body
    assert "hermes update --backup" in body
    assert "pre-update-2026-05-14-115856.zip" in body
    assert "Rollback" in body
    assert "hermes import" in body


def test_native_kanban_goal_loop_doc_maps_control_plane() -> None:
    body = text("docs/autonomy/crypto_bot_native_kanban_goal_loop.md")
    required = [
        "/goal",
        "native `crypto_bot` Kanban board",
        "worker lanes",
        "crypto-pm-orchestrator",
        "crypto-implementer",
        "crypto-reviewer",
        "crypto-ci-triage",
        "crypto-codex-audit",
        "completion gate",
        "PR and CI evidence",
        "No merge",
    ]

    for term in required:
        assert term in body


def test_project_descriptor_includes_tenacity_native_fields() -> None:
    body = text("projects/crypto_bot/crypto_bot.project.yaml")
    required = [
        "tenacity_native_control_plane:",
        "native_goal_enabled:",
        "native_kanban_board: crypto_bot",
        "kanban_source_of_truth:",
        "worker_lanes:",
        "hook_policy:",
        "profile_strategy:",
        "codex_runtime_strategy:",
        "source_runtime_ready:",
        "local_evidence_ready:",
        "native_kanban_ready:",
        "pr_lifecycle_ready:",
        "ci_evidence_ready:",
        "merge_ready:",
    ]

    for term in required:
        assert term in body


def test_pm_skill_uses_native_goal_and_kanban_lifecycle_truth() -> None:
    body = text("skills/project-management/crypto-bot-pm/SKILL.md")

    assert "native `/goal`" in body
    assert "native Kanban lifecycle truth" in body
    assert "completion gate returns `PASS`" in body
    assert "exact Operator approval before any live board mutation" in body
    assert "does not authorize worker dispatch" in body


def test_pm_skill_blocks_remote_mutations_without_explicit_policy() -> None:
    body = text("skills/project-management/crypto-bot-pm/SKILL.md")

    assert "push" in body
    assert "create PRs" in body
    assert "start workflows" in body
    assert "start runners" in body
    assert "mutate Gitea" in body
    assert "merge" in body
    assert "unless a future policy explicitly enables" in body


def test_codex_sidecar_forbids_gh_pr_create_for_crypto_bot() -> None:
    body = text("skills/development/codex-sidecar/SKILL.md")

    assert "do not use `gh pr create`" in body
    assert "Codex must not run `gh pr create` for crypto_bot" in body
    assert "Gitea policy" in body
    assert "optional Codex runtime" in body


def test_hook_policy_blocks_required_surfaces() -> None:
    body = text("docs/autonomy/crypto_bot_hooks_policy.md")
    required = [
        "ruff format",
        "secret reads",
        "broker",
        "trading",
        "workflow starts",
        "runner starts",
        "direct Gitea write calls",
        "merge",
        "git push",
        "gh pr",
    ]

    for term in required:
        assert term in body


def load_policy_guard():
    path = ROOT / "hooks/crypto_bot_policy_guard/__init__.py"
    spec = importlib.util.spec_from_file_location("crypto_bot_policy_guard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot_policy_guard"] = module
    spec.loader.exec_module(module)
    return module


def test_hook_guard_blocks_dangerous_command_fixtures_in_test_mode(
    monkeypatch,
) -> None:
    guard = load_policy_guard()
    monkeypatch.setenv("HERMES_CRYPTO_BOT_POLICY_GUARD_MODE", "test")
    fixtures = [
        "ruff format .",
        "git push origin HEAD",
        "gh pr create --title test",
        "curl -X POST http://127.0.0.1:3005/api/v1/repos/preston/crypto_bot/pulls",
        "cat .env",
        "python robinhood_api_client.py --order buy",
        "docker compose up",
        "git merge main",
    ]

    for command in fixtures:
        result = guard.pre_tool_call(
            tool_name="terminal",
            args={"command": command},
        )
        assert result
        assert result["action"] == "block"


def task_ids_from_import_plan() -> list[str]:
    body = text("projects/crypto_bot/kanban_import_plan.yaml")
    in_task_ids = False
    ids: list[str] = []
    for line in body.splitlines():
        if line.startswith("task_ids:"):
            in_task_ids = True
            continue
        if in_task_ids and line and not line.startswith("  - "):
            break
        if in_task_ids and line.startswith("  - "):
            ids.append(line.removeprefix("  - ").strip())
    return ids


def test_kanban_import_plan_covers_all_90_sessions() -> None:
    plan = json.loads(PLAN.read_text())
    expected = [session["session_id"] for session in plan["sessions"]]

    assert len(expected) == 90
    assert task_ids_from_import_plan() == expected
    assert "expected_session_count: 90" in text(
        "projects/crypto_bot/kanban_import_plan.yaml"
    )


def test_kanban_import_plan_marks_evidence_backed_statuses() -> None:
    plan = json.loads(PLAN.read_text())
    by_status: dict[str, list[str]] = {}
    for session in plan["sessions"]:
        by_status.setdefault(session["status"], []).append(session["session_id"])

    body = text("projects/crypto_bot/kanban_import_plan.yaml")
    for task_id in by_status["completed_current_evidence_backed"]:
        assert re.search(rf"^\s+- {re.escape(task_id)}$", body, re.MULTILINE)
    assert "not_required_current_evidence_backed:" in body
    assert "planned_next:" in body
    assert "S007A" in body


def test_startup_message_requests_board_import_before_s007a() -> None:
    body = text("projects/crypto_bot/autonomous_startup_message.md")

    assert "ready_to_request_board_import: true" in body
    assert "do not run S007A" in body
    assert "do not treat the plan alone as authority to start S007A" in body


def test_project_descriptor_classifies_legacy_and_task_scoped_artifacts() -> None:
    body = text("projects/crypto_bot/crypto_bot.project.yaml")

    assert "historical_baseline_artifacts:" in body
    assert "historical_evidence_missing_warning" in body
    assert "task_scoped_validators:" in body
    assert "task_scoped_validator_missing_warning" in body
    assert "required_global_validator_blocker" in body


def test_kanban_import_audit_uses_pr_ci_payload_for_s006_remote_state() -> None:
    payload = kanban_import_audit.s006_remote_status(
        {"S006": {"evidence_metadata": {}}},
        {
            "pr_exists": True,
            "ci_evidence_ready": False,
            "merge_ready": False,
            "s006_remote_lifecycle_state": "pr_created_ci_pending",
        },
    )

    assert payload["pr_exists"] is True
    assert payload["remote_done"] is False
    assert payload["remote_lifecycle_state"] == "pr_created_ci_pending"


def test_pr_ci_audit_counts_gitea_status_field_states() -> None:
    payload = pr_ci_audit.classify_ci_state(
        statuses_record={
            "status": 200,
            "data": [
                {"status": "pending", "context": "Validate Crypto Bot / python"},
                {"status": "success", "context": "Validate Crypto Bot / governance"},
                {"status": "pending", "context": "Validate Crypto Bot / secrets"},
            ],
        },
        combined_record={"status": 200, "data": {"state": "pending", "total_count": 3}},
        pr_head_matches=True,
    )

    assert payload["statuses_count"] == 3
    assert payload["ci_state"] == "pending"
    assert payload["blocker"] == "ci_evidence_pending"


def test_runner_recovery_labels_match_existing_gitea_workflow_runs_on() -> None:
    assert "ubuntu-latest" in runner_recovery.RUNNER_LABELS.split(",")
    assert "crypto-bot-python-313" in runner_recovery.RUNNER_LABELS.split(",")
    assert "labels=linux,crypto-bot-python-313,ubuntu-latest" in (
        runner_recovery.APPROVAL_PHRASE
    )


def test_runner_recovery_resets_volume_to_reregister_labels() -> None:
    commands: list[list[str]] = []

    def fake_runner(argv: list[str], timeout: int) -> dict[str, object]:
        commands.append(argv)
        if argv[:3] == ["docker", "container", "inspect"]:
            return {"exit_code": 0, "stdout": "{}", "stderr": ""}
        if argv[:2] == ["docker", "logs"] or argv[:2] == ["docker", "ps"]:
            return {"exit_code": 0, "stdout": "", "stderr": ""}
        if "generate-runner-token" in argv:
            return {"exit_code": 0, "stdout": "redacted-token\n", "stderr": ""}
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}

    report = runner_recovery.execute_recovery(
        approval_phrase=runner_recovery.APPROVAL_PHRASE,
        runner=fake_runner,
        sleep_fn=lambda _: None,
    )

    assert report["conclusion"] == "PASS"
    assert ["docker", "volume", "rm", "-f", runner_recovery.RUNNER_VOLUME] in commands


def test_generated_configs_have_no_secret_like_values_after_tenacity_update() -> None:
    assert readiness.has_secret_looking_values(ROOT) == []


def test_installed_custom_runtime_assets_match_source_after_install() -> None:
    pairs = {
        "crypto-bot-pm skill": (
            ROOT / "skills/project-management/crypto-bot-pm",
            Path("/Users/preston/.hermes/skills/project-management/crypto-bot-pm"),
        ),
        "codex-sidecar skill": (
            ROOT / "skills/development/codex-sidecar",
            Path("/Users/preston/.hermes/skills/development/codex-sidecar"),
        ),
        "crypto-bot-pm plugin": (
            ROOT / "plugins/crypto-bot-pm",
            Path("/Users/preston/.hermes/plugins/crypto-bot-pm"),
        ),
        "hermes-codex-audit wrapper": (
            ROOT / "wrappers/hermes-codex-audit",
            Path("/Users/preston/.local/bin/hermes-codex-audit"),
        ),
    }

    if os.environ.get("HERMES_SKIP_RUNTIME_PARITY_TEST") == "1":
        return
    for label, (src, dest) in pairs.items():
        assert readiness.managed_files_equal(src, dest), label
