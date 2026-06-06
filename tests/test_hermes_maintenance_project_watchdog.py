from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import subprocess


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hermes_maintenance_project_watchdog.py"
spec = importlib.util.spec_from_file_location("project_watchdog", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
project_watchdog = importlib.util.module_from_spec(spec)
sys.modules["project_watchdog"] = project_watchdog
spec.loader.exec_module(project_watchdog)


def test_kanban_snapshot_flags_missing_board_and_dispatch_owner_metadata() -> None:
    snapshot = project_watchdog.KanbanSnapshot(
        db_readable=True,
        board_exists=False,
        board_slug="hermes-maintenance",
        dispatch_owner=None,
        tasks=[],
    )

    findings = project_watchdog.evaluate_kanban_health(snapshot)

    assert ("kanban.board_missing", "blocked") in [(finding.code, finding.status) for finding in findings]
    assert ("kanban.dispatch_owner_missing", "blocked") in [(finding.code, finding.status) for finding in findings]


def test_kanban_snapshot_classifies_non_human_waits_as_waiting_and_review_as_blocked() -> None:
    snapshot = project_watchdog.KanbanSnapshot(
        db_readable=True,
        board_exists=True,
        board_slug="hermes-maintenance",
        dispatch_owner="responsible_project_autonomy",
        tasks=[
            project_watchdog.KanbanTaskSnapshot(
                task_id="t_waiting",
                title="waiting dependency",
                status="todo",
                updated_at=100,
                pr_refs=[],
            ),
            project_watchdog.KanbanTaskSnapshot(
                task_id="t_review",
                title="review needed",
                status="blocked",
                updated_at=200,
                block_reason="review-required: inspect diff",
                pr_refs=["https://github.com/yunuenmf/hermes-agent/pull/7"],
            ),
            project_watchdog.KanbanTaskSnapshot(
                task_id="t_ready_stale",
                title="ready too long",
                status="ready",
                updated_at=0,
                pr_refs=[],
            ),
        ],
    )

    findings = project_watchdog.evaluate_kanban_health(snapshot, now=10_000, stale_seconds=300)
    by_code = {finding.code: finding for finding in findings}

    assert by_code["kanban.waiting_tasks"].status == "waiting"
    assert "t_waiting" in by_code["kanban.waiting_tasks"].details["task_ids"]
    assert by_code["kanban.review_required"].status == "blocked"
    assert by_code["kanban.stale_ready_tasks"].status == "waiting"
    assert "t_ready_stale" in by_code["kanban.stale_ready_tasks"].details["task_ids"]


def test_matrix_snapshot_reports_route_registry_strict_validator_and_degraded_fallback() -> None:
    snapshot = project_watchdog.MatrixSnapshot(
        project_slug="hermes-maintenance",
        coordinator_room_id="!coord:matrix.example",
        project_room_id=None,
        gateway_route_registered=False,
        profile_route_registered=True,
        strict_validator_ok=False,
        degraded_fallback="direct-profile-runner",
    )

    findings = project_watchdog.evaluate_matrix_health(snapshot)
    by_code = {finding.code: finding for finding in findings}

    assert by_code["matrix.project_room_missing"].status == "blocked"
    assert by_code["matrix.gateway_route_missing"].status == "blocked"
    assert by_code["matrix.strict_validator_failed"].status == "blocked"
    assert by_code["matrix.degraded_fallback_active"].status == "waiting"
    assert by_code["matrix.degraded_fallback_active"].details["fallback"] == "direct-profile-runner"


def test_github_snapshot_reports_remote_auth_branch_worktree_pr_and_kanban_drift() -> None:
    snapshot = project_watchdog.GitHubSnapshot(
        repo_path="/repo/hermes-agent",
        remote_url="https://github.com/yunuenmf/hermes-agent.git",
        remote_reachable=False,
        auth_ok=False,
        branch_name="downstream/watchdog-live-health-adapters",
        branch_upstream="yunuenmf/feature/gateway-fragmentation-finalize-reset",
        worktree_clean=False,
        open_prs=[
            project_watchdog.PullRequestSnapshot(number=7, state="open", head_branch="other-branch", linked_task_ids=["t_other"]),
        ],
        kanban_task_id="t_28a15448",
    )

    findings = project_watchdog.evaluate_github_health(snapshot)
    by_code = {finding.code: finding for finding in findings}

    assert by_code["github.remote_unreachable"].status == "blocked"
    assert by_code["github.auth_failed"].status == "blocked"
    assert by_code["github.worktree_dirty"].status == "waiting"
    assert by_code["github.pr_task_drift"].status == "waiting"
    assert by_code["github.pr_branch_drift"].details["pr_numbers"] == [7]


def test_project_health_summary_picks_most_severe_status_deterministically() -> None:
    snapshot = project_watchdog.ProjectHealthSnapshot(
        project_slug="hermes-maintenance",
        kanban=project_watchdog.KanbanSnapshot(
            db_readable=True,
            board_exists=True,
            board_slug="hermes-maintenance",
            dispatch_owner="responsible_project_autonomy",
            tasks=[],
        ),
        matrix=project_watchdog.MatrixSnapshot(
            project_slug="hermes-maintenance",
            coordinator_room_id="!coord:matrix.example",
            project_room_id="!project:matrix.example",
            gateway_route_registered=True,
            profile_route_registered=True,
            strict_validator_ok=True,
        ),
        github=project_watchdog.GitHubSnapshot(
            repo_path="/repo/hermes-agent",
            remote_url="https://github.com/yunuenmf/hermes-agent.git",
            remote_reachable=True,
            auth_ok=True,
            branch_name="downstream/watchdog-live-health-adapters",
            branch_upstream="yunuenmf/feature/gateway-fragmentation-finalize-reset",
            worktree_clean=True,
            open_prs=[],
            kanban_task_id="t_28a15448",
        ),
    )

    summary = project_watchdog.evaluate_project_health(snapshot)

    assert summary.status == "working"
    assert summary.project_slug == "hermes-maintenance"
    assert summary.findings == []


def test_cli_snapshot_loader_rehydrates_nested_dataclasses(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        """
        {
          "project_slug": "hermes-maintenance",
          "kanban": {
            "db_readable": true,
            "board_exists": true,
            "board_slug": "hermes-maintenance",
            "dispatch_owner": "coordinator_hermes_maintenance",
            "tasks": [{"task_id": "t_waiting", "title": "dependency", "status": "todo", "updated_at": 100, "block_reason": null, "pr_refs": []}]
          },
          "matrix": {
            "project_slug": "hermes-maintenance",
            "coordinator_room_id": "!coord:matrix.example",
            "project_room_id": "!project:matrix.example",
            "gateway_route_registered": true,
            "profile_route_registered": true,
            "strict_validator_ok": true
          },
          "github": {
            "repo_path": "/repo/hermes-agent",
            "remote_url": "https://github.com/yunuenmf/hermes-agent.git",
            "remote_reachable": true,
            "auth_ok": true,
            "branch_name": "downstream/watchdog-live-health-adapters",
            "branch_upstream": "yunuenmf/feature/gateway-fragmentation-finalize-reset",
            "worktree_clean": true,
            "open_prs": [{"number": 7, "state": "open", "head_branch": "downstream/watchdog-live-health-adapters", "linked_task_ids": ["t_28a15448"]}],
            "kanban_task_id": "t_28a15448"
          }
        }
        """,
        encoding="utf-8",
    )

    assert project_watchdog.main(["--snapshot-json", str(snapshot_path)]) == 0


def test_matrix_registry_adapter_uses_injected_registry_without_live_matrix() -> None:
    snapshot = project_watchdog.load_matrix_snapshot_from_registry(
        "hermes-maintenance",
        {
            "path": "/tmp/routes.json",
            "hermes-maintenance": {
                "coordinator_room_id": "!coord:matrix.example",
                "project_room_id": "!project:matrix.example",
            },
            "routes": {"hermes-maintenance": "!project:matrix.example"},
            "profile_routes": {"hermes-maintenance": "!project:matrix.example"},
        },
        strict_validator_ok=True,
    )

    assert snapshot.project_room_id == "!project:matrix.example"
    assert snapshot.gateway_route_registered is True
    assert snapshot.profile_route_registered is True
    assert project_watchdog.evaluate_matrix_health(snapshot) == []


class FakeGitClient:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, args: tuple[str, ...], cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        outputs = {
            ("git", "branch", "--show-current"): (0, "downstream/watchdog-live-health-adapters\n", ""),
            ("git", "remote", "get-url", "--push", "origin"): (0, "https://github.com/yunuenmf/hermes-agent.git\n", ""),
            ("git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): (0, "yunuenmf/feature/gateway-fragmentation-finalize-reset\n", ""),
            ("git", "status", "--porcelain"): (0, "", ""),
            ("git", "ls-remote", "--exit-code", "origin", "HEAD"): (0, "ref\n", ""),
            ("gh", "auth", "status"): (0, "ok\n", ""),
            ("gh", "pr", "list", "--state", "open", "--json", "number,state,headRefName,body,title"): (
                0,
                '[{"number": 8, "state": "OPEN", "headRefName": "downstream/watchdog-live-health-adapters", "title": "watchdog t_28a15448", "body": "links t_28a15448"}]',
                "",
            ),
        }
        returncode, stdout, stderr = outputs[args]
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def test_github_adapter_uses_injected_command_client_without_live_credentials(tmp_path: Path) -> None:
    client = FakeGitClient()

    snapshot = project_watchdog.load_github_snapshot(tmp_path, kanban_task_id="t_28a15448", client=client)

    assert snapshot.remote_reachable is True
    assert snapshot.auth_ok is True
    assert snapshot.worktree_clean is True
    assert snapshot.open_prs[0].linked_task_ids == ["t_28a15448"]
    assert project_watchdog.evaluate_github_health(snapshot) == []
