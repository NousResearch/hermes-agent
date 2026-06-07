from __future__ import annotations

import importlib.util
import json
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "multi_ai_workflow.py"
CHECK_PATH = Path(__file__).resolve().parents[2] / "scripts" / "multi_ai_workflow_check.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cli():
    return _load_script(SCRIPT_PATH, "_multi_ai_workflow")


def _load_check():
    return _load_script(CHECK_PATH, "_multi_ai_workflow_check_for_cli_tests")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _init_git_repo(root: Path) -> None:
    assert _run(["git", "init"], root).returncode == 0
    assert _run(["git", "config", "user.email", "test@example.com"], root).returncode == 0
    assert _run(["git", "config", "user.name", "Test User"], root).returncode == 0
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    assert _run(["git", "add", "README.md"], root).returncode == 0
    assert _run(["git", "commit", "-m", "initial"], root).returncode == 0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_init_project_bootstraps_readiness_files(tmp_path):
    cli = _load_cli()
    check = _load_check()

    result = cli.init_project(tmp_path, force=False)

    assert result["created_count"] >= 10
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / "QWEN.md").exists()
    assert (tmp_path / "GEMINI.md").exists()
    assert (tmp_path / ".cursor" / "rules" / "multi-ai-workflow.mdc").exists()
    assert (tmp_path / ".hermes" / "issues" / "README.md").exists()
    assert (tmp_path / ".hermes" / "plans" / "README.md").exists()
    assert (tmp_path / "docs" / "multi-ai-workflow" / "templates" / "opus-plan.md").exists()
    assert check.inspect_project(tmp_path)["ok"] is True


def test_create_issue_writes_complete_issue_file(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)

    issue_path = cli.create_issue(
        project=tmp_path,
        issue_id="phase-001-worktree-cli",
        phase="P1",
        title="Add worktree CLI",
        owner_role="AI Workflow Architect",
        assigned_ai="Codex",
        goal="Create a tested workflow CLI.",
        scope="CLI commands for issue creation and comply reporting.",
        out_of_scope="Dashboard and MCP server.",
        verify_commands=["pytest tests/scripts/test_multi_ai_workflow_cli.py -q"],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="ai/phase-001-worktree-cli",
        worktree_path="../worktrees/hermes-agent/phase-001-codex",
        reviewer_ai="Claude Code",
        force=False,
    )

    text = issue_path.read_text(encoding="utf-8")
    assert "issue_id: phase-001-worktree-cli" in text
    assert "assigned_ai: Codex" in text
    assert "reviewer_ai: Claude Code" in text
    assert "done_percent: 0" in text
    assert "remaining_percent: 100" in text
    assert "pytest tests/scripts/test_multi_ai_workflow_cli.py -q" in text


def test_claim_issue_updates_assignment_and_status(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_issue(
        project=tmp_path,
        issue_id="phase-002-review",
        phase="P2",
        title="Review workflow",
        owner_role="Reviewer",
        assigned_ai="unassigned",
        goal="Review the implementation.",
        scope="Review only.",
        out_of_scope="Implementation.",
        verify_commands=[],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="",
        worktree_path="",
        reviewer_ai="",
        force=False,
    )

    issue_path = cli.claim_issue(
        project=tmp_path,
        issue_id="phase-002-review",
        assigned_ai="Codex",
        branch="ai/phase-002-review",
        worktree_path="../worktrees/hermes-agent/phase-002-codex",
    )

    text = issue_path.read_text(encoding="utf-8")
    assert "assigned_ai: Codex" in text
    assert "branch: ai/phase-002-review" in text
    assert "worktree_path: ../worktrees/hermes-agent/phase-002-codex" in text
    assert "status: claimed" in text


def test_comply_report_summarizes_issue_percentages(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    first = cli.create_issue(
        project=tmp_path,
        issue_id="phase-001-a",
        phase="P1",
        title="A",
        owner_role="Implementer",
        assigned_ai="Codex",
        goal="A",
        scope="A",
        out_of_scope="B",
        verify_commands=[],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="",
        worktree_path="",
        reviewer_ai="",
        force=False,
    )
    second = cli.create_issue(
        project=tmp_path,
        issue_id="phase-001-b",
        phase="P1",
        title="B",
        owner_role="Reviewer",
        assigned_ai="Claude Code",
        goal="B",
        scope="B",
        out_of_scope="A",
        verify_commands=[],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="",
        worktree_path="",
        reviewer_ai="",
        force=False,
    )
    cli.update_issue_fields(first, {"done_percent": "100", "remaining_percent": "0", "status": "verified"})
    cli.update_issue_fields(second, {"done_percent": "50", "remaining_percent": "50", "status": "in_progress"})

    report = cli.build_comply_report(tmp_path)

    assert report["summary"]["issue_count"] == 2
    assert report["summary"]["done_percent"] == 75.0
    assert report["summary"]["remaining_percent"] == 25.0
    assert [issue["issue_id"] for issue in report["issues"]] == ["phase-001-a", "phase-001-b"]


def test_update_issue_status_records_evidence(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_issue(
        project=tmp_path,
        issue_id="phase-003-verify",
        phase="P3",
        title="Verify workflow",
        owner_role="Verification",
        assigned_ai="Codex",
        goal="Verify commands.",
        scope="Verification only.",
        out_of_scope="Runtime deploy.",
        verify_commands=[],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="",
        worktree_path="",
        reviewer_ai="",
        force=False,
    )

    issue_path = cli.update_issue_status(
        project=tmp_path,
        issue_id="phase-003-verify",
        status="verified",
        done_percent="100",
        remaining_percent="0",
        evidence="pytest passed 5/5",
    )

    text = issue_path.read_text(encoding="utf-8")
    assert "status: verified" in text
    assert "done_percent: 100" in text
    assert "remaining_percent: 0" in text
    assert "evidence: pytest passed 5/5" in text


def test_create_worktree_executes_git_worktree_and_claims_issue(tmp_path):
    cli = _load_cli()
    _init_git_repo(tmp_path)
    cli.init_project(tmp_path, force=False)
    cli.create_issue(
        project=tmp_path,
        issue_id="phase-004-worktree",
        phase="P4",
        title="Create worktree",
        owner_role="Implementer",
        assigned_ai="unassigned",
        goal="Create an isolated workspace.",
        scope="Worktree only.",
        out_of_scope="Deployment.",
        verify_commands=[],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="",
        worktree_path="",
        reviewer_ai="",
        force=False,
    )
    worktree_path = tmp_path.parent / "demo-worktrees" / "phase-004-codex"

    result = cli.create_worktree(
        project=tmp_path,
        issue_id="phase-004-worktree",
        assigned_ai="Codex",
        branch="ai/phase-004-worktree",
        worktree_path=worktree_path,
        execute=True,
    )

    assert result["executed"] is True
    assert result["returncode"] == 0
    assert (worktree_path / "README.md").exists()
    issue_text = (tmp_path / ".hermes" / "issues" / "phase-004-worktree.md").read_text(encoding="utf-8")
    assert "status: claimed" in issue_text
    assert f"worktree_path: {worktree_path}" in issue_text


def test_github_issue_sync_dry_run_builds_gh_command(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_issue(
        project=tmp_path,
        issue_id="phase-005-github",
        phase="P5",
        title="Sync to GitHub",
        owner_role="AI Workflow Architect",
        assigned_ai="Codex",
        goal="Prepare GitHub issue command.",
        scope="Dry-run command only.",
        out_of_scope="Network calls.",
        verify_commands=["pytest"],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="ai/phase-005-github",
        worktree_path="../worktrees/demo",
        reviewer_ai="Owner",
        force=False,
    )

    result = cli.github_issue_sync(
        project=tmp_path,
        issue_id="phase-005-github",
        execute=False,
    )

    assert result["executed"] is False
    assert result["command"][:3] == ["gh", "issue", "create"]
    assert "--title" in result["command"]
    assert "Sync to GitHub" in result["command"]
    assert "issue_id: phase-005-github" in result["body"]


def test_write_handoff_updates_project_handoff(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)

    handoff_path = cli.write_handoff(
        project=tmp_path,
        task="Continue implementation",
        issue_id="phase-006-handoff",
        phase="P6",
        latest_state="Implementation verified.",
        next_agent="Reviewer",
        next_step="Review the diff.",
        verification_run="tests passed",
        localhost_result="not applicable",
        vps_result="not applicable",
        remaining_risk="none",
    )

    text = handoff_path.read_text(encoding="utf-8")
    assert "task: Continue implementation" in text
    assert "issue_id: phase-006-handoff" in text
    assert "verification_run: tests passed" in text


def test_status_payload_includes_readiness_and_comply(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_issue(
        project=tmp_path,
        issue_id="phase-007-status",
        phase="P7",
        title="Status",
        owner_role="Verification",
        assigned_ai="Codex",
        goal="Expose status.",
        scope="Read-only status.",
        out_of_scope="Mutation.",
        verify_commands=[],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="",
        worktree_path="",
        reviewer_ai="",
        force=False,
    )

    payload = cli.build_status_payload(tmp_path)

    assert payload["readiness"]["ok"] is True
    assert payload["comply"]["summary"]["issue_count"] == 1
    assert payload["comply"]["issues"][0]["issue_id"] == "phase-007-status"


def test_status_server_serves_health_and_comply(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_issue(
        project=tmp_path,
        issue_id="phase-008-server",
        phase="P8",
        title="Server",
        owner_role="Verification",
        assigned_ai="Codex",
        goal="Serve status.",
        scope="Localhost only.",
        out_of_scope="Production.",
        verify_commands=[],
        localhost_check="GET /health",
        vps_check="not applicable",
        branch="",
        worktree_path="",
        reviewer_ai="",
        force=False,
    )
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "serve",
            "--project",
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 5
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                    health = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:  # pragma: no cover - timing dependent
                last_error = exc
                time.sleep(0.1)
        else:
            raise AssertionError(f"server did not start: {last_error}")

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/comply", timeout=1) as response:
            comply = json.loads(response.read().decode("utf-8"))

        assert health["ok"] is True
        assert comply["summary"]["issue_count"] == 1
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_route_plan_recommends_codex_for_python_cli_tests(tmp_path):
    cli = _load_cli()
    plan = """
planner_ai: Opus 4.8
task: Implement Python CLI with pytest coverage, git worktree safety, and code review.
needs: backend, tests, scripts, repository changes
"""

    result = cli.recommend_executor(plan)

    assert result["primary"]["id"] == "codex_app"
    assert result["primary"]["tool"] == "Codex App"
    assert result["primary"]["suitability_percent"] == 100.0
    assert "pytest" in " ".join(result["primary"]["matched_signals"]).lower()


def test_route_plan_recommends_qwen_for_cursor_frontend_work(tmp_path):
    cli = _load_cli()
    plan = """
planner_ai: Opus 4.8
task: Update React TypeScript UI components in Cursor, adjust styling, and polish forms.
needs: frontend, cursor, component editing
"""

    result = cli.recommend_executor(plan)

    assert result["primary"]["id"] == "qwen_cursor"
    assert result["primary"]["tool"] == "Qwen on Cursor"


def test_route_plan_recommends_gemini_for_antigravity_browser_work(tmp_path):
    cli = _load_cli()
    plan = """
planner_ai: Opus 4.8
task: Use Antigravity to inspect browser behavior, compare app flows, and validate UX.
needs: browser exploration, multimodal review, large-context analysis
"""

    result = cli.recommend_executor(plan)

    assert result["primary"]["id"] == "gemini_antigravity"
    assert result["primary"]["tool"] == "Gemini on Antigravity"


def test_route_plan_file_writes_recommendation(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    plan_path = tmp_path / ".hermes" / "plans" / "opus-phase-009.md"
    plan_path.write_text(
        "planner_ai: Opus 4.8\n"
        "task: Build Python scripts with pytest tests and review git changes.\n",
        encoding="utf-8",
    )

    result = cli.route_plan_file(project=tmp_path, plan_file=plan_path, write=True)

    assert result["recommendation"]["primary"]["id"] == "codex_app"
    output_path = Path(result["output_path"])
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["recommendation"]["primary"]["id"] == "codex_app"
    assert "Prompt for Codex App" in written["handoff_prompt"]


def test_complete_issue_creates_opus_review_request(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_issue(
        project=tmp_path,
        issue_id="phase-009-opus-review",
        phase="P9",
        title="Return to Opus",
        owner_role="Implementer",
        assigned_ai="Codex",
        goal="Finish work and request Opus review.",
        scope="Review request.",
        out_of_scope="External notification.",
        verify_commands=[],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="ai/phase-009-opus-review",
        worktree_path="../worktrees/demo",
        reviewer_ai="Opus 4.8",
        force=False,
    )

    result = cli.complete_issue_for_review(
        project=tmp_path,
        issue_id="phase-009-opus-review",
        completed_by="Codex App",
        evidence="tests passed 18/18",
        review_ai="Opus 4.8",
    )

    issue_text = (tmp_path / ".hermes" / "issues" / "phase-009-opus-review.md").read_text(encoding="utf-8")
    request_path = Path(result["review_request_path"])
    request_text = request_path.read_text(encoding="utf-8")
    assert "status: ready_for_opus_review" in issue_text
    assert "done_percent: 100" in issue_text
    assert "remaining_percent: 0" in issue_text
    assert request_path.exists()
    assert "review_ai: Opus 4.8" in request_text
    assert "owner_notification:" in request_text


def test_status_payload_includes_review_requests(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    review_dir = tmp_path / ".hermes" / "review-requests"
    review_dir.mkdir(parents=True)
    (review_dir / "phase-010.md").write_text(
        "issue_id: phase-010\nreview_ai: Opus 4.8\nstatus: pending\n",
        encoding="utf-8",
    )

    payload = cli.build_status_payload(tmp_path)

    assert payload["review_requests"]["pending_count"] == 1
    assert payload["review_requests"]["items"][0]["issue_id"] == "phase-010"


def test_compact_workflow_archives_completed_files(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    issue_path = cli.create_issue(
        project=tmp_path,
        issue_id="phase-011-archive",
        phase="P11",
        title="Archive old files",
        owner_role="Context Keeper",
        assigned_ai="Codex",
        goal="Archive completed workflow files.",
        scope="Archive only.",
        out_of_scope="Active work.",
        verify_commands=[],
        localhost_check="not applicable",
        vps_check="not applicable",
        branch="",
        worktree_path="",
        reviewer_ai="Opus 4.8",
        force=False,
    )
    cli.update_issue_status(
        project=tmp_path,
        issue_id="phase-011-archive",
        status="verified",
        done_percent="100",
        remaining_percent="0",
        evidence="done",
    )
    plan_path = tmp_path / ".hermes" / "plans" / "old-plan.md"
    route_path = tmp_path / ".hermes" / "routes" / "old-plan.json"
    plan_path.write_text("planner_ai: Opus 4.8\n", encoding="utf-8")
    route_path.write_text('{"ok": true}\n', encoding="utf-8")

    dry_run = cli.compact_workflow(project=tmp_path, execute=False)
    result = cli.compact_workflow(project=tmp_path, execute=True)

    assert dry_run["executed"] is False
    assert result["executed"] is True
    assert result["archived_count"] == 3
    archive_path = Path(result["archive_path"])
    assert archive_path.exists()
    archive_text = archive_path.read_text(encoding="utf-8")
    assert "phase-011-archive.md" in archive_text
    assert "old-plan.md" in archive_text
    assert "old-plan.json" in archive_text
    assert not issue_path.exists()
    assert not plan_path.exists()
    assert not route_path.exists()


def test_create_ai_pair_job_writes_state_and_templates(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)

    result = cli.create_ai_pair_job(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        task="Add Use AI Pair shortcut",
        coder_ai="Codecode",
        reviewer_ai="Codex",
        branch="ai-pair/use-ai-pair",
        gitlab_host="https://gitlab.dev.jigsawgroups.work/",
        force=False,
    )

    pair_dir = tmp_path / ".hermes" / "ai-pair" / "pair-001-use-ai-pair"
    assert result["pair_dir"] == str(pair_dir)
    assert result["status"] == "pair_selected"
    assert (pair_dir / "pair-state.json").exists()
    assert (pair_dir / "coder-plan.md").exists()
    assert (pair_dir / "coder-brief.md").exists()
    assert (pair_dir / "review-packet.md").exists()
    assert (pair_dir / "review-result.md").exists()
    assert (pair_dir / "gitlab-gate.md").exists()
    state = json.loads((pair_dir / "pair-state.json").read_text(encoding="utf-8"))
    assert state["coder_ai"] == "Codecode"
    assert state["reviewer_ai"] == "Codex"
    assert state["reviewer_mode"] == "read_only"
    assert state["gitlab_host"] == "https://gitlab.dev.jigsawgroups.work/"


def test_propose_ai_pair_branch_returns_safe_branch_for_clean_repo(tmp_path):
    cli = _load_cli()
    _init_git_repo(tmp_path)

    result = cli.propose_ai_pair_branch(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        task="Use AI Pair shortcut",
    )

    assert result["ok"] is True
    assert result["dirty"] is False
    assert result["branch"] == "ai-pair/pair-001-use-ai-pair"
    assert result["requires_owner_approval"] is True


def test_propose_ai_pair_branch_blocks_dirty_repo(tmp_path):
    cli = _load_cli()
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("# Demo\nchanged\n", encoding="utf-8")

    result = cli.propose_ai_pair_branch(
        project=tmp_path,
        issue_id="pair-002-dirty",
        task="Dirty repo task",
    )

    assert result["ok"] is False
    assert result["dirty"] is True
    assert "worktree has uncommitted changes" in result["reason"]


def test_ai_pair_cli_init_creates_job(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)

    result = _run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "ai-pair",
            "init",
            "--project",
            str(tmp_path),
            "--issue-id",
            "pair-001-use-ai-pair",
            "--task",
            "Add Use AI Pair",
            "--coder-ai",
            "Codecode",
            "--reviewer-ai",
            "Codex",
            "--branch",
            "ai-pair/pair-001-use-ai-pair",
            "--gitlab-host",
            "https://gitlab.dev.jigsawgroups.work/",
            "--format",
            "json",
        ],
        tmp_path,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "pair_selected"
    assert payload["coder_ai"] == "Codecode"
    assert payload["reviewer_ai"] == "Codex"


def test_ai_pair_cli_branch_proposal_is_json(tmp_path):
    _init_git_repo(tmp_path)

    result = _run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "ai-pair",
            "branch",
            "--project",
            str(tmp_path),
            "--issue-id",
            "pair-001-use-ai-pair",
            "--task",
            "Add Use AI Pair",
        ],
        tmp_path,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["branch"] == "ai-pair/pair-001-use-ai-pair"
    assert payload["requires_owner_approval"] is True


def test_ai_pair_run_blocks_when_coder_runtime_is_missing(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_ai_pair_job(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        task="Add Use AI Pair",
        coder_ai="Codecode",
        reviewer_ai="Codex",
        branch="ai-pair/pair-001-use-ai-pair",
        gitlab_host="https://gitlab.dev.jigsawgroups.work/",
        force=False,
    )

    result = cli.run_ai_pair_coder_plan(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        execute=True,
    )

    assert result["ok"] is False
    assert result["status"] == "blocked_missing_coder_runtime"
    assert result["required_env"] == "HERMES_AI_PAIR_CODECODE_COMMAND"
    pair_dir = tmp_path / ".hermes" / "ai-pair" / "pair-001-use-ai-pair"
    assert (pair_dir / "automation-blocker.md").exists()
    state = json.loads((pair_dir / "pair-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "blocked_missing_coder_runtime"


def test_ai_pair_run_executes_configured_coder_command(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_ai_pair_job(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        task="Add Use AI Pair",
        coder_ai="Codecode",
        reviewer_ai="Codex",
        branch="ai-pair/pair-001-use-ai-pair",
        gitlab_host="https://gitlab.dev.jigsawgroups.work/",
        force=False,
    )
    fake_coder = tmp_path / "fake_coder.py"
    fake_coder.write_text(
        "\n".join(
            [
                "import sys",
                "_ = sys.stdin.read()",
                "print('# Coder Plan')",
                "print('issue_id: pair-001-use-ai-pair')",
                "print('status: plan_ready_for_owner')",
                "print('approved_by_owner: no')",
            ]
        ),
        encoding="utf-8",
    )

    result = cli.run_ai_pair_coder_plan(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        execute=True,
        coder_command=f"{shlex.quote(sys.executable)} {shlex.quote(str(fake_coder))}",
    )

    assert result["ok"] is True
    assert result["status"] == "coder_plan_ready_for_owner"
    pair_dir = tmp_path / ".hermes" / "ai-pair" / "pair-001-use-ai-pair"
    assert "status: plan_ready_for_owner" in (pair_dir / "coder-plan.md").read_text(encoding="utf-8")
    state = json.loads((pair_dir / "pair-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "coder_plan_ready_for_owner"


def test_render_ai_pair_review_packet_includes_read_only_rules(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_ai_pair_job(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        task="Add Use AI Pair",
        coder_ai="Codecode",
        reviewer_ai="Codex",
        branch="ai-pair/pair-001-use-ai-pair",
        gitlab_host="https://gitlab.dev.jigsawgroups.work/",
        force=False,
    )
    pair_dir = tmp_path / ".hermes" / "ai-pair" / "pair-001-use-ai-pair"
    (pair_dir / "coder-plan.md").write_text(
        "# Plan\n\napproved_by_owner: yes\napproval_note: Owner approved test plan.\n",
        encoding="utf-8",
    )
    (pair_dir / "coder-brief.md").write_text(
        "\n".join(
            [
                "# Brief",
                "diff_summary: changed prompt registry",
                "files_changed: docs and scripts",
                "commands_run: pytest tests/scripts/test_multi_ai_workflow_cli.py -q",
                "results: passed",
                "review_focus: confirm read-only reviewer packet",
                "",
            ]
        ),
        encoding="utf-8",
    )

    packet_path = cli.render_ai_pair_review_packet(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        diff_summary="Modified prompt shortcut files only.",
        verification_evidence="pytest passed 22/22",
    )

    text = packet_path.read_text(encoding="utf-8")
    assert "reviewer_ai: Codex" in text
    assert "reviewer_mode: read_only" in text
    assert "Reviewer must not edit files." in text
    assert "Modified prompt shortcut files only." in text
    assert "pytest passed 22/22" in text


def test_render_ai_pair_review_packet_blocks_unapproved_plan_and_empty_brief(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_ai_pair_job(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        task="Add Use AI Pair",
        coder_ai="Codecode",
        reviewer_ai="Codex",
        branch="ai-pair/pair-001-use-ai-pair",
        gitlab_host="https://gitlab.dev.jigsawgroups.work/",
        force=False,
    )

    with pytest.raises(ValueError) as exc:
        cli.render_ai_pair_review_packet(
            project=tmp_path,
            issue_id="pair-001-use-ai-pair",
            diff_summary="Modified prompt shortcut files only.",
            verification_evidence="pytest passed 22/22",
        )

    message = str(exc.value)
    assert "approved_by_owner: yes" in message
    assert "coder-brief.md missing required value" in message
    state = json.loads(
        (tmp_path / ".hermes" / "ai-pair" / "pair-001-use-ai-pair" / "pair-state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "blocked_missing_review_gate"


def test_gitlab_gate_dry_run_never_prints_token(tmp_path):
    cli = _load_cli()
    cli.init_project(tmp_path, force=False)
    cli.create_ai_pair_job(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        task="Add Use AI Pair",
        coder_ai="Codecode",
        reviewer_ai="Codex",
        branch="ai-pair/pair-001-use-ai-pair",
        gitlab_host="https://gitlab.dev.jigsawgroups.work/",
        force=False,
    )

    result = cli.gitlab_gate_dry_run(
        project=tmp_path,
        issue_id="pair-001-use-ai-pair",
        project_path="tools/hermes-agent",
        merge_request_iid="7",
        token_env="GITLAB_TOKEN",
    )

    assert result["executed"] is False
    assert result["gitlab_host"] == "https://gitlab.dev.jigsawgroups.work/"
    assert result["project_path"] == "tools/hermes-agent"
    assert result["merge_request_iid"] == "7"
    rendered = json.dumps(result)
    assert "PRIVATE-TOKEN" not in rendered
    assert "token-value" not in rendered
    assert "GITLAB_TOKEN" in rendered
