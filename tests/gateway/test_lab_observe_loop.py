from __future__ import annotations

import json
import os
import pytest
import shutil
import subprocess
from pathlib import Path

from gateway.dev_control.dogfood_backlog import dogfood_scope_check
from gateway.dev_control.lab_process_isolation import audit_process_isolation
from gateway.dev_control.lab_loop import (
    DEFAULT_LAB_VERIFICATION_COMMAND,
    DevLabLoopStore,
    _await_implementation_terminal,
    _candidate_acceptance_criteria,
    _await_review_terminal,
    _normalize_lab_commit_author_for_ci,
    _prepare_lab_branch_base_for_publish,
    _recover_lab_verification_from_transcript,
    _touched_paths_from_worktree,
    _verification_timeout_seconds,
    _worker_session_usage_cost,
    finalize_pending_lab_ci_outcomes,
    loop_health,
    observe_profile_preflight,
    run_lab_observe_profile,
    run_lab_loop_pass,
)
from gateway.dev_control.acceptance_verification import DevVerificationStore
from gateway.dev_control.reliability import DevReliabilityStore, scorecard
from gateway.subagent_events import SubagentEventStore
from gateway.dev_execution import DevExecutionStore
from scripts.seed_dev_lab_data import seed_lab_data
from tools.ao_bridge import AOSession


def _env(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    lab_home = tmp_path / "lab"
    db_path = lab_home / "hermes-home" / "state.db"
    stable_db = tmp_path / "stable" / "state.db"
    stable_db.parent.mkdir(parents=True)
    stable_db.write_text("stable", encoding="utf-8")
    (lab_home / "repos" / "hermes-agent").mkdir(parents=True)
    (lab_home / "repos" / "Oryn").mkdir(parents=True)
    (lab_home / "worktrees").mkdir(parents=True)
    monkeypatch.setenv("ORYN_LAB_HOME", str(lab_home))
    monkeypatch.setenv("HERMES_HOME", str(lab_home / "hermes-home"))
    monkeypatch.setenv("API_SERVER_PORT", "8662")
    monkeypatch.delenv("HERMES_DEV_MERGE_EXECUTOR_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_DEV_BRANCH_PROTECTION_CONFIRMED", raising=False)
    monkeypatch.setenv("HERMES_DEV_LAB_MIN_TERMINAL_SECONDS", "0")
    monkeypatch.setattr(
        "gateway.dev_control.lab_loop.audit_current_process_isolation",
        lambda extra_pids=None: {
            "ok": True,
            "object": "hermes.dev_lab_process_isolation",
            "pids": [os.getpid(), *(extra_pids or [])],
            "write_handles": [],
            "offending_paths": [],
            "warnings": [],
            "authoritative": True,
        },
    )
    return db_path, stable_db


class _FakeLabRouter:
    def __init__(
        self,
        lab_home: Path,
        *,
        diff_paths: list[str] | None = None,
        status: str = "done",
        status_sequence: list[str] | None = None,
        transcript: str | None = None,
        spawn_error: Exception | None = None,
    ):
        self.lab_home = lab_home
        self.diff_paths = diff_paths or []
        self.status_value = status
        self.status_sequence = list(status_sequence or [])
        self.transcript = transcript
        self.spawn_error = spawn_error
        self.spawned = []
        self.sessions: dict[str, AOSession] = {}

    def spawn(self, *args, **kwargs):
        if self.spawn_error:
            raise self.spawn_error
        index = len(self.spawned) + 1
        workspace = self.lab_home / "worktrees" / f"dogfood-{index}"
        _init_git_repo(workspace)
        for rel in self.diff_paths:
            path = workspace / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"lab change for {rel}\n", encoding="utf-8")
        session = AOSession(
            id=f"lab-session-{index}",
            project_id=kwargs.get("project_id"),
            status=self.status_value,
            branch=kwargs.get("branch"),
            workspace_path=str(workspace),
            agent=kwargs.get("agent") or "codex",
            model=kwargs.get("model") or "gpt-5.5",
            reasoning_effort=kwargs.get("reasoning_effort"),
            summary="PHASE16_FIXTURE_OK_DONE Completed the lab dogfood task with scoped evidence.",
        )
        self.spawned.append({"args": args, "kwargs": kwargs, "session": session})
        self.sessions[session.id] = session
        return session

    def status(self, *args):
        session_id = args[-1]
        session = self.sessions.get(session_id)
        if session and self.status_sequence:
            session.status = self.status_sequence.pop(0)
        return session

    def list(self, *args, **kwargs):
        return list(self.sessions.values())

    def runtime_health(self, *args):
        return {"runtime_health": "ok", "runtime_warning": None}

    def capture_output(self, *args, **kwargs):
        return self.transcript or "PHASE16_FIXTURE_OK_DONE Completed the lab dogfood task with scoped evidence."


class _CostReportingFakeLabRouter(_FakeLabRouter):
    def spawn(self, *args, **kwargs):
        session = super().spawn(*args, **kwargs)
        _write_codex_usage_session(
            self.lab_home,
            workspace_path=Path(session.workspace_path or ""),
            input_tokens=1000,
            cached_input_tokens=400,
            output_tokens=120,
            total_tokens=1120,
        )
        return session


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "lab@example.test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Hermes Lab"], cwd=path, check=True)
    (path / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=path, check=True)


def _write_codex_usage_session(
    lab_home: Path,
    *,
    workspace_path: Path,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> Path:
    session_dir = lab_home / ".codex" / "sessions" / "2026" / "05" / "30"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"rollout-test-{workspace_path.name}.jsonl"
    rows = [
        {
            "timestamp": "2026-05-30T00:00:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": f"codex-{workspace_path.name}",
                "cwd": str(workspace_path),
                "model_provider": "openai",
                "model": None,
            },
        },
        {
            "timestamp": "2026-05-30T00:00:01.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": cached_input_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_output_tokens": 25,
                        "total_tokens": total_tokens,
                    }
                },
            },
        },
        {
            "timestamp": "2026-05-30T00:00:02.000Z",
            "type": "event_msg",
            "payload": {"type": "task_complete", "completed_at": 1780099202},
        },
    ]
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    return path


def test_preapproved_dogfood_pass_writes_real_outcome_and_keeps_stable_db(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    candidate = store.upsert_candidate({
        "prompt": "Add a small docs note for lab dogfood.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab-dogfood.md"],
        "source": "docs",
    }, approved=True)
    before = stable_db.stat().st_mtime

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/lab-dogfood.md"]),
    )

    assert report["status"] == "completed"
    assert report["isolation"]["ok"] is True
    assert report["stable_db_telemetry"]["authoritative"] is False
    assert report["candidate_id"] == candidate["candidate_id"]
    assert report["stable_db_unchanged"] is True
    assert stable_db.stat().st_mtime == before
    outcomes = DevReliabilityStore(db_path).list_outcomes(limit=20)
    assert outcomes
    assert outcomes[0]["source_refs"]["source"] == "dogfood_lab_loop"
    assert outcomes[0]["source_refs"]["seeded"] is False
    assert outcomes[0]["source_refs"]["implement_session_id"] == "lab-session-1"
    assert outcomes[0]["source_refs"]["draft_pr_only"] is True
    assert outcomes[0]["ci_state"] == "unknown"
    assert outcomes[0]["code_review_verdict"] == "unknown"
    assert loop_health(db_path=db_path)["real_outcome_count"] == 1


def test_stable_db_mtime_change_is_informational_not_a_gate(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Exercise mtime telemetry.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab.md"],
        "source": "docs",
    }, approved=True)

    def _executor(_candidate, _context):
        stable_db.write_text("stable changed by unrelated live service", encoding="utf-8")
        return {"status": "completed", "duration_seconds": 0.1}

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        executor=_executor,
        max_consecutive_failures=10,
    )

    assert report["status"] == "completed"
    assert report["stable_db_unchanged"] is False
    assert report["stable_db_telemetry"]["authoritative"] is False
    assert report["isolation"]["ok"] is True


def test_open_file_isolation_audit_flags_non_lab_write_handle(monkeypatch, tmp_path):
    if not shutil.which("lsof"):
        pytest.skip("lsof is required for open-file isolation audit")
    db_path, stable_db = _env(monkeypatch, tmp_path)
    outside = tmp_path / "outside-stable" / "state.db"
    outside.parent.mkdir(parents=True)
    handle = outside.open("w", encoding="utf-8")
    try:
        handle.write("open")
        handle.flush()
        audit = audit_process_isolation(pids=[os.getpid()])
    finally:
        handle.close()

    assert not audit["ok"]
    assert any(str(outside.resolve(strict=False)) == item["path"] for item in audit["offending_paths"])


def test_open_file_isolation_audit_allows_temp_stdout_handle(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    temp_output = Path("/private/tmp/hermes-lab-test/run-output.json")
    monkeypatch.setattr(
        "gateway.dev_control.lab_process_isolation._write_handles_for_pid",
        lambda _pid: [
            {"pid": os.getpid(), "fd": "1w", "type": "REG", "path": str(temp_output)},
            {"pid": os.getpid(), "fd": "3u", "type": "REG", "path": str(db_path)},
        ],
    )

    audit = audit_process_isolation(pids=[os.getpid()])

    assert audit["ok"] is True
    assert audit["offending_paths"] == []
    assert any(item["path"] == str(temp_output) for item in audit["write_handles"])


def test_lab_pass_hard_stops_on_non_lab_write_handle(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    store.upsert_candidate({
        "prompt": "Exercise isolation breaker.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab.md"],
        "source": "docs",
    }, approved=True)
    outside = tmp_path / "outside-stable" / "state.db"
    outside.parent.mkdir(parents=True)
    monkeypatch.setattr(
        "gateway.dev_control.lab_loop.audit_current_process_isolation",
        lambda extra_pids=None: {
            "ok": False,
            "object": "hermes.dev_lab_process_isolation",
            "pids": [os.getpid()],
            "write_handles": [{"pid": os.getpid(), "fd": "9u", "type": "REG", "path": str(outside)}],
            "offending_paths": [{"pid": os.getpid(), "fd": "9u", "type": "REG", "path": str(outside), "in_forbidden_root": True}],
            "warnings": [],
            "authoritative": True,
        },
    )

    def _executor(_candidate, _context):
        return {"status": "completed", "duration_seconds": 0.1}

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        executor=_executor,
        max_consecutive_failures=10,
    )

    assert report["status"] == "loop_halted"
    assert report["breaker_reason"] == "isolation_breach"
    assert not report["isolation"]["ok"]
    assert store.get_state()["status"] == "halted"


def test_lab_pass_does_not_halt_on_non_forbidden_stdout_write(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    store.upsert_candidate({
        "prompt": "Exercise benign output redirection telemetry.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab.md"],
        "source": "docs",
    }, approved=True)
    stdout_path = tmp_path / "run-output.json"
    monkeypatch.setattr(
        "gateway.dev_control.lab_loop.audit_current_process_isolation",
        lambda extra_pids=None: {
            "ok": False,
            "object": "hermes.dev_lab_process_isolation",
            "pids": [os.getpid()],
            "write_handles": [{"pid": os.getpid(), "fd": "1w", "type": "REG", "path": str(stdout_path)}],
            "offending_paths": [{"pid": os.getpid(), "fd": "1w", "type": "REG", "path": str(stdout_path), "in_forbidden_root": False}],
            "warnings": [],
            "authoritative": True,
        },
    )

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        executor=lambda _candidate, _context: {"status": "completed", "duration_seconds": 0.1},
        max_consecutive_failures=10,
    )

    assert report["status"] == "completed"
    assert report.get("breaker_reason") is None
    assert report["isolation"]["offending_paths"][0]["in_forbidden_root"] is False
    assert store.get_state()["status"] == "idle"


def test_lab_executor_derives_verified_outcome_from_measured_verification(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure a passing verification fixture.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
        "payload": {
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["gateway/dev_control/lab_loop.py"]),
    )

    assert report["status"] == "completed"
    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert outcome["verification_verdict"] == "verified"
    assert outcome["merged"] is False
    assert outcome["source_refs"]["draft_pr_only"] is True
    assert outcome["source_refs"]["gates"]["ci"] == "not_measured"
    assert outcome["source_refs"]["gates"]["review"] == "not_measured"
    assert outcome["success"] is True

    ready_payload = dict(outcome)
    ready_payload.pop("outcome_id", None)
    ready = DevReliabilityStore(db_path).upsert_outcome({
        **ready_payload,
        "plan_id": "ready-plan",
        "task_id": "ready-task",
        "terminal_status": "completed",
        "verification_verdict": "verified",
        "ci_state": "success",
        "code_review_verdict": "approved",
        "source_refs": {**outcome["source_refs"], "draft_pr_ready": True},
    })
    assert ready["success"] is True


def test_lab_executor_derives_ci_state_from_draft_head_sha(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    calls = []

    def fake_fetch_ci_status(*, repo, ref):
        calls.append({"repo": repo, "ref": ref})
        return {"state": "success", "repo": repo, "ref": ref, "warnings": []}

    monkeypatch.setattr("gateway.dev_control.lab_loop.fetch_ci_status", fake_fetch_ci_status)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure CI after a passing verification fixture.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
        "payload": {
            "ci_repo": "Felippen/Oryn",
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["gateway/dev_control/lab_loop.py"]),
    )

    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert report["status"] == "completed"
    assert report["execution"]["ci_state"] == "success"
    assert report["execution"]["ci_status"]["measured"] is True
    assert outcome["ci_state"] == "success"
    assert outcome["source_refs"]["gates"]["ci"] == "success"
    assert outcome["success"] is True
    assert calls == [{"repo": "Felippen/Oryn", "ref": report["execution"]["head_sha"]}]


def test_lab_executor_uses_published_head_sha_after_author_rewrite(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    calls = []

    def fake_fetch_ci_status(*, repo, ref):
        calls.append({"repo": repo, "ref": ref})
        return {"state": "success", "repo": repo, "ref": ref, "warnings": []}

    def fake_publish(*, candidate, workspace_path, branch):
        return {
            "ready": True,
            "status": "created",
            "repo": "Felippen/Oryn",
            "branch": branch,
            "head_sha": "published-head-after-author-rewrite",
            "pr_url": "https://github.com/Felippen/Oryn/pull/123",
            "warnings": [],
        }

    monkeypatch.setattr("gateway.dev_control.lab_loop.fetch_ci_status", fake_fetch_ci_status)
    monkeypatch.setattr("gateway.dev_control.lab_loop._publish_lab_draft_pr", fake_publish)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure CI against the published draft PR head.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/ci-head.md"],
        "source": "docs",
        "payload": {
            "ci_repo": "Felippen/Oryn",
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/ci-head.md"]),
    )

    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert report["execution"]["head_sha"] == "published-head-after-author-rewrite"
    assert outcome["source_refs"]["head_sha"] == "published-head-after-author-rewrite"
    assert outcome["source_refs"]["implementation_head_sha"] != "published-head-after-author-rewrite"
    assert calls == [{"repo": "Felippen/Oryn", "ref": "published-head-after-author-rewrite"}]


def test_lab_executor_records_approved_code_review_and_contract_score(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure approved R4 review after a passing verification fixture.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/review-approved.md"],
        "source": "docs",
        "payload": {
            "ci_status": {"state": "success", "repo": "Felippen/Oryn"},
            "code_review_result": {
                "object": "hermes.dev_code_review_result",
                "verdict": "approved",
                "findings": [],
                "summary": "Reviewed the scoped docs diff against the task intent.",
                "evidence_refs": ["docs/review-approved.md"],
            },
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/review-approved.md"]),
    )

    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert report["status"] == "completed"
    assert report["gate_verdicts"]["review"] == "approved"
    assert report["gate_verdicts"]["contract_score"] == 1.0
    assert outcome["code_review_verdict"] == "approved"
    assert outcome["output_contract_score"] == 1.0
    assert outcome["source_refs"]["gates"]["review"] == "approved"
    assert outcome["success"] is True


def test_lab_executor_launches_review_worker_without_issue_binding(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    commands = []

    def fake_run_command(args, *, timeout=30.0):
        commands.append(args)
        if args[:2] == ["gh", "pr"]:
            return {"returncode": 0, "stdout": "https://github.com/Felippen/Oryn/pull/456\n", "stderr": ""}
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("gateway.dev_control.lab_loop._run_command", fake_run_command)
    transcript = """
```json DEV_WORKER_EVIDENCE
{"structured_summary":"Implemented docs task.","findings":[],"verification_status":"passed","verification_evidence":["fixture"],"files_changed":["docs/review-live.md"],"commands_run":[]}
```
```json DEV_CODE_REVIEW_RESULT
{"object":"hermes.dev_code_review_result","verdict":"approved","findings":[],"summary":"Review approved.","evidence_refs":["docs/review-live.md"]}
```
"""
    bridge = _FakeLabRouter(tmp_path / "lab", diff_paths=["docs/review-live.md"], transcript=transcript)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure live R4 review worker launch.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/review-live.md"],
        "source": "docs",
        "payload": {
            "branch": "lab/dogfood/review-live",
            "ci_status": {"state": "success", "repo": "Felippen/Oryn"},
            "draft_pr_repo": "Felippen/Oryn",
            "draft_pr_remote": "lab-origin",
            "draft_pr_base": "main",
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=bridge,
    )

    review_spawn = bridge.spawned[-1]
    assert len(bridge.spawned) == 2
    assert review_spawn["kwargs"]["issue_id"] is None
    assert review_spawn["kwargs"]["branch"] is None
    assert "Profile: review; permissions: review_only." in review_spawn["kwargs"]["prompt"]
    assert "Do not invoke slash commands, /review, CodeRabbit, coderabbit" in review_spawn["kwargs"]["prompt"]
    assert "gh pr diff 456 --repo Felippen/Oryn --patch" in review_spawn["kwargs"]["prompt"]
    assert "run only the exact gh pr view/diff commands listed above" in review_spawn["kwargs"]["prompt"]
    assert "Do not start background terminals or long-running tools" in review_spawn["kwargs"]["prompt"]
    assert "```json DEV_CODE_REVIEW_RESULT" in review_spawn["kwargs"]["prompt"]
    assert "Hermes will not count the review as measured if this fenced block is missing or invalid" in review_spawn["kwargs"]["prompt"]
    assert "Do not use GitHub connector/MCP tools" in review_spawn["kwargs"]["prompt"]
    assert "Measured gate evidence supplied by Hermes:" in review_spawn["kwargs"]["prompt"]
    assert "Verification verdict: verified" in review_spawn["kwargs"]["prompt"]
    assert "CI state: success" in review_spawn["kwargs"]["prompt"]
    assert "Do not return verdict commented merely because you did not run tests/builds" in review_spawn["kwargs"]["prompt"]
    assert report["gate_verdicts"]["review"] == "approved"
    assert report["execution"]["code_review"]["cleanup"]["cleaned"] is True


def test_lab_review_await_ignores_empty_commented_result(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_DEV_LAB_MIN_TERMINAL_SECONDS", "0")
    transcript = """
{
  "object": "hermes.dev_code_review_result",
  "verdict": "commented",
  "findings": [],
  "summary": "",
  "evidence_refs": []
}
"""
    bridge = _FakeLabRouter(
        tmp_path / "lab",
        status="working",
        status_sequence=["working", "working"],
        transcript=transcript,
    )
    session = AOSession(id="review-session-empty-commented", status="working", workspace_path=str(tmp_path / "lab" / "review"))
    bridge.sessions[session.id] = session

    terminal = _await_review_terminal(
        bridge=bridge,
        runtime="ao",
        session=session,
        timeout_seconds=0.01,
    )

    assert terminal["status"] == "timed_out"
    assert terminal["timed_out"] is True


def test_lab_review_await_requires_fenced_review_json_contract(tmp_path):
    transcript = """
{
  "object": "hermes.dev_code_review_result",
  "verdict": "approved",
  "findings": [],
  "summary": "Review approved.",
  "evidence_refs": ["docs/review-live.md:1"]
}
"""
    bridge = _FakeLabRouter(
        tmp_path / "lab",
        status="working",
        status_sequence=["working", "working", "working"],
        transcript=transcript,
    )
    session = AOSession(id="review-session", status="working", workspace_path=str(tmp_path / "lab" / "review"))
    bridge.sessions[session.id] = session

    terminal = _await_review_terminal(
        bridge=bridge,
        runtime="ao",
        session=session,
        timeout_seconds=0.01,
    )

    assert terminal["status"] == "timed_out"
    assert terminal["timed_out"] is True
    assert "hermes.dev_code_review_result" in terminal["transcript"]


def test_lab_review_await_completes_when_fenced_review_json_appears(tmp_path):
    transcript = """
```json DEV_CODE_REVIEW_RESULT
{
  "object": "hermes.dev_code_review_result",
  "verdict": "approved",
  "findings": [],
  "summary": "Review approved.",
  "evidence_refs": ["docs/review-live.md:1"]
}
```
"""
    bridge = _FakeLabRouter(
        tmp_path / "lab",
        status="working",
        status_sequence=["working", "working", "working"],
        transcript=transcript,
    )
    session = AOSession(id="review-session-fenced", status="working", workspace_path=str(tmp_path / "lab" / "review"))
    bridge.sessions[session.id] = session

    terminal = _await_review_terminal(
        bridge=bridge,
        runtime="ao",
        session=session,
        timeout_seconds=30.0,
    )

    assert terminal["status"] == "completed_from_transcript"
    assert terminal["timed_out"] is False
    assert "DEV_CODE_REVIEW_RESULT" in terminal["transcript"]


def test_lab_executor_marks_recovered_review_json_unmeasured(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    commands = []

    def fake_run_command(args, *, timeout=30.0):
        commands.append(args)
        if args[:2] == ["gh", "pr"]:
            return {"returncode": 0, "stdout": "https://github.com/Felippen/Oryn/pull/457\n", "stderr": ""}
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("gateway.dev_control.lab_loop._run_command", fake_run_command)
    transcript = """
```json DEV_WORKER_EVIDENCE
{"structured_summary":"Implemented docs task.","findings":[],"verification_status":"passed","verification_evidence":["fixture"],"files_changed":["docs/review-unfenced.md"],"commands_run":[]}
```
{
  "object": "hermes.dev_code_review_result",
  "verdict": "approved",
  "findings": [],
  "summary": "Review approved.",
  "evidence_refs": ["docs/review-unfenced.md"]
}
"""
    bridge = _FakeLabRouter(tmp_path / "lab", diff_paths=["docs/review-unfenced.md"], transcript=transcript)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure recovered R4 review output.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/review-unfenced.md"],
        "source": "docs",
        "payload": {
            "branch": "lab/dogfood/review-unfenced",
            "ci_status": {"state": "success", "repo": "Felippen/Oryn"},
            "draft_pr_repo": "Felippen/Oryn",
            "draft_pr_remote": "lab-origin",
            "draft_pr_base": "main",
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=bridge,
    )

    review = report["execution"]["code_review"]
    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert review["status"] == "needs_attention"
    assert review["measured"] is False
    assert review["verdict"] == "unknown"
    assert "parsed review JSON is advisory only" in " ".join(review["warnings"])
    assert report["gate_verdicts"]["review"] == "unknown"
    assert outcome["code_review_verdict"] == "unknown"


def test_lab_executor_marks_template_review_json_unmeasured(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)

    def fake_run_command(args, *, timeout=30.0):
        if args[:2] == ["gh", "pr"]:
            return {"returncode": 0, "stdout": "https://github.com/Felippen/Oryn/pull/458\n", "stderr": ""}
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("gateway.dev_control.lab_loop._run_command", fake_run_command)
    transcript = """
```json DEV_WORKER_EVIDENCE
{"structured_summary":"Implemented docs task.","findings":[],"verification_status":"passed","verification_evidence":["fixture"],"files_changed":["docs/review-template.md"],"commands_run":[]}
```
```json DEV_CODE_REVIEW_RESULT
{
  "object": "hermes.dev_code_review_result",
  "verdict": "approved",
  "findings": [],
  "summary": "One sentence explaining the review decision.",
  "evidence_refs": ["gh pr diff <number> --repo <owner/repo> --patch"]
}
```
"""
    bridge = _FakeLabRouter(tmp_path / "lab", diff_paths=["docs/review-template.md"], transcript=transcript)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure templated R4 review output.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/review-template.md"],
        "source": "docs",
        "payload": {
            "branch": "lab/dogfood/review-template",
            "ci_status": {"state": "success", "repo": "Felippen/Oryn"},
            "draft_pr_repo": "Felippen/Oryn",
            "draft_pr_remote": "lab-origin",
            "draft_pr_base": "main",
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=bridge,
    )

    review = report["execution"]["code_review"]
    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert review["status"] == "needs_attention"
    assert review["measured"] is False
    assert review["verdict"] == "unknown"
    assert "parsed review JSON is advisory only" in " ".join(review["warnings"])
    assert report["gate_verdicts"]["review"] == "unknown"
    assert outcome["code_review_verdict"] == "unknown"


def test_lab_executor_marks_changes_requested_review_as_failed_outcome(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure changes-requested R4 review after a passing verification fixture.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/review-changes.md"],
        "source": "docs",
        "payload": {
            "ci_status": {"state": "success", "repo": "Felippen/Oryn"},
            "code_review_result": {
                "object": "hermes.dev_code_review_result",
                "verdict": "changes_requested",
                "findings": [{"severity": "major", "file": "docs/review-changes.md", "line": 1, "note": "Requested change."}],
                "summary": "The review found a concrete issue.",
                "evidence_refs": ["docs/review-changes.md:1"],
            },
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        max_consecutive_failures=10,
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/review-changes.md"]),
    )

    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert report["status"] == "failed"
    assert report["gate_verdicts"]["review"] == "changes_requested"
    assert outcome["code_review_verdict"] == "changes_requested"
    assert outcome["source_refs"]["gates"]["review"] == "changes_requested"
    assert outcome["success"] is False


def test_lab_ci_finalizer_updates_pending_outcome_to_success(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    store = DevReliabilityStore(db_path)
    store.upsert_outcome({
        "plan_id": "plan-ci-finalize",
        "task_id": "task-ci-finalize",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "completed",
        "merged": False,
        "verification_verdict": "verified",
        "ci_state": "pending",
        "code_review_verdict": "unknown",
        "source_refs": {
            "source": "dogfood_lab_loop",
            "draft_pr_only": True,
            "draft_pr_ready": True,
            "ci_status": {"repo": "Felippen/Oryn", "ref": "abc123", "state": "pending"},
            "gates": {"verification": "completed", "ci": "pending", "review": "not_measured"},
        },
    })

    result = finalize_pending_lab_ci_outcomes(
        db_path=db_path,
        fetcher=lambda *, repo, ref: {"state": "success", "repo": repo, "ref": ref, "warnings": []},
        now=1234.0,
    )

    outcome = store.get_outcome(plan_id="plan-ci-finalize", task_id="task-ci-finalize")
    assert result["counts"]["refreshed"] == 1
    assert outcome["ci_state"] == "success"
    assert outcome["source_refs"]["gates"]["ci"] == "success"
    assert outcome["source_refs"]["ci_finalized_at"] == 1234.0
    assert outcome["success"] is True


def test_lab_ci_finalizer_updates_pending_outcome_to_failure(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    store = DevReliabilityStore(db_path)
    store.upsert_outcome({
        "plan_id": "plan-ci-fail",
        "task_id": "task-ci-fail",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "completed",
        "merged": False,
        "verification_verdict": "verified",
        "ci_state": "pending",
        "code_review_verdict": "unknown",
        "source_refs": {
            "source": "dogfood_lab_loop",
            "draft_pr_only": True,
            "draft_pr_ready": True,
            "draft_artifact": {"head_sha": "def456", "publish": {"repo": "Felippen/Oryn"}},
            "gates": {"verification": "completed", "ci": "pending", "review": "not_measured"},
        },
    })

    finalize_pending_lab_ci_outcomes(
        db_path=db_path,
        fetcher=lambda *, repo, ref: {"state": "failure", "repo": repo, "ref": ref, "warnings": []},
    )

    outcome = store.get_outcome(plan_id="plan-ci-fail", task_id="task-ci-fail")
    assert outcome["terminal_status"] == "failed"
    assert outcome["ci_state"] == "failure"
    assert outcome["source_refs"]["gates"]["ci"] == "failure"
    assert outcome["success"] is False


def test_lab_ci_finalizer_uses_current_pr_head_for_unfinalized_draft(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    store = DevReliabilityStore(db_path)
    store.upsert_outcome({
        "plan_id": "plan-ci-pr-head",
        "task_id": "task-ci-pr-head",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "completed",
        "merged": False,
        "verification_verdict": "verified",
        "ci_state": "success",
        "code_review_verdict": "approved",
        "source_refs": {
            "source": "dogfood_lab_loop",
            "draft_pr_only": True,
            "draft_pr_ready": True,
            "ci_status": {"repo": "Felippen/Oryn", "ref": "stale-head", "state": "success"},
            "draft_artifact": {
                "head_sha": "stale-head",
                "pr_number": 123,
                "pr_url": "https://github.com/Felippen/Oryn/pull/123",
                "publish": {"repo": "Felippen/Oryn", "head_sha": "stale-head", "pr_number": 123},
            },
            "gates": {"verification": "completed", "ci": "success", "review": "approved"},
        },
    })
    calls = []
    monkeypatch.setattr(
        "gateway.dev_control.lab_loop._resolve_lab_draft_pr_ref",
        lambda *, repo, draft_artifact: {"pr_number": 123, "head_sha": "current-head", "warnings": []},
    )

    result = finalize_pending_lab_ci_outcomes(
        db_path=db_path,
        fetcher=lambda *, repo, ref: calls.append({"repo": repo, "ref": ref}) or {"state": "success", "repo": repo, "ref": ref},
        now=2468.0,
    )

    outcome = store.get_outcome(plan_id="plan-ci-pr-head", task_id="task-ci-pr-head")
    assert result["counts"]["refreshed"] == 1
    assert calls == [{"repo": "Felippen/Oryn", "ref": "current-head"}]
    assert outcome["ci_state"] == "success"
    assert outcome["source_refs"]["ci_status"]["ref"] == "current-head"
    assert outcome["source_refs"]["draft_artifact"]["head_sha"] == "current-head"
    assert outcome["source_refs"]["draft_artifact"]["publish"]["head_sha"] == "current-head"
    assert outcome["source_refs"]["ci_finalized_at"] == 2468.0


def test_lab_ci_finalizer_keeps_pending_when_checks_still_running(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    store = DevReliabilityStore(db_path)
    store.upsert_outcome({
        "plan_id": "plan-ci-pending",
        "task_id": "task-ci-pending",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "completed",
        "merged": False,
        "verification_verdict": "verified",
        "ci_state": "pending",
        "code_review_verdict": "unknown",
        "source_refs": {
            "source": "dogfood_lab_loop",
            "draft_pr_only": True,
            "draft_pr_ready": True,
            "ci_status": {"repo": "Felippen/Oryn", "ref": "ghi789", "state": "pending"},
            "gates": {"verification": "completed", "ci": "pending", "review": "not_measured"},
        },
    })

    finalize_pending_lab_ci_outcomes(
        db_path=db_path,
        fetcher=lambda *, repo, ref: {"state": "pending", "repo": repo, "ref": ref, "warnings": []},
        now=5678.0,
    )

    outcome = store.get_outcome(plan_id="plan-ci-pending", task_id="task-ci-pending")
    assert outcome["ci_state"] == "pending"
    assert outcome["source_refs"]["gates"]["ci"] == "pending"
    assert outcome["source_refs"]["ci_finalized_at"] is None
    assert outcome["success"] is False


def test_lab_ci_finalizer_leaves_unknown_unmeasured_outcomes_alone(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    store = DevReliabilityStore(db_path)
    store.upsert_outcome({
        "plan_id": "plan-ci-unknown",
        "task_id": "task-ci-unknown",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "completed",
        "merged": False,
        "verification_verdict": "verified",
        "ci_state": "unknown",
        "code_review_verdict": "unknown",
        "source_refs": {
            "source": "dogfood_lab_loop",
            "draft_pr_only": True,
            "draft_pr_ready": True,
            "ci_status": {"repo": "Felippen/Oryn", "ref": "unknown-ref", "state": "unknown"},
            "gates": {"verification": "completed", "ci": "not_measured", "review": "not_measured"},
        },
    })
    calls = []

    result = finalize_pending_lab_ci_outcomes(
        db_path=db_path,
        fetcher=lambda *, repo, ref: calls.append({"repo": repo, "ref": ref}) or {"state": "success"},
    )

    outcome = store.get_outcome(plan_id="plan-ci-unknown", task_id="task-ci-unknown")
    assert result["counts"]["refreshed"] == 0
    assert result["skipped"][0]["reason"] == "ci_state_not_pending:unknown"
    assert calls == []
    assert outcome["ci_state"] == "unknown"
    assert outcome["source_refs"]["gates"]["ci"] == "not_measured"


def test_observe_profile_preflight_blocks_halted_state(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    store.halt("fixture halt")

    result = observe_profile_preflight(
        db_path=db_path,
        active_session_lister=lambda: [],
        active_worktree_lister=lambda: [],
    )

    assert result["ok"] is False
    assert result["blockers"][0]["code"] == "loop_halted"


def test_observe_profile_preflight_blocks_recent_forbidden_isolation(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    store.record_pass({
        "status": "completed",
        "isolation": {
            "ok": False,
            "offending_paths": [{"path": str(tmp_path / "stable.db"), "in_forbidden_root": True}],
        },
    })

    result = observe_profile_preflight(
        db_path=db_path,
        active_session_lister=lambda: [],
        active_worktree_lister=lambda: [],
    )

    assert result["ok"] is False
    assert result["blockers"][0]["code"] == "recent_isolation_breach"


def test_observe_profile_preflight_requires_docs_or_tests_candidates(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Modify implementation code.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
    }, approved=True)

    result = observe_profile_preflight(
        db_path=db_path,
        active_session_lister=lambda: [],
        active_worktree_lister=lambda: [],
    )

    assert result["ok"] is False
    assert result["blockers"][0]["code"] == "candidate_outside_observe_profile"


def test_observe_profile_preflight_requires_idle_lab_processes(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)

    result = observe_profile_preflight(
        db_path=db_path,
        active_session_lister=lambda: ["lab-hermes-agent-busy"],
        active_worktree_lister=lambda: [str(tmp_path / "lab" / "worktrees" / "busy")],
    )

    assert result["ok"] is False
    assert {item["code"] for item in result["blockers"]} == {"active_lab_sessions", "active_lab_worktrees"}


def test_observe_profile_runs_bounded_docs_pass_and_reports_summary(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    monkeypatch.setattr("gateway.dev_control.lab_loop._active_lab_tmux_sessions", lambda: [])
    monkeypatch.setattr("gateway.dev_control.lab_loop._active_lab_worktrees", lambda: [])
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Add an observe-profile docs note.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/observe-profile.md"],
        "source": "docs",
        "payload": {
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    result = run_lab_observe_profile(
        db_path=db_path,
        stable_db_path=stable_db,
        max_passes=5,
        sources=["reliability"],
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/observe-profile.md"]),
    )

    assert result["ok"] is True
    assert result["profile"] == "observe"
    assert result["run"]["pass_count"] == 2
    assert result["run"]["passes"][0]["status"] == "completed"
    assert result["run"]["passes"][1]["status"] == "idle"
    assert result["summary"]["passes"][0]["verification"] == "verified"
    assert result["summary"]["passes"][0]["diff_scope"] == "in_scope"
    assert result["summary"]["forbidden_root_writes"] == []
    assert result["summary"]["merge_executed"] is False


def test_observe_profile_finalizes_pending_ci_before_running(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    monkeypatch.setattr("gateway.dev_control.lab_loop._active_lab_tmux_sessions", lambda: [])
    monkeypatch.setattr("gateway.dev_control.lab_loop._active_lab_worktrees", lambda: [])
    store = DevReliabilityStore(db_path)
    store.upsert_outcome({
        "plan_id": "plan-observe-ci",
        "task_id": "task-observe-ci",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "completed",
        "merged": False,
        "verification_verdict": "verified",
        "ci_state": "pending",
        "code_review_verdict": "approved",
        "source_refs": {
            "source": "dogfood_lab_loop",
            "draft_pr_only": True,
            "draft_pr_ready": True,
            "ci_status": {"repo": "Felippen/Oryn", "ref": "abc123", "state": "pending"},
            "gates": {"verification": "completed", "ci": "pending", "review": "approved"},
        },
    })

    result = run_lab_observe_profile(
        db_path=db_path,
        stable_db_path=stable_db,
        max_passes=1,
        sources=["reliability"],
        executor=lambda _candidate, _context: {"status": "completed"},
        ci_fetcher=lambda *, repo, ref: {"state": "success", "repo": repo, "ref": ref, "warnings": []},
    )

    outcome = store.get_outcome(plan_id="plan-observe-ci", task_id="task-observe-ci")
    assert result["ok"] is True
    assert result["ci_finalization"]["before"]["counts"]["refreshed"] == 1
    assert result["summary"]["ci_finalization"]["before"]["refreshed"] == 1
    assert outcome["ci_state"] == "success"


def test_lab_executor_publishes_configured_draft_pr_before_ci(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    commands = []

    def fake_run_command(args, *, timeout=30.0):
        commands.append(args)
        if args[:6] == ["git", "-C", str(Path(args[2])), "merge-base", "--is-ancestor", "codex/fixed-lab-base"]:
            return {"returncode": 1, "stdout": "", "stderr": ""}
        if args[:2] == ["gh", "pr"]:
            return {"returncode": 0, "stdout": "https://github.com/Felippen/Oryn/pull/123\n", "stderr": ""}
        return {"returncode": 0, "stdout": "", "stderr": ""}

    def fake_fetch_ci_status(*, repo, ref):
        return {"state": "success", "repo": repo, "ref": ref, "warnings": []}

    monkeypatch.setattr("gateway.dev_control.lab_loop._run_command", fake_run_command)
    monkeypatch.setattr("gateway.dev_control.lab_loop.fetch_ci_status", fake_fetch_ci_status)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Publish a draft PR artifact for a lab docs task.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab-draft-pr.md"],
        "source": "docs",
        "payload": {
            "branch": "lab/dogfood/draft-pr-fixture",
            "ci_repo": "Felippen/Oryn",
            "draft_pr_remote": "lab-origin",
            "draft_pr_base": "main",
            "branch_base_ref": "codex/fixed-lab-base",
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/lab-draft-pr.md"]),
    )

    artifact = report["execution"]["draft_artifact"]
    assert report["status"] == "completed"
    assert artifact["type"] == "draft_pr"
    assert artifact["pr_url"] == "https://github.com/Felippen/Oryn/pull/123"
    assert artifact["pr_number"] == 123
    assert artifact["publish"]["status"] == "created"
    assert artifact["publish"]["pr_number"] == 123
    assert artifact["publish"]["branch_base"]["base_ref"] == "codex/fixed-lab-base"
    assert any(command[:4] == ["git", "-C", str(Path(report["execution"]["workspace_path"])), "push"] for command in commands)
    assert any(command[:4] == ["git", "-C", str(Path(report["execution"]["workspace_path"])), "rebase"] and command[-1] == "codex/fixed-lab-base" for command in commands)
    assert any(command[:3] == ["gh", "pr", "create"] and "--draft" in command for command in commands)


def test_prepare_lab_branch_base_rebases_before_publish(monkeypatch, tmp_path):
    commands = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fake_run_command(args, *, timeout=30.0):
        commands.append(args)
        if args[3:6] == ["merge-base", "--is-ancestor", "codex/fixed-lab-base"]:
            return {"returncode": 1, "stdout": "", "stderr": ""}
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("gateway.dev_control.lab_loop._run_command", fake_run_command)

    result = _prepare_lab_branch_base_for_publish(workspace, "codex/fixed-lab-base")

    assert result == {"status": "rebased", "base_ref": "codex/fixed-lab-base", "warnings": []}
    assert commands == [
        ["git", "-C", str(workspace), "rev-parse", "--verify", "codex/fixed-lab-base^{commit}"],
        ["git", "-C", str(workspace), "merge-base", "--is-ancestor", "codex/fixed-lab-base", "HEAD"],
        ["git", "-C", str(workspace), "rebase", "codex/fixed-lab-base"],
    ]


def test_normalize_lab_commit_author_uses_configured_base_ref(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    commands = []

    monkeypatch.setattr("gateway.dev_control.lab_loop._git_head_sha", lambda _workspace: "head-sha")

    def fake_git_scalar(_workspace, args):
        if args == ["merge-base", "codex/fixed-lab-base", "HEAD"]:
            return "base-sha"
        if args == ["log", "-1", "--format=%s"]:
            return "test: lab change"
        return ""

    def fake_git_lines(_workspace, args):
        if args == ["log", "base-sha..HEAD", "--format=%ae"]:
            return ["felipe@mac.home"]
        return []

    def fake_run_command(args, *, timeout=30.0):
        commands.append(args)
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("gateway.dev_control.lab_loop._git_scalar", fake_git_scalar)
    monkeypatch.setattr("gateway.dev_control.lab_loop._git_lines", fake_git_lines)
    monkeypatch.setattr("gateway.dev_control.lab_loop._run_command", fake_run_command)

    result = _normalize_lab_commit_author_for_ci(workspace, base_ref_name="codex/fixed-lab-base")

    assert result["status"] == "normalized"
    assert result["base_ref"] == "codex/fixed-lab-base"
    assert ["git", "-C", str(workspace), "reset", "--soft", "base-sha"] in commands


def test_lab_executor_keeps_local_branch_when_draft_pr_remote_missing(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    commands = []
    monkeypatch.setattr("gateway.dev_control.lab_loop._run_command", lambda args, *, timeout=30.0: commands.append(args) or {"returncode": 0, "stdout": "", "stderr": ""})
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Keep local branch artifact when draft PR remote is absent.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/local-branch.md"],
        "source": "docs",
        "payload": {
            "ci_repo": "Felippen/Oryn",
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/local-branch.md"]),
    )

    artifact = report["execution"]["draft_artifact"]
    assert artifact["type"] == "local_branch"
    assert artifact["pr_url"] is None
    assert artifact["publish"]["status"] == "not_configured"
    assert commands == []


def test_lab_executor_ci_failure_is_real_failed_outcome(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure failing CI after a passing verification fixture.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
        "payload": {
            "ci_status": {"state": "failure", "repo": "Felippen/Oryn"},
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        max_consecutive_failures=10,
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["gateway/dev_control/lab_loop.py"]),
    )

    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert report["status"] == "failed"
    assert report["execution"]["ci_state"] == "failure"
    assert outcome["verification_verdict"] == "verified"
    assert outcome["ci_state"] == "failure"
    assert outcome["success"] is False


def test_lab_executor_can_produce_bad_score_from_failed_verification(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Measure a failing verification fixture.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
        "payload": {
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "failed",
                "command_run": "make test",
                "exit_code": 1,
                "output_excerpt": "1 failed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        max_consecutive_failures=10,
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["gateway/dev_control/lab_loop.py"]),
    )

    assert report["status"] == "failed"
    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert outcome["verification_verdict"] == "failed"
    assert outcome["success"] is False


def test_out_of_scope_engine_task_is_skipped_not_failed(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    store.upsert_candidate({
        "prompt": "Modify the conversation loop.",
        "profile_id": "platform.implement",
        "risk_level": "high",
        "target_paths": ["agent/conversation_loop.py"],
        "source": "manual",
    }, approved=True)

    report = run_lab_loop_pass(db_path=db_path, stable_db_path=stable_db, max_consecutive_out_of_scope=10)

    assert report["status"] == "skipped"
    assert report["skip_reason"] == "out_of_scope"
    assert DevReliabilityStore(db_path).list_outcomes(limit=20) == []


def test_circuit_breaker_halts_on_consecutive_failure(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    store.upsert_candidate({
        "prompt": "Exercise failure breaker.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        executor=lambda _candidate, _context: {"status": "failed", "error": "fixture"},
        max_consecutive_failures=1,
    )

    assert report["status"] == "loop_halted"
    assert report["breaker_reason"] == "consecutive_failures:1"
    assert store.get_state()["status"] == "halted"


def test_circuit_breaker_halts_on_cost_budget(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    store.upsert_candidate({
        "prompt": "Exercise cost breaker.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        executor=lambda _candidate, _context: {"status": "completed", "cost_usd": 3.0},
        max_cost_usd=1.0,
    )

    assert report["status"] == "loop_halted"
    assert report["breaker_reason"] == "cost_budget_exceeded:3.0000"


def test_lab_cost_telemetry_reads_codex_session_usage(monkeypatch, tmp_path):
    lab_home = tmp_path / "lab"
    monkeypatch.setenv("ORYN_LAB_HOME", str(lab_home))
    workspace = lab_home / "worktrees" / "dogfood-usage"
    workspace.mkdir(parents=True)
    session_file = _write_codex_usage_session(
        lab_home,
        workspace_path=workspace,
        input_tokens=1200,
        cached_input_tokens=500,
        output_tokens=150,
        total_tokens=1350,
    )

    cost = _worker_session_usage_cost({
        "runtime": "ao",
        "workspace_path": str(workspace),
        "launch": {"session": {"agent": "codex", "model": "gpt-5.5"}},
    })

    assert cost is not None
    assert cost["source"] == "codex_session_jsonl"
    assert cost["status"] == "included"
    assert cost["measured"] is True
    assert cost["cost_usd"] == 0.0
    assert cost["provider"] == "openai-codex"
    assert cost["model"] == "gpt-5.5"
    assert cost["session_path"] == str(session_file)
    assert cost["usage"]["input_tokens"] == 700
    assert cost["usage"]["cache_read_tokens"] == 500
    assert cost["usage"]["output_tokens"] == 150
    assert cost["usage"]["total_tokens"] == 1350


def test_lab_loop_uses_codex_session_cost_for_budget(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Exercise Codex session cost telemetry.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/cost.md"],
        "source": "docs",
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        bridge=_CostReportingFakeLabRouter(tmp_path / "lab", diff_paths=["docs/cost.md"]),
        sources=["reliability"],
        max_cost_usd=0.0,
    )

    assert report["status"] == "completed"
    assert report.get("breaker_reason") is None
    cost = report["execution"]["cost"]
    assert cost["status"] == "included"
    assert cost["measured"] is True
    assert cost["cost_usd"] == 0.0
    assert cost["usage"]["total_tokens"] == 1120
    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert outcome["source_refs"]["cost"]["status"] == "included"


def test_circuit_breaker_halts_when_cost_budget_requires_missing_cost(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    store.upsert_candidate({
        "prompt": "Exercise unavailable cost breaker.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        executor=lambda _candidate, _context: {
            "status": "completed",
            "cost_status": "unavailable",
            "cost": {"status": "unavailable", "measured": False, "cost_usd": None},
        },
        max_cost_usd=1.0,
    )

    assert report["status"] == "loop_halted"
    assert report["breaker_reason"] == "cost_unavailable"
    assert "missing cost as zero" in report["warnings"][0]


def test_seeded_data_remains_distinguishable_from_real_outcomes(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    seed = seed_lab_data(db_path)
    assert seed["ok"] is True
    assert all((item["source_refs"] or {}).get("seeded") is True for item in DevReliabilityStore(db_path).list_outcomes(limit=20))
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Add docs dogfood.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab.md"],
        "source": "docs",
    }, approved=True)

    run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        sources=["reliability"],
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/lab.md"]),
    )

    health = loop_health(db_path=db_path)
    assert health["real_outcome_count"] == 1
    assert health["scorecard_summary"]["sample_count"] >= 4


def test_invalid_lab_outcomes_are_excluded_from_scorecard(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    store = DevReliabilityStore(db_path)
    store.upsert_outcome({
        "outcome_id": "devrel-out-042c159df0",
        "plan_id": "bad-plan",
        "task_id": "bad-task",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "failed",
        "verification_verdict": "unknown",
        "ci_state": "unknown",
        "code_review_verdict": "unknown",
        "source_refs": {"source": "dogfood_lab_loop", "draft_pr_only": True},
    })
    store.upsert_outcome({
        "outcome_id": "devrel-out-valid",
        "plan_id": "valid-plan",
        "task_id": "valid-task",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "failed",
        "verification_verdict": "unknown",
        "ci_state": "unknown",
        "code_review_verdict": "unknown",
        "source_refs": {"source": "dogfood_lab_loop", "draft_pr_only": True},
    })

    card = scorecard(store.list_outcomes(limit=20))
    health = loop_health(db_path=db_path)

    assert card["summary"]["sample_count"] == 1
    assert health["real_outcome_count"] == 1


def test_runner_invalid_execution_does_not_write_scorecard_outcome(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Runner abort fixture.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab.md"],
        "source": "docs",
    }, approved=True)

    def _executor(_candidate, _context):
        return {
            "status": "runner_aborted",
            "reason": "runner_defect:premature_terminal",
            "invalid_outcome": True,
            "scorecard_excluded": True,
        }

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        executor=_executor,
        max_consecutive_failures=10,
    )

    assert report["status"] == "failed"
    assert report["outcome_id"] is None
    assert DevReliabilityStore(db_path).list_outcomes(limit=20) == []


def test_scope_filter_rejects_engine_paths():
    assert dogfood_scope_check(["gateway/dev_control/lab_loop.py"])["ok"] is True
    rejected = dogfood_scope_check(["agent/conversation_loop.py"])
    assert rejected["ok"] is False
    assert rejected["status"] == "out_of_scope"


def test_lab_executor_dispatches_worker_and_records_diff_artifact(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    store = DevLabLoopStore(db_path)
    store.upsert_candidate({
        "prompt": "Make a scoped dev_control change.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
        "payload": {
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)
    router = _FakeLabRouter(tmp_path / "lab", diff_paths=["gateway/dev_control/lab_loop.py"])

    report = run_lab_loop_pass(db_path=db_path, stable_db_path=stable_db, bridge=router, sources=["reliability"])

    assert report["status"] == "completed"
    assert router.spawned
    assert router.spawned[0]["kwargs"]["project_id"] == "HermesAgentLab"
    assert router.spawned[0]["kwargs"]["branch"].startswith("lab/dogfood/")
    assert report["implement_session_id"] == "lab-session-1"
    assert report["diff_scope"]["status"] == "in_scope"
    assert report["draft_artifact"]["type"] == "local_branch"
    assert not Path(router.sessions["lab-session-1"].workspace_path).exists()


def test_lab_await_uses_authoritative_terminal_not_transcript_inference(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_DEV_LAB_WORKER_TIMEOUT_SECONDS", "0.01")
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    plan = execution_store.create_plan(
        title="Transcript inference guard",
        vision_brief="Do not accept transcript-only completion.",
        tasks=[{
            "goal": "Guard terminal detection.",
            "prompt": "Guard terminal detection.",
            "profile_id": "platform.implement",
            "project_id": "HermesAgentLab",
        }],
    )
    task = plan["tasks"][0]
    execution_store.update_task_launch(plan_id=plan["plan_id"], task_id=task["task_id"], ao_session_id="lab-session-inferred")
    event_store.append_event({
        "event": "subagent.complete",
        "ao_session_id": "lab-session-inferred",
        "subagent_id": "ao:lab-session-inferred",
        "status": "completed",
        "message": "Starting MCP servers (0/7): codex_apps",
        "transcript_inferred_completion": True,
        "launch_plan_id": plan["plan_id"],
        "launch_task_id": task["task_id"],
    })
    router = _FakeLabRouter(tmp_path / "lab", status="running")
    router.sessions["lab-session-inferred"] = AOSession(
        id="lab-session-inferred",
        project_id="HermesAgentLab",
        status="running",
        workspace_path=str(tmp_path / "lab" / "worktrees" / "inferred"),
    )

    terminal = _await_implementation_terminal(
        execution_store=execution_store,
        event_store=event_store,
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        bridge=router,
        timeout_seconds=0.01,
    )

    assert terminal["timed_out"] is True
    assert terminal["authoritative_terminal"] is False


def test_lab_await_waits_for_runtime_terminal_status(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    plan = execution_store.create_plan(
        title="Runtime terminal",
        vision_brief="Wait until the runtime reports done.",
        tasks=[{
            "goal": "Guard runtime terminal detection.",
            "prompt": "Guard runtime terminal detection.",
            "profile_id": "platform.implement",
            "project_id": "HermesAgentLab",
        }],
    )
    task = plan["tasks"][0]
    execution_store.update_task_launch(plan_id=plan["plan_id"], task_id=task["task_id"], ao_session_id="lab-session-terminal")
    router = _FakeLabRouter(tmp_path / "lab", status_sequence=["running", "running", "done"])
    router.sessions["lab-session-terminal"] = AOSession(
        id="lab-session-terminal",
        project_id="HermesAgentLab",
        status="running",
        workspace_path=str(tmp_path / "lab" / "worktrees" / "terminal"),
    )

    terminal = _await_implementation_terminal(
        execution_store=execution_store,
        event_store=event_store,
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        bridge=router,
        timeout_seconds=2.0,
    )

    assert terminal.get("timed_out") is not True
    assert terminal["authoritative_terminal"] is True
    assert terminal["status"] == "completed"


def test_lab_await_rejects_implausibly_early_completed_session(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_DEV_LAB_MIN_TERMINAL_SECONDS", "60")
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    plan = execution_store.create_plan(
        title="Minimum liveness guard",
        vision_brief="Do not accept an immediate completed status.",
        tasks=[{
            "goal": "Guard terminal floor.",
            "prompt": "Guard terminal floor.",
            "profile_id": "platform.implement",
            "project_id": "HermesAgentLab",
        }],
    )
    task = plan["tasks"][0]
    execution_store.update_task_launch(plan_id=plan["plan_id"], task_id=task["task_id"], ao_session_id="lab-session-too-fast")
    router = _FakeLabRouter(tmp_path / "lab", status="done")
    router.sessions["lab-session-too-fast"] = AOSession(
        id="lab-session-too-fast",
        project_id="HermesAgentLab",
        status="done",
        workspace_path=str(tmp_path / "lab" / "worktrees" / "too-fast"),
    )

    terminal = _await_implementation_terminal(
        execution_store=execution_store,
        event_store=event_store,
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        bridge=router,
        timeout_seconds=0.01,
    )

    assert terminal["timed_out"] is True
    assert terminal["reason"] == "worker_timeout:0.0s"


def test_lab_await_accepts_valid_worker_evidence_transcript(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    plan = execution_store.create_plan(
        title="Transcript evidence terminal",
        vision_brief="Accept valid direct AO worker evidence.",
        tasks=[{
            "goal": "Append a docs note.",
            "prompt": "Append a docs note.",
            "profile_id": "platform.implement",
            "project_id": "HermesAgentLab",
            "target_paths": ["docs/lab-dogfood-supervised.md"],
        }],
    )
    task = plan["tasks"][0]
    execution_store.update_task_launch(plan_id=plan["plan_id"], task_id=task["task_id"], ao_session_id="lab-session-evidence")
    transcript = """Worker finished.
```json DEV_WORKER_EVIDENCE
{
  "summary": "Added the lab dogfood supervised note.",
  "findings": ["Commit changes only the requested docs file."],
  "files_read": ["docs/lab-dogfood-supervised.md"],
  "files_changed": ["docs/lab-dogfood-supervised.md"],
  "commands_run": ["git show --stat HEAD"],
  "verification": {
    "status": "passed",
    "evidence": ["git show --stat reports one docs file changed."]
  },
  "unresolved_gaps": [],
  "confidence": 0.92,
  "final_marker": null
}
```
"""
    router = _FakeLabRouter(tmp_path / "lab", status="running", transcript=transcript)
    router.sessions["lab-session-evidence"] = AOSession(
        id="lab-session-evidence",
        project_id="HermesAgentLab",
        status="running",
        workspace_path=str(tmp_path / "lab" / "worktrees" / "evidence"),
    )

    terminal = _await_implementation_terminal(
        execution_store=execution_store,
        event_store=event_store,
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        bridge=router,
        timeout_seconds=2.0,
    )

    assert terminal["authoritative_terminal"] is True
    assert terminal["status"] == "completed"
    assert terminal["task"]["files_changed"] == ["docs/lab-dogfood-supervised.md"]
    events = event_store.list_events(ao_session_id="lab-session-evidence", limit=20)
    assert any(event.get("transcript_evidence_completion") for event in events)


def test_lab_await_accepts_unfenced_worker_evidence_transcript(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    plan = execution_store.create_plan(
        title="Unfenced transcript evidence terminal",
        vision_brief="Accept direct AO worker evidence rendered without fences.",
        tasks=[{
            "goal": "Append a docs note.",
            "prompt": "Append a docs note.",
            "profile_id": "platform.implement",
            "project_id": "HermesAgentLab",
            "target_paths": ["docs/lab-dogfood-supervised.md"],
        }],
    )
    task = plan["tasks"][0]
    execution_store.update_task_launch(plan_id=plan["plan_id"], task_id=task["task_id"], ao_session_id="lab-session-unfenced")
    transcript = """
{
  "summary": "Added the one-line dated lab dogfood verification note at the requested docs path and committed it.",
  "findings": [
    "Commit e941a7056 changes only docs/lab-dogfood-supervised.md."
  ],
  "files_read": [
    "docs/lab-dogfood-supervised.md"
  ],
  "files_changed": [
    "docs/lab-dogfood-supervised.md"
  ],
  "commands_run": [
    "git diff-tree --no-commit-id --name-only -r HEAD"
  ],
  "verification": {
    "status": "passed",
    "evidence": [
      "git diff-tree returned only docs/lab-dogfood-supervised.md."
    ]
  },
  "unresolved_gaps": [],
  "confidence": 0.97,
  "final_marker": null
}
"""
    router = _FakeLabRouter(tmp_path / "lab", status="running", transcript=transcript)
    router.sessions["lab-session-unfenced"] = AOSession(
        id="lab-session-unfenced",
        project_id="HermesAgentLab",
        status="running",
        workspace_path=str(tmp_path / "lab" / "worktrees" / "unfenced"),
    )

    terminal = _await_implementation_terminal(
        execution_store=execution_store,
        event_store=event_store,
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        bridge=router,
        timeout_seconds=2.0,
    )

    assert terminal["authoritative_terminal"] is True
    assert terminal["status"] == "completed"
    assert terminal["task"]["files_changed"] == ["docs/lab-dogfood-supervised.md"]


def test_lab_await_ignores_prompt_template_before_actual_evidence(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    plan = execution_store.create_plan(
        title="Prompt template then actual evidence",
        vision_brief="Accept actual evidence after the prompt template.",
        tasks=[{
            "goal": "Append a docs note.",
            "prompt": "Append a docs note.",
            "profile_id": "platform.implement",
            "project_id": "HermesAgentLab",
            "target_paths": ["docs/lab-dogfood-supervised.md"],
        }],
    )
    task = plan["tasks"][0]
    execution_store.update_task_launch(plan_id=plan["plan_id"], task_id=task["task_id"], ao_session_id="lab-session-template-then-real")
    transcript = """
```json DEV_WORKER_EVIDENCE
{
  "summary": "What you concluded or changed.",
  "findings": ["Concrete finding or result."],
  "files_read": ["path/or/file.ext"],
  "files_changed": [],
  "commands_run": ["command --if-any"],
  "verification": {"status": "passed", "evidence": ["What proves the result."]},
  "unresolved_gaps": [],
  "confidence": 0.86,
  "final_marker": null
}
```
{
  "summary": "Added the requested one-line dated lab dogfood note and committed only docs/lab-dogfood-supervised.md.",
  "findings": ["No Hermes engine files were touched."],
  "files_read": ["docs/lab-dogfood-supervised.md"],
  "files_changed": ["docs/lab-dogfood-supervised.md"],
  "commands_run": ["pytest tests/gateway/test_api_server_runs.py -q"],
  "verification": {"status": "passed", "evidence": ["22 passed, 22 warnings in 3.10s."]},
  "unresolved_gaps": [],
  "confidence": 0.99,
  "final_marker": null
}
"""
    router = _FakeLabRouter(tmp_path / "lab", status="running", transcript=transcript)
    router.sessions["lab-session-template-then-real"] = AOSession(
        id="lab-session-template-then-real",
        project_id="HermesAgentLab",
        status="running",
        workspace_path=str(tmp_path / "lab" / "worktrees" / "template-then-real"),
    )

    terminal = _await_implementation_terminal(
        execution_store=execution_store,
        event_store=event_store,
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        bridge=router,
        timeout_seconds=2.0,
    )

    assert terminal["authoritative_terminal"] is True
    assert terminal["task"]["files_changed"] == ["docs/lab-dogfood-supervised.md"]


def test_lab_await_rejects_prompt_example_evidence_block(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    plan = execution_store.create_plan(
        title="Prompt example rejection",
        vision_brief="Do not accept the worker contract example.",
        tasks=[{
            "goal": "Append a docs note.",
            "prompt": "Append a docs note.",
            "profile_id": "platform.implement",
            "project_id": "HermesAgentLab",
            "target_paths": ["docs/lab-dogfood-supervised.md"],
        }],
    )
    task = plan["tasks"][0]
    execution_store.update_task_launch(plan_id=plan["plan_id"], task_id=task["task_id"], ao_session_id="lab-session-example")
    transcript = """Worker Output Contract v2
```json DEV_WORKER_EVIDENCE
{
  "summary": "What you concluded or changed.",
  "findings": ["Concrete finding or result."],
  "files_read": ["path/or/file.ext"],
  "files_changed": [],
  "commands_run": ["command --if-any"],
  "verification": {"status": "passed", "evidence": ["What proves the result."]},
  "unresolved_gaps": [],
  "confidence": 0.86,
  "final_marker": null
}
```
"""
    router = _FakeLabRouter(tmp_path / "lab", status="running", transcript=transcript)
    router.sessions["lab-session-example"] = AOSession(
        id="lab-session-example",
        project_id="HermesAgentLab",
        status="running",
        workspace_path=str(tmp_path / "lab" / "worktrees" / "example"),
    )

    terminal = _await_implementation_terminal(
        execution_store=execution_store,
        event_store=event_store,
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        bridge=router,
        timeout_seconds=0.01,
    )

    assert terminal["timed_out"] is True
    assert terminal["authoritative_terminal"] is False


def test_lab_diff_scope_ignores_bootstrap_venv(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Worker makes a docs change with a bootstrap symlink present.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab.md"],
        "source": "docs",
        "payload": {
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["venv", "docs/lab.md"]),
        sources=["reliability"],
    )

    assert report["diff_scope"]["status"] == "in_scope"
    assert report["execution"]["touched_paths"] == ["docs/lab.md"]
    assert report["quarantined"] is False


def test_lab_executor_preserves_structured_acceptance_criteria(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    criterion = {
        "statement": "The lab observe-loop tests pass.",
        "verification_method": "test",
        "verification_detail": "scripts/run_tests.sh tests/gateway/test_lab_observe_loop.py -- -q",
        "machine_checkable": True,
    }
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Worker makes a docs change with an executable criterion.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab.md"],
        "source": "docs",
        "payload": {
            "acceptance_criteria": [criterion],
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "scripts/run_tests.sh tests/gateway/test_lab_observe_loop.py -- -q",
                "exit_code": 0,
                "output_excerpt": "24 passed in 1.0s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/lab.md"]),
        sources=["reliability"],
    )

    task_criteria = report["execution"]["implement"]["plan"]["tasks"][0]["acceptance_criteria"]
    assert task_criteria == [criterion]
    assert isinstance(task_criteria[0], dict)
    assert report["execution"]["pre_verification_cleanup"]["cleaned"] is True


def test_lab_acceptance_criteria_rewrite_direct_pytest_to_repo_wrapper(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Worker adds a small test.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["tests/gateway/test_lab_dogfood_example.py"],
        "source": "tests",
        "payload": {
            "acceptance_criteria": [{
                "statement": "The generated lab dogfood test passes.",
                "verification_method": "test",
                "verification_detail": "pytest tests/gateway/test_lab_dogfood_example.py",
                "machine_checkable": True,
            }],
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "scripts/run_tests.sh tests/gateway/test_lab_dogfood_example.py -- -q",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
        },
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["tests/gateway/test_lab_dogfood_example.py"]),
        sources=["reliability"],
    )

    task_criteria = report["execution"]["implement"]["plan"]["tasks"][0]["acceptance_criteria"]
    assert task_criteria[0]["verification_detail"] == "scripts/run_tests.sh tests/gateway/test_lab_dogfood_example.py -- -q"


def test_lab_default_acceptance_criterion_is_executable(monkeypatch):
    monkeypatch.delenv("HERMES_DEV_LAB_DEFAULT_VERIFICATION_COMMAND", raising=False)

    criteria = _candidate_acceptance_criteria({"payload": {}})

    assert criteria == [{
        "statement": "The lab dogfood task has executable verification evidence.",
        "verification_method": "test",
        "verification_detail": DEFAULT_LAB_VERIFICATION_COMMAND,
        "machine_checkable": True,
    }]


def test_lab_default_acceptance_criterion_can_be_overridden(monkeypatch):
    monkeypatch.setenv("HERMES_DEV_LAB_DEFAULT_VERIFICATION_COMMAND", "make test")

    criteria = _candidate_acceptance_criteria({"payload": {}})

    assert criteria[0]["verification_detail"] == "make test"
    assert criteria[0]["machine_checkable"] is True


def test_lab_commit_author_is_normalized_for_ci(monkeypatch, tmp_path):
    workspace = tmp_path / "repo"
    _init_git_repo(workspace)
    (workspace / "tests" / "gateway").mkdir(parents=True)
    (workspace / "tests" / "gateway" / "test_lab_dogfood_example.py").write_text(
        "def test_marker():\n    assert True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "tests/gateway/test_lab_dogfood_example.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test: add lab dogfood marker"],
        cwd=workspace,
        check=True,
    )
    previous_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()

    result = _normalize_lab_commit_author_for_ci(workspace)

    author_email = subprocess.check_output(["git", "log", "-1", "--format=%ae"], cwd=workspace, text=True).strip()
    assert result["status"] == "normalized"
    assert result["previous_author_emails"] == ["lab@example.test"]
    assert result["unsafe_author_emails"] == ["lab@example.test"]
    assert result["previous_head_sha"] == previous_head
    assert result["head_sha"] != previous_head
    assert author_email == "41898282+github-actions[bot]@users.noreply.github.com"


def test_lab_verification_recovers_from_direct_transcript(monkeypatch, tmp_path):
    db_path, _stable_db = _env(monkeypatch, tmp_path)
    verification_store = DevVerificationStore(db_path)
    command = "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q"
    run = verification_store.create_run(
        plan_id="plan-direct-recovery",
        task_id="task-direct-recovery",
        target_type="task",
        status="launched",
        results=[{
            "criterion_id": "crit-1",
            "statement": "Gateway API tests pass.",
            "verification_method": "test",
            "verification_detail": command,
            "machine_checkable": True,
            "status": "pending",
            "command_run": command,
            "exit_code": None,
            "passed": None,
            "output_excerpt": "",
            "notes": "",
            "warnings": [],
        }],
        executable_commands=[{
            "criterion_id": "crit-1",
            "command": command,
            "cwd": ".",
            "relative_cwd": ".",
        }],
        verified_against={"workspace_path": str(tmp_path / "lab" / "worktrees" / "verify")},
        verification_session_id="lab-session-verify",
        verification_runtime="ao",
    )
    transcript = """
Worker completed verification.
```json DEV_VERIFICATION_RESULTS
{
  "object": "hermes.dev_verification_results",
  "results": [
    {
      "criterion_id": "crit-1",
      "command_run": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q",
      "cwd": ".",
      "exit_code": 0,
      "output_excerpt": "=== Summary: 1 files, 22 tests passed, 0 failed in 0.8s",
      "notes": ""
    }
  ]
}
```
"""
    router = _FakeLabRouter(tmp_path / "lab", status="running", transcript=transcript)
    router.sessions["lab-session-verify"] = AOSession(
        id="lab-session-verify",
        project_id="HermesAgentLab",
        status="running",
        workspace_path=str(tmp_path / "lab" / "worktrees" / "verify"),
    )

    recovered = _recover_lab_verification_from_transcript(
        verification_store=verification_store,
        run=run,
        bridge=router,
    )

    assert recovered["status"] == "completed"
    assert recovered["verdict"] == "verified"
    assert recovered["counts"]["passed"] == 1
    assert recovered["results"][0]["passed"] is True


def test_lab_verification_timeout_uses_remaining_pass_budget(monkeypatch):
    monkeypatch.setenv("HERMES_DEV_LAB_VERIFY_TIMEOUT_SECONDS", "900")
    monkeypatch.setattr("gateway.dev_control.lab_loop.time.time", lambda: 250.0)

    timeout = _verification_timeout_seconds({"started_at": 100.0, "max_seconds": 200.0})

    assert timeout == 50.0


def test_lab_executor_quarantines_out_of_scope_worker_diff(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    commands = []
    monkeypatch.setattr("gateway.dev_control.lab_loop._run_command", lambda args, *, timeout=30.0: commands.append(args) or {"returncode": 0, "stdout": "", "stderr": ""})
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Worker attempts an engine edit despite scoped intent.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["gateway/dev_control/lab_loop.py"],
        "source": "todo",
        "payload": {"ci_repo": "Felippen/Oryn", "draft_pr_remote": "lab-origin"},
    }, approved=True)
    router = _FakeLabRouter(tmp_path / "lab", diff_paths=["agent/conversation_loop.py"])

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        bridge=router,
        sources=["reliability"],
        max_consecutive_failures=10,
    )

    assert report["status"] == "failed"
    assert report["diff_scope"]["status"] == "out_of_scope"
    assert "agent/conversation_loop.py" in report["diff_scope"]["rejected_paths"]
    assert report["draft_artifact"] is None
    assert commands == []
    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert outcome["source_refs"]["quarantined"] is True
    assert outcome["source_refs"]["draft_pr_ready"] is False


def test_lab_adversarial_fixture_proves_post_diff_quarantine(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Compliant worker makes a docs change; fixture simulates a forbidden engine diff.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab.md"],
        "source": "docs",
        "payload": {
            "adversarial_diff_paths": ["agent/conversation_loop.py"],
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "fixture would pass if quarantine did not suppress it",
            }],
        },
    }, approved=True)
    router = _FakeLabRouter(tmp_path / "lab", diff_paths=["docs/lab.md"])

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        bridge=router,
        sources=["reliability"],
        enable_adversarial_fixture=True,
        max_consecutive_failures=10,
    )

    assert report["status"] == "failed"
    assert report["quarantined"] is True
    assert report["diff_scope"]["status"] == "out_of_scope"
    assert report["diff_scope"]["rejected_paths"] == ["agent/conversation_loop.py"]
    assert report["draft_artifact"] is None
    assert report["execution"]["verification"]["status"] == "quarantined"
    assert report["execution"]["verification"]["verdict"] == "unknown"
    assert report["execution"]["verification"]["measured"] is False
    fixture = report["execution"]["adversarial_fixture"]
    assert fixture["applied"] is True
    assert fixture["paths"] == ["agent/conversation_loop.py"]
    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert outcome["source_refs"]["quarantined"] is True
    assert outcome["source_refs"]["adversarial_fixture"]["applied"] is True


def test_lab_adversarial_fixture_requires_explicit_enable(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Fixture request should be inert unless explicitly enabled.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/lab.md"],
        "source": "docs",
        "payload": {
            "verification_results": [{
                "criterion_id": "crit-1",
                "status": "passed",
                "command_run": "make test",
                "exit_code": 0,
                "output_excerpt": "1 passed in 0.1s",
            }],
            "adversarial_diff_paths": ["agent/conversation_loop.py"],
        },
    }, approved=True)
    router = _FakeLabRouter(tmp_path / "lab", diff_paths=["docs/lab.md"])

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        bridge=router,
        sources=["reliability"],
    )

    assert report["status"] == "completed"
    assert report["quarantined"] is False
    assert report["diff_scope"]["status"] == "in_scope"
    fixture = report["execution"]["adversarial_fixture"]
    assert fixture["requested"] is True
    assert fixture["enabled"] is False
    assert fixture["applied"] is False


def test_touched_paths_preserve_porcelain_paths_with_leading_status_space(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / "agent" / "conversation_loop.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('seed')\n", encoding="utf-8")
    subprocess.run(["git", "add", "agent/conversation_loop.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed agent file"], cwd=repo, check=True)

    target.write_text("print('changed')\n", encoding="utf-8")
    paths = _touched_paths_from_worktree(repo)

    assert "agent/conversation_loop.py" in paths
    assert "gent/conversation_loop.py" not in paths


def test_lab_executor_records_empty_diff_as_failure(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Worker makes no changes.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/noop.md"],
        "source": "docs",
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=[]),
        sources=["reliability"],
        max_consecutive_failures=10,
    )

    assert report["status"] == "failed"
    assert report["empty_diff"] is True
    assert report["draft_artifact"] is None
    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert outcome["source_refs"]["empty_diff"] is True


def test_lab_executor_worker_timeout_is_failed_outcome(monkeypatch, tmp_path):
    db_path, stable_db = _env(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_DEV_LAB_WORKER_TIMEOUT_SECONDS", "0.01")
    DevLabLoopStore(db_path).upsert_candidate({
        "prompt": "Worker never reaches terminal state.",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "target_paths": ["docs/timeout.md"],
        "source": "docs",
    }, approved=True)

    report = run_lab_loop_pass(
        db_path=db_path,
        stable_db_path=stable_db,
        bridge=_FakeLabRouter(tmp_path / "lab", diff_paths=["docs/timeout.md"], status="running"),
        sources=["reliability"],
        max_consecutive_failures=10,
    )

    assert report["status"] == "failed"
    assert report["execution"]["implement"]["timed_out"] is True
    outcome = DevReliabilityStore(db_path).list_outcomes(limit=1)[0]
    assert outcome["terminal_status"] == "failed"
