from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli import control_db as cp
from hermes_cli.control_worker import (
    DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S,
    DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S,
    _default_runner,
    build_agent_prompt,
    ControlDispatchWorker,
    DispatchWorkItem,
    run_agent_dispatch,
    run_deterministic_dispatch,
    validate_control_result,
)


def _payload(root: Path, parent: str | None = None):
    return {
        "schema": "statute_dispatch_v1",
        "silo": "statute",
        "repo_root": str(root),
        "allowed_paths": [str(root)],
        "task_type": "generic",
        "task_permissions": ["read", "test"],
        "parent_dispatch_id": parent,
        "instructions": "work",
        "constraints": {"no_live_db_mutation": True, "no_push": True},
    }


def _wave_payload(root: Path, *, parent: str | None = "disp_parent", write: bool = True, allowed_paths: list[str] | None = None):
    payload = _payload(root, parent=parent)
    payload["task_permissions"] = ["read", "test", *(["write"] if write else [])]
    payload["allowed_paths"] = allowed_paths or [str(root)]
    payload["constraints"] = {
        "no_live_db_mutation": True,
        "no_push": True,
        "wave": "F.1-F.5",
        "sprint_ids": ["F.1", "F.2", "F.3", "F.4", "F.5"],
    }
    return payload


def test_control_worker_claims_records_artifact_and_completes(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        pm = "statutepm:bootstrap"
        did = cp.create_dispatch_from_instance(conn, sender_instance_id=pm, receiver_profile="statute-worker", payload=_payload(repo, parent="disp_parent"))
    finally:
        conn.close()

    result = run_deterministic_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:test", dispatch_id=did)
    assert result["lease_epoch"] == 1
    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "completed"
        assert cp.get_latest_dispatch_result(conn, did)["result"]["status"] == "completed"
        artifacts = cp.list_artifacts(conn, did)
        assert artifacts
        artifact_path = Path(artifacts[0]["path"])
        assert artifact_path.exists()
        artifact = json.loads(artifact_path.read_text())
        assert artifact["dispatch_id"] == did
        assert artifact["status"] == "completed"
        inst = conn.execute("SELECT status, lease_expires_at_ms FROM cp_profile_instances WHERE instance_id='statute-worker:test'").fetchone()
        assert inst["status"] == "offline"
        assert inst["lease_expires_at_ms"] is not None
    finally:
        conn.close()


def test_claim_next_returns_none_without_work(tmp_path):
    worker = ControlDispatchWorker("statute-worker", "statute-worker:test", tmp_path / ".hermes")
    worker.heartbeat_once()
    assert worker.claim_next() is None


def test_agent_worker_builds_prompt_runs_subprocess_and_records_result(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        did = cp.create_dispatch_from_instance(conn, sender_instance_id="statutepm:bootstrap", receiver_profile="statute-worker", payload=_payload(repo, parent="disp_parent"))
    finally:
        conn.close()

    calls = {}

    def fake_runner(cmd, *, env, input_text, timeout_s, cwd):
        calls["cmd"] = cmd
        calls["env"] = env
        calls["input_text"] = input_text
        calls["timeout_s"] = timeout_s
        calls["cwd"] = cwd
        result = {
            "schema": "control_result_v1",
            "status": "completed",
            "summary": "agent completed",
            "artifacts": [],
            "tests": [{"command": "fake", "exit_code": 0}],
            "blockers": [],
        }
        return {"returncode": 0, "stdout": "CONTROL_RESULT_JSON:" + json.dumps(result), "stderr": ""}

    result = run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:agent", dispatch_id=did, runner=fake_runner, timeout_s=5)
    assert result["result"]["summary"] == "agent completed"
    assert "statute_dispatch_v1" in calls["input_text"]
    assert calls["env"]["HERMES_CONTROL_DISPATCH_ID"] == did
    assert calls["env"]["HERMES_CONTROL_LEASE_EPOCH"] == "1"
    assert calls["cmd"][-2:] == ["--query", "-"]
    assert "--ignore-rules" not in calls["cmd"]
    assert "--yolo" not in calls["cmd"]
    assert "--resume" not in calls["cmd"]
    assert "Post-wave closeout requirements" not in calls["input_text"]

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "completed"
        latest = cp.get_latest_dispatch_result(conn, did)["result"]
        assert latest["status"] == "completed"
        inst = conn.execute("SELECT status FROM cp_profile_instances WHERE instance_id='statute-worker:agent'").fetchone()
        assert inst["status"] == "offline"
    finally:
        conn.close()


def test_bounded_wave_prompt_includes_post_wave_closeout_requirements(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    item = DispatchWorkItem(
        dispatch_id="disp_wave",
        sender_profile="statutepm",
        receiver_profile="statute-worker",
        lease_epoch=1,
        payload=_wave_payload(repo),
    )

    prompt = build_agent_prompt(item)

    assert "Post-wave closeout requirements" in prompt
    assert "research -> diagnose -> plan -> write executable proposal -> looped oppositional review" in prompt
    assert "Identify the next wave" in prompt
    assert "Determine the next wave boundary" in prompt
    assert "do not stop at a one-sprint prompt" in prompt
    assert "autonomous_contract.py ready --db .contract-ledger/state.sqlite" in prompt
    assert "docs/dispatches/" in prompt
    assert "next_dispatch_prompt_missing" in prompt


def test_single_sprint_prompt_does_not_include_post_wave_closeout_requirements(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    payload = _wave_payload(repo)
    payload["constraints"].pop("wave")
    payload["constraints"]["sprint_ids"] = ["F.1"]
    item = DispatchWorkItem(
        dispatch_id="disp_single",
        sender_profile="statutepm",
        receiver_profile="statute-worker",
        lease_epoch=1,
        payload=payload,
    )

    prompt = build_agent_prompt(item)

    assert "Post-wave closeout requirements" not in prompt


def test_agent_worker_fails_malformed_success_output(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        did = cp.create_dispatch_from_instance(conn, sender_instance_id="statutepm:bootstrap", receiver_profile="statute-worker", payload=_payload(repo, parent="disp_parent"))
    finally:
        conn.close()

    def fake_runner(cmd, *, env, input_text, timeout_s, cwd):
        return {"returncode": 0, "stdout": "looks fine", "stderr": ""}

    result = run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:agent-bad", dispatch_id=did, runner=fake_runner, timeout_s=5)
    assert result["result"]["status"] == "failed"
    assert result["result"]["blockers"][0]["kind"] == "runtime_error"


def _make_dispatch(root: Path, repo: Path) -> str:
    return _make_dispatch_with_payload(root, _payload(repo, parent="disp_parent"))


def _make_dispatch_with_payload(root: Path, payload: dict) -> str:
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        return cp.create_dispatch_from_instance(
            conn,
            sender_instance_id="statutepm:bootstrap",
            receiver_profile="statute-worker",
            payload=payload,
        )
    finally:
        conn.close()


def _runner_for_result(result: dict):
    def fake_runner(cmd, *, env, input_text, timeout_s, cwd):
        return {"returncode": 0, "stdout": "CONTROL_RESULT_JSON:" + json.dumps(result), "stderr": ""}

    return fake_runner


def test_bounded_wave_success_without_dispatch_markdown_blocks_child(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    did = _make_dispatch_with_payload(root, _wave_payload(repo))
    result = {
        "schema": "control_result_v1",
        "status": "completed",
        "summary": "wave complete",
        "artifacts": [],
        "tests": [{"command": "fake", "exit_code": 0}],
        "blockers": [],
    }

    actual = run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:wave-missing", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    assert actual["result"]["status"] == "action_required"
    blocker = actual["result"]["blockers"][-1]
    assert blocker["kind"] == "next_dispatch_prompt_missing"
    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "blocked"
    finally:
        conn.close()


def test_bounded_wave_success_with_dispatch_markdown_completes(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    dispatch_dir = repo / "docs" / "dispatches"
    dispatch_dir.mkdir(parents=True)
    dispatch_file = dispatch_dir / "g1-next-wave-dispatch.md"
    dispatch_file.write_text("# next dispatch\n", encoding="utf-8")
    did = _make_dispatch_with_payload(root, _wave_payload(repo))
    result = {
        "schema": "control_result_v1",
        "status": "completed",
        "summary": "wave complete",
        "artifacts": [{"path": str(dispatch_file), "summary": "next dispatch"}],
        "tests": [{"command": "fake", "exit_code": 0}],
        "blockers": [],
    }

    actual = run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:wave-ok", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    assert actual["result"]["status"] == "completed"
    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "completed"
    finally:
        conn.close()


def test_bounded_wave_success_with_repo_relative_dispatch_markdown_completes(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    dispatch_dir = repo / "docs" / "dispatches"
    dispatch_dir.mkdir(parents=True)
    dispatch_file = dispatch_dir / "g1-next-wave-dispatch.md"
    dispatch_file.write_text("# next dispatch\n", encoding="utf-8")
    did = _make_dispatch_with_payload(root, _wave_payload(repo))
    result = {
        "schema": "control_result_v1",
        "status": "completed",
        "summary": "wave complete",
        "artifacts": [{"path": "docs/dispatches/g1-next-wave-dispatch.md", "summary": "next dispatch"}],
        "tests": [{"command": "fake", "exit_code": 0}],
        "blockers": [],
    }

    actual = run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:wave-relative-ok", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    assert actual["result"]["status"] == "completed"


def test_bounded_wave_success_without_write_permission_blocks_even_with_file(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    dispatch_dir = repo / "docs" / "dispatches"
    dispatch_dir.mkdir(parents=True)
    dispatch_file = dispatch_dir / "g1-next-wave-dispatch.md"
    dispatch_file.write_text("# next dispatch\n", encoding="utf-8")
    did = _make_dispatch_with_payload(root, _wave_payload(repo, write=False))
    result = {
        "schema": "control_result_v1",
        "status": "completed",
        "summary": "wave complete",
        "artifacts": [{"path": str(dispatch_file), "summary": "next dispatch"}],
        "tests": [],
        "blockers": [],
    }

    actual = run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:wave-no-write", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    assert actual["result"]["status"] == "action_required"
    assert actual["result"]["blockers"][-1]["kind"] == "next_dispatch_prompt_missing"
    assert "write permission" in actual["result"]["blockers"][-1]["message"]


def test_non_wave_success_without_dispatch_markdown_still_completes(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    did = _make_dispatch(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "completed",
        "summary": "single sprint complete",
        "artifacts": [],
        "tests": [],
        "blockers": [],
    }

    actual = run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:single-ok", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    assert actual["result"]["status"] == "completed"


def test_bounded_wave_action_required_result_is_preserved(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    did = _make_dispatch_with_payload(root, _wave_payload(repo))
    result = {
        "schema": "control_result_v1",
        "status": "action_required",
        "summary": "CodeRabbit auth required",
        "artifacts": [],
        "tests": [],
        "blockers": [{"kind": "auth", "message": "CodeRabbit auth required"}],
    }

    actual = run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:wave-action", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    assert actual["result"]["status"] == "action_required"
    assert [blocker["kind"] for blocker in actual["result"]["blockers"]] == ["auth"]


def test_validate_control_result_accepts_completed_with_warnings():
    result = {
        "schema": "control_result_v1",
        "status": "completed_with_warnings",
        "summary": "done with caveats",
        "artifacts": [],
        "tests": [],
        "blockers": [],
    }
    validated = validate_control_result(result)
    assert validated["status"] == "completed_with_warnings"


def test_validate_control_result_normalizes_known_blocked_alias_and_string_blockers():
    result = {
        "schema": "control_result_v1",
        "status": "blocked_action_required",
        "summary": "blocked on CodeRabbit",
        "artifacts": [],
        "tests": ["pytest tests/hermes_cli/test_control_worker.py"],
        "blockers": ["CodeRabbit credits exhausted"],
    }
    validated = validate_control_result(result)
    assert validated["status"] == "action_required"
    assert validated["tests"][0]["command"] == "pytest tests/hermes_cli/test_control_worker.py"
    assert validated["blockers"][0]["message"] == "CodeRabbit credits exhausted"
    marker = validated["blockers"][-1]
    assert marker["kind"] == "control_result_status_normalized"
    assert marker["raw_status"] == "blocked_action_required"


def test_validate_control_result_normalizes_unknown_status_with_blockers_to_action_required():
    result = {
        "schema": "control_result_v1",
        "status": "paused_for_supervisor",
        "summary": "cannot continue",
        "artifacts": [],
        "tests": [],
        "blockers": [{"kind": "auth", "message": "CodeRabbit auth required"}],
    }
    validated = validate_control_result(result)
    assert validated["status"] == "action_required"
    assert validated["blockers"][0]["message"] == "CodeRabbit auth required"
    assert validated["blockers"][-1]["raw_status"] == "paused_for_supervisor"


def test_validate_control_result_rejects_unknown_status_without_blockers():
    result = {
        "schema": "control_result_v1",
        "status": "paused_for_supervisor",
        "summary": "cannot continue",
        "artifacts": [],
        "tests": [],
        "blockers": [],
    }
    try:
        validate_control_result(result)
    except ValueError as exc:
        assert "control result status invalid" in str(exc)
    else:
        raise AssertionError("unknown status without blockers should be rejected")


def test_validate_control_result_normalizes_unknown_status_with_blockers_and_missing_arrays():
    result = {
        "schema": "control_result_v1",
        "status": "waiting_for_gate",
        "summary": "blocked",
        "blockers": ["manual review required"],
    }
    validated = validate_control_result(result)
    assert validated["status"] == "action_required"
    assert validated["artifacts"] == []
    assert validated["tests"] == []
    assert validated["blockers"][0]["message"] == "manual review required"
    assert validated["blockers"][-1]["raw_status"] == "waiting_for_gate"


def test_validate_control_result_failed_string_blockers_use_runtime_error_kind():
    result = {
        "schema": "control_result_v1",
        "status": "failed",
        "summary": "hard failure",
        "artifacts": [],
        "tests": [],
        "blockers": ["unit tests failed"],
    }
    validated = validate_control_result(result)
    assert validated["status"] == "failed"
    assert validated["blockers"][0]["kind"] == "runtime_error"


def test_agent_worker_completed_with_warnings_marks_dispatch_completed_and_preserves_exact_result_status(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    did = _make_dispatch(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "completed_with_warnings",
        "summary": "agent completed with warnings",
        "artifacts": [],
        "tests": [{"command": "fake", "exit_code": 0}],
        "blockers": [],
    }

    run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:warn", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "completed"
        latest = cp.get_latest_dispatch_result(conn, did)["result"]
        assert latest["status"] == "completed_with_warnings"
    finally:
        conn.close()


def test_agent_worker_action_required_records_blocked_dispatch_and_preserves_action_required_result(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    did = _make_dispatch(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "action_required",
        "summary": "needs supervisor",
        "artifacts": [],
        "tests": [],
        "blockers": [{"kind": "decision", "message": "choose"}],
    }

    run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:ar", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status,last_error FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "blocked"
        assert row["last_error"] == "needs supervisor"
        latest = cp.get_latest_dispatch_result(conn, did)["result"]
        assert latest["status"] == "action_required"
        assert latest["blockers"][0]["kind"] == "decision"
        status_event = conn.execute("SELECT status,summary FROM cp_status_events WHERE dispatch_id=? ORDER BY created_at_ms DESC LIMIT 1", (did,)).fetchone()
        assert status_event["status"] == "blocked"
        assert status_event["summary"] == "needs supervisor"
    finally:
        conn.close()


def test_agent_worker_unknown_blocked_status_records_action_required_with_raw_status_evidence(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    did = _make_dispatch(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "awaiting_supervisor_gate",
        "summary": "needs supervisor gate",
        "artifacts": ["non-file artifact note"],
        "tests": [],
        "blockers": [{"kind": "approval", "message": "approve CodeRabbit retry"}],
    }

    run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:unknown-status", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "blocked"
        latest_row = cp.get_latest_dispatch_result(conn, did)
        assert latest_row is not None
        latest = latest_row["result"]
        assert latest["status"] == "action_required"
        assert latest["artifacts"][0]["summary"] == "non-file artifact note"
        assert latest["artifacts"][0]["metadata"]["raw_artifact"] == "non-file artifact note"
        assert latest["blockers"][0]["message"] == "approve CodeRabbit retry"
        marker = latest["blockers"][-1]
        assert marker["kind"] == "control_result_status_normalized"
        assert marker["raw_status"] == "awaiting_supervisor_gate"
        artifacts = cp.list_artifacts(conn, did)
        assert all(artifact["path"] for artifact in artifacts)
    finally:
        conn.close()


def test_agent_worker_completed_with_blockers_does_not_mark_dispatch_completed(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    did = _make_dispatch(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "completed",
        "summary": "contradictory",
        "artifacts": [],
        "tests": [],
        "blockers": [{"kind": "runtime_error", "message": "still blocked"}],
    }

    run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:block", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status, last_error FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "failed"
        assert "successful control result cannot include blockers" in row["last_error"]
        latest = cp.get_latest_dispatch_result(conn, did)["result"]
        assert latest["status"] == "failed"
    finally:
        conn.close()


def test_agent_worker_invalid_but_parseable_control_result_is_preserved_in_runtime_artifact_redacted(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    did = _make_dispatch(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "bogus",
        "summary": "bad token secret=supersecretvalue",
        "artifacts": [],
        "tests": [],
        "blockers": [],
        "api_key": "supersecretvalue",
    }

    run_agent_dispatch(root=root, profile_id="statute-worker", instance_id="statute-worker:bad-json", dispatch_id=did, runner=_runner_for_result(result), timeout_s=5)

    artifact = root / "control-plane" / "agent-runs" / f"{did}-1.json"
    data = json.loads(artifact.read_text())
    assert data["invalid_control_result"]["status"] == "bogus"
    assert data["invalid_control_result"]["api_key"] == "***"
    assert "supersecretvalue" not in artifact.read_text()


def test_agent_worker_timeout_records_action_required_result_and_runtime_artifact_with_redacted_byte_tails(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    did = _make_dispatch(root, repo)

    def timeout_runner(cmd, *, env, input_text, timeout_s, cwd):
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout_s,
            output=b"partial stdout before timeout api_key=supersecretvalue\n",
            stderr=b"partial stderr before timeout password=supersecretvalue\n",
        )

    result = run_agent_dispatch(
        root=root,
        profile_id="statute-worker",
        instance_id="statute-worker:timeout",
        dispatch_id=did,
        runner=timeout_runner,
        timeout_s=5,
    )

    assert result["result"]["status"] == "action_required"
    assert result["result"]["blockers"][0]["kind"] == "hard_timeout"
    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "failed"
        latest_row = cp.get_latest_dispatch_result(conn, did)
        assert latest_row is not None
        latest = latest_row["result"]
        assert latest["status"] == "action_required"
        assert latest["artifacts"]
    finally:
        conn.close()

    artifact = root / "control-plane" / "agent-runs" / f"{did}-1.json"
    data = json.loads(artifact.read_text())
    assert data["schema"] == "control_agent_run_v1"
    assert data["timed_out"] is True
    assert data["hard_timed_out"] is True
    assert data["runner_kind"] == "custom"
    assert data["timeout_s"] == 5
    assert data["hard_timeout_s"] == 5
    assert data["soft_timeout_checks"] == []
    assert data["returncode"] is None
    assert "partial stdout before timeout" in data["stdout_tail"]
    assert "partial stderr before timeout" in data["stderr_tail"]
    assert "supersecretvalue" not in artifact.read_text()


def test_default_agent_runner_records_soft_checks_and_hard_kills_process_group(tmp_path):
    soft_checks = []
    script = (
        "import sys, time\n"
        "sys.stdin.read()\n"
        "print('started secret=supersecretvalue', flush=True)\n"
        "time.sleep(5)\n"
    )

    result = _default_runner(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        input_text="prompt",
        timeout_s=0.6,
        soft_timeout_s=0.05,
        cwd=str(tmp_path),
        lease_extender=lambda: soft_checks.append("extended"),
    )

    assert result["returncode"] is None
    assert result["timed_out"] is True
    assert result["hard_timed_out"] is True
    assert result["terminated"] is True
    assert result["killed"] in {True, False}
    assert result["timeout_s"] == 0.6
    assert result["hard_timeout_s"] == 0.6
    assert result["soft_timeout_s"] == 0.05
    assert result["soft_timeout_checks"]
    assert soft_checks
    assert "started" in result["stdout"]


def test_agent_worker_defaults_are_ten_minute_soft_and_fifty_minute_hard():
    assert DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S == 600.0
    assert DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S == 3000.0
