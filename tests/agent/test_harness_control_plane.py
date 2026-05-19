import json
from pathlib import Path

from agent.harness_control_plane import (
    classify_memory_admission,
    ensure_profile_eval_suite,
    evaluate_promotion_gate,
    learning_health_unavailable_summary,
    learning_health_summary,
    learning_snapshot_summary,
    promotion_gate_summary,
    promote_skill,
    record_approval_decision,
    record_eval_suite,
    record_goal_decision,
    record_harness_event,
    record_memory_admission,
    record_mutation_contract,
    record_replay_case,
    record_skill_load,
    record_skill_mutation,
    record_turn_result,
    replay_corpus_summary,
    run_core_harness,
)
from hermes_constants import get_hermes_home


class _Agent:
    api_mode = "codex_app_server"
    session_id = "sess-1"
    platform = "cli"
    provider = "openai"
    model = "gpt-test"


def test_record_turn_result_emits_codex_trace(_isolate_hermes_home):
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "terminal"}},
                    {"function": {"name": "skill_view"}},
                ],
            },
            {"role": "tool", "tool_name": "terminal"},
        ],
        "completed": True,
        "api_calls": 2,
        "codex_thread_id": "thread-1",
        "codex_turn_id": "turn-1",
    }

    record = record_turn_result(_Agent(), result)

    assert record["runtime"] == "codex_app_server"
    assert record["tool_call_count"] == 3
    assert record["tool_names"] == ["skill_view", "terminal"]
    assert record["trace_id"].startswith("turn_")
    trace_file = get_hermes_home() / "harness" / "turn-traces.jsonl"
    assert trace_file.exists()
    assert json.loads(trace_file.read_text().splitlines()[0])["codex_turn_id"] == "turn-1"
    event_file = get_hermes_home() / "harness" / "harness-events.jsonl"
    assert any(
        json.loads(line)["event_type"] == "turn.finish"
        for line in event_file.read_text().splitlines()
    )


def test_harness_event_redacts_and_summarizes_payload(_isolate_hermes_home):
    record_harness_event(
        "tool.start",
        {
            "api_key": "sk-" + "x" * 64,
            "value": "sk-" + "y" * 64,
            "preview": "long text " * 80,
            "nested": {"authorization": "Bearer secret"},
        },
        session_id="sess-1",
        component="test",
    )

    event_file = get_hermes_home() / "harness" / "harness-events.jsonl"
    row = json.loads(event_file.read_text().splitlines()[0])

    assert row["event_type"] == "tool.start"
    assert row["payload"]["api_key"] == "[REDACTED]"
    assert row["payload"]["value"] == "[REDACTED]"
    assert row["payload"]["nested"]["authorization"] == "[REDACTED]"
    assert "sha256:" in row["payload"]["preview"]


def test_goal_skill_and_approval_events_appear_in_learning_health(_isolate_hermes_home):
    record_goal_decision(
        session_id="sess-1",
        action="set",
        goal="make /goal use loaded skills",
        should_continue=True,
        message="private setup reason: the database password was rotated",
        turn_count=0,
    )
    record_goal_decision(
        session_id="sess-1",
        action="judge",
        goal="make /goal use loaded skills",
        should_continue=False,
        message="private judge reason: deploy token failed",
        turn_count=1,
    )
    record_skill_load(
        session_id="sess-1",
        name="hermes-agent",
        command="/hermes-agent",
        arg="fix goal",
        source="goal",
    )
    record_approval_decision(
        session_id="sess-1",
        choice="approved",
        resolved=1,
        resolve_all=False,
    )

    health = learning_health_summary()

    assert health["events"]["goal_events"] == 2
    assert health["events"]["skill_loads"] == 1
    assert health["events"]["approval_events"] == 1
    assert health["events"]["by_type"]["skill.loaded"] == 1

    raw_events = (get_hermes_home() / "harness" / "harness-events.jsonl").read_text()
    assert "private setup reason" not in raw_events
    assert "private judge reason" not in raw_events
    assert "deploy token failed" not in raw_events
    rows = [json.loads(line) for line in raw_events.splitlines()]
    goal_payloads = [row["payload"] for row in rows if str(row.get("event_type", "")).startswith("goal.")]
    assert all("message" not in payload for payload in goal_payloads)
    assert all(payload["message_present"] is True for payload in goal_payloads)
    assert all(payload["message_chars"] > 0 for payload in goal_payloads)


def test_turn_trace_records_error_metadata_without_raw_content(_isolate_hermes_home):
    private_error = "private failure: bearer token for customer abc was rejected"

    record = record_turn_result(
        _Agent(),
        {
            "messages": [],
            "completed": False,
            "partial": True,
            "api_calls": 1,
            "error": private_error,
        },
    )

    assert "error" not in record
    assert record["error_present"] is True
    assert record["error_chars"] == len(private_error)
    assert record["error_sha256"]

    trace_file = get_hermes_home() / "harness" / "turn-traces.jsonl"
    event_file = get_hermes_home() / "harness" / "harness-events.jsonl"
    raw = trace_file.read_text() + event_file.read_text()
    assert private_error not in raw
    assert "bearer token for customer" not in raw

    health = learning_health_summary()
    assert health["traces"]["failure_count"] == 1


def test_memory_tool_rejects_task_progress_before_durable_write(_isolate_hermes_home):
    from tools.memory_tool import memory_tool

    class _Store:
        def __init__(self):
            self.added = []

        def add(self, target, content):
            self.added.append((target, content))
            return {"success": True}

    store = _Store()
    raw = memory_tool(
        "add",
        target="memory",
        content="Fixed issue #1234 and committed sha abcdef123456.",
        store=store,  # type: ignore[arg-type]
    )
    result = json.loads(raw)

    assert result["success"] is False
    assert result["admission"]["decision"] == "reject"
    assert result["admission"]["route"] == "session_search"
    assert result["admission_recorded"] is True
    assert store.added == []

    admissions = (get_hermes_home() / "harness" / "memory-admissions.jsonl").read_text()
    assert "Fixed issue" not in admissions
    assert "abcdef123456" not in admissions


def test_memory_admission_classifies_task_progress_as_session_search(_isolate_hermes_home):
    admission = classify_memory_admission(
        action="add",
        target="memory",
        content="Fixed issue #1234 and committed sha abcdef123456.",
    )

    assert admission["decision"] == "reject"
    assert admission["kind"] == "task_progress_or_session_outcome"
    assert admission["route"] == "session_search"

    record = record_memory_admission(
        action="add",
        target="memory",
        content="Fixed issue #1234 and committed sha abcdef123456.",
        result={"success": True},
        admission=admission,
    )
    assert record["decision"] == "reject"


def test_skill_registry_tracks_draft_and_promotion(_isolate_hermes_home):
    record_skill_mutation(
        action="patch",
        name="harness-skill",
        result={"success": True},
        file_path="SKILL.md",
        origin="foreground",
    )

    health = learning_health_summary()
    assert health["skills"]["registered"] == 1
    assert health["skills"]["needs_verification"] == 1
    assert health["mutations"]["draft"] == 1


def test_mutation_contract_summary_tracks_component_status(_isolate_hermes_home):
    record_mutation_contract(
        component="gateway",
        action="patch",
        target="tui_gateway/server.py",
        evidence=["goal skill expansion failed"],
        prediction="Goal slash-skill prompts become real skill invocations.",
        rollback="revert tui_gateway/server.py patch",
        verification=["pytest tests/tui_gateway/test_goal_command.py -q"],
        status="draft",
    )

    health = learning_health_summary()

    assert health["mutations"]["total"] == 1
    assert health["mutations"]["draft"] == 1
    assert health["mutations"]["by_component"]["gateway"] == 1


def test_profile_eval_suite_metadata_is_profile_scoped(monkeypatch, _isolate_hermes_home):
    monkeypatch.setenv("HERMES_PROFILE", "founder")

    ensure_profile_eval_suite()
    suite = record_eval_suite(
        profile=None,
        name="harness-core",
        status="passed",
        checks=["trace"],
        result="ok",
    )

    assert suite["status"] == "passed"
    health = learning_health_summary()
    assert health["profile"] == "founder"
    assert health["evals"]["suite_count"] == 1
    assert health["evals"]["missing_run_count"] == 0


def test_profile_eval_suite_registers_all_core_cases(_isolate_hermes_home):
    state = ensure_profile_eval_suite()

    suite = state["profiles"]["default"]["suites"]["harness-core"]

    assert suite["checks"] == [
        "turn trace emitted",
        "harness event emitted",
        "memory admission classified",
        "skill mutation registered",
        "mutation contract registered",
        "goal skill expansion covered",
        "dashboard health endpoint responds",
    ]
    assert {case["id"] for case in suite["cases"]} == {
        "goal-skill-expansion",
        "codex-trace-projection",
        "harness-event-safety",
        "dashboard-learning-health",
        "memory-admission-routing",
        "skill-mutation-contract",
        "mutation-contract-summary",
    }


def test_learning_health_degrades_when_eval_suite_write_fails(monkeypatch, _isolate_hermes_home):
    import agent.harness_control_plane as hcp

    def fail_eval_suite(profile=None):
        raise OSError("read-only harness")

    monkeypatch.setattr(hcp, "ensure_profile_eval_suite", fail_eval_suite)

    health = hcp.learning_health_summary()

    assert health["degraded"] is True
    assert "read-only harness" in health["error"]
    assert health["profile"] == "default"
    assert "traces" in health
    assert "evals" in health


def test_learning_health_unavailable_summary_is_complete(_isolate_hermes_home):
    health = learning_health_unavailable_summary("boom")

    assert health["degraded"] is True
    assert health["error"] == "boom"
    assert health["traces"]["total"] == 0
    assert health["events"]["total"] == 0
    assert health["memory"]["total"] == 0
    assert health["skills"]["registered"] == 0
    assert health["mutations"]["total"] == 0
    assert health["replay_corpus"]["total"] == 0
    assert health["promotion_gates"]["total"] == 0


def test_turn_trace_has_standard_shape_and_failure_taxonomy(_isolate_hermes_home):
    private_error = "private failure: customer token sk-" + "x" * 32
    private_reason = f"error_near_max_iterations({private_error})"

    record = record_turn_result(
        _Agent(),
        {
            "messages": [],
            "completed": False,
            "api_calls": 1,
            "turn_exit_reason": private_reason,
            "error": private_error,
        },
    )

    assert record["trace_schema"] == {
        "name": "hermes.turn_trace",
        "version": 1,
        "content_policy": "metadata_only",
    }
    assert record["turn_exit_reason"] == "error_near_max_iterations"
    assert record["turn_exit_reason_raw_present"] is True
    assert record["turn_exit_reason_raw_sha256"]
    assert record["failure_kind"] == "iteration_budget"

    harness_dir = get_hermes_home() / "harness"
    raw = "\n".join(
        path.read_text()
        for path in [harness_dir / "turn-traces.jsonl", harness_dir / "harness-events.jsonl"]
    )
    assert private_error not in raw
    assert private_reason not in raw
    assert "customer token" not in raw

    snapshot = learning_snapshot_summary()
    assert snapshot["content_policy"] == "metadata_only"
    assert snapshot["failure_taxonomy"]["iteration_budget"] == 1
    assert snapshot["trace_schema"]["name"] == "hermes.turn_trace"
    assert "hermes_home" not in snapshot


def test_replay_corpus_records_failure_cases_without_raw_content(_isolate_hermes_home):
    private_note = "replay this failure; bearer token was rejected"
    private_check = "private merger plan replay"

    replay = record_replay_case(
        source_trace_id="turn_abc123",
        failure_kind="runtime_error",
        checks=[private_check],
        status="candidate",
        note=private_note,
    )

    assert replay["replay_id"].startswith("replay_")
    assert replay["source_trace_id"] == "turn_abc123"
    assert replay["failure_kind"] == "runtime_error"
    assert replay["checks_count"] == 1
    assert replay["checks_sha256"]
    assert "checks" not in replay
    assert replay["note_present"] is True
    assert replay["note_sha256"]
    assert "note" not in replay

    raw = (get_hermes_home() / "harness" / "replay-corpus.jsonl").read_text()
    assert private_note not in raw
    assert private_check not in raw
    assert "bearer token" not in raw
    assert "private merger" not in raw

    summary = replay_corpus_summary()
    assert summary["total"] == 1
    assert summary["by_status"] == {"candidate": 1}
    assert summary["by_failure_kind"] == {"runtime_error": 1}


def test_mutation_contracts_and_core_outputs_are_fingerprinted_not_raw(_isolate_hermes_home):
    private_evidence = "goal failed because database password rotated"
    private_prediction = "future turns avoid leaking customer token"
    private_target = "/home/alice/private-project/secret-skill"
    private_command = "scripts/run_tests.sh tests/agent/test_harness_control_plane.py --token sk-" + "z" * 32
    private_output = "pytest failed with OPENAI_API_KEY=sk-" + "y" * 32

    mutation = record_mutation_contract(
        component="skill",
        action="patch",
        target=private_target,
        evidence=[private_evidence],
        prediction=private_prediction,
        rollback="restore private branch path",
        verification=["scripts/run_tests.sh tests/agent/test_harness_control_plane.py -q"],
        status="draft",
    )

    assert mutation["target_present"] is True
    assert mutation["target_sha256"]
    assert "target" not in mutation
    assert mutation["evidence_count"] == 1
    assert mutation["prediction_present"] is True
    assert mutation["prediction_sha256"]
    assert "evidence" not in mutation
    assert "prediction" not in mutation

    result = run_core_harness(
        case_ids=["harness-event-safety"],
        runner=lambda case: {
            "status": "failed",
            "command": private_command,
            "output_tail": private_output,
        },
    )

    raw = "\n".join(
        path.read_text()
        for path in (get_hermes_home() / "harness").glob("*.json*")
    )
    assert private_evidence not in raw
    assert private_prediction not in raw
    assert private_target not in raw
    assert private_command not in raw
    assert private_output not in raw
    assert "database password" not in raw
    assert "private-project" not in raw
    assert "OPENAI_API_KEY" not in raw
    assert result["cases"][0]["command_present"] is True
    assert result["cases"][0]["command_sha256"]
    assert "command" not in result["cases"][0]
    assert result["cases"][0]["output_tail_present"] is True
    assert result["cases"][0]["output_tail_sha256"]
    assert "output_tail" not in result["cases"][0]


def test_promotion_gate_blocks_until_required_offline_eval_passes(_isolate_hermes_home):
    blocked = evaluate_promotion_gate(
        component="skill",
        target="/home/alice/private-project/secret-skill",
        required_suites=["harness-core"],
    )

    assert blocked["status"] == "blocked"
    assert blocked["target_present"] is True
    assert blocked["target_sha256"]
    assert "target" not in blocked
    assert blocked["missing_suites"] == ["harness-core"]

    denied = promote_skill("harness-skill", evidence="SECRET_EVAL_OUTPUT customer token abc")
    assert denied["promotion_status"] == "blocked"
    assert denied["status"] == "draft"
    assert denied["promotion_gate_status"] == "blocked"

    record_eval_suite(
        profile=None,
        name="harness-core",
        status="passed",
        checks=["trace", "replay"],
        result="SECRET_EVAL_OUTPUT customer token abc",
    )
    passed = evaluate_promotion_gate(
        component="skill",
        target="harness-skill",
        required_suites=["harness-core"],
    )

    assert passed["status"] == "passed"
    assert passed["missing_suites"] == []

    promoted = promote_skill("harness-skill", evidence="SECRET_EVAL_OUTPUT customer token abc")
    assert promoted["status"] == "promoted"
    assert promoted["promotion_status"] == "verified"
    assert promoted["promotion_evidence_present"] is True
    assert "promotion_evidence" not in promoted

    summary = promotion_gate_summary()
    assert summary["total"] == 4
    assert summary["blocked"] == 2
    assert summary["passed"] == 2
    assert summary["by_component"] == {"skill": 4}

    snapshot = learning_snapshot_summary()
    raw_snapshot = json.dumps(snapshot, sort_keys=True)
    assert "SECRET_EVAL_OUTPUT" not in raw_snapshot
    assert "/home/alice/private-project" not in raw_snapshot
    assert "customer token" not in raw_snapshot
