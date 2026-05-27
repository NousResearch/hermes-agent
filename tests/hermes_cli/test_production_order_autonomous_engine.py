from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.production_order_db import (
    WORKFLOW_SPEC_SOURCE,
    list_production_orders,
    run_architect_spec_bridge,
    run_full_bridge,
    run_orchestrator_triage_bridge,
)
from hermes_cli.production_order_dispatch import (
    DispatchManifestError,
    apply_accepted_result_action,
    build_dispatch_manifest,
    build_manual_fallback_handoff,
    build_profile_task_envelope,
    execute_profile_dispatch,
    ingest_profile_result_packet,
)
from hermes_cli.production_order_autonomous import (
    apply_profile_result_return,
    brief_packet_from_approved_action_envelope,
    collect_profile_result_packet,
    create_approved_action_envelope,
    hermes_profile_runner,
    invoke_profile_task,
    run_approved_action_envelope_autonomously,
    run_production_order_autonomously,
    validate_approved_action_envelope,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    yield


@pytest.fixture
def conn(kanban_home):
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def sample_brief() -> dict:
    return {
        "title": "Test authentication feature",
        "objective": "Add JWT authentication to the Relay demo",
        "target repo or workspace": "relay-go-app",
        "scope": "Implement /login and /register endpoints",
        "out of scope": "OAuth, password reset",
        "acceptance criteria": "All tests pass, protected routes reject unauthenticated requests",
        "stop conditions": "Secret management requires external service",
        "approval boundaries": "No spending, no publishing",
        "constraints": "Use existing Go module",
        "expected output": "Working auth endpoints with tests",
    }


def architect_spec_packet(production_order_id: str) -> dict:
    return {
        "production_order_id": production_order_id,
        "packet_type": "architect_spec_packet",
        "stage": "architect_spec",
        "owner_profile": "architect_os",
        "source_state": "ARCHITECT_SPEC",
        "objective": "Specify the bounded production-order handoff.",
        "source_truth": [WORKFLOW_SPEC_SOURCE],
        "scope": ["Attach a DevOS handoff packet and advance the production order state."],
        "out_of_scope": ["DevOS implementation execution"],
        "acceptance_criteria": [
            "Production order transitions to ARCHITECT_READY_FOR_DEV.",
            "Current owner becomes dev_os.",
        ],
        "devos_task": "Prepare for implementation from the approved spec.",
        "files_or_areas_allowed": [
            "hermes_cli/production_order_db.py",
            "hermes_cli/production_order_dispatch.py",
            "tests/hermes_cli/test_production_order_autonomous_engine.py",
        ],
        "stop_conditions": [
            "Production order is not in ARCHITECT_SPEC.",
            "Current owner is not architect_os.",
        ],
        "approval_boundaries": ["Do not trigger unapproved execution."],
        "artifact_references": ["architect-spec.json"],
        "next_state": "ARCHITECT_READY_FOR_DEV",
    }


def devos_build_packet(production_order_id: str) -> dict:
    return {
        "production_order_id": production_order_id,
        "packet_type": "devos_build_packet",
        "owner_profile": "dev_os",
        "source_state": "ARCHITECT_READY_FOR_DEV",
        "result_type": "build_complete",
        "summary": "Implemented the approved bridge and preserved graph semantics.",
        "files_changed": [
            "hermes_cli/production_order_db.py",
            "hermes_cli/production_order_dispatch.py",
            "tests/hermes_cli/test_production_order_autonomous_engine.py",
        ],
        "tests_run": [
            "pytest tests/hermes_cli/test_production_order_autonomous_engine.py -q",
        ],
        "test_status": "green",
        "limitations_or_notes": ["AuditOS should verify smoke evidence against the existing board."],
        "next_handoff_target": "audit_os",
    }


def audit_review_packet(production_order_id: str, source_state: str = "DEV_COMPLETE") -> dict:
    return {
        "production_order_id": production_order_id,
        "owner_profile": "audit_os",
        "source_state": source_state,
        "review_result": "passed",
        "summary": "Audit verified the implementation and evidence.",
        "evidence": ["tests passed", "changed files reviewed"],
        "tests_reviewed": ["pytest tests/hermes_cli/test_production_order_autonomous_engine.py -q"],
        "verdict": "PASS",
        "risks_or_notes": ["No blocking risks found."],
        "next_handoff_target": "architect_os",
    }


def architect_reconcile_packet(production_order_id: str, source_state: str = "AUDIT_PASSED") -> dict:
    return {
        "production_order_id": production_order_id,
        "packet_type": "architect_reconcile_packet",
        "owner_profile": "architect_os",
        "source_state": source_state,
        "reconcile_result": "accepted",
        "summary": "Implementation remains aligned with the approved architecture.",
        "architecture_alignment": "aligned",
        "drift_assessment": "No architecture drift.",
        "spec_patch_needed": False,
        "risks_or_notes": ["No rework needed."],
        "next_handoff_target": "default",
    }


def final_review_packet(production_order_id: str, source_state: str = "ARCHITECT_ACCEPTED") -> dict:
    return {
        "production_order_id": production_order_id,
        "owner_profile": "default",
        "source_state": source_state,
        "final_review_result": "accepted",
        "summary": "Final review confirms the approved brief is complete.",
        "original_brief_alignment": "Matches the approved brief.",
        "artifacts_reviewed": ["DevOS result", "AuditOS result", "ArchitectOS reconcile result"],
        "evidence_summary": "All stage evidence is present and accepted.",
        "final_status": "DONE",
        "next_action": "report_done_to_jarren",
    }


def _reload_order(conn, production_order_id: str):
    for order in list_production_orders(conn):
        if order.production_order_id == production_order_id:
            return order
    raise AssertionError(f"production order not found: {production_order_id}")


def create_architect_spec_order(conn, sample_brief):
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    return run_orchestrator_triage_bridge(conn, production_order_id=po.production_order_id)


def create_ready_for_dev_order(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)
    return run_architect_spec_bridge(
        conn,
        production_order_id=po.production_order_id,
        architect_packet=architect_spec_packet(po.production_order_id),
    )


def test_execute_profile_dispatch_explicitly_reports_manual_fallback_until_safe_invoker_exists(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)

    result = execute_profile_dispatch(conn, po.production_order_id)

    assert result["executed"] is False
    assert result["fallback_required"] is True
    assert result["target_profile"] == "dev_os"
    assert result["result_packet"] is None
    assert result["manual_fallback"]["target_profile"] == "dev_os"
    assert result["manual_fallback"]["bridge_function"] == "run_devos_complete_bridge"
    assert "No safe synchronous profile invocation mechanism is available" in result["error"]
    assert result["next_action"] == "manual_fallback_required"


def test_manual_fallback_contract_remains_available_for_orchestrator_direct_bridge_routes(conn, sample_brief):
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    manifest = build_dispatch_manifest(conn, po.production_order_id)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    handoff = build_manual_fallback_handoff(conn, po.production_order_id)

    assert manifest.target_profile == "orchestrator_os"
    assert manifest.bridge_function == "run_orchestrator_triage_bridge"
    assert envelope.expected_output_packet["bridge_function"] == "run_orchestrator_triage_bridge"
    assert handoff.target_profile == "orchestrator_os"
    assert handoff.bridge_function == "run_orchestrator_triage_bridge"
    assert "Call run_orchestrator_triage_bridge(" in handoff.result_return_action


def test_downstream_result_action_requires_ingestion_before_application(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    packet = devos_build_packet(po.production_order_id)

    with pytest.raises(DispatchManifestError, match="accepted"):
        apply_accepted_result_action(conn, po.production_order_id, result_packet=packet)

    ingestion = ingest_profile_result_packet(conn, po.production_order_id, packet)
    applied = apply_accepted_result_action(conn, po.production_order_id, result_packet=packet)

    assert ingestion["accepted"] is True
    assert ingestion["runtime_action"].startswith("run_devos_complete_bridge(")
    assert applied["applied"] is True
    assert applied["bridge_function"] == "run_devos_complete_bridge"
    assert applied["to_state"] == "DEV_COMPLETE"


def test_invoke_profile_task_accepts_fake_runner_and_collects_single_packet(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)

    def fake_runner(payload: dict) -> dict:
        assert payload["production_order_id"] == po.production_order_id
        assert payload["target_profile"] == "dev_os"
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 17,
            "result_channel": devos_build_packet(po.production_order_id),
        }

    invocation = invoke_profile_task(envelope, runner=fake_runner, timeout_seconds=30)
    packet = collect_profile_result_packet(invocation, envelope)

    assert invocation.exit_code == 0
    assert invocation.duration_ms == 17
    assert invocation.log_ref.startswith("profile-invocation:")
    assert packet["production_order_id"] == po.production_order_id
    assert packet["owner_profile"] == "dev_os"
    assert packet["source_state"] == envelope.source_state


def test_extracts_one_raw_json_object_from_stdout(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    packet = devos_build_packet(po.production_order_id)

    invocation = invoke_profile_task(
        envelope,
        runner=lambda payload: {
            "stdout": json.dumps(packet),
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 5,
        },
        timeout_seconds=30,
    )

    result = collect_profile_result_packet(invocation, envelope)
    assert result["production_order_id"] == po.production_order_id


def test_extracts_one_json_object_from_markdown_code_fence(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    packet = devos_build_packet(po.production_order_id)
    fenced = "Some commentary.```json\n" + json.dumps(packet) + "\n```"

    invocation = invoke_profile_task(
        envelope,
        runner=lambda payload: {
            "stdout": fenced,
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 5,
        },
        timeout_seconds=30,
    )

    result = collect_profile_result_packet(invocation, envelope)
    assert result["packet_type"] == "devos_build_packet"


def test_extracts_one_json_object_surrounded_by_prose(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    packet = devos_build_packet(po.production_order_id)
    prose = "Result below:\n" + json.dumps(packet) + "\nEnd of message."

    invocation = invoke_profile_task(
        envelope,
        runner=lambda payload: {
            "stdout": prose,
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 5,
        },
        timeout_seconds=30,
    )

    result = collect_profile_result_packet(invocation, envelope)
    assert result["packet_type"] == "devos_build_packet"


def test_rejects_multiple_json_objects_embedded_in_text(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    p1 = devos_build_packet(po.production_order_id)
    p2 = devos_build_packet(po.production_order_id)
    multi = json.dumps(p1) + "\n" + json.dumps(p2)

    invocation = invoke_profile_task(
        envelope,
        runner=lambda payload: {
            "stdout": multi,
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 5,
        },
        timeout_seconds=30,
    )

    with pytest.raises(ValueError, match="multiple competing packets"):
        collect_profile_result_packet(invocation, envelope)


@pytest.mark.parametrize(
    ("runner_payload", "error_match"),
    [
        ({"stdout": "done", "stderr": "", "exit_code": 0, "duration_ms": 5}, "free-text-only output"),
        ({"stdout": "{not-json}", "stderr": "", "exit_code": 0, "duration_ms": 5}, "malformed JSON"),
        (
            {
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "duration_ms": 5,
                "result_channel": [
                    {"packet_type": "devos_build_packet"},
                    {"packet_type": "devos_build_packet"},
                ],
            },
            "multiple competing packets",
        ),
    ],
)
def test_collect_profile_result_packet_rejects_ambiguous_or_non_structured_output(
    conn,
    sample_brief,
    runner_payload,
    error_match,
):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)

    invocation = invoke_profile_task(
        envelope,
        runner=lambda payload: runner_payload,
        timeout_seconds=30,
    )

    with pytest.raises(ValueError, match=error_match):
        collect_profile_result_packet(invocation, envelope)


def test_malformed_profile_output_records_bounded_diagnostics(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)

    result = run_production_order_autonomously(
        conn,
        po.production_order_id,
        runner=lambda payload: {
            "stdout": "{not-json}",
            "stderr": "the-stderr",
            "exit_code": 0,
            "duration_ms": 3,
            "resolved_hermes_home": "/tmp/hermes-profile",
            "resolved_model_default": "gpt-5.4-mini",
            "resolved_model_provider": "openai-codex",
            "resolved_model_base_url": "https://chatgpt.com/backend-api/codex",
        },
        max_steps=5,
        max_retries=1,
        timeout_seconds=30,
    )

    # Ensure run stopped with validation failure
    assert result.terminal_reason == "validation_failed"

    # Find the failed dispatch event and inspect diagnostics
    from hermes_cli.production_order_dispatch import list_dispatch_events

    events = list_dispatch_events(conn, po.production_order_id)
    failed = [e for e in events if e["event_type"] == "dispatch_failed"]
    assert failed, "no dispatch_failed event recorded"
    evt = failed[-1]

    # result field holds JSON diagnostics
    diag = {}
    try:
        diag = json.loads(evt["result"] or "{}")
    except Exception:
        pytest.fail("dispatch event result is not valid JSON diagnostics")

    assert "parse_error" in diag
    assert "invocation_log_ref" in diag
    assert "stdout_preview" in diag
    assert "stderr_preview" in diag
    assert "result_channel_preview" in diag
    assert diag["resolved_hermes_home"] == "/tmp/hermes-profile"
    assert diag["resolved_model_default"] == "gpt-5.4-mini"
    assert diag["resolved_model_provider"] == "openai-codex"
    assert diag["resolved_model_base_url"] == "https://chatgpt.com/backend-api/codex"

    # Previews must be capped to 4000 chars
    for key in ("stdout_preview", "stderr_preview", "result_channel_preview"):
        assert isinstance(diag[key], str)
        assert len(diag[key]) <= 4000


def test_profile_runtime_fails_fast_when_profile_config_has_blank_provider_or_model(
    conn,
    sample_brief,
    tmp_path,
    monkeypatch,
):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    profile_home = tmp_path / "architect_os"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text(
        "model:\n"
        "  default: ''\n"
        "  provider: ''\n"
        "  base_url: https://chatgpt.com/backend-api/codex\n",
        encoding="utf-8",
    )

    import hermes_cli.production_order_autonomous as autonomous

    monkeypatch.setattr(autonomous, "resolve_profile_env", lambda target_profile: profile_home)

    agent_called = {"value": False}

    class ExplodingAgent:
        def __init__(self, *args, **kwargs):
            agent_called["value"] = True
            raise AssertionError("AIAgent must not be constructed when runtime config is blank")

    monkeypatch.setattr(autonomous, "AIAgent", ExplodingAgent)

    result = autonomous._run_profile_agent_once(
        envelope,
        timeout_seconds=30,
        runtime_session_id="blank-config-test",
    )

    assert result["exit_code"] == 1
    assert "profile runtime config missing" in result["stderr"]
    assert result["resolved_hermes_home"] == str(profile_home)
    assert result["resolved_model_default"] == ""
    assert result["resolved_model_provider"] == ""
    assert result["resolved_model_base_url"] == "https://chatgpt.com/backend-api/codex"
    assert agent_called["value"] is False


@pytest.mark.parametrize(
    ("mutator", "error_match"),
    [
        (lambda packet: packet.__setitem__("production_order_id", "PO-wrong"), "production_order_id"),
        (lambda packet: packet.__setitem__("owner_profile", "architect_os"), "owner_profile"),
        (lambda packet: packet.__setitem__("source_state", "ARCHITECT_SPEC"), "source_state"),
        (lambda packet: packet.__setitem__("current_owner_profile", "audit_os"), "mutate workflow state directly"),
    ],
)
def test_collect_profile_result_packet_rejects_invalid_packet_contract_fields(
    conn,
    sample_brief,
    mutator,
    error_match,
):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    packet = devos_build_packet(po.production_order_id)
    mutator(packet)

    invocation = invoke_profile_task(
        envelope,
        runner=lambda payload: {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 9,
            "result_channel": packet,
        },
        timeout_seconds=30,
    )

    with pytest.raises(ValueError, match=error_match):
        collect_profile_result_packet(invocation, envelope)


def test_apply_profile_result_return_ingests_before_applying(conn, sample_brief, monkeypatch):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    packet = devos_build_packet(po.production_order_id)
    calls: list[str] = []

    import hermes_cli.production_order_autonomous as autonomous

    original_ingest = autonomous.ingest_profile_result_packet
    original_apply = autonomous.apply_accepted_result_action

    def wrapped_ingest(*args, **kwargs):
        calls.append("ingest")
        return original_ingest(*args, **kwargs)

    def wrapped_apply(*args, **kwargs):
        calls.append("apply")
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(autonomous, "ingest_profile_result_packet", wrapped_ingest)
    monkeypatch.setattr(autonomous, "apply_accepted_result_action", wrapped_apply)

    result = apply_profile_result_return(conn, po.production_order_id, envelope, packet)

    assert calls == ["ingest", "apply"]
    assert result["ingestion"]["accepted"] is True
    assert result["applied"]["applied"] is True
    assert result["applied"]["to_state"] == "DEV_COMPLETE"


def test_run_production_order_autonomously_reaches_done_on_happy_path(conn, sample_brief):
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    seen_calls: list[tuple[str, str]] = []

    def fake_runner(payload: dict) -> dict:
        seen_calls.append((payload["target_profile"], payload["source_state"]))
        packet_map = {
            ("architect_os", "ARCHITECT_SPEC"): architect_spec_packet(po.production_order_id),
            ("dev_os", "ARCHITECT_READY_FOR_DEV"): devos_build_packet(po.production_order_id),
            ("audit_os", "DEV_COMPLETE"): audit_review_packet(po.production_order_id),
            ("architect_os", "AUDIT_PASSED"): architect_reconcile_packet(po.production_order_id),
            ("default", "ARCHITECT_ACCEPTED"): final_review_packet(po.production_order_id),
        }
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 11,
            "result_channel": packet_map[(payload["target_profile"], payload["source_state"])],
        }

    result = run_production_order_autonomously(
        conn,
        po.production_order_id,
        runner=fake_runner,
        max_steps=10,
        max_retries=1,
        timeout_seconds=30,
    )

    refreshed = _reload_order(conn, po.production_order_id)
    assert refreshed.current_state == "DONE"
    assert result.done is True
    assert result.final_state == "DONE"
    assert result.final_owner_profile == "default"
    assert result.terminal_reason == "done"
    assert result.steps_run == 6
    assert seen_calls == [
        ("architect_os", "ARCHITECT_SPEC"),
        ("dev_os", "ARCHITECT_READY_FOR_DEV"),
        ("audit_os", "DEV_COMPLETE"),
        ("architect_os", "AUDIT_PASSED"),
        ("default", "ARCHITECT_ACCEPTED"),
    ]
    assert result.applied_actions[0]["bridge_function"] == "run_orchestrator_triage_bridge"
    assert result.applied_actions[-1]["to_state"] == "DONE"


def test_run_production_order_autonomously_uses_direct_orchestrator_bridge_without_runner_call(conn, sample_brief):
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    runner_calls: list[dict] = []

    def fake_runner(payload: dict) -> dict:
        runner_calls.append(payload)
        packet_map = {
            ("architect_os", "ARCHITECT_SPEC"): architect_spec_packet(po.production_order_id),
            ("dev_os", "ARCHITECT_READY_FOR_DEV"): devos_build_packet(po.production_order_id),
            ("audit_os", "DEV_COMPLETE"): audit_review_packet(po.production_order_id),
            ("architect_os", "AUDIT_PASSED"): architect_reconcile_packet(po.production_order_id),
            ("default", "ARCHITECT_ACCEPTED"): final_review_packet(po.production_order_id),
        }
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 7,
            "result_channel": packet_map[(payload["target_profile"], payload["source_state"])],
        }

    result = run_production_order_autonomously(
        conn,
        po.production_order_id,
        runner=fake_runner,
        max_steps=1,
        max_retries=1,
        timeout_seconds=30,
    )

    refreshed = _reload_order(conn, po.production_order_id)
    assert refreshed.current_state == "ARCHITECT_SPEC"
    assert result.steps_run == 1
    assert result.done is False
    assert result.terminal_reason == "max_steps_exceeded"
    assert result.applied_actions == [
        {
            "bridge_function": "run_orchestrator_triage_bridge",
            "from_state": "ORCHESTRATOR_TRIAGE",
            "to_state": "ARCHITECT_SPEC",
            "target_profile": "orchestrator_os",
            "mode": "direct_bridge",
        }
    ]
    assert runner_calls == []


def test_run_production_order_autonomously_stops_on_malformed_output_without_state_mutation(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)

    result = run_production_order_autonomously(
        conn,
        po.production_order_id,
        runner=lambda payload: {
            "stdout": "done",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 3,
        },
        max_steps=5,
        max_retries=1,
        timeout_seconds=30,
    )

    refreshed = _reload_order(conn, po.production_order_id)
    assert refreshed.current_state == "ARCHITECT_READY_FOR_DEV"
    assert result.done is False
    assert result.final_state == "ARCHITECT_READY_FOR_DEV"
    assert result.terminal_reason == "validation_failed"
    assert any("free-text-only output" in error for error in result.errors)


def test_run_production_order_autonomously_stops_safely_at_max_steps(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)

    result = run_production_order_autonomously(
        conn,
        po.production_order_id,
        runner=lambda payload: {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 3,
            "result_channel": devos_build_packet(po.production_order_id),
        },
        max_steps=0,
        max_retries=1,
        timeout_seconds=30,
    )

    refreshed = _reload_order(conn, po.production_order_id)
    assert refreshed.current_state == "ARCHITECT_READY_FOR_DEV"
    assert result.steps_run == 0
    assert result.terminal_reason == "max_steps_exceeded"
    assert result.final_state == "ARCHITECT_READY_FOR_DEV"


def test_run_production_order_autonomously_respects_retry_limit_on_failed_invocation(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    attempts = {"count": 0}

    def failing_runner(payload: dict) -> dict:
        attempts["count"] += 1
        return {
            "stdout": "",
            "stderr": "runner failed",
            "exit_code": 1,
            "duration_ms": 2,
        }

    result = run_production_order_autonomously(
        conn,
        po.production_order_id,
        runner=failing_runner,
        max_steps=5,
        max_retries=1,
        timeout_seconds=30,
    )

    refreshed = _reload_order(conn, po.production_order_id)
    assert refreshed.current_state == "ARCHITECT_READY_FOR_DEV"
    assert attempts["count"] == 2
    assert result.done is False
    assert result.terminal_reason == "retry_limit_exceeded"
    assert any("exit_code=1" in error for error in result.errors)


def test_hermes_profile_runner_fails_safe_when_real_runtime_is_disabled(conn, sample_brief, monkeypatch):
    monkeypatch.delenv("HERMES_ENABLE_PRODUCTION_PROFILE_RUNTIME", raising=False)
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)

    result = hermes_profile_runner(envelope.to_dict(), timeout_seconds=15)

    assert result["exit_code"] == 1
    assert result["stdout"] == ""
    assert "disabled" in result["stderr"].lower()
    assert result["log_ref"] == f"profile-runtime:{envelope.target_profile}:disabled"


def test_validate_approved_action_envelope_rejects_raw_chat_and_wrong_control_object_type():
    envelope = create_approved_action_envelope(
        approved_brief={"objective": "Ship the approved slice."},
        approved_by="Jarren",
        approved_at="2026-05-26T00:00:00Z",
        approval_phrase="I approve this production workflow brief.",
        priority_lane="Hermes OS",
        repo_or_workspace="alphathetacoding/hermes-agent",
        scope=["Implement the bounded runtime slice."],
        out_of_scope=["No daemon changes."],
        acceptance_criteria=["Return structured production results."],
        approval_boundaries=["Pause before publishing or deployment."],
        stop_conditions=["Stop if architecture changes are required."],
        source_truth=[WORKFLOW_SPEC_SOURCE],
        silence_protocol={"mode": "silent"},
        idempotency_key="aae-test-1",
    )

    with pytest.raises(ValueError, match="rejects raw chat"):
        validate_approved_action_envelope({**envelope, "raw_chat": "please just do it"})

    with pytest.raises(ValueError, match="control_object_type"):
        validate_approved_action_envelope(
            {**envelope, "control_object_type": "freeform_chat_request"}
        )


def test_brief_packet_from_approved_action_envelope_maps_runtime_brief_fields():
    envelope = create_approved_action_envelope(
        approved_brief={"objective": "Implement Slice 13 safely."},
        approved_by="Jarren",
        approved_at="2026-05-26T00:00:00Z",
        approval_phrase="I approve this production workflow brief.",
        priority_lane="Hermes OS",
        repo_or_workspace="alphathetacoding/hermes-agent",
        scope=["Implement approved envelope entrypoint."],
        out_of_scope=["Do not add background workers."],
        acceptance_criteria=["Use approved brief as frozen source."],
        approval_boundaries=["Pause before broad architecture change."],
        stop_conditions=["Stop if real runtime would be faked."],
        source_truth=[WORKFLOW_SPEC_SOURCE],
        silence_protocol={"mode": "silent"},
        idempotency_key="aae-test-2",
    )

    packet = brief_packet_from_approved_action_envelope(envelope)

    assert packet["target repo or workspace"] == "alphathetacoding/hermes-agent"
    assert packet["out of scope"] == ["Do not add background workers."]
    assert packet["approval boundaries"] == ["Pause before broad architecture change."]
    assert packet["idempotency_key"] == "aae-test-2"


def test_run_approved_action_envelope_autonomously_creates_or_reuses_order_and_preserves_silence_protocol(conn):
    envelope = create_approved_action_envelope(
        approved_brief={
            "title": "Approved autonomous entrypoint",
            "objective": "Run the approved brief through the existing production-order runtime.",
            "constraints": "No broad architecture changes.",
            "expected output": "Structured production run result.",
        },
        approved_by="Jarren",
        approved_at="2026-05-26T00:00:00Z",
        approval_phrase="I approve this production workflow brief.",
        priority_lane="Hermes OS",
        repo_or_workspace="alphathetacoding/hermes-agent",
        scope=["Implement slices 11.5B and 13."],
        out_of_scope=["No slice 14 or slice 15 work."],
        acceptance_criteria=["Create or reuse production order idempotently."],
        approval_boundaries=["Pause before deployment or publishing."],
        stop_conditions=["Stop if real runtime would be faked."],
        source_truth=[WORKFLOW_SPEC_SOURCE],
        silence_protocol={"mode": "silent", "channel": "none"},
        idempotency_key="approved-envelope-idempotency",
    )

    def fake_runner(payload: dict) -> dict:
        packet_map = {
            ("architect_os", "ARCHITECT_SPEC"): architect_spec_packet(payload["production_order_id"]),
            ("dev_os", "ARCHITECT_READY_FOR_DEV"): devos_build_packet(payload["production_order_id"]),
            ("audit_os", "DEV_COMPLETE"): audit_review_packet(payload["production_order_id"]),
            ("architect_os", "AUDIT_PASSED"): architect_reconcile_packet(payload["production_order_id"]),
            ("default", "ARCHITECT_ACCEPTED"): final_review_packet(payload["production_order_id"]),
        }
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 5,
            "result_channel": packet_map[(payload["target_profile"], payload["source_state"])],
        }

    first = run_approved_action_envelope_autonomously(
        conn,
        envelope,
        runner=fake_runner,
        max_steps=10,
        max_retries=1,
        timeout_seconds=30,
    )
    second = run_approved_action_envelope_autonomously(
        conn,
        envelope,
        runner=fake_runner,
        max_steps=10,
        max_retries=1,
        timeout_seconds=30,
    )

    assert first["silence_protocol"] == {"mode": "silent", "channel": "none"}
    assert first["approved_action_envelope_id"] == envelope["approved_action_envelope_id"]
    assert first["production_run_result"]["done"] is True
    assert second["approved_action_envelope_id"] == envelope["approved_action_envelope_id"]
    assert second["production_order_id"] == first["production_order_id"]
    assert second["production_order"]["production_order_id"] == first["production_order"]["production_order_id"]
    assert second["production_run_result"] == {
        "production_order_id": first["production_order_id"],
        "final_state": "DONE",
        "final_owner_profile": "default",
        "steps_run": 0,
        "terminal_reason": "already_terminal",
        "applied_actions": [],
        "errors": [],
        "blocked_reason": None,
        "done": True,
    }
