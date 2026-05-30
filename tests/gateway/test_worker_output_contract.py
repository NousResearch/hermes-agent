from __future__ import annotations

from gateway.dev_control.worker_output_contract import (
    append_worker_output_contract,
    parse_worker_output_contract,
    worker_output_contract_score,
)
from gateway.dev_execution import DevExecutionStore, derive_execution_plan_status
from gateway.subagent_events import SubagentEventStore


VALID_OUTPUT = """
Completed the inspection.

```json DEV_WORKER_EVIDENCE
{
  "summary": "Verified runtime metadata decoding.",
  "findings": ["WorkspaceSubagentActivity decodes runtime fields."],
  "files_read": ["apps/oryn-workspace/Sources/OrynWorkspaceCore/Models/WorkspaceSubagentActivity.swift"],
  "files_changed": [],
  "commands_run": ["swift test --filter WorkspaceSubagentActivityTests"],
  "verification": {
    "status": "passed",
    "evidence": ["WorkspaceSubagentActivityTests passed."]
  },
  "unresolved_gaps": [],
  "confidence": 0.86,
  "final_marker": "PHASE26_DONE"
}
```

FINAL_MARKER: PHASE26_DONE
"""


def test_worker_output_contract_parser_accepts_valid_markdown_json():
    parsed = parse_worker_output_contract(VALID_OUTPUT)

    assert parsed["output_contract_version"] == 2
    assert parsed["output_contract_status"] == "ok"
    assert parsed["structured_summary"] == "Verified runtime metadata decoding."
    assert parsed["findings"] == ["WorkspaceSubagentActivity decodes runtime fields."]
    assert parsed["files_read"] == ["apps/oryn-workspace/Sources/OrynWorkspaceCore/Models/WorkspaceSubagentActivity.swift"]
    assert parsed["commands_run"] == ["swift test --filter WorkspaceSubagentActivityTests"]
    assert parsed["verification_status"] == "passed"
    assert parsed["verification_evidence"] == ["WorkspaceSubagentActivityTests passed."]
    assert parsed["worker_confidence"] == 0.86
    assert parsed["final_marker"] == "PHASE26_DONE"
    assert worker_output_contract_score(parsed, required_marker="PHASE26_DONE") == 1.0


def test_worker_output_contract_parser_reports_missing_and_invalid_without_crashing():
    missing = parse_worker_output_contract("Finished without structured evidence.")
    invalid = parse_worker_output_contract("```json DEV_WORKER_EVIDENCE\n{\"summary\": \n```")

    assert missing["output_contract_status"] == "missing"
    assert "did not include" in missing["output_contract_warning"]
    assert invalid["output_contract_status"] == "invalid"
    assert "no valid JSON object" in invalid["output_contract_warning"]


def test_worker_output_contract_prompt_helper_is_idempotent():
    prompt = append_worker_output_contract("Do the work.")

    assert "Worker Output Contract v2" in prompt
    assert append_worker_output_contract(prompt) == prompt


def test_subagent_complete_persists_and_status_uses_structured_evidence(tmp_path):
    store = DevExecutionStore(tmp_path / "state.db")
    event_store = SubagentEventStore(tmp_path / "state.db")
    plan = store.create_plan(
        title="Structured evidence plan",
        vision_brief=None,
        tasks=[{
            "goal": "Return PHASE26_DONE",
            "prompt": "Inspect and return PHASE26_DONE.",
        }],
    )
    task = plan["tasks"][0]
    store.update_task_launch(
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        ao_session_id="fixture-phase-26",
    )

    event = event_store.append_event({
        "event": "subagent.complete",
        "subagent_id": "fixture:phase-26",
        "ao_session_id": "fixture-phase-26",
        "runtime": "fixture",
        "status": "completed",
        "summary": VALID_OUTPUT,
        "goal": "Return PHASE26_DONE",
        "launch_plan_id": plan["plan_id"],
        "launch_task_id": task["task_id"],
    })
    status = derive_execution_plan_status(
        store=store,
        plan_id=plan["plan_id"],
        event_store=event_store,
    )

    task_status = status["tasks"][0]
    assert event["output_contract_status"] == "ok"
    assert status["status"] == "completed"
    assert task_status["summary"] == "Verified runtime metadata decoding."
    assert task_status["summary_quality"] == "ok"
    assert task_status["output_contract_status"] == "ok"
    assert task_status["files_read"] == ["apps/oryn-workspace/Sources/OrynWorkspaceCore/Models/WorkspaceSubagentActivity.swift"]
    assert "WorkspaceSubagentActivityTests passed." in task_status["verification_evidence"]


def test_missing_structured_evidence_is_warning_first_for_good_summary(tmp_path):
    store = DevExecutionStore(tmp_path / "state.db")
    event_store = SubagentEventStore(tmp_path / "state.db")
    plan = store.create_plan(
        title="Warning-first plan",
        vision_brief=None,
        tasks=[{
            "goal": "Inspect without marker",
            "prompt": "Inspect without marker.",
        }],
    )
    task = plan["tasks"][0]
    store.update_task_launch(
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        ao_session_id="fixture-phase-26-warning",
    )
    event_store.append_event({
        "event": "subagent.complete",
        "subagent_id": "fixture:phase-26-warning",
        "ao_session_id": "fixture-phase-26-warning",
        "runtime": "fixture",
        "status": "completed",
        "summary": "Verified the relevant behavior with concrete evidence and no unresolved gaps.",
        "goal": "Inspect without marker",
        "launch_plan_id": plan["plan_id"],
        "launch_task_id": task["task_id"],
    })

    status = derive_execution_plan_status(
        store=store,
        plan_id=plan["plan_id"],
        event_store=event_store,
    )

    task_status = status["tasks"][0]
    assert status["status"] == "completed"
    assert task_status["summary_quality"] == "ok"
    assert task_status["summary_warning"] is None
    assert task_status["output_contract_status"] == "missing"
    assert "DEV_WORKER_EVIDENCE" in task_status["output_contract_warning"]


def test_structured_unresolved_gaps_make_completed_task_reviewable(tmp_path):
    store = DevExecutionStore(tmp_path / "state.db")
    event_store = SubagentEventStore(tmp_path / "state.db")
    plan = store.create_plan(
        title="Structured gaps plan",
        vision_brief=None,
        tasks=[{
            "goal": "Return PHASE26_DONE",
            "prompt": "Inspect and return PHASE26_DONE.",
        }],
    )
    task = plan["tasks"][0]
    store.update_task_launch(
        plan_id=plan["plan_id"],
        task_id=task["task_id"],
        ao_session_id="fixture-phase-26-gaps",
    )

    event_store.append_event({
        "event": "subagent.complete",
        "subagent_id": "fixture:phase-26-gaps",
        "ao_session_id": "fixture-phase-26-gaps",
        "runtime": "fixture",
        "status": "completed",
        "summary": VALID_OUTPUT.replace('"unresolved_gaps": []', '"unresolved_gaps": ["Need product owner confirmation."]'),
        "goal": "Return PHASE26_DONE",
        "launch_plan_id": plan["plan_id"],
        "launch_task_id": task["task_id"],
    })
    status = derive_execution_plan_status(
        store=store,
        plan_id=plan["plan_id"],
        event_store=event_store,
    )

    task_status = status["tasks"][0]
    assert status["status"] == "needs_review"
    assert task_status["summary_warning"] == "Worker reported unresolved gaps in structured evidence."
    assert task_status["unresolved_gaps"] == ["Need product owner confirmation."]
