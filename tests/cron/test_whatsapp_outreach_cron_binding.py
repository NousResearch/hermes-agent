from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from cron.jobs import create_job
from cron.scheduler import tick
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.whatsapp_approved_outreach import (
    bind_whatsapp_outreach_plan_to_cron_job,
    load_whatsapp_outreach_state,
    _write_whatsapp_outreach_state,
)


def _seed_plan_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / ".hermes" / "cron")
    monkeypatch.setattr(
        "cron.jobs.JOBS_FILE", tmp_path / ".hermes" / "cron" / "jobs.json"
    )
    monkeypatch.setattr(
        "cron.jobs.OUTPUT_DIR", tmp_path / ".hermes" / "cron" / "output"
    )
    monkeypatch.setattr("cron.scheduler._hermes_home", tmp_path / ".hermes")
    from gateway.whatsapp_message_store import append_whatsapp_record
    from datetime import datetime

    base_dir = tmp_path / ".hermes" / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    append_whatsapp_record(
        {
            "record_kind": "conversation_record",
            "record_id": "record-1",
            "conversation_key": "whatsapp:dm:15551230000",
            "destination_key": "whatsapp:dm:15551230000",
            "destination_context_type": "direct_message",
            "destination_chat_id": "15551230000@s.whatsapp.net",
            "destination_target_id": "15551230000",
            "group_chat_id": None,
            "dm_counterparty_id": "15551230000",
            "direction": "inbound",
            "effective_event_at_utc": "2024-06-02T09:01:00Z",
            "record_sequence": 1,
            "participant_role": "external_party",
            "message_classification": "conversational_only",
            "command_authority_scope": "none",
            "sender_id": "15551230000",
            "sender_name": "Vendor",
            "text": "Prior vendor thread context.",
            "message_id": "msg-1",
            "media_types": [],
        },
        effective_event_at=datetime.fromisoformat("2024-06-02T09:01:00+00:00"),
        base_dir=base_dir,
    )

    from gateway.whatsapp_approved_outreach import execute_whatsapp_approved_outreach
    import asyncio

    adapter = AsyncMock(
        return_value=SendResult(success=True, message_id="seed-msg", raw_response={})
    )
    asyncio.run(
        execute_whatsapp_approved_outreach(
            {
                "destination_key": "whatsapp:dm:15551230000",
                "operator_objective": "request the revised quote",
                "message_text": "Following up on the revised quote.",
            },
            authorized=True,
            adapter=adapter,
        )
    )
    return load_whatsapp_outreach_state()["plans"][0]["plan_id"]


def test_tick_runs_bound_whatsapp_outreach_through_shared_execution_path(
    tmp_path, monkeypatch
):
    plan_id = _seed_plan_state(tmp_path, monkeypatch)
    job = create_job(
        prompt="Scheduled follow-up.",
        schedule="every 1h",
        deliver="telegram:-100ops",
        workflow_binding_type="whatsapp_outreach_plan",
        workflow_binding_id=plan_id,
    )
    bind_whatsapp_outreach_plan_to_cron_job(plan_id=plan_id, cron_job_id=job["id"])

    adapter_instance = type("Adapter", (), {})()
    adapter_instance.send = AsyncMock(
        return_value=SendResult(
            success=True,
            message_id="cron-msg",
            raw_response={"dispatch_group_id": "dispatch-cron"},
        )
    )

    fake_db = MagicMock()

    with (
        patch("cron.scheduler.get_due_jobs", return_value=[job]),
        patch("cron.scheduler.advance_next_run"),
        patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"),
        patch("cron.scheduler._deliver_result", return_value=None),
        patch("cron.scheduler.mark_job_run") as mark_mock,
        patch("hermes_state.SessionDB", return_value=fake_db),
    ):
        executed = tick(
            verbose=False,
            adapters={Platform.WHATSAPP: adapter_instance},
        )

    assert executed == 1
    adapter_instance.send.assert_awaited_once_with(
        "15551230000@s.whatsapp.net",
        "Scheduled follow-up.",
    )
    mark_mock.assert_called_once()
    assert mark_mock.call_args[0][1] is True

    state = load_whatsapp_outreach_state()
    run_row = state["runs"][-1]
    assert run_row["workflow_binding_type"] == "whatsapp_outreach_plan"
    assert run_row["workflow_binding_id"] == plan_id
    assert run_row["trigger_source"] == "cron_job"
    assert run_row["trigger_reference_id"] == job["id"]
    assert run_row["report_delivery_target"] == "telegram:-100ops"


def test_tick_fails_closed_when_bound_plan_is_paused(tmp_path, monkeypatch):
    plan_id = _seed_plan_state(tmp_path, monkeypatch)
    job = create_job(
        prompt="Scheduled follow-up.",
        schedule="every 1h",
        workflow_binding_type="whatsapp_outreach_plan",
        workflow_binding_id=plan_id,
    )
    bind_whatsapp_outreach_plan_to_cron_job(plan_id=plan_id, cron_job_id=job["id"])

    from gateway.whatsapp_approved_outreach import _write_whatsapp_outreach_state

    state = load_whatsapp_outreach_state()
    state["plans"][0]["plan_status"] = "paused"
    _write_whatsapp_outreach_state(state)

    adapter_instance = type("Adapter", (), {})()
    adapter_instance.send = AsyncMock()
    fake_db = MagicMock()

    with (
        patch("cron.scheduler.get_due_jobs", return_value=[job]),
        patch("cron.scheduler.advance_next_run"),
        patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"),
        patch("cron.scheduler._deliver_result", return_value=None),
        patch("cron.scheduler.mark_job_run") as mark_mock,
        patch("hermes_state.SessionDB", return_value=fake_db),
    ):
        executed = tick(
            verbose=False,
            adapters={Platform.WHATSAPP: adapter_instance},
        )

    assert executed == 1
    adapter_instance.send.assert_not_awaited()
    mark_mock.assert_called_once()
    assert mark_mock.call_args[0][1] is False
    assert "paused" in mark_mock.call_args[0][2]


def test_tick_partitions_bound_whatsapp_outreach_jobs_sequentially(
    tmp_path, monkeypatch
):
    plan_id = _seed_plan_state(tmp_path, monkeypatch)
    job = create_job(
        prompt="Scheduled follow-up.",
        schedule="every 1h",
        workflow_binding_type="whatsapp_outreach_plan",
        workflow_binding_id=plan_id,
    )

    calls = []

    def fake_run_job(passed_job):
        calls.append(passed_job["id"])
        return True, "output", "response", None

    with (
        patch("cron.scheduler.get_due_jobs", return_value=[job]),
        patch("cron.scheduler.advance_next_run"),
        patch("cron.scheduler.run_job", side_effect=fake_run_job),
        patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"),
        patch("cron.scheduler._deliver_result", return_value=None),
        patch("cron.scheduler.mark_job_run"),
    ):
        executed = tick(verbose=False)

    assert executed == 1
    assert calls == [job["id"]]


def test_tick_runs_multi_target_bound_whatsapp_outreach_and_persists_partial_report(
    tmp_path, monkeypatch
):
    plan_id = _seed_plan_state(tmp_path, monkeypatch)
    state = load_whatsapp_outreach_state()
    state["plan_targets"].append({
        "plan_target_id": "watarget-extra",
        "plan_id": plan_id,
        "target_status": "active",
        "conversation_key": "whatsapp:dm:15551230001",
        "destination_key": "whatsapp:dm:15551230001",
        "group_chat_id": None,
        "dm_counterparty_id": "15551230001",
        "target_objective_override": None,
        "max_outbound_messages_per_run": 1,
        "last_resolved_conversation_key": None,
        "last_observed_message_at_utc": None,
    })
    state["plans"][0]["plan_status"] = "approved"
    from gateway.whatsapp_message_store import append_whatsapp_record
    from datetime import datetime

    append_whatsapp_record(
        {
            "record_kind": "conversation_record",
            "record_id": "record-2",
            "conversation_key": "whatsapp:dm:15551230001",
            "destination_key": "whatsapp:dm:15551230001",
            "destination_context_type": "direct_message",
            "destination_chat_id": "15551230001@s.whatsapp.net",
            "destination_target_id": "15551230001",
            "group_chat_id": None,
            "dm_counterparty_id": "15551230001",
            "direction": "inbound",
            "effective_event_at_utc": "2024-06-02T09:02:00Z",
            "record_sequence": 2,
            "participant_role": "external_party",
            "message_classification": "conversational_only",
            "command_authority_scope": "none",
            "sender_id": "15551230001",
            "sender_name": "Vendor B",
            "text": "Prior vendor thread context B.",
            "message_id": "msg-2",
            "media_types": [],
        },
        effective_event_at=datetime.fromisoformat("2024-06-02T09:02:00+00:00"),
        base_dir=tmp_path / ".hermes" / "gateway" / "whatsapp-records",
    )
    _write_whatsapp_outreach_state(state)

    job = create_job(
        prompt="Scheduled follow-up.",
        schedule="every 1h",
        deliver="telegram:-100ops",
        workflow_binding_type="whatsapp_outreach_plan",
        workflow_binding_id=plan_id,
    )
    bind_whatsapp_outreach_plan_to_cron_job(plan_id=plan_id, cron_job_id=job["id"])

    async def _send(chat_id, _message):
        if chat_id == "15551230000@s.whatsapp.net":
            return SendResult(success=True, message_id="cron-msg-a", raw_response={})
        return SendResult(success=False, error="bridge down")

    adapter_instance = type("Adapter", (), {})()
    adapter_instance.send = AsyncMock(side_effect=_send)

    fake_db = MagicMock()

    with (
        patch("cron.scheduler.get_due_jobs", return_value=[job]),
        patch("cron.scheduler.advance_next_run"),
        patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"),
        patch("cron.scheduler._deliver_result", return_value=None),
        patch("cron.scheduler.mark_job_run") as mark_mock,
        patch("hermes_state.SessionDB", return_value=fake_db),
    ):
        executed = tick(
            verbose=False,
            adapters={Platform.WHATSAPP: adapter_instance},
        )

    assert executed == 1
    assert adapter_instance.send.await_count == 2
    assert mark_mock.call_args[0][1] is True

    final_state = load_whatsapp_outreach_state()
    latest_run = final_state["runs"][-1]
    latest_report = final_state["reports"][-1]
    assert latest_run["run_status"] == "completed_with_failures"
    assert latest_run["target_count"] == 2
    assert latest_run["failed_target_count"] == 1
    assert (
        len([
            r
            for r in final_state["target_executions"]
            if r["run_id"] == latest_run["run_id"]
        ])
        == 2
    )
    assert latest_report["report_status"] == "partial"
    assert len(latest_report["target_rows"]) == 2
