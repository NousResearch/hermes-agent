"""A Desktop catch-up script must not lose its only transition before plugin registration."""
import sys
from types import SimpleNamespace

import pytest

from cron import delivery_queue, executions, jobs, scheduler
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry


@pytest.fixture
def startup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "cron" / "deliveries.db")
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    entries = {}
    monkeypatch.setattr(platform_registry, "get", entries.get)
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig(platforms={
        Platform.DISCORD: PlatformConfig(enabled=True, token="test-token"),
        Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token"),
    }))
    sent = []

    async def sender(_config, chat_id, content, **kwargs):
        sent.append((chat_id, content))
        return {"success": True, "message_id": "receipt"}

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entry = PlatformEntry(name="discord", label="Discord", adapter_factory=lambda _: None,
                          check_fn=lambda: True, standalone_sender_fn=sender)
    with jobs.use_cron_store(tmp_path):
        yield SimpleNamespace(home=tmp_path, entries=entries, entry=entry, sent=sent)


def make_script_job(startup, *, output="transition", deliver="discord:123", exit_code=0, **kwargs):
    script = startup.home / "scripts" / "watchdog.py"
    script.write_text(
        "from pathlib import Path\n"
        "state = Path(__file__).with_suffix('.state')\n"
        "state.write_text(state.read_text() + 'run\\n' if state.exists() else 'run\\n')\n"
        f"print({output!r})\nraise SystemExit({exit_code})\n", encoding="utf-8",
    )
    return jobs.create_job(prompt="", schedule="every 1h", script=script.name,
                           no_agent=True, deliver=deliver, interpreter=sys.executable, **kwargs)


def test_transition_waits_for_registration_and_survives_restart_once(startup, monkeypatch):
    # Arrange
    job = make_script_job(startup)

    # Act
    assert scheduler.run_one_job(job)
    execution = executions.latest_execution(job["id"])
    assert execution["status"] == "completed", execution
    execution_id = execution["id"]
    row = delivery_queue.get_status(execution_id)

    # Assert
    assert row is not None and row["status"] == "pending"
    assert row["content"] == "transition"
    assert jobs.get_job(job["id"])["last_status"] == "delivery_queued"
    assert execution["delivery_outcome"] == "queued"
    assert scheduler.drain_delivery_queue({}, None) == 0
    assert delivery_queue.get_status(execution_id)["status"] == "pending"
    assert startup.sent == []
    saved_outputs = list((startup.home / "cron" / "output" / job["id"]).glob("*.md"))
    assert len(saved_outputs) == 1
    assert "transition" in saved_outputs[0].read_text(encoding="utf-8")

    # A new scheduler process sees the durable row, not a reason to execute the script again.
    monkeypatch.setattr(delivery_queue, "_PROCESS_ID", "replacement-gateway")
    startup.entries["discord"] = startup.entry
    assert scheduler.drain_delivery_queue({}, None) == 1
    assert startup.sent == [("123", "transition")]
    assert delivery_queue.get_status(execution_id)["status"] == "delivered"
    assert jobs.get_job(job["id"])["last_status"] == "ok"
    assert executions.get_execution(execution_id)["delivery_outcome"] == "delivered"
    delivery_queue.enqueue(execution_id, job, "must not replace saved output")
    monkeypatch.setattr(delivery_queue, "_PROCESS_ID", "another-gateway")
    assert scheduler.drain_delivery_queue({}, None) == 0
    assert startup.sent == [("123", "transition")]
    assert (startup.home / "scripts" / "watchdog.state").read_text() == "run\n"
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 1


@pytest.mark.parametrize("output", ["", "  ", '[SILENT]'])
def test_empty_script_output_never_waits_or_sends(startup, output):
    # Arrange
    job = make_script_job(startup, output=output)

    # Act
    assert scheduler.run_one_job(job)

    # Assert
    execution = executions.latest_execution(job["id"])
    assert execution["status"] == "completed", execution
    assert delivery_queue.get_status(execution["id"]) is None
    assert execution["delivery_outcome"] == "suppressed"
    assert scheduler.drain_delivery_queue({}, None) == 0
    assert startup.sent == []


def test_ready_script_keeps_direct_delivery(startup):
    # Arrange
    startup.entries["discord"] = startup.entry
    job = make_script_job(startup)

    # Act
    assert scheduler.run_one_job(job)

    # Assert
    assert startup.sent == [("123", "transition")]
    execution = executions.latest_execution(job["id"])
    assert delivery_queue.get_status(execution["id"]) is None
    assert execution["delivery_outcome"] == "delivered"


def test_unready_row_does_not_block_other_deliveries(startup):
    # Arrange
    job = make_script_job(startup)
    assert scheduler.run_one_job(job)
    startup.entries["telegram"] = PlatformEntry(
        name="telegram", label="Telegram", adapter_factory=lambda _: None,
        check_fn=lambda: True, standalone_sender_fn=startup.entry.standalone_sender_fn)
    delivery_queue.enqueue("ready", {"id": "ready-job", "deliver": "telegram:456"}, "ready")

    # Act
    assert scheduler.drain_delivery_queue({}, None) == 1

    # Assert
    pending = delivery_queue.get_status(executions.latest_execution(job["id"])["id"])
    assert pending is not None and pending["status"] == "pending"
    assert delivery_queue.get_status("ready")["status"] == "delivered"
    assert startup.sent == [("456", "ready")]


def test_claimed_startup_delivery_is_never_replayed_after_uncertain_send(startup, monkeypatch):
    # Arrange
    job = make_script_job(startup)
    assert scheduler.run_one_job(job)
    execution_id = executions.latest_execution(job["id"])["id"]
    assert delivery_queue.claim_next()["execution_id"] == execution_id
    startup.entries["discord"] = startup.entry
    monkeypatch.setattr(delivery_queue, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(delivery_queue, "_owner_is_live", lambda *_: False)

    # Act
    assert scheduler.drain_delivery_queue({}, None) == 0

    # Assert
    assert delivery_queue.get_status(execution_id)["status"] == "unknown"
    assert jobs.get_job(job["id"])["last_status"] == "delivery_failed"
    assert startup.sent == []
    assert (startup.home / "scripts" / "watchdog.state").read_text() == "run\n"


def test_failure_lane_waits_without_rerunning_failed_script(startup):
    # Arrange
    job = make_script_job(startup, deliver="local", failure_deliver="discord:123", exit_code=1)
    assert scheduler.run_one_job(job)
    execution_id = executions.latest_execution(job["id"])["id"]
    row = delivery_queue.get_status(execution_id)
    assert row is not None and row["status"] == "pending" and row["for_failure"]
    content = row["content"]
    assert content
    assert scheduler.drain_delivery_queue({}, None) == 0
    startup.entries["discord"] = startup.entry

    # Act
    assert scheduler.drain_delivery_queue({}, None) == 1

    # Assert
    assert startup.sent == [("123", content)]
    assert executions.get_execution(execution_id)["delivery_outcome"] == "delivered"
    assert jobs.get_job(job["id"])["last_status"] == "error"
    assert (startup.home / "scripts" / "watchdog.state").read_text() == "run\n"
