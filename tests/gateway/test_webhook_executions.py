"""WebhookAdapter execution-state regressions for Task 11."""

import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


def _adapter(tmp_path):
    route = {
        "secret": "secret",
        "signature_mode": "github",
        "prompt": "{event_type}",
        "deliver": "log",
    }
    adapter = WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 0,
                "routes": {"route": route},
            },
        )
    )
    # Isolate the durable registry: the adapter defaults to the real
    # Hermes-home ledger, which would leak state between tests.
    adapter._execution_registry.path = tmp_path / "executions.json"
    adapter._execution_registry._records.clear()
    adapter._execution_registry._persist()
    return adapter


def test_record_execution_sets_accepted(tmp_path):
    adapter = _adapter(tmp_path)
    rec = adapter._record_execution("d1", "route", None, "github", "webhook:route:d1")
    assert rec["execution_id"]
    assert rec["state"] == "accepted"
    assert adapter._execution_registry.public(rec["execution_id"])["state"] == "accepted"


def test_finish_marks_completed(tmp_path):
    adapter = _adapter(tmp_path)
    rec = adapter._record_execution("d1", "route", None, "github", "webhook:route:d1")
    assert adapter._execution_registry.finish(rec["execution_id"], "completed")
    assert adapter._execution_registry.public(rec["execution_id"])["state"] == "completed"


def test_prune_removes_stale(tmp_path):
    adapter = _adapter(tmp_path)
    rec = adapter._record_execution("old", "route", None, "github", "webhook:route:old")
    adapter._execution_registry.finish(rec["execution_id"], "completed")
    # finished_at must be a truthy old value: the registry treats a falsy
    # finished_at as "use created_at", which is not stale.
    adapter._execution_registry._records[rec["execution_id"]]["finished_at"] = 1.0
    adapter._prune_executions(99999.0)
    with pytest.raises(KeyError):
        adapter._execution_registry.public(rec["execution_id"])


def test_request_cancel_reports_cancelling(tmp_path):
    adapter = _adapter(tmp_path)
    rec = adapter._record_execution("d1", "route", "default", "github", "webhook:route:d1")
    assert adapter._execution_registry.request_cancel(rec["execution_id"]) == "cancelling"


@pytest.mark.asyncio
async def test_bound_real_task_completes_only_on_finish(tmp_path):
    adapter = _adapter(tmp_path)
    rec = adapter._record_execution("d1", "route", None, "github", "webhook:route:d1")
    gate = asyncio.Event()

    async def actual_run():
        await gate.wait()

    task = asyncio.create_task(actual_run())
    assert adapter._execution_registry.bind(rec["execution_id"], task)
    await asyncio.sleep(0)
    assert adapter._execution_registry.public(rec["execution_id"])["state"] == "running"
    gate.set()
    await task
