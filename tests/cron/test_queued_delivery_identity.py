"""A queued cron send carries its execution's identity into the gateway drain.

Every agent fire hands its send to the durable delivery queue, and the gateway drains it later from
its housekeeping thread. The drain ran outside any cron identity, so a platform that admits a send
only for its own job (``ctx.current_cron_execution()``) refused every queued run, on both the live
lane and the standalone fallback.
"""
import asyncio
import threading
from unittest.mock import patch

import pytest

from cron import delivery_queue, scheduler
from cron.execution_identity import current_cron_execution
from cron.executions import create_execution, mark_execution_running
from gateway.config import Platform
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture()
def home(tmp_path):
    root = tmp_path / "home"
    (root / "cron").mkdir(parents=True)
    token = set_hermes_home_override(str(root))
    try:
        yield root
    finally:
        reset_hermes_home_override(token)


@pytest.fixture()
def gateway_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)


class _IdentityProbeAdapter:
    """A live adapter that, like the Oi reports platform, reads the cron identity at send time."""

    def __init__(self):
        self.seen = []

    async def send(self, chat_id, content, metadata=None):
        self.seen.append((chat_id, current_cron_execution()))
        return {"success": True, "message_id": "m1"}


def _queue(execution_job_id: str, queued_job_id: str) -> str:
    execution = create_execution(execution_job_id, source="builtin")
    mark_execution_running(execution["id"])
    delivery_queue.enqueue(
        execution["id"],
        {"id": queued_job_id, "name": "probe job", "deliver": "discord:C1",
         "execution_id": execution["id"]},
        "report body")
    return execution["id"]


def _drain(adapter, loop):
    with patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
        return scheduler.drain_delivery_queue({Platform.DISCORD: adapter}, loop)


def test_queued_send_sees_its_own_execution_identity(home, gateway_loop):
    execution_id = _queue("job1", "job1")
    adapter = _IdentityProbeAdapter()

    assert _drain(adapter, gateway_loop) == 1

    ((chat_id, ident),) = adapter.seen
    assert chat_id == "C1"
    assert ident is not None, "queued cron send was drained without its cron identity"
    assert ident.job_id == "job1"
    assert ident.job_name == "probe job"
    assert ident.execution_id == execution_id
    assert ident.source == "builtin"
    # Bound only for that send: the housekeeping thread is not left inside the fire.
    assert current_cron_execution() is None
    assert delivery_queue.get_status(execution_id)["status"] == "delivered"


def test_queued_send_for_another_jobs_execution_binds_nothing(home, gateway_loop):
    """The ledger, not the queued payload, decides: a row naming an execution of another job
    sends with no identity, so identity-gated platforms fail closed."""
    _queue("other-job", "job1")
    adapter = _IdentityProbeAdapter()

    _drain(adapter, gateway_loop)

    ((_, ident),) = adapter.seen
    assert ident is None


def test_queued_send_without_a_ledger_row_binds_nothing(home, gateway_loop):
    delivery_queue.enqueue(
        "missing-exec", {"id": "job1", "name": "probe job", "deliver": "discord:C1",
                         "execution_id": "missing-exec"}, "report body")
    adapter = _IdentityProbeAdapter()

    _drain(adapter, gateway_loop)

    ((_, ident),) = adapter.seen
    assert ident is None
