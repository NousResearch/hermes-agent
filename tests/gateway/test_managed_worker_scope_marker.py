"""The managed worker stamps the marker the background-executor scope gate reads (#144495).

Terminal tools run in ``agent.managed_worker``, a child of the gateway, so a background
command it starts inherits the worker's pid and NOT the gateway's. Stamping the worker's own
pid is what lets ``tools.process_registry`` give those jobs a systemd scope instead of leaving
them in ``hermes-gateway.service``'s cgroup, where ``MemoryHigh`` throttling stalls the gateway
event loop and every profile times out with ``worker_receipt_timeout``.
"""
import os

import pytest

from agent import managed_worker
from hermes_constants import MANAGED_WORKER_PID_ENV
from tools import process_registry


def test_claim_managed_worker_scope_stamps_own_pid(monkeypatch):
    monkeypatch.delenv(MANAGED_WORKER_PID_ENV, raising=False)

    managed_worker.claim_managed_worker_scope()

    assert os.environ[MANAGED_WORKER_PID_ENV] == str(os.getpid())


def test_main_stamps_the_marker_before_serving(monkeypatch):
    """The marker must be in place before the worker can run a turn."""
    monkeypatch.delenv(MANAGED_WORKER_PID_ENV, raising=False)

    def stop_at_channel(_stream):
        raise RuntimeError("worker bootstrap stopped for the test")

    monkeypatch.setattr(managed_worker, "WorkerChannel", stop_at_channel)

    with pytest.raises(RuntimeError, match="stopped for the test"):
        managed_worker.main()

    assert os.environ[MANAGED_WORKER_PID_ENV] == str(os.getpid())


@pytest.mark.platforms("linux")
def test_stamped_worker_is_recognised_by_the_scope_gate(monkeypatch):
    """End to end over the seam: the stamp is exactly what the gate compares."""
    gateway_pid = os.getpid() + 4242
    monkeypatch.delenv(MANAGED_WORKER_PID_ENV, raising=False)
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
    monkeypatch.setattr(
        "gateway.status.get_running_pid",
        lambda *, cleanup_stale=False: gateway_pid,
    )
    monkeypatch.setattr(
        "tools.process_registry._has_gateway_ancestor",
        lambda pid: pid == gateway_pid,
    )

    # In the gateway tree, but not the worker: no scope (the pre-#144495 behaviour).
    assert not process_registry._is_supervised_gateway_process()
    assert not process_registry._is_in_supervised_gateway_tree()

    managed_worker.claim_managed_worker_scope()

    assert process_registry._is_managed_worker_process()
    assert process_registry._is_in_supervised_gateway_tree()


def test_worker_marker_does_not_make_this_process_the_gateway(monkeypatch):
    """Scoping is the only thing the marker unlocks.

    ``_is_supervised_gateway_process`` still keys on owning the gateway pid file: it gates
    cron/kanban dispatch and the self-kill guards, which a worker must never satisfy.
    """
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setattr(
        "gateway.status.get_running_pid",
        lambda *, cleanup_stale=False: os.getpid() + 4242,
    )
    managed_worker.claim_managed_worker_scope()

    assert not process_registry._is_supervised_gateway_process()
