"""#122813: the API-server adapter refreshes gateway_state.json's active_agents
whenever its work-count contributions change (pending reservations, in-flight
agent turns, /v1/runs tasks) — until now those changes never reached the file,
so the gateway read as idle mid-run and busy after it."""
from unittest.mock import Mock

from gateway.platforms.api_server import (
    APIServerAdapter,
    _release_pending_api_work,
    _reserve_pending_api_work,
)
from gateway.platforms.api_server_runs import _persist_adapter_active_work


def _make_adapter():
    adapter = APIServerAdapter.__new__(APIServerAdapter)  # skip __init__ wiring
    adapter._pending_agent_requests = 0
    adapter._inflight_agent_runs = 0
    adapter._active_run_tasks = {}
    return adapter


def test_reservation_cycle_persists():
    adapter = _make_adapter()
    runner = Mock()
    runner._persist_active_agents = Mock()
    adapter.gateway_runner = runner

    with _reserve_pending_api_work(adapter):
        assert runner._persist_active_agents.call_count == 1
    assert runner._persist_active_agents.call_count == 2


def test_release_persists_and_is_once_only():
    adapter = _make_adapter()
    runner = Mock()
    runner._persist_active_agents = Mock()
    adapter.gateway_runner = runner
    reservation = {"active": True}
    adapter._pending_agent_requests += 1

    _release_pending_api_work(adapter, reservation)
    _release_pending_api_work(adapter, reservation)  # idempotent release
    assert runner._persist_active_agents.call_count == 1


def test_no_gateway_runner_is_noop():
    adapter = _make_adapter()
    adapter.gateway_runner = None
    with _reserve_pending_api_work(adapter):
        pass  # must not raise
    assert adapter._pending_agent_requests == 0


def test_persist_failure_is_swallowed():
    adapter = _make_adapter()
    runner = Mock()
    runner._persist_active_agents = Mock(side_effect=RuntimeError("disk gone"))
    adapter.gateway_runner = runner
    with _reserve_pending_api_work(adapter):
        pass  # persist failure must not break the request path


def test_persist_helper_guards_and_delegates():
    persist = Mock()
    adapter = SimpleNamespaceAdapter(persist)
    _persist_adapter_active_work(adapter)
    assert persist.call_count == 1

    _persist_adapter_active_work(object())  # no helper: no-op, no raise

    class _Boom:
        def _persist_active_work(self):
            raise RuntimeError("boom")

    _persist_adapter_active_work(_Boom())  # failure swallowed


class SimpleNamespaceAdapter:
    def __init__(self, persist):
        self._persist = persist

    def _persist_active_work(self):
        self._persist()
