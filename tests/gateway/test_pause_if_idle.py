"""``pause-if-idle``: the gateway begins an update drain only while it holds no work, deciding on
its loop in one step so no turn is admitted between the check and the pause."""

from __future__ import annotations

import asyncio

import pytest

from gateway.control_socket import GatewayControlServer, pause_gateway_if_idle
from gateway.drain_control import pause_if_idle_verb


class _Runner:
    def __init__(self, work):
        self.work, self._draining, self._restart_task_started = list(work), False, False
        self.session_runtime_descriptor: dict = {}

    def _describe_active_work(self):
        return list(self.work)

    def _active_work_count(self):
        return len(self.work)

    def request_restart(self, *, detached=False, via_service=False):
        self._draining = self._restart_task_started = True
        return True


@pytest.mark.platforms("posix")  # unix socket transport
def test_a_busy_gateway_refuses_and_an_idle_one_pauses_over_the_real_socket(tmp_path):
    runner = _Runner([{"kind": "chat", "session": "s1"}])

    async def scenario():
        loop = asyncio.get_running_loop()
        server = GatewayControlServer(home=tmp_path, verb_handlers={"pause-if-idle": pause_if_idle_verb(runner, loop)})
        assert await server.start()
        try:
            busy = await loop.run_in_executor(None, lambda: pause_gateway_if_idle(tmp_path, target="v0.30.0"))
            busy_state = runner._draining
            runner.work.clear()
            paused = await loop.run_in_executor(None, lambda: pause_gateway_if_idle(tmp_path, target="v0.30.0"))
            again = await loop.run_in_executor(None, lambda: pause_gateway_if_idle(tmp_path))
        finally:
            await server.stop()
        return busy, busy_state, paused, again

    busy, busy_state, paused, again = asyncio.run(scenario())
    assert busy["busy"] is True and not busy.get("pausing")
    assert busy["active_work"] == [{"kind": "chat", "session": "s1"}]
    assert busy_state is False, "a busy gateway began draining"
    assert runner.session_runtime_descriptor.get("drain_target") == "v0.30.0"
    assert paused["pausing"] is True and runner._draining is True
    assert again == {**again, "pausing": False, "already_stopping": True}
