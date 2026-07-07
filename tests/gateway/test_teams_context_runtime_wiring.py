from __future__ import annotations

from types import SimpleNamespace

from gateway.config import Platform
from plugins.teams_context.runtime import bind_gateway_runtime


def test_bind_gateway_runtime_registers_named_scheduler(monkeypatch):
    class FakeAdapter:
        def __init__(self):
            self.schedulers = {}

        def register_notification_scheduler(self, name, scheduler):
            self.schedulers[name] = scheduler

    adapter = FakeAdapter()
    gateway = SimpleNamespace(
        adapters={Platform.MSGRAPH_WEBHOOK: adapter},
        config=SimpleNamespace(),
        _teams_context_runtime=None,
        _teams_context_runtime_error=None,
    )
    runtime = SimpleNamespace()
    monkeypatch.setattr("plugins.teams_context.runtime.build_runtime", lambda _config: runtime)

    assert bind_gateway_runtime(gateway) is True
    assert gateway._teams_context_runtime is runtime
    assert "teams_context" in adapter.schedulers
