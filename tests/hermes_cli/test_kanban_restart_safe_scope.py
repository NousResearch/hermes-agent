"""The Linux Kanban worker boundary refuses an unavailable restart-safe scope."""
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db_dispatch
from tools import process_registry


@pytest.mark.platforms("linux")
def test_missing_user_bus_refuses_restart_safe_worker(monkeypatch):
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway")
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: False)
    task = SimpleNamespace(id="scope-refusal", current_run_id=1)
    with pytest.raises(process_registry.RestartSafeScopeUnavailable, match="enable-linger"):
        kanban_db_dispatch._restart_safe_worker_argv(task, ["hermes", "chat"])
