"""An unsuccessful explicit kind switch must not publish new routing."""
from pathlib import Path
import runpy

import pytest


@pytest.mark.linux_only
@pytest.mark.parametrize("stage", ["prerequisites", "boot"])
def test_failed_kind_switch_preserves_previous_mode_and_kind(tmp_path, monkeypatch, stage):
    load = runpy.run_path(str(Path(__file__).resolve().parents[2] / "plugins/hermes-realms/realms/_binding.py"))["load_runtime"]
    integration = load("integration")
    service = integration.RealmIntegration(tmp_path)
    owner = service.bind(session_id="chat", runtime_session_id="runtime")
    service.owners.set_mode(owner, "ask")
    stopped = []
    monkeypatch.setattr(service, "stop", lambda value: stopped.append(value))
    monkeypatch.setattr(integration, "vm_setup_status", lambda home: {"ready": stage == "boot", "message": "missing fixture dependency"})
    monkeypatch.setattr(service, "ready", lambda owner: (_ for _ in ()).throw(RuntimeError("fixture boot failure")))

    with pytest.raises((integration.SetupError, RuntimeError)):
        service.command("on omarchy", session_id="chat", runtime_session_id="runtime")

    assert service.kind(owner) == "realm"
    assert service.owners.mode(owner, "realm") == "ask"
    if stage == "prerequisites":
        assert stopped == []
