"""An elevated (UAC) client never starts an elevated gateway.

The unmanaged start inherits the client's token. A Desktop or terminal run as administrator, with no
Scheduled Task registered, started a gateway every normal Hermes window then refused as
``gateway_elevated``, until someone stopped it from an admin prompt. The registered task starts the
gateway without elevation whatever the client's token, so that path stays open.
"""
import time

import pytest


@pytest.fixture
def elevated(monkeypatch):
    import hermes_dir_acl
    from hermes_cli import gateway_runtime_start as start

    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    spawned = []
    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: spawned.append(a) or pytest.fail("spawned"))
    return spawned


def test_ensure_from_an_elevated_client_refuses_instead_of_starting(tmp_path, monkeypatch, elevated):
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service

    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    result = runtime.ensure_gateway_runtime(tmp_path / "profile", timeout=5.0)

    assert (result.state, result.reason_code) == ("inaccessible", "client_elevated")
    assert "administrator" in result.detail and "hermes gateway install" in result.detail
    assert elevated == []


def test_detached_restart_from_an_elevated_client_names_the_remedy(tmp_path, capsys, elevated):
    from hermes_cli import gateway_runtime_start as start

    assert start.start_detached_replacement(tmp_path / "profile") == 1
    err = capsys.readouterr().err
    assert "client_elevated" in err and "hermes gateway install" in err and elevated == []


def test_an_elevated_client_still_starts_the_registered_task(tmp_path, monkeypatch, elevated):
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service

    task = service.ExistingService("windows", ("schtasks.exe", "/Run", "/TN", "Hermes_Gateway"))
    started = []
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: task)
    monkeypatch.setattr(service, "start_existing_gateway_service", lambda svc, **k: started.append(svc))
    before = time.monotonic()
    result = runtime.ensure_gateway_runtime(tmp_path / "profile", timeout=1.0)

    assert started == [task] and result.reason_code != "client_elevated"
    assert time.monotonic() - before < 10 and elevated == []
