"""An elevated (UAC) client never starts an elevated gateway.

The unmanaged start inherits the client's token. A Desktop or terminal run as administrator, with no
Scheduled Task registered, started a gateway every normal Hermes window then refused as
``gateway_elevated``, until someone stopped it from an admin prompt. A registered task starts under
its own principal, not the client's token, so that path stays open. The remedy names only a normal
window: a task registered through the UAC hand-off can itself start the gateway elevated (#152218).
An administrator with no split token (UAC off, the built-in Administrator) has no normal window to
lock out, so that client still starts its gateway.
"""
import time

import pytest


@pytest.fixture(params=[2, None], ids=["uac-full-token", "type-unreadable"])
def elevated(request, monkeypatch):
    import hermes_dir_acl
    from hermes_agent.hermes_cli import gateway_runtime_start as start

    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "process_elevation_type", lambda: request.param)
    spawned = []
    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: spawned.append(a) or pytest.fail("spawned"))
    return spawned


def test_ensure_from_an_elevated_client_refuses_instead_of_starting(tmp_path, monkeypatch, elevated):
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service

    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    result = runtime.ensure_gateway_runtime(tmp_path / "profile", timeout=5.0)

    assert (result.state, result.reason_code) == ("inaccessible", "client_elevated")
    assert "non-administrator" in result.detail and "gateway install" not in result.detail
    assert elevated == []


def test_detached_restart_from_an_elevated_client_names_the_remedy(tmp_path, capsys, elevated):
    from hermes_agent.hermes_cli import gateway_runtime_start as start

    assert start.start_detached_replacement(tmp_path / "profile") == 1
    err = capsys.readouterr().err
    assert "client_elevated" in err and "non-administrator" in err and "gateway install" not in err
    assert elevated == []


def test_an_elevated_client_still_starts_the_registered_task(tmp_path, monkeypatch, elevated):
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service

    task = service.ExistingService("windows", ("schtasks.exe", "/Run", "/TN", "Hermes_Gateway"))
    started = []
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: task)
    monkeypatch.setattr(service, "start_existing_gateway_service", lambda svc, **k: started.append(svc))
    before = time.monotonic()
    result = runtime.ensure_gateway_runtime(tmp_path / "profile", timeout=1.0)

    assert started == [task] and result.reason_code != "client_elevated"
    assert time.monotonic() - before < 10 and elevated == []


def test_an_elevated_client_without_a_split_token_still_starts(tmp_path, monkeypatch):
    import hermes_dir_acl
    from hermes_agent.hermes_cli import gateway_runtime_start as start

    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "process_elevation_type", lambda: hermes_dir_acl.TOKEN_ELEVATION_TYPE_DEFAULT)
    monkeypatch.setattr(start, "start_gateway_job", lambda *a, **k: None)  # macOS: the direct spawn, too
    spawned = []
    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: spawned.append(a) or object())
    start.spawn_unmanaged_gateway(tmp_path / "profile", deadline=time.monotonic() + 5)
    assert len(spawned) == 1


@pytest.mark.platforms("windows")
def test_the_gate_reads_this_hosts_real_token(tmp_path, monkeypatch):
    """No pin: the real TokenElevation and TokenElevationType reads. The gate refuses exactly the
    UAC-elevated box (Full); whichever box this runner's token is in, the two reads agree."""
    import hermes_dir_acl
    from hermes_agent.hermes_cli import gateway_runtime_start as start
    from hermes_agent.hermes_cli.gateway_runtime_service import RuntimeStartError

    elevation_type = hermes_dir_acl.process_elevation_type()
    assert elevation_type in (1, 2, 3)  # Default, Full, Limited
    if elevation_type != hermes_dir_acl.TOKEN_ELEVATION_TYPE_DEFAULT:
        assert hermes_dir_acl.process_elevation() is (elevation_type == 2)
    spawned = []
    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: spawned.append(a) or object())
    try:
        start.spawn_unmanaged_gateway(tmp_path / "profile", deadline=time.monotonic() + 5)
        refused = False
    except RuntimeStartError as exc:
        assert exc.reason == "client_elevated"
        refused = True
    assert refused is (elevation_type == 2) and len(spawned) == (0 if refused else 1)
