"""Data deletion must establish quiescence, never infer it from a signal count."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import uninstall
from tests.hermes_cli.test_data_uninstall import layout  # noqa: F401 — isolated layout


@pytest.mark.parametrize("mode", ["confirmed", "cancel", "dry-run"])
def test_unconfirmed_removal_never_contacts_the_gateway(layout, monkeypatch, mode):
    home, _, data = layout
    calls = []

    def identify(target, **kwargs):
        calls.append(target)
        return {"pid": 42, "hermes_home": str(home), "supervisor": "systemd"}

    monkeypatch.setattr("gateway.control_socket.identify_gateway", identify)
    monkeypatch.setattr("builtins.input", lambda *args: "no")
    args = SimpleNamespace(yes=mode != "cancel", dry_run=mode == "dry-run")
    if mode == "confirmed":
        with pytest.raises(SystemExit):
            uninstall.run_data_uninstall(args)
        assert calls == [home]
    else:
        uninstall.run_data_uninstall(args)
        assert calls == []
    assert all(path.exists() for path in data)


def test_live_chat_lease_blocks_data_deletion(layout):
    from hermes_cli.active_sessions import release_active_session, try_acquire_active_session

    home, _, data = layout
    lease, refusal = try_acquire_active_session(
        session_id="fixture-session", surface="cli", config={}, registry_home=home,
        track_liveness=True,
    )
    assert lease is not None and refusal is None
    try:
        with pytest.raises(SystemExit):
            uninstall.run_data_uninstall(SimpleNamespace(yes=True))
        assert all(path.exists() for path in data)
    finally:
        release_active_session(lease)


def test_manual_gateway_drains_over_real_control_transport_before_deletion(layout):
    import asyncio
    import subprocess
    import sys
    from threading import Event, Thread
    from gateway.control_socket import GatewayControlServer
    from gateway.status import get_process_start_time

    home, _, data = layout
    # A disposable child is the declared process. Control dispatch is real;
    # the fixture handler stops only that child, never any running gateway.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], cwd=home)
    loop = asyncio.new_event_loop()
    ready = Event()
    paused = Event()
    holders = []

    def identify():
        return {"pid": child.pid, "start_time": get_process_start_time(child.pid),
                "hermes_home": str(home), "supervisor": "manual"}

    def drain():
        paused.set()
        child.terminate()
        child.wait(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        return {"pid": child.pid, "pausing": True, "drain_timeout": 0}

    server = GatewayControlServer(home, verb_handlers={"identify": identify, "pause-for-update": drain})

    def serve():
        asyncio.set_event_loop(loop)
        started = loop.run_until_complete(server.start())
        holders.append(started)
        ready.set()
        if started:
            loop.run_forever()
        loop.run_until_complete(server.stop())
        loop.close()

    thread = Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=10) and holders == [True]
        uninstall.run_data_uninstall(SimpleNamespace(yes=True))
        assert paused.is_set() and child.poll() is not None
        assert all(not path.exists() for path in data)
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        if thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
    assert not thread.is_alive()


@pytest.mark.parametrize("profile", ["", "sibling"])
def test_backend_initial_profile_is_not_its_write_scope(layout, profile):
    from hermes_cli.process_identity import register_self

    _, _, data = layout
    assert register_self("serve", project_root=uninstall.get_project_root(), detail={"profile": profile})
    with pytest.raises(SystemExit):
        uninstall.run_data_uninstall(SimpleNamespace(yes=True))
    assert all(path.exists() for path in data)


@pytest.mark.parametrize("terminal", [False, True])
def test_detached_cron_attempt_must_finish_before_data_removal(layout, monkeypatch, terminal):
    from cron import executions

    _, _, data = layout
    monkeypatch.setattr(executions, "_emit_execution_state", lambda *args, **kwargs: None)
    record = executions.create_execution("temporary-job", source="manual")
    if terminal:
        executions.finish_execution(record["id"], success=True)
        uninstall.run_data_uninstall(SimpleNamespace(yes=True))
        assert all(not path.exists() for path in data)
    else:
        with pytest.raises(SystemExit):
            uninstall.run_data_uninstall(SimpleNamespace(yes=True))
        assert all(path.exists() for path in data)


def test_named_profile_cannot_delete_data_served_by_the_default_multiplexer(layout, monkeypatch):
    home, _, _ = layout
    profile = home / "profiles" / "active"
    profile.mkdir()
    config = profile / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: profile)

    def identify(target, **kwargs):
        if target == home:
            return {"pid": 42, "hermes_home": str(home), "served_profiles": ["active"]}
        return None

    monkeypatch.setattr("gateway.control_socket.identify_gateway", identify)
    monkeypatch.setattr("gateway.control_socket.pause_gateway_for_update",
                        lambda *args, **kwargs: pytest.fail("must not stop other profiles' gateway"))
    with pytest.raises(SystemExit):
        uninstall.run_data_uninstall(SimpleNamespace(yes=True))
    assert config.read_text(encoding="utf-8") == "{}"


def test_data_removal_respects_the_checkpoint_store_owner(layout):
    from tools.checkpoint_pruning import store_lock

    home, _, data = layout
    with store_lock(home / "checkpoints"):
        with pytest.raises(SystemExit):
            uninstall.run_data_uninstall(SimpleNamespace(yes=True))
        assert all(path.exists() for path in data)
