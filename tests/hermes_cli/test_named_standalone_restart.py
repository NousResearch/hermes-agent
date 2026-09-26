"""#120871: a standalone named gateway restarts its OWN service; only a true satellite refuses.

A per-profile standalone fleet member (own service, own host record, no default gateway)
hit the multiplexer guard on ``gateway restart``: ``named_profile_served_by_running_multiplexer``
counted the profile's OWN host record as "served by a multiplexer", so the restart exited 78
pointing at ``-p default``. A profile genuinely served by another home's host process must
still be refused (no double-poll).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import host_attach
from gateway import host_rendezvous as hr
from hermes_cli import gateway as gw


def _reset_probe_memo():
    getattr(host_attach, "invalidate_host_gateway_cache", lambda: None)()


@pytest.fixture(autouse=True)
def _no_memo():
    _reset_probe_memo()
    yield
    _reset_probe_memo()


def _write_record(pid, home, profiles):
    payload = {
        "role": hr.ROLE_GATEWAY,
        "home": str(home),
        "pid": pid,
        "createTime": hr.process_create_time(pid),
        "host": "",
        "port": None,
        "protocolVersion": hr.HOST_PROTOCOL_VERSION,
        "tokenFingerprint": "",
        "profiles": list(profiles),
        "updatedAt": "",
    }
    path = hr.record_path(hr.ROLE_GATEWAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def live_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield child
    finally:
        child.terminate()
        child.wait(timeout=10)


def _arrange_named_home(monkeypatch, tmp_path, own_home):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("HERMES_HOME", str(own_home))
    own_home.mkdir(parents=True, exist_ok=True)
    (own_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "gateway.status._get_process_hermes_home", lambda: Path(own_home)
    )


def _publish_host(monkeypatch, child, record_home, served):
    _write_record(child.pid, record_home, served)
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda dialled, **kw: {
            "pid": child.pid,
            "hermes_home": str(record_home),
            "served_profiles": list(served),
        },
    )


def test_own_standalone_host_record_does_not_refuse_restart(
    monkeypatch, tmp_path, live_child
):
    """The profile's OWN host record is not a multiplexer serving it: restart is allowed."""
    own_home = tmp_path / "root" / "profiles" / "argus"
    _arrange_named_home(monkeypatch, tmp_path, own_home)
    _publish_host(monkeypatch, live_child, own_home, ("argus",))
    monkeypatch.setattr(gw, "_is_service_installed", lambda: True)

    assert gw._served_by_another_host_gateway() is None
    assert gw.named_profile_served_by_running_multiplexer() is False
    assert gw._named_profile_refused_under_multiplexer() is False


def test_foreign_host_record_still_refuses_restart(
    monkeypatch, tmp_path, live_child, capsys
):
    """A profile genuinely served by another home's host process is still refused."""
    default_home = tmp_path / "root"
    own_home = tmp_path / "root" / "profiles" / "argus"
    _arrange_named_home(monkeypatch, tmp_path, own_home)
    _publish_host(monkeypatch, live_child, default_home, ("default", "argus"))
    monkeypatch.setattr(gw, "_is_service_installed", lambda: False)

    assert gw.named_profile_served_by_running_multiplexer() is True
    assert gw._named_profile_refused_under_multiplexer() is True
    out = capsys.readouterr().out
    assert "hermes -p default gateway restart" in out


def test_cron_restart_hint_names_own_standalone_scheduler(monkeypatch):
    """cron status points at the profile's own service when IT is the scheduler host."""
    from hermes_cli import cron as cron_mod

    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 4242)
    assert (
        cron_mod._cron_scheduler_restart_command(4242, "argus")
        == "hermes --profile argus gateway restart"
    )
    assert (
        cron_mod._cron_scheduler_restart_command(9999, "argus")
        == "hermes --profile default gateway restart"
    )
