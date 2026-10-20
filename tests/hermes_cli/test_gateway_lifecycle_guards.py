"""Gateway lifecycle verbs never take down a gateway they must not touch.

Two P0 invariants:

* A lifecycle call never acts on a service definition pinning another HERMES_HOME. A probe run with
  ``HOME=<scratch>`` resolved the bare ``hermes-gateway`` name (its scratch ``.hermes`` looked native),
  found the ACCOUNT's real unit through the account-anchored unit dir, and ``systemctl --user stop``
  took the production gateway down.
* ``gateway stop``/``restart`` refuse while the gateway runs live sessions, naming them, unless the
  caller passes ``--interrupt``.

Every ``systemctl``/``launchctl``/kill seam is replaced by a recorder; nothing here executes one.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gw


def _unit(path, home) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[Service]\nEnvironment="HERMES_HOME={home}"\nExecStart=/usr/bin/hermes gateway run\n',
                    encoding="utf-8")


@pytest.fixture
def recorded(monkeypatch):
    """Record every service-manager and process-kill call instead of running it."""
    calls: list = []
    monkeypatch.setattr(gw, "_run_systemctl", lambda argv, **kw: calls.append(("systemctl", list(argv))))
    monkeypatch.setattr(gw.subprocess, "run", lambda argv, *a, **kw: calls.append(("run", list(argv))))
    monkeypatch.setattr(gw, "terminate_pid", lambda pid, **kw: calls.append(("kill", pid)))
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "_preflight_user_systemd", lambda: None)
    monkeypatch.setattr(gw, "refresh_systemd_unit_if_needed", lambda system=False: False)
    return calls


# ── Invariant 1: never act on another home's service ─────────────────────────────────────────────


@pytest.mark.platforms("linux")
def test_scratch_home_does_not_resolve_the_accounts_bare_gateway_unit(tmp_path, monkeypatch):
    """The incident's resolution: HOME=<scratch>, HERMES_HOME unset, the account's real bare unit
    pinning the account home. The scratch home must not name that unit."""
    account, scratch = tmp_path / "account", tmp_path / "scratch"
    (scratch / ".hermes").mkdir(parents=True)
    real_unit = account / ".config" / "systemd" / "user" / "hermes-gateway.service"
    _unit(real_unit, account / ".hermes")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(scratch))
    monkeypatch.setenv("HERMES_REAL_HOME", str(account))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)

    assert gw.get_hermes_home() == scratch / ".hermes"
    assert gw.get_service_name() != "hermes-gateway"
    assert gw.get_systemd_unit_path() != real_unit
    assert not gw.get_systemd_unit_path().exists()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("verb", ["stop", "restart", "start", "uninstall"])
def test_lifecycle_verb_refuses_a_user_unit_pinning_another_home(tmp_path, monkeypatch, recorded, capsys, verb):
    """The incident's scope (USER unit; #141908's tests cover the system unit). Two homes can still share
    a unit name (a custom root's ``profiles/coder`` and the account's): the verb refuses the unit that
    pins the other one, with no systemctl call at all and without adopting its home."""
    account = tmp_path / "account"
    custom_profile = tmp_path / "custom" / "profiles" / "coder"
    custom_profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(custom_profile))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(account / ".config"))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    unit = gw.get_systemd_unit_path()
    assert unit.name == "hermes-gateway-coder.service"
    _unit(unit, account / ".hermes" / "profiles" / "coder")

    if verb == "uninstall":
        gw._service_call("systemd", verb, False)
    else:
        with pytest.raises(SystemExit) as exc:
            gw._service_call("systemd", verb, False)
        assert exc.value.code == 1

    assert recorded == []
    captured = capsys.readouterr()
    assert "Refusing to" in captured.out + captured.err
    assert unit.exists()
    assert os.environ["HERMES_HOME"] == str(custom_profile)


# ── Invariant 2: stop/restart refuse while sessions are live ─────────────────────────────────────


def _live_gateway(monkeypatch, work):
    """A gateway answering the ``active-work`` control verb with *work*."""
    from gateway import control_socket
    monkeypatch.setattr(control_socket, "query_gateway_control",
                        lambda home, verb, **kw: work if verb == "active-work" else None)
    monkeypatch.setattr(gw, "_host_multiplexer_for_all_verb", lambda: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda *a, **k: [4242])


_WORK = {"active_agents": 2, "active_work": [
    {"kind": "chat", "session": "agent:main:telegram:dm:7", "pid": 4242, "elapsed_s": 75.0},
    {"kind": "delegation", "delegation_id": "deleg_ab12cd34", "goal": "review the PR", "pid": 4242},
]}


def test_stop_refuses_while_sessions_are_live_and_names_them(monkeypatch, recorded, capsys):
    _live_gateway(monkeypatch, _WORK)
    stopped: list = []
    monkeypatch.setattr(gw, "_stop_installed_service", lambda system: stopped.append(system) or True)
    monkeypatch.setattr(gw, "stop_profile_gateway", lambda: stopped.append("process") or True)

    with pytest.raises(SystemExit) as exc:
        gw._cmd_stop(SimpleNamespace(all=False, system=False, interrupt=False))

    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert stopped == [] and recorded == []
    assert "2 live session(s)" in out
    assert "chat turn agent:main:telegram:dm:7" in out
    assert "background delegation deleg_ab12cd34" in out
    assert "hermes gateway stop --interrupt" in out

    gw._cmd_stop(SimpleNamespace(all=False, system=False, interrupt=True))
    assert stopped == [False]
    assert "--interrupt: stop proceeds: 2 live session(s) would be interrupted" in capsys.readouterr().out


def test_restart_refuses_on_the_published_active_agents_of_an_older_gateway(monkeypatch, recorded, capsys):
    """A gateway without the control verb still publishes ``active_agents`` in gateway_state.json."""
    from gateway import control_socket
    from gateway.status import write_runtime_status
    write_runtime_status(gateway_state="running", active_agents=1)
    monkeypatch.setattr(control_socket, "query_gateway_control", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_host_multiplexer_for_all_verb", lambda: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda force=False: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda probe: "systemd")
    service: list = []
    monkeypatch.setattr(gw, "_service_call", lambda *a: service.append(a))
    monkeypatch.setattr(gw, "_reap_unsupervised_gateway_orphans", lambda: None)

    with pytest.raises(SystemExit):
        gw._cmd_restart(SimpleNamespace(all=False, system=False, force=False, interrupt=False))

    assert service == [] and recorded == []
    assert "1 active agent run(s)" in capsys.readouterr().out


def test_stop_all_leaves_another_homes_gateway_process_alone(monkeypatch, recorded, tmp_path):
    """``kill_gateway_processes(all_profiles=True)`` scans every gateway of this account; a scratch-home
    ``stop --all`` must not SIGTERM a gateway running on another Hermes home."""
    from hermes_cli import dashboard_procs
    own, other = str(gw.get_hermes_home()), str(tmp_path / "account" / ".hermes")
    monkeypatch.setattr(gw, "find_gateway_pids", lambda *a, **k: [111, 222])
    monkeypatch.setattr(dashboard_procs, "_hermes_home_for_pid", {111: own, 222: other}.get)

    assert gw.kill_gateway_processes(all_profiles=True) == 1
    assert recorded == [("kill", 111)]


def test_dashboard_stop_refuses_while_sessions_are_live(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from gateway import host_attach
    from hermes_cli import web_server, web_server_gateway

    _live_gateway(monkeypatch, _WORK)
    monkeypatch.setattr(host_attach, "host_gateway", lambda **kw: None)
    monkeypatch.setattr(web_server_gateway, "multiplexed_profile_refusal", lambda *a: None)
    spawned: list = []
    monkeypatch.setattr(web_server_gateway, "_spawn_hermes_action",
                        lambda argv, name: spawned.append(argv) or SimpleNamespace(pid=1))
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    refused = client.post("/api/gateway/stop")
    assert refused.status_code == 409
    assert "chat turn agent:main:telegram:dm:7" in refused.json()["detail"]
    assert spawned == []

    assert client.post("/api/gateway/stop?interrupt=true").status_code == 200
    assert spawned and spawned[0][-1] == "--interrupt"


@pytest.mark.platforms("linux")
def test_sudo_system_stop_checks_the_units_home_not_roots(tmp_path, monkeypatch, recorded, capsys):
    """``sudo hermes gateway stop --system``: HOME=/root and HERMES_HOME stripped, while the system unit
    runs the invoking user's home. The live-work check must ask the unit's home, without adopting it."""
    import hermes_constants
    root_home, user_home = tmp_path / "root", tmp_path / "user" / ".hermes"
    (root_home / ".hermes").mkdir(parents=True)
    user_home.mkdir(parents=True)
    _unit(tmp_path / "systemd" / "hermes-gateway.service", user_home)
    monkeypatch.setattr(gw, "_SYSTEM_UNIT_DIR", tmp_path / "systemd")
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.delenv("SUDO_USER", raising=False)
    monkeypatch.setenv("HOME", str(root_home))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(hermes_constants, "_account_home", lambda: root_home)
    _live_gateway(monkeypatch, None)
    from gateway import control_socket
    monkeypatch.setattr(control_socket, "query_gateway_control", lambda home, verb, **kw: (
        _WORK if verb == "active-work" and Path(home).resolve() == user_home.resolve() else None))
    monkeypatch.setattr(gw, "_served_by_another_host_gateway", lambda: None)
    monkeypatch.setattr(gw, "named_profile_served_by_running_multiplexer", lambda: False)
    stopped: list = []
    monkeypatch.setattr(gw, "_stop_installed_service", lambda system: stopped.append(system) or True)

    with pytest.raises(SystemExit) as exc:
        gw._cmd_stop(SimpleNamespace(all=False, system=True, interrupt=False))

    assert exc.value.code == 1 and stopped == [] and recorded == []
    assert "chat turn agent:main:telegram:dm:7" in capsys.readouterr().out
    assert "HERMES_HOME" not in os.environ  # the check never adopted the unit's home

