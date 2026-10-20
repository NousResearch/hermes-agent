"""`hermes gateway stop|restart|start` from a sandbox home must never operate ANOTHER install's systemd unit.

Real incident: an e2e helper ran `hermes gateway stop` with a throwaway ``HOME`` and
``HERMES_HOME=$HOME/.hermes`` on a host whose production gateway is the system unit
``hermes-gateway.service`` (``Environment=HERMES_HOME=/root/.hermes``). The sandbox home is the
"default" home relative to its own ``HOME``, so the service name resolved to the bare
``hermes-gateway``; system units are global, so that unit "existed"; ``_sync_hermes_home_from_systemd_unit``
then adopted the production home and ``systemctl stop hermes-gateway`` took every bot offline.
"""
import os
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli import gateway_service_owner


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A production system unit pinned to ``prod``; the account's own default root is ``account``."""
    prod = tmp_path / "prod" / ".hermes"
    account = tmp_path / "account" / ".hermes"
    sandbox_home = tmp_path / "sandbox"
    for p in (prod, account, sandbox_home / ".hermes"):
        p.mkdir(parents=True)
    unit_dir = tmp_path / "systemd"
    unit_dir.mkdir()
    (unit_dir / f"{gateway_cli._SERVICE_BASE}.service").write_text(
        f'[Service]\nEnvironment="HERMES_HOME={prod}"\nExecStart=/bin/true\n', encoding="utf-8")
    monkeypatch.setattr(gateway_cli, "_SYSTEM_UNIT_DIR", unit_dir)
    monkeypatch.setattr(gateway_cli, "user_systemd_unit_dir", lambda: tmp_path / "no-user-units")
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_account_home", lambda: account.parent)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    # Never reach a real systemd: preamble/privilege checks are not under test, systemctl is recorded.
    monkeypatch.setattr(gateway_cli, "_systemd_scope_preamble", lambda action, system, **_k: system)
    monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: None)
    monkeypatch.setattr(gateway_cli, "_systemctl_show", lambda props, system=False: {})
    monkeypatch.setattr(gateway_cli, "_mark_planned_stop", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(gateway_cli, "_run_systemctl", lambda args, **kw: calls.append(list(args)))
    return {"prod": prod, "account": account, "sandbox_home": sandbox_home, "calls": calls}


def _sandbox(monkeypatch, host):
    """A sandbox home whose resolved unit name collides with a unit pinning the production home. The
    naming root no longer gives a throwaway HOME the bare name, but names still collide (a custom root's
    ``profiles/<name>``, a pre-pinning unit), so the guard is what keeps the verb off that unit."""
    monkeypatch.setenv("HOME", str(host["sandbox_home"]))
    monkeypatch.setenv("HERMES_HOME", str(host["sandbox_home"] / ".hermes"))
    collided = gateway_cli._SYSTEM_UNIT_DIR / f"{gateway_cli.get_service_name()}.service"
    collided.write_text(f'[Service]\nEnvironment="HERMES_HOME={host["prod"]}"\nExecStart=/bin/true\n', encoding="utf-8")


@pytest.mark.platforms("linux")
def test_sandbox_home_does_not_name_the_bare_unit_and_a_colliding_unit_is_not_installed_for_it(host, monkeypatch):
    _sandbox(monkeypatch, host)
    assert gateway_cli.get_service_name() != gateway_cli._SERVICE_BASE      # naming root: account, not $HOME
    assert gateway_service_owner.foreign_systemd_unit_home(True) == str(host["prod"])
    assert gateway_cli._systemd_unit_installed() is False                   # `gateway stop` -> PID path
    assert gateway_cli._installed_service_kind_for(lambda: False) is None


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("verb", ["stop", "restart", "start"])
def test_direct_service_verbs_refuse_a_foreign_unit_without_touching_systemd(host, monkeypatch, verb):
    _sandbox(monkeypatch, host)
    with pytest.raises(SystemExit) as exc:
        getattr(gateway_cli, f"systemd_{verb}")(system=True)
    assert exc.value.code == 1
    assert host["calls"] == []
    assert os.environ["HERMES_HOME"] == str(host["sandbox_home"] / ".hermes")   # never adopted prod's home


@pytest.mark.platforms("linux")
def test_uninstall_guard_checks_before_the_home_sync(host, monkeypatch):
    _sandbox(monkeypatch, host)
    assert gateway_cli._systemd_unit_belongs_to_current_home(system=True) is False
    assert os.environ["HERMES_HOME"] == str(host["sandbox_home"] / ".hermes")


@pytest.mark.platforms("linux")
def test_sudo_from_the_account_default_home_still_operates_the_unit(host, monkeypatch):
    """#108674: root's default home driving a unit that pins the invoking user's home is the operator."""
    monkeypatch.setenv("HOME", str(host["account"].parent))
    monkeypatch.setenv("HERMES_HOME", str(host["account"]))
    assert gateway_service_owner.foreign_systemd_unit_home(True) is None
    gateway_cli.systemd_stop(system=True)
    assert host["calls"] == [["stop", gateway_cli.get_service_name()]]
    assert Path(os.environ["HERMES_HOME"]) == host["prod"]                       # adopted, as before


@pytest.mark.platforms("linux")
def test_the_units_own_home_operates_it(host, monkeypatch):
    monkeypatch.setenv("HOME", str(host["prod"].parent))
    monkeypatch.setenv("HERMES_HOME", str(host["prod"]))
    assert gateway_cli._systemd_unit_installed() is True
    gateway_cli.systemd_stop(system=True)
    assert host["calls"] == [["stop", gateway_cli._SERVICE_BASE]]
