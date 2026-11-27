"""The environment the gateway's service will start with, which the ``.env`` sign-in template check compares
against the old backend's. A first move installs the gateway's service, so the service the move creates is
modelled: the user manager's environment plus the ``Environment=`` lines ``hermes gateway install`` writes
(or the LaunchAgent's ``EnvironmentVariables``). An installed unit's own ``UnsetEnvironment=`` is applied last,
as systemd does, on the gateway's side and the old backend's. Follow-up to #157403.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import gateway as gw
from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli import gateway_migrate_backends_host as backends_host
from hermes_cli.gateway_migrate_backends_host import SystemHost
from tests.hermes_cli.gateway_migrate_backends_apply.test_launcher_templates import (
    TemplateHost,
    _dotenv,
    _template_blockers,
)
from tests.hermes_cli.test_gateway_migrate_backends_apply import _plan

_TEMPLATE = "HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${SOMEVAR}\n"


class FakeSystemctl:
    """``systemctl --user``: the manager's environment, and ``show`` output per unit (a unit not listed is
    not installed: systemd 255 prints empty properties and exits 0, as measured)."""

    def __init__(self, manager="", units=None, manager_rc=0):
        self.manager, self.units, self.manager_rc, self.calls = manager, dict(units or {}), manager_rc, []

    def __call__(self, scope, args, **kw):
        self.calls.append(list(args))
        if args[:1] == ["show-environment"]:
            return subprocess.CompletedProcess(args, self.manager_rc, self.manager, "")
        if args[:1] == ["show"]:
            return subprocess.CompletedProcess(
                args, 0, self.units.get(args[1], "Environment=\nEnvironmentFiles=\nUnsetEnvironment=\n"), "")
        return subprocess.CompletedProcess(args, 1, "", "unexpected")


@pytest.fixture
def no_gateway_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: tmp_path / "absent" / "gw.service")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: tmp_path / "absent" / "gw.plist")


class FirstMoveHost(TemplateHost):
    """The old backend's unit starts it with *old_env*; the gateway's side is read for real, through *systemctl*."""

    def __init__(self, home, old_env, **kw):
        super().__init__(home, **kw)
        self.old_env = dict(old_env)
        self.old["active"] = False  # read from the unit, not the process

    def process_env(self, pid):
        return None

    def unit_environment(self, unit, scope):
        if unit == "hermes-dashboard.service":
            return dict(self.old_env), None
        return SystemHost.unit_environment(self, unit, scope)

    def unit_launcher_env(self, unit, scope):
        return dict(self.old_env)

    gateway_environment = SystemHost.gateway_environment  # read for real


def _installed(home) -> str:
    return str(home.resolve())


def test_a_gateway_unit_not_installed_yet_starts_with_the_manager_and_what_install_writes(home, monkeypatch,
                                                                                         no_gateway_unit):
    monkeypatch.setattr(backends_host, "_systemctl", FakeSystemctl("SOMEVAR=same\nPATH=/usr/bin\n"))

    env, why = SystemHost(home).gateway_environment()

    assert why is None
    assert env["SOMEVAR"] == "same"  # the user manager's
    assert env["HERMES_HOME"] == _installed(home) and env["HERMES_SUPERVISED_CHILD"] == "1"  # the installer's
    assert env["PATH"] != "/usr/bin"  # the unit's Environment=PATH wins over the manager's


def test_the_installer_unit_environment_is_read_with_systemd_quoting():
    text = ('[Service]\nEnvironment="A=two words"\nEnvironment="B=back\\\\slash \\"q\\" 100%%"\n'
            'ExecStart=/bin/true\n')

    assert backends_host.unit_text_environment(text) == ({"A": "two words", "B": 'back\\slash "q" 100%'}, None)
    assert backends_host.unit_text_environment('Environment="A=%h"\n')[1]  # a specifier is not resolved here


@pytest.mark.parametrize(("manager", "dotenv", "refused"), [
    ("SOMEVAR=same\n", _TEMPLATE, False),  # the manager gives the gateway the old backend's value
    ("SOMEVAR=other\n", _TEMPLATE, True),
    ("", _TEMPLATE, True),
    # the old unit and the unit `gateway install` writes both set HERMES_HOME to this home
    ("", "HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${HERMES_HOME}-pw\n",
     False),
])
def test_a_first_move_compares_with_the_service_it_installs(home, capsys, monkeypatch, no_gateway_unit,
                                                           manager, dotenv, refused):
    _dotenv(home, dotenv)
    monkeypatch.setattr(backends_host, "_systemctl", FakeSystemctl(manager))
    host = FirstMoveHost(home, {"HERMES_HOME": _installed(home), "SOMEVAR": "same"})

    code, out = _plan(host, capsys)

    assert (code, bool(_template_blockers(out))) == ((engine.EXIT_BLOCKED, True) if refused else (engine.EXIT_OK, False))


def test_a_first_move_refuses_a_template_when_the_manager_environment_cannot_be_read(home, capsys, monkeypatch,
                                                                                     no_gateway_unit):
    _dotenv(home, _TEMPLATE)
    monkeypatch.setattr(backends_host, "_systemctl", FakeSystemctl("SOMEVAR=same\n", manager_rc=1))

    code, out = _plan(FirstMoveHost(home, {"HERMES_HOME": _installed(home), "SOMEVAR": "same"}), capsys)

    assert code == engine.EXIT_BLOCKED
    (blocker,) = _template_blockers(out)
    assert "show-environment" in blocker


def test_a_launchagent_not_installed_yet_starts_with_what_install_writes(home, monkeypatch, no_gateway_unit):
    host = SystemHost(home)
    monkeypatch.setattr(SystemHost, "gateway_kind", lambda self: "launchd")

    env, why = host.gateway_environment()

    assert why is None and env["HERMES_HOME"] == _installed(home) and env["HERMES_SUPERVISED_CHILD"] == "1"


def test_an_installed_gateway_unit_is_still_read_from_systemd(home, monkeypatch, tmp_path):
    unit = tmp_path / "gw.service"
    unit.write_text("[Service]\n", encoding="utf-8")
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit)
    fake = FakeSystemctl("SOMEVAR=same\n", {gw.get_service_name(): "Environment=X=1\nEnvironmentFiles=\n"})
    monkeypatch.setattr(backends_host, "_systemctl", fake)

    env, why = SystemHost(home).gateway_environment()

    assert why is None and env == {"SOMEVAR": "same", "X": "1"}


def test_a_units_unset_environment_is_applied_last(home, monkeypatch):
    """systemd.exec(5): the list is applied last; NAME drops the variable, NAME=VALUE only that exact assignment."""
    fake = FakeSystemctl("MGR=m\n", {"x.service": "Environment=SOMEVAR=same KEEP=1 EXACT=2\nEnvironmentFiles=\n"
                                                  "UnsetEnvironment=SOMEVAR KEEP=other EXACT=2 MGR\n"})
    monkeypatch.setattr(backends_host, "_systemctl", fake)

    assert SystemHost(home).unit_environment("x.service", "user") == ({"KEEP": "1"}, None)


class RealUnitsHost(FirstMoveHost):
    """Both units are read for real through *systemctl*; the gateway unit is installed."""

    def unit_environment(self, unit, scope):
        return SystemHost.unit_environment(self, unit, scope)

    def unit_launcher_env(self, unit, scope):
        return SystemHost.unit_launcher_env(self, unit, scope)

    def backends(self):
        from hermes_cli.gateway_migrate_backends import BackendItem
        return [BackendItem(supervisor="systemd", running=False, unit="hermes-dashboard.service", scope="user",
                            state="enabled", argv=list(self.argv))]


@pytest.fixture
def gateway_unit(tmp_path, monkeypatch):
    unit = tmp_path / "gw.service"
    unit.write_text("[Service]\n", encoding="utf-8")
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit)
    return gw.get_service_name()


def _show(env, unset=""):
    return f"Environment={env}\nEnvironmentFiles=\nUnsetEnvironment={unset}\n"


@pytest.mark.parametrize(("old", "gateway", "refused"), [
    (_show("HOME_PIN SOMEVAR=same"), _show("SOMEVAR=same", "SOMEVAR"), True),  # the gateway unit unsets it
    (_show("HOME_PIN SOMEVAR=same", "SOMEVAR"), _show("SOMEVAR=same"), True),  # the old unit unsets it
    (_show("HOME_PIN SOMEVAR=same", "SOMEVAR=other"), _show("SOMEVAR=same"), False),  # not that assignment
    (_show("HOME_PIN SOMEVAR=same"), _show("SOMEVAR=same"), False),  # control: nothing unset
])
def test_unset_environment_is_applied_on_both_sides_before_comparing(home, capsys, monkeypatch, gateway_unit,
                                                                     old, gateway, refused):
    _dotenv(home, _TEMPLATE)
    old = old.replace("HOME_PIN", f"HERMES_HOME={_installed(home)}")
    monkeypatch.setattr(backends_host, "_systemctl",
                        FakeSystemctl("", {"hermes-dashboard.service": old, gateway_unit: gateway}))

    code, out = _plan(RealUnitsHost(home, {}), capsys)

    assert (code, bool(_template_blockers(out))) == ((engine.EXIT_BLOCKED, True) if refused else (engine.EXIT_OK, False))


def test_a_unit_that_unsets_its_hermes_home_is_not_this_homes(home, capsys, monkeypatch, gateway_unit, tmp_path):
    """``Environment=HERMES_HOME=<this home>`` with ``UnsetEnvironment=HERMES_HOME`` runs the account's default home."""
    old = _show(f"HERMES_HOME={_installed(home)} HOME={tmp_path / 'acct'}", "HERMES_HOME")
    monkeypatch.setattr(backends_host, "_systemctl",
                        FakeSystemctl("", {"hermes-dashboard.service": old, gateway_unit: _show("")}))

    code, out = _plan(RealUnitsHost(home, {}), capsys)

    assert (code, out["plan"]["blockers"], out["steps"]) == (engine.EXIT_OK, [], [])


@pytest.mark.parametrize("kind", ["systemd", "launchd"])
def test_a_gateway_service_the_installer_cannot_generate_is_unknown(home, monkeypatch, no_gateway_unit, kind):
    from hermes_cli import gateway_launchd

    def fail():
        raise OSError("unreadable")

    monkeypatch.setattr(gw, "generate_systemd_unit", fail)
    monkeypatch.setattr(gateway_launchd, "generate_launchd_plist", fail)
    monkeypatch.setattr(backends_host, "_systemctl", FakeSystemctl("SOMEVAR=same\n"))
    monkeypatch.setattr(SystemHost, "gateway_kind", lambda self: kind)

    why = SystemHost(home).gateway_environment()[1]

    assert why and "cannot be generated here (OSError)" in why
