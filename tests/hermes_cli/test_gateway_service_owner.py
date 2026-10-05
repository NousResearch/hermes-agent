"""Gateway service-definition writes belong to the home the definition pins.

Two homes can resolve to one service name: ``<root>/profiles/<name>`` and ``~/.hermes/profiles/<name>``
both take ``hermes-gateway-<name>``. Every gateway boot refreshes "its" unit, so a scratch or E2E gateway
started from such a home rewrote the real install's unit (WorkingDirectory and HERMES_HOME pointed at the
scratch dir) and the real gateway failed on its next restart. The collision is simulated here by pointing
the unit/plist path at a file that pins another home; ``test_colliding_profile_names_share_a_unit`` proves
the collision itself on the real name resolution.
"""

import argparse
import plistlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gw
from hermes_cli import gateway_launchd, gateway_service_owner
from hermes_cli.subcommands.gateway import build_gateway_parser

REAL_UNIT = (
    "[Service]\n"
    "ExecStart=/home/ace/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run\n"
    "WorkingDirectory={home}\n"
    'Environment="HERMES_HOME={home}"\n'
)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    real = tmp_path / "account" / ".hermes" / "profiles" / "coder"
    scratch = tmp_path / "ci-scratch" / "run-1" / "profiles" / "coder"
    real.mkdir(parents=True)
    scratch.mkdir(parents=True)
    # The process is the scratch gateway. A scratch dir is not always under a temp root (CI scratch
    # volumes), so the temp-home guard is out of the way: the ownership check is what is under test.
    monkeypatch.setenv("HERMES_HOME", str(scratch))
    monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda definition, kind: False)
    return SimpleNamespace(real=real, scratch=scratch)


def test_colliding_profile_names_share_a_unit(homes, tmp_path, monkeypatch):
    """The premise: a custom root's profile resolves to the same unit as the account's profile."""
    monkeypatch.setattr(gw, "_bare_unit_pinned_home", lambda: None)
    monkeypatch.setattr(gw, "user_systemd_unit_dir", lambda: tmp_path / "systemd-user")
    monkeypatch.setattr(gw, "_native_service_homes", lambda: {(tmp_path / "account" / ".hermes").resolve()})
    scratch_unit = gw.get_systemd_unit_path(system=False)
    monkeypatch.setenv("HERMES_HOME", str(homes.real))
    assert gw.get_systemd_unit_path(system=False) == scratch_unit


@pytest.fixture
def systemd_unit(tmp_path, homes, monkeypatch):
    unit = tmp_path / "systemd" / "hermes-gateway-coder.service"
    unit.parent.mkdir()
    unit.write_text(REAL_UNIT.format(home=homes.real), encoding="utf-8")
    calls = []
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit)
    monkeypatch.setattr(gw, "systemd_unit_is_current", lambda system=False: False)
    monkeypatch.setattr(gw, "_retire_hermes_replace_dropin", lambda system=False: False)
    monkeypatch.setattr(gw, "_prepare_service_launcher", lambda system=False, run_as_user=None: None)
    monkeypatch.setattr(gw, "_run_systemctl", lambda args, **kw: calls.append(tuple(args)) or
                        SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(gw, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(gw, "_ensure_linger_enabled", lambda *a, **k: True)
    monkeypatch.setattr(gw, "_ensure_system_service_linger", lambda *a, **k: None)
    monkeypatch.setattr(gw, "print_systemd_scope_conflict_warning", lambda: None)
    monkeypatch.setattr(gw, "print_legacy_unit_warning", lambda: None)
    monkeypatch.setattr(gw, "generate_systemd_unit", lambda system=False, run_as_user=None:
                        REAL_UNIT.format(home=Path(gw.get_hermes_home())))
    return SimpleNamespace(path=unit, calls=calls, original=unit.read_text(encoding="utf-8"))


class TestSystemdWriters:
    def test_boot_refresh_leaves_a_unit_pinned_to_another_home_untouched(self, systemd_unit, capsys):
        assert gw.refresh_systemd_unit_if_needed(system=False) is False
        assert systemd_unit.path.read_text(encoding="utf-8") == systemd_unit.original
        assert ("daemon-reload",) not in systemd_unit.calls
        assert "Refusing to rewrite" in capsys.readouterr().out

    @pytest.fixture
    def marker_free_unit(self, monkeypatch):
        # refresh's own test belt refuses a generated unit naming a pytest tmpdir; any marker-free body works.
        monkeypatch.setattr(gw, "generate_systemd_unit", lambda system=False, run_as_user=None: "ExecStart=new\n")

    def test_boot_refresh_still_rewrites_its_own_stale_unit(self, systemd_unit, homes, marker_free_unit,
                                                            monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(homes.real))
        assert gw.refresh_systemd_unit_if_needed(system=False) is True
        assert systemd_unit.path.read_text(encoding="utf-8") == "ExecStart=new\n"
        assert ("daemon-reload",) in systemd_unit.calls

    def test_unpinned_unit_is_nobody_elses(self, systemd_unit, marker_free_unit):
        systemd_unit.path.write_text("[Service]\nExecStart=/usr/bin/hermes gateway run\n", encoding="utf-8")
        assert gw.refresh_systemd_unit_if_needed(system=False) is True

    def test_force_install_does_not_overwrite_another_homes_unit(self, systemd_unit):
        with pytest.raises(SystemExit) as exc:
            gw.systemd_install(force=True, non_interactive=True)
        assert exc.value.code == 1
        assert systemd_unit.path.read_text(encoding="utf-8") == systemd_unit.original
        assert systemd_unit.calls == []

    def test_refused_install_never_starts_the_other_homes_service(self, systemd_unit, monkeypatch):
        started = []
        monkeypatch.setattr(gw, "systemd_start", lambda system=False: started.append(system))
        args = SimpleNamespace(start_now=True, start_on_login=True, force_unit_path=False)
        with pytest.raises(SystemExit):
            gw._install_systemd_from_cli(args, force=False, system=False, run_as_user=None)
        assert started == []

    def test_force_unit_path_repoints_the_unit(self, systemd_unit, homes):
        gw.systemd_install(force_unit_path=True, non_interactive=True)
        assert f"HERMES_HOME={homes.scratch}" in systemd_unit.path.read_text(encoding="utf-8")

    def test_system_refresh_checks_the_callers_home_before_adopting_the_units(
            self, systemd_unit, homes, monkeypatch):
        # systemd_unit_is_current adopts a system unit's HERMES_HOME into os.environ (sudo strips it);
        # checking after that adoption compared the unit with itself.
        monkeypatch.setattr(gw, "_read_systemd_user_from_unit", lambda path: None)

        def is_current(system=False):
            gw._sync_hermes_home_from_systemd_unit(system=system)
            return False
        monkeypatch.setattr(gw, "systemd_unit_is_current", is_current)
        assert gw.refresh_systemd_unit_if_needed(system=True) is False
        assert systemd_unit.path.read_text(encoding="utf-8") == systemd_unit.original
        assert Path(gw.get_hermes_home()) == homes.scratch

    def test_sudo_stripped_home_still_refreshes_the_system_unit(self, systemd_unit, homes, marker_free_unit,
                                                                monkeypatch):
        # Under sudo there is no explicit HERMES_HOME: the unit's own pinned home names the install.
        monkeypatch.delenv("HERMES_HOME")
        monkeypatch.setattr(gw, "_read_systemd_user_from_unit", lambda path: None)
        assert gw.refresh_systemd_unit_if_needed(system=True) is True


class TestLaunchdWriters:
    @pytest.fixture
    def plist(self, tmp_path, homes, monkeypatch):
        path = tmp_path / "LaunchAgents" / "ai.hermes.gateway-coder.plist"
        path.parent.mkdir()
        data = {"Label": "ai.hermes.gateway-coder", "EnvironmentVariables": {"HERMES_HOME": str(homes.real)}}
        path.write_bytes(plistlib.dumps(data))
        monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: path)
        monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway-coder")
        monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
        monkeypatch.setattr(gw, "generate_launchd_plist", lambda: "<plist>new</plist>")
        monkeypatch.setattr(gw, "_prepare_service_launcher", lambda *a, **k: None)
        monkeypatch.setattr(gw, "_launchctl_label_supervising_process", lambda label: None)
        monkeypatch.setattr(gateway_launchd.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
        return SimpleNamespace(path=path, original=path.read_bytes())

    def test_refresh_leaves_a_plist_pinned_to_another_home_untouched(self, plist):
        assert gateway_launchd.refresh_launchd_plist_if_needed() is False
        assert plist.path.read_bytes() == plist.original

    def test_force_install_does_not_overwrite_another_homes_plist(self, plist):
        with pytest.raises(SystemExit):
            gateway_launchd.launchd_install(force=True, start_now=False)
        assert plist.path.read_bytes() == plist.original

    def test_force_unit_path_repoints_the_plist(self, plist):
        gateway_launchd.launchd_install(start_now=False, force_unit_path=True)
        assert plist.path.read_text(encoding="utf-8") == "<plist>new</plist>"


class TestInstallAdmission:
    """`gateway install` registers a persistent host service; a scratch/E2E home must not get one."""

    @pytest.fixture
    def install_cli(self, homes, tmp_path, monkeypatch):
        installs = []
        monkeypatch.setattr(gw, "is_managed", lambda: False)
        monkeypatch.setattr(gw, "_service_mgmt_blocked", lambda: False)
        monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda force: None)
        monkeypatch.setattr(gw, "_service_backend", lambda: "launchd")
        monkeypatch.setattr(gw, "is_macos", lambda: True)
        monkeypatch.setattr(gw, "_native_service_homes", lambda: {(tmp_path / "account" / ".hermes").resolve()})
        monkeypatch.setattr(gw, "_bare_unit_pinned_home", lambda: None)
        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: tmp_path / "none.service")
        monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: tmp_path / "none.plist")
        monkeypatch.setattr(gw, "launchd_install", lambda force, start_now, force_unit_path=False:
                            installs.append(force_unit_path))
        return installs

    @staticmethod
    def _args(**kw):
        return SimpleNamespace(if_missing=False, force=False, system=False, run_as_user=None, **kw)

    def test_scratch_home_is_refused(self, install_cli):
        with pytest.raises(SystemExit) as exc:
            gw._cmd_install(self._args(force_unit_path=False))
        assert exc.value.code == 1
        assert install_cli == []

    def test_force_unit_path_admits_a_custom_home(self, install_cli):
        gw._cmd_install(self._args(force_unit_path=True))
        assert install_cli == [True]

    def test_account_profile_is_admitted(self, install_cli, homes, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(homes.real))
        gw._cmd_install(self._args(force_unit_path=False))
        assert install_cli == [False]

    def test_custom_home_with_its_own_installed_service_is_admitted(self, install_cli, homes, tmp_path,
                                                                    monkeypatch):
        own = tmp_path / "own.plist"
        own.write_bytes(plistlib.dumps({"EnvironmentVariables": {"HERMES_HOME": str(homes.scratch)}}))
        monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: own)
        gw._cmd_install(self._args(force_unit_path=False))
        assert install_cli == [False]

    def test_parser_accepts_force_unit_path(self):
        parser = argparse.ArgumentParser()
        build_gateway_parser(parser.add_subparsers(dest="command"), cmd_gateway=lambda a: None,
                             cmd_proxy=lambda a: None, cmd_gateway_enroll=lambda a: None)
        assert parser.parse_args(["gateway", "install", "--force-unit-path"]).force_unit_path is True


def test_pinned_home_reads_both_definition_kinds(tmp_path):
    unit = tmp_path / "g.service"
    unit.write_text('[Service]\nEnvironment="HERMES_HOME=/srv/a"\n', encoding="utf-8")
    plist = tmp_path / "g.plist"
    plist.write_bytes(plistlib.dumps({"EnvironmentVariables": {"HERMES_HOME": "/srv/b"}}))
    broken = tmp_path / "broken.plist"
    broken.write_text("not a plist", encoding="utf-8")
    assert gateway_service_owner.pinned_home(unit) == "/srv/a"
    assert gateway_service_owner.pinned_home(plist) == "/srv/b"
    assert gateway_service_owner.pinned_home(broken) is None
