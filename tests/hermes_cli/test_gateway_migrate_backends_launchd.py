"""The backend migration with a hand-written launchd LaunchAgent as the old backend (macOS).

Same fake host as the engine's safety-rule tests (service manager, process table and network stand-ins;
config.yaml, .env, the manifest and the old job's plist are real files written and read by the real code),
with a launchd job instead of a systemd unit and the gateway as a launchd agent. macOS is planned and
applied automatically (exit 0), not exit 6.
"""

from __future__ import annotations

import os
import plistlib

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli.gateway_migrate_backends import BackendItem
from hermes_cli.gateway_migrate_backends_plan import manual_steps
from tests.hermes_cli.test_gateway_migrate_backends_apply import _BACKEND_PID, _PASSWORD, FakeHost, _args, _files


@pytest.fixture(autouse=True)
def _standard_root(monkeypatch):
    """The homes here stand for the user's standard Hermes root, so steps say `hermes -p NAME`; the root
    pin for any other root is tested in test_gateway_migrate_backends_root_pin.py."""
    monkeypatch.setattr("hermes_cli.gateway_migrate_backends_plan._is_standard_root", lambda root: True)

_LABEL = "local.hermes-dashboard"
_DOMAIN = "gui/501"


class LaunchdHost(FakeHost):
    def __init__(self, home, plist, *, running=True):
        super().__init__(home, supervisor="launchd", scope=_DOMAIN)
        self.plist = plist
        self.old["active"] = running

    def backends(self):
        # The report lists a launchd job only while it is loaded; it has a pid only while it runs.
        if not self.old["enabled"]:
            return []
        return [BackendItem(supervisor="launchd", running=self.old["active"],
                            pid=_BACKEND_PID if self.old["active"] else None, argv=list(self.argv),
                            unit=_LABEL, scope=_DOMAIN, unit_path=str(self.plist), hermes_home=None,
                            this_home=True if self.old["active"] else None)]

    def default_home(self, env=None):
        return True

    def process_env(self, pid):
        return super().process_env(pid) if self.old["active"] else None

    def gateway_kind(self):
        return "launchd"

    def gateway_state(self):
        return {**super().gateway_state(), "kind": "launchd", "unit": "ai.hermes.gateway"}

    def linger(self):
        return None


def _env_keys(home):
    return {line.split("=", 1)[0] for line in (home / ".env").read_text(encoding="utf-8").splitlines() if "=" in line}


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes-home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    (h / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    (h / ".env").write_text("OPENAI_API_KEY=***\n", encoding="utf-8")
    os.chmod(h / ".env", 0o600)
    monkeypatch.setattr(engine, "VERIFY_SECONDS", 0.0)
    monkeypatch.setattr(engine, "ROLLBACK_VERIFY_SECONDS", 0.0)
    return h


@pytest.fixture
def plist(tmp_path):
    path = tmp_path / "LaunchAgents" / f"{_LABEL}.plist"
    path.parent.mkdir()
    path.write_bytes(plistlib.dumps({
        "Label": _LABEL, "RunAtLoad": True, "KeepAlive": True,
        "ProgramArguments": ["/Users/u/.local/bin/hermes", "dashboard", "--no-open", "--host", "0.0.0.0"],
        "EnvironmentVariables": {"HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "spike",
                                 "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": _PASSWORD}}))
    return path


def test_printed_steps_keep_the_job_from_loading_at_the_next_login(plist):
    item = BackendItem(supervisor="launchd", running=True, unit=_LABEL, scope=_DOMAIN, unit_path=str(plist))

    steps = manual_steps("0.0.0.0", 9119, item)

    assert f"launchctl bootout {_DOMAIN}/{_LABEL}; launchctl disable {_DOMAIN}/{_LABEL}" in steps


def test_the_dry_run_names_the_plist_credentials_of_a_loaded_job_that_is_not_running(home, plist, capsys):
    host = LaunchdHost(home, plist, running=False)
    before = _files(home)

    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_OK

    out = capsys.readouterr().out
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in out and _PASSWORD not in out
    assert host.calls == [] and _files(home) == before and not (home / "backend_migration.json").exists()


def test_the_dry_run_plans_the_launchd_agent_and_never_lingering(home, plist, capsys):
    host = LaunchdHost(home, plist)

    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_OK

    out = capsys.readouterr().out
    assert "install the launchd agent (hermes -p default gateway install; it starts at login)" in out
    assert "inger" not in out


def test_a_loaded_job_that_is_not_running_still_carries_its_plist_credentials(home, plist, capsys):
    host = LaunchdHost(home, plist, running=False)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    assert {"HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"} <= _env_keys(home)
    assert _PASSWORD not in capsys.readouterr().out


def test_rollback_loads_again_a_job_that_was_loaded_but_not_running(home, plist):
    host = LaunchdHost(home, plist, running=False)
    host.fail_at["gateway_start"] = RuntimeError("the gateway agent did not start")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert host.calls[-2:] == ["old_enable", "old_start"]
    assert engine.read_manifest(home)["old_backend"]["was_loaded"] is True


def test_the_apply_output_names_the_launchd_agent_and_never_lingering(home, plist, capsys):
    host = LaunchdHost(home, plist)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    out = capsys.readouterr().out
    assert "install the launchd agent (hermes -p default gateway install; it starts at login)" in out
    assert "launchd agent of this user: like the old backend, it starts when you log in" in out
    assert "inger" not in out


def test_the_manual_rollback_steps_enable_the_label_before_loading_it(home, plist):
    host = LaunchdHost(home, plist)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    steps = engine._manual_rollback_steps(engine.read_manifest(home))

    assert steps[-1] == (f"mv {plist}.disabled {plist}; launchctl enable {_DOMAIN}/{_LABEL}; "
                         f"launchctl bootstrap {_DOMAIN} {plist}")


def test_restore_enables_the_label_before_it_loads_the_job_again(home, plist):
    host = LaunchdHost(home, plist)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    assert host.old == {"enabled": False, "active": False}

    assert engine.restore(host, yes=True) == engine.EXIT_OK

    assert host.calls[-2:] == ["old_enable", "old_start"] and host.old == {"enabled": True, "active": True}
    assert engine.status_payload(host)["state"] == "restored"


class _TwoJobsHost(LaunchdHost):
    """The moved job, plus a second job for this home that launchd has loaded but is not running now."""

    def backends(self):
        idle = BackendItem(supervisor="launchd", running=False, argv=["/Users/u/.local/bin/hermes", "serve"],
                           unit="local.hermes-serve", scope=_DOMAIN, unit_path="/Users/u/Library/x.plist",
                           hermes_home=None, this_home=None)
        return [*super().backends(), idle]


def test_a_loaded_idle_launch_agent_for_this_home_is_a_second_owner(home, plist, capsys):
    host = _TwoJobsHost(home, plist)

    assert engine.cmd_apply(host, _args(address="0.0.0.0:9119")) == engine.EXIT_BLOCKED

    assert host.calls == [] and "Another old backend also serves this home: launchd local.hermes-serve" in \
        capsys.readouterr().out

