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


_SYSTEM_LABEL = "com.example.hermes-dashboard"
_SYSTEM_PLIST = f"/Library/LaunchDaemons/{_SYSTEM_LABEL}.plist"


class SystemLaunchDaemonHost(LaunchdHost):
    """A macOS system-domain LaunchDaemon (/Library/LaunchDaemons, scope='system')."""

    def __init__(self, home, plist_path=_SYSTEM_PLIST, *, scope="system", running=True, unreadable_proc_env=False):
        super().__init__(home, plist_path, running=running)
        self.scope = scope
        self.unreadable_proc_env = unreadable_proc_env

    def backends(self):
        if not self.old["enabled"]:
            return []
        return [BackendItem(supervisor="launchd", running=self.old["active"],
                            pid=_BACKEND_PID if self.old["active"] else None, argv=list(self.argv),
                            unit=_SYSTEM_LABEL, scope=self.scope, unit_path=str(self.plist) if self.plist else None,
                            hermes_home=str(self.home), this_home=True)]

    def process_env(self, pid):
        return None if self.unreadable_proc_env else super().process_env(pid)


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_system_launchdaemon_is_exit_6_with_user_steps_and_one_sudo_line(home, capsys, dry_run):
    host = SystemLaunchDaemonHost(home)
    before = _files(home)

    code = engine.cmd_apply(host, _args(dry_run=dry_run))

    assert code == engine.EXIT_UNSUPPORTED and host.calls == []
    assert _files(home) == before and not (home / "backend_migration.json").exists()
    out = capsys.readouterr().out
    lines = [line.strip() for line in out.splitlines()]
    steps = lines[lines.index("Steps, in this order (one command per line):") + 1:]
    sudo = [line for line in steps if line.startswith("sudo ")]
    assert sudo == [f"sudo launchctl bootout system/{_SYSTEM_LABEL}; sudo launchctl disable system/{_SYSTEM_LABEL}"]
    before_stop = steps[:steps.index(sudo[0])]
    after_stop = steps[steps.index(sudo[0]) + 1:]
    assert not any(" gateway " in s for s in before_stop)
    assert after_stop[0] == "hermes -p default gateway status"
    assert after_stop[-1] == "hermes -p default webapp --status"
    assert not any("sudo hermes" in line or "sudo -E" in line for line in lines)
    assert any("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in s and _PASSWORD not in s for s in steps)
    assert _PASSWORD not in out


def test_a_system_launchdaemon_reads_plist_env_keys_when_root_process_env_is_unreadable(home, tmp_path, capsys):
    daemon_plist = tmp_path / "LaunchDaemons" / f"{_SYSTEM_LABEL}.plist"
    daemon_plist.parent.mkdir()
    daemon_plist.write_bytes(plistlib.dumps({
        "Label": _SYSTEM_LABEL, "RunAtLoad": True, "KeepAlive": True,
        "ProgramArguments": ["/Users/u/.local/bin/hermes", "dashboard", "--no-open", "--host", "0.0.0.0"],
        "EnvironmentVariables": {"HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "spike",
                                 "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": _PASSWORD}}))
    host = SystemLaunchDaemonHost(home, daemon_plist, unreadable_proc_env=True)

    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_UNSUPPORTED

    out = capsys.readouterr().out
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in out and _PASSWORD not in out
    assert f"sudo launchctl bootout system/{_SYSTEM_LABEL}; sudo launchctl disable system/{_SYSTEM_LABEL}" in out


def test_a_system_launchdaemon_detach_exits_6_without_spawning(home, capsys):
    host = SystemLaunchDaemonHost(home)

    assert engine.cmd_apply(host, _args(detach=True)) == engine.EXIT_UNSUPPORTED

    assert host.calls == [] and host.spawned == [] and not (home / "backend_migration.json").exists()
    assert f"sudo launchctl bootout system/{_SYSTEM_LABEL}; sudo launchctl disable system/{_SYSTEM_LABEL}" in \
        capsys.readouterr().out


@pytest.mark.parametrize(("scope", "unit_path"), [
    ("system", None),
    (None, f"/Library/LaunchDaemons/{_SYSTEM_LABEL}.plist"),
    ("gui/0", f"/System/Library/LaunchDaemons/{_SYSTEM_LABEL}.plist"),
])
def test_launchdaemon_detection_catches_system_scope_and_daemon_paths(home, capsys, scope, unit_path):
    host = SystemLaunchDaemonHost(home, unit_path, scope=scope)

    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_UNSUPPORTED
    assert host.calls == []
    domain = scope or "system"
    assert f"sudo launchctl bootout {domain}/{_SYSTEM_LABEL}; sudo launchctl disable {domain}/{_SYSTEM_LABEL}" in \
        capsys.readouterr().out


def test_user_domain_launchagent_still_migrates_automatically(home, plist):
    class UserDomainLaunchdHost(LaunchdHost):
        def __init__(self, h, p):
            super().__init__(h, p)
            self.scope = "user/501"

        def backends(self):
            return [BackendItem(**{**item.__dict__, "scope": "user/501"}) for item in super().backends()]

    host = UserDomainLaunchdHost(home, plist)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    steps = manual_steps("0.0.0.0", 9119, BackendItem(supervisor="launchd", running=True, unit=_LABEL,
                                                      scope="user/501", unit_path=str(plist)))
    assert f"launchctl bootout user/501/{_LABEL}; launchctl disable user/501/{_LABEL}" in steps
    assert not any("sudo " in s for s in steps)


def test_printed_steps_use_sudo_for_system_launchdaemon():
    item = BackendItem(supervisor="launchd", running=True, unit=_SYSTEM_LABEL, scope="system",
                       unit_path=_SYSTEM_PLIST)

    steps = manual_steps("0.0.0.0", 9119, item)

    assert f"sudo launchctl bootout system/{_SYSTEM_LABEL}; sudo launchctl disable system/{_SYSTEM_LABEL}" in steps


def _system_manifest(home, *, stopped, disabled, stop_state="failed", legacy=False):
    m = {
        "version": 1, "kind": "apply", "state": "failed", "step": "stop_old_backend",
        "home": str(home), "old_address": "0.0.0.0:9119", "new_address": "0.0.0.0:9119",
        "entered": ["manifest", "write_config", "gateway_service", "stop_old_backend"],
        "steps": [{"name": "stop_old_backend", "state": stop_state}],
        "backups": {"config": str(home / "backend_migration" / "config.yaml.bak"),
                    "env": str(home / "backend_migration" / "env.bak")},
        "gateway": {"installed": False, "enabled": False, "active": False, "kind": "launchd"},
        "old_backend": {"supervisor": "launchd", "unit": _SYSTEM_LABEL, "scope": "system",
                        "unit_path": _SYSTEM_PLIST, "was_active": True, "was_loaded": True},
    }
    if not legacy:
        m["old_backend_stopped"] = stopped
        m["old_backend_disabled"] = disabled
    return m


def test_manual_rollback_steps_use_sudo_for_system_launchdaemon_when_disabled(home):
    m = _system_manifest(home, stopped=True, disabled=True, stop_state="done")

    steps = engine._manual_rollback_steps(m)

    assert steps[-1] == (f"sudo mv {_SYSTEM_PLIST}.disabled {_SYSTEM_PLIST}; "
                         f"sudo launchctl enable system/{_SYSTEM_LABEL}; "
                         f"sudo launchctl bootstrap system {_SYSTEM_PLIST}")


@pytest.mark.parametrize("legacy", [False, True])
def test_manual_rollback_steps_for_system_launchdaemon_failed_at_stop_omit_mv_and_use_sudo(home, legacy):
    m = _system_manifest(home, stopped=False, disabled=False, stop_state="failed", legacy=legacy)

    steps = engine._manual_rollback_steps(m)

    assert steps[-1] == (f"sudo launchctl kickstart system/{_SYSTEM_LABEL} || "
                         f"sudo launchctl bootstrap system {_SYSTEM_PLIST}")
    assert ".disabled" not in steps[-1] and "launchctl enable" not in steps[-1]


def test_manual_rollback_steps_when_old_stop_fails_omit_mv_and_skip_old_enable(home, plist):
    host = LaunchdHost(home, plist)
    host.fail_at["old_stop"] = RuntimeError("Operation not permitted")
    host.fail_at["uninstall_gateway"] = RuntimeError("gateway uninstall blocked")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    assert "old_enable" not in host.calls
    steps = engine.status_payload(host)["manual_steps"]
    assert steps[-1] == f"launchctl kickstart {_DOMAIN}/{_LABEL} || launchctl bootstrap {_DOMAIN} {plist}"
    assert ".disabled" not in steps[-1] and "launchctl enable" not in steps[-1] and "sudo " not in steps[-1]


def test_manual_rollback_steps_when_old_disable_fails_before_rename_omit_mv(home, plist):
    host = LaunchdHost(home, plist)
    host.fail_at["old_disable"] = RuntimeError("disable failed")
    host.fail_at["old_enable"] = RuntimeError("enable failed")
    host.fail_at["old_start"] = RuntimeError("cannot bootstrap disabled job")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    steps = engine.status_payload(host)["manual_steps"]
    assert steps[-1] == f"launchctl enable {_DOMAIN}/{_LABEL}; launchctl bootstrap {_DOMAIN} {plist}"
    assert ".disabled" not in steps[-1]


def test_manual_rollback_steps_when_old_disable_fails_after_rename_keep_mv(home, plist):
    host = LaunchdHost(home, plist)

    def rename_then_fail(rec):
        host._change("old_disable")
        plist.rename(plist.with_name(plist.name + ".disabled"))
        raise RuntimeError("post-rename failure")

    host.old_disable = rename_then_fail
    host.fail_at["old_enable"] = RuntimeError("enable failed")
    host.fail_at["old_start"] = RuntimeError("plist missing")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    steps = engine.status_payload(host)["manual_steps"]
    assert steps[-1] == (f"mv {plist}.disabled {plist}; launchctl enable {_DOMAIN}/{_LABEL}; "
                         f"launchctl bootstrap {_DOMAIN} {plist}")


def test_manual_rollback_steps_when_old_enable_succeeded_and_old_start_failed_omit_mv_and_enable(home, plist):
    host = LaunchdHost(home, plist)
    host.fail_at["gateway_start"] = RuntimeError("gateway start failed")
    host.fail_at["old_start"] = RuntimeError("bootstrap failed")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    steps = engine.status_payload(host)["manual_steps"]
    assert steps[-1] == f"launchctl bootstrap {_DOMAIN} {plist}"
    assert ".disabled" not in steps[-1] and "launchctl enable" not in steps[-1]


def test_manual_rollback_steps_before_stop_old_backend_emit_no_launchctl_step(home, plist):
    host = LaunchdHost(home, plist)

    def partial_install():
        host._change("install_gateway")
        host.gw["installed"] = True
        raise RuntimeError("install failed partway")

    host.install_gateway = partial_install
    host.fail_at["uninstall_gateway"] = RuntimeError("uninstall failed")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    steps = engine.status_payload(host)["manual_steps"]
    assert not any("launchctl" in s or ".disabled" in s for s in steps)

