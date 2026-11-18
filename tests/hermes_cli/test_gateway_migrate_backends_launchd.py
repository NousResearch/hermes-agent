"""The backend migration with a hand-written launchd LaunchAgent as the old backend (macOS).

Same fake host as the dry-run tests (service manager, process table and network stand-ins; config.yaml,
.env and the old job's plist are real files read by the real code), with a launchd job instead of a
systemd unit and the gateway as a launchd agent. macOS is planned automatically (exit 0), not exit 6.
"""

from __future__ import annotations

import os
import plistlib

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli.gateway_migrate_backends import BackendItem
from hermes_cli.gateway_migrate_backends_plan import manual_steps
from tests.hermes_cli.test_gateway_migrate_backends_apply import _BACKEND_PID, _PASSWORD, FakeHost, _args, _files

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


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes-home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    (h / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    (h / ".env").write_text("OPENAI_API_KEY=***\n", encoding="utf-8")
    os.chmod(h / ".env", 0o600)
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
    assert _files(home) == before and not (home / "backend_migration.json").exists()


def test_the_dry_run_plans_the_launchd_agent_and_never_lingering(home, plist, capsys):
    host = LaunchdHost(home, plist)

    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_OK

    out = capsys.readouterr().out
    assert "install the launchd agent (hermes -p default gateway install; it starts at login)" in out
    assert "inger" not in out
