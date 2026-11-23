"""What the move stops or replaces besides the old backend: a gateway of this home running outside its service
(a chat window's own, or one started by hand) only with consent (``--force-interrupt``, the dashboard's Move
confirmation), and a pre-rename ``hermes.service``, replaced with its environment kept and put back by a
rollback or ``--restore``."""

from __future__ import annotations

import json
import os
import stat

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli import gateway_migrate_backends_host as host_mod
from hermes_cli.gateway_migrate_backends_host import SystemHost
from tests.hermes_cli.test_gateway_migrate_backends_apply import (
    FakeHost,
    _args,
    _config_files,
    _nothing_applied,
    _plan,
)

_STRAY = 6041


def _stray_host(home, **stray):
    host = FakeHost(home)
    host.stray_pid = _STRAY
    host.stray = {_STRAY: {"idle_exit": True, "chats": 0, **stray}}
    return host


def test_a_gateway_outside_its_service_is_a_consent_blocker_named_in_takes_over(home, capsys):
    code, out = _plan(_stray_host(home), capsys)

    plan = out["plan"]
    (blocker,) = plan["blockers"]
    assert code == engine.EXIT_BLOCKED and f"pid {_STRAY}, started by a chat window" in blocker
    assert plan["forceable_blockers"] == [blocker]
    assert any(f"pid {_STRAY}" in t for t in plan["takes_over"])


def test_with_consent_the_gateway_outside_its_service_is_stopped_after_the_old_backend(home, capsys):
    host = _stray_host(home, idle_exit=False, chats=2)

    code, out = _plan(host, capsys, force_interrupt=True)
    assert code == engine.EXIT_OK and out["plan"]["blockers"] == []
    assert "2 chat(s) running on it" in " ".join(out["plan"]["takes_over"])

    assert engine.cmd_apply(host, _args(force_interrupt=True)) == engine.EXIT_OK

    assert host.calls.index("stop_stray_gateway") > host.calls.index("old_stop")
    assert host.calls.index("stop_stray_gateway") < host.calls.index("gateway_start")
    assert host.stray_pid is None and engine.status_payload(host)["state"] == "done"


def test_without_consent_the_apply_stops_nothing(home, capsys):
    host = _stray_host(home)
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    _nothing_applied(host, home, before)
    assert host.stray_pid == _STRAY


def test_a_gateway_started_after_the_check_holds_the_stop_without_consent(home, capsys):
    host = FakeHost(home)
    real = host.install_gateway

    def install_then_chat():  # a chat window starts its own gateway while the move runs
        real()
        host.stray_pid = _STRAY
    host.install_gateway = install_then_chat

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert "old_stop" not in host.calls and host.old == {"enabled": True, "active": True}
    assert f"started outside its service after the check (pid {_STRAY})" in engine.status_payload(host)["message"]


def _legacy_host(home, tmp_path, **unit):
    path = tmp_path / "units" / "hermes.service"
    path.parent.mkdir()
    path.write_text("[Service]\nExecStart=/venv/bin/python -m hermes_cli.main gateway run\n", encoding="utf-8")
    host = FakeHost(home)
    host.legacy_records = [{"unit": "hermes.service", "path": str(path), "system": False, "enabled": "enabled",
                            "active": True, "main_pid": 7070, **unit}]
    host.legacy_env = ['Environment="TELEGRAM_HOME_CHANNEL=42"', "EnvironmentFile=/home/u/bot.env"]
    host.unit_environment = lambda unit, scope: ({}, None)  # it pins no HERMES_HOME
    host.default_home = lambda env: True  # ... and its HOME's default home is this one
    return host, path


def test_a_legacy_unit_is_replaced_and_its_environment_kept(home, tmp_path, capsys):
    host, path = _legacy_host(home, tmp_path)

    code, out = _plan(host, capsys)
    assert code == engine.EXIT_OK and out["plan"]["blockers"] == []
    assert any("Replaces the old hermes.service" in t for t in out["plan"]["takes_over"])

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    assert host.calls.index("remove_legacy_units") < host.calls.index("install_gateway")
    m = json.loads(engine.manifest_path(home).read_text(encoding="utf-8"))
    dropin = m["gateway_env_dropin"]
    assert open(dropin, encoding="utf-8").read().splitlines()[1:] == host.legacy_env
    (rec,) = m["legacy_units"]
    assert open(rec["backup"], encoding="utf-8").read() == path.read_text(encoding="utf-8")
    assert stat.S_IMODE(os.stat(rec["backup"]).st_mode) == 0o600


def test_restore_puts_the_legacy_unit_back_and_removes_the_kept_environment(home, tmp_path, capsys):
    host, _ = _legacy_host(home, tmp_path)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    dropin = json.loads(engine.manifest_path(home).read_text(encoding="utf-8"))["gateway_env_dropin"]

    assert engine.restore(host, yes=True) == engine.EXIT_OK

    assert "remove_gateway_env_dropin" in host.calls and "restore_legacy_unit:hermes.service" in host.calls
    assert host.calls.index("restore_legacy_unit:hermes.service") > host.calls.index("uninstall_gateway")
    assert not os.path.exists(dropin)
    assert [u["unit"] for u in host.legacy_records] == ["hermes.service"]


def test_a_failed_move_puts_the_legacy_unit_back(home, tmp_path):
    host, _ = _legacy_host(home, tmp_path)
    host.gateway_lies = True  # the gateway never answers on the old address

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert "restore_legacy_unit:hermes.service" in host.calls
    assert [u["unit"] for u in host.legacy_records] == ["hermes.service"]


def test_the_legacy_units_own_gateway_is_replaced_not_reported_as_outside_its_service(home, tmp_path, capsys):
    host, _ = _legacy_host(home, tmp_path)
    host.stray_pid = 7070  # the legacy unit's MainPID

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_OK and out["plan"]["blockers"] == [] and out["plan"]["forceable_blockers"] == []


@pytest.mark.parametrize(("unit", "env", "expected"), [
    ({"system": True}, ({}, None), "legacy system-wide gateway unit"),
    ({}, ({"HERMES_HOME": "/srv/other"}, None), "it serves another Hermes home (/srv/other)"),
    ({}, ({}, "an EnvironmentFile cannot be read"), "which Hermes home it serves cannot be told"),
])
def test_a_legacy_unit_it_cannot_replace_blocks(home, tmp_path, capsys, unit, env, expected):
    host, _ = _legacy_host(home, tmp_path, **unit)
    host.unit_environment = lambda u, scope: env
    host.same_home = lambda value: False
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    assert expected in capsys.readouterr().out
    _nothing_applied(host, home, before)


def test_a_run_inside_the_legacy_unit_is_blocked_and_pointed_at_detach(home, tmp_path, capsys):
    host, _ = _legacy_host(home, tmp_path)
    host.own_unit = lambda: "hermes.service"  # an agent's terminal under the legacy gateway
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    out = capsys.readouterr().out
    assert "runs inside hermes.service, which the move stops and replaces" in out and "--detach" in out
    _nothing_applied(host, home, before)


# What systemd 255 prints for a unit whose drop-in reset Environment= and set more after it, with a continued
# line, quoting, an escaped quote, a C escape, a literal % and an optional EnvironmentFile and a required one
# whose path holds a % (systemd prints it expanded).
_SHOW = ('Environment=HOME=/home/u "A=one two" B=cont "Q=say \\"hi\\"" P=100% "T=tab\\there" PATH=/usr/bin TOKEN=x\n'
         "EnvironmentFiles=/home/u/bot.env (ignore_errors=yes)\n"
         "EnvironmentFiles=/home/u/100%/must.env (ignore_errors=no)\n")


def test_the_legacy_environment_is_what_systemd_resolved_carried_verbatim(home, monkeypatch):
    calls = []

    def systemctl(scope, args, **kw):
        calls.append((scope, args))
        return type("R", (), {"returncode": 0, "stdout": _SHOW, "stderr": ""})()
    monkeypatch.setattr(host_mod, "_systemctl", systemctl)

    lines = SystemHost(home).legacy_environment_lines("hermes.service")

    assert calls == [("user", ["show", "hermes.service", "--no-pager", "--property=Environment",
                               "--property=EnvironmentFiles"])]
    assert lines == ['Environment="A=one two" B=cont "Q=say \\"hi\\"" P=100%% "T=tab\\there" TOKEN=x',
                     "EnvironmentFile=-/home/u/bot.env", "EnvironmentFile=/home/u/100%%/must.env"]


def test_a_legacy_unit_whose_environment_was_reset_carries_none(home, monkeypatch):
    monkeypatch.setattr(host_mod, "_systemctl", lambda scope, args, **kw: type(
        "R", (), {"returncode": 0, "stdout": "Environment=\n", "stderr": ""})())

    assert SystemHost(home).legacy_environment_lines("hermes.service") == []
