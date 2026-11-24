"""The printed by-hand steps match what ``hermes gateway status`` and ``hermes webapp --status`` print.

The branch step reads "# then by what it prints: 'A' or 'B' -> install; ... -> start; ... -> restart".
Each case runs the real status function with controlled inputs on the host it belongs to (systemd on
Linux, launchd on macOS, the Windows service on Windows), captures its output, and checks that the
first group of the step whose quoted line appears there names the right command. The functions are the
systemd and launchd service status, the no-service and by-hand paths of ``hermes gateway status``, and
the Windows service status. The closing ``webapp --status`` runs while a just-started gateway may still
be binding, so the step before it quotes what ``webapp --status`` prints then (captured the same way)
and says to wait and run it again.
"""

from __future__ import annotations

import contextlib
import re
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_migrate_backends_plan as plan
from hermes_cli import gateway_migrate_backends_windows as win


@pytest.fixture(autouse=True)
def _standard_root(monkeypatch):
    """The homes here stand for the user's standard Hermes root, so steps say `hermes -p NAME`; the root
    pin for any other root is tested in test_gateway_migrate_backends_root_pin.py."""
    monkeypatch.setattr("hermes_cli.gateway_migrate_backends_plan._is_standard_root", lambda root: True)

_H = "hermes -p work"


def _groups(steps: list[str]) -> list[tuple[list[str], str]]:
    """``[(quoted lines, command)]`` of the branch step, in the order a reader checks them."""
    (line,) = [s for s in steps if s.startswith("# then by what it prints")]
    groups = []
    # Each alternative starts with a quoted status line, so "; '" separates them even when a command pins
    # HERMES_HOME for PowerShell ("$env:HERMES_HOME='…'; hermes -p NAME …").
    for part in re.split(r";\s*(?=')", line.split(":", 1)[1]):
        quoted, _, command = part.partition(" -> ")
        groups.append((re.findall(r"'([^']+)'", quoted), command.strip()))
    return groups


def _command_for(steps: list[str], printed: str) -> str | None:
    return next((command for quoted, command in _groups(steps) if any(q in printed for q in quoted)), None)


def _system_unit_steps():
    item = SimpleNamespace(supervisor="systemd", scope="system", unit="hermes-dashboard.service", pid=None)
    return plan.manual_steps("127.0.0.1", 9119, item, env_keys=(), home="/home/u/.hermes/profiles/work")


def _windows_steps():
    item = SimpleNamespace(supervisor="scheduled-task", unit="Hermes Dashboard", unit_path=None, pid=11)
    return win.manual_steps("127.0.0.1", 9119, item, home="/home/u/.hermes/profiles/work")


def _printed(capsys, call) -> str:
    # The status line comes first; probes after it (health, logs) may not run in a test and are not checked.
    with contextlib.suppress(Exception):
        call()
    return capsys.readouterr().out


def _systemd(monkeypatch, tmp_path, active: bool):
    from hermes_cli import gateway as gw
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text("[Service]\n", encoding="utf-8")
    monkeypatch.setattr(gw, "_select_systemd_scope", lambda system=False: system)
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit)
    monkeypatch.setattr(gw, "has_conflicting_systemd_units", lambda: False)
    monkeypatch.setattr(gw, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(gw, "systemd_unit_is_current", lambda system=False: True)
    monkeypatch.setattr("hermes_cli.gateway_status_notes.print_scrubbed_output", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_run_systemctl", lambda args, **k: SimpleNamespace(
        returncode=0, stdout="active" if active else "inactive", stderr=""))
    return gw.systemd_status


def _launchd(monkeypatch, state: str):
    from hermes_cli import gateway_launchd as ld
    listed = {"not-loaded": (1, ""), "registered": (0, '{\n\t"Label" = "ai.hermes.gateway";\n};'),
              "supervised": (0, '{\n\t"PID" = 4321;\n\t"Label" = "ai.hermes.gateway";\n};')}
    code, out = listed[state]
    fake = SimpleNamespace(get_launchd_plist_path=lambda: "/Users/u/Library/LaunchAgents/ai.hermes.gateway.plist",
                           get_launchd_label=lambda: "ai.hermes.gateway", _CAPTURE_TEXT={},
                           _parse_launchd_pid_from_list_output=lambda text: 4321 if '"PID"' in text else None,
                           _launchd_unsupported_marker_exists=lambda: False, launchd_plist_is_current=lambda: True)
    monkeypatch.setattr(ld, "_gw", lambda: fake)
    monkeypatch.setattr(ld.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=code, stdout=out, stderr=""))
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **k: None)
    return ld.launchd_status


def _no_service(monkeypatch, pids: list[int]):
    from hermes_cli import gateway as gw
    snapshot = SimpleNamespace(running=bool(pids), gateway_pids=pids, service_running=False, manager=None)
    monkeypatch.setattr(gw, "get_gateway_runtime_snapshot", lambda system=False: snapshot)
    monkeypatch.setattr("hermes_cli.gateway_profile_lifecycle.print_parked_status", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway_status_notes.report_incompatible_owner", lambda pids: False)
    monkeypatch.setattr(gw, "named_profile_served_by_running_multiplexer", lambda: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda probe: None)
    monkeypatch.setattr(gw, "_print_runtime_health", lambda *a, **k: None)
    return lambda: gw._cmd_status(SimpleNamespace(deep=False, full=False, system=False))


def _windows(monkeypatch, tmp_path, installed: str | None, pids: list[int]):
    """The whole of `hermes gateway status` on Windows: the service ladder in ``_cmd_status`` picks
    ``gateway_windows.status()`` only when a Scheduled Task or Startup entry exists. Only the machine's
    state is set here (task, Startup entry, gateway pids); the host checks are the real ones."""
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_windows as gwin
    entry = tmp_path / "Hermes Gateway.lnk"
    entry.write_text("", encoding="utf-8")
    for name in ("_print_start_attestation_warning", "_print_scheduled_task_drift"):
        monkeypatch.setattr(gwin, name, lambda *a, **k: None)
    monkeypatch.setattr(gwin, "get_task_name", lambda: "Hermes Gateway")
    monkeypatch.setattr(gwin, "is_task_registered", lambda: installed == "task")
    monkeypatch.setattr(gwin, "is_startup_entry_installed", lambda: installed == "startup")
    monkeypatch.setattr(gwin, "get_startup_entry_path", lambda: entry)
    monkeypatch.setattr(gwin, "_gateway_pids", lambda: list(pids))
    monkeypatch.setattr(gwin, "query_task_status", dict)
    monkeypatch.setattr("hermes_cli.gateway_windows_legacy.warn_legacy_launchers", lambda: None)
    snapshot = SimpleNamespace(running=bool(pids), gateway_pids=pids, service_running=False, manager=None)
    monkeypatch.setattr(gw, "get_gateway_runtime_snapshot", lambda system=False: snapshot)
    monkeypatch.setattr("hermes_cli.gateway_profile_lifecycle.print_parked_status", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway_status_notes.report_incompatible_owner", lambda pids: False)
    monkeypatch.setattr(gw, "named_profile_served_by_running_multiplexer", lambda: False)
    monkeypatch.setattr(gw, "_print_runtime_health", lambda *a, **k: None)
    return lambda: gw._cmd_status(SimpleNamespace(deep=False, full=False, system=False))


def _assert_posix_branch(capsys, call, verb: str) -> None:
    printed = _printed(capsys, call)
    steps = _system_unit_steps()
    assert "not installed ->" not in "\n".join(steps)  # `hermes gateway status` never prints that there
    assert _command_for(steps, printed) == f"{_H} gateway {verb}", printed


@pytest.mark.platforms("linux")
@pytest.mark.parametrize(("active", "verb"), [(False, "start"), (True, "restart")])
def test_the_posix_branch_names_the_right_command_for_what_systemd_status_prints(active, verb, monkeypatch,
                                                                                   tmp_path, capsys):
    _assert_posix_branch(capsys, _systemd(monkeypatch, tmp_path, active), verb)


@pytest.mark.platforms("macos")
@pytest.mark.parametrize(("state", "verb"), [("not-loaded", "start"), ("registered", "start"),
                                              ("supervised", "restart")])
def test_the_posix_branch_names_the_right_command_for_what_launchd_status_prints(state, verb, monkeypatch, capsys):
    _assert_posix_branch(capsys, _launchd(monkeypatch, state), verb)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("pids", "verb"), [([], "install"), ([4321], "restart")])  # no service; run by hand
def test_the_posix_branch_names_the_right_command_without_a_service(pids, verb, monkeypatch, capsys):
    _assert_posix_branch(capsys, _no_service(monkeypatch, pids), verb)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(("installed", "pids", "verb"), [
    (None, [], "install"),         # no service: "✗ Gateway is not running" (the VM run's P1)
    (None, [77], "restart"),       # run by hand: "(Running manually, not as a system service)"
    ("task", [], "start"), ("task", [77], "restart"),
    ("startup", [], "start"), ("startup", [77], "restart"),
])
def test_the_windows_branch_names_the_right_command_for_what_status_prints(installed, pids, verb, monkeypatch,
                                                                             tmp_path, capsys):
    printed = _printed(capsys, _windows(monkeypatch, tmp_path, installed, pids))

    assert "Gateway" in printed, printed  # the status ran: its first line is captured
    assert _command_for(_windows_steps(), printed) == f"{_H} gateway {verb}", printed


@pytest.mark.parametrize("steps", [_system_unit_steps, _windows_steps,
                                   lambda: plan.manual_steps("127.0.0.1", 9119, None, env_keys=(),
                                                             home="/home/u/.hermes/profiles/work", no_service=True)])
def test_the_closing_check_quotes_what_webapp_status_prints_while_the_gateway_starts(steps, monkeypatch,
                                                                                     tmp_path, capsys):
    from gateway import webapp_listener
    from hermes_cli import webapp_launcher

    def unreachable(*a, **k):
        raise webapp_launcher.GatewayUnreachable("not up yet")

    monkeypatch.setattr(webapp_launcher, "_control", unreachable)
    monkeypatch.setattr(webapp_listener, "webapp_settings", lambda *a, **k: webapp_listener.WebappSettings(
        enabled=True))
    assert webapp_launcher._status(tmp_path) == 1
    printed = capsys.readouterr().out

    got = steps()
    assert got[-1] == f"{_H} webapp --status"
    wait = got[-2]
    assert wait.startswith("# ") and "wait" in wait and "again" in wait
    quoted = re.findall(r"'([^']+)'", wait)
    assert quoted and all(q in printed for q in quoted), printed
