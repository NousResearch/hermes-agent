"""launchd query failures must not turn unknown supervision into a stopped/manual verdict.

Adapted from the regression coverage in #73069 for the facade/sibling layout.
"""
from types import SimpleNamespace
import subprocess

import pytest

from hermes_cli import gateway as gateway_cli
from hermes_cli import status as status_cli
from hermes_cli.dump import _gateway_status
from hermes_cli.service_manager import LaunchdServiceManager

pytestmark = pytest.mark.platforms("macos")

LABEL = "ai.hermes.gateway-example"
PID = 24680


@pytest.fixture
def status_environment(tmp_path, monkeypatch):
    plist = tmp_path / f"{LABEL}.plist"
    plist.write_text("<plist/>", encoding="utf-8")
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: LABEL)
    monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: True)
    monkeypatch.setattr(gateway_cli, "_launchd_unsupported_marker_exists", lambda: False)
    monkeypatch.setattr(gateway_cli, "find_gateway_pids", lambda: [PID])
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **kw: PID)
    monkeypatch.setattr(gateway_cli, "_installed_service_kind_for", lambda probe: "launchd")
    monkeypatch.setattr(gateway_cli, "named_profile_served_by_running_multiplexer", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway_profile_lifecycle.print_parked_status", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway_status_notes.report_incompatible_owner", lambda pids: False)
    monkeypatch.setattr("hermes_cli.gateway_status_notes.print_standalone_by_config", lambda: None)
    monkeypatch.setattr("hermes_cli.gateway_multiplex_served.multiplexer_served_secondaries", list)
    for name in ("_print_multiplex_standalone_reason", "_print_served_ingress_urls",
                 "_print_unserved_shared_ingress", "_print_duplicate_credential_warnings",
                 "_print_other_profiles_gateway_status"):
        monkeypatch.setattr(gateway_cli, name, lambda *a, **kw: None)

    def set_query(outcome):
        def run(command, **kwargs):
            # A status query must never start, stop, or rewrite a real service.
            assert command == ["launchctl", "list", LABEL]
            if isinstance(outcome, Exception):
                raise outcome
            return subprocess.CompletedProcess(command, *outcome)
        monkeypatch.setattr(gateway_cli.subprocess, "run", run)

    return set_query


@pytest.mark.parametrize("outcome", [
    (1, "", "Operation not permitted"),
    (5, "", "Input/output error"),
    (125, "", "Domain does not support specified action"),
    subprocess.TimeoutExpired(["launchctl", "list", LABEL], 10),
    PermissionError("launchctl is unavailable"),
    FileNotFoundError("launchctl is missing"),
], ids=["denied", "io-error", "unsupported-domain", "timeout", "permission-error", "missing-executable"])
@pytest.mark.parametrize("process_visible", [True, False], ids=["live-pid", "no-visible-pid"])
def test_unknown_launchd_status_survives_snapshot_and_renderers(
    status_environment, monkeypatch, capsys, outcome, process_visible,
):
    status_environment(outcome)
    if not process_visible:
        monkeypatch.setattr(gateway_cli, "find_gateway_pids", list)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda **kw: None)

    assert gateway_cli._probe_launchd_service_running() is None
    snapshot = gateway_cli.get_gateway_runtime_snapshot()
    assert snapshot.service_status_known is False
    assert snapshot.running is process_visible
    assert not snapshot.has_process_service_mismatch
    assert LaunchdServiceManager().is_running("gateway") is False
    diagnostic = _gateway_status()
    assert "unknown" in diagnostic
    assert "stopped" not in diagnostic
    assert "manual" not in diagnostic

    gateway_cli._cmd_status(SimpleNamespace(deep=False, full=False, system=False))
    detailed = capsys.readouterr().out
    assert "Unable to verify launchd service state" in detailed
    assert "not loaded" not in detailed
    assert "detached" not in detailed
    assert "manual foreground" not in detailed
    if process_visible:
        assert str(PID) in detailed

    status_cli._render_gateway(None)
    summary = capsys.readouterr().out
    assert "supervision could not be verified" in summary
    assert "stopped" not in summary
    assert "not managing" not in summary
    if process_visible:
        assert str(PID) in summary
        assert "running" in summary
    else:
        assert "unknown" in summary


@pytest.mark.parametrize("outcome,running", [
    ((0, f'{{\n    "PID" = {PID};\n}}', ""), True),
    ((0, '{ "Label" = "example"; }', ""), False),
    ((113, "", "Could not find service"), False),
    ((3, "", "No such process"), False),
    ((1, "", "Could not find service"), False),
], ids=["supervised", "registered-idle", "missing-113", "missing-3", "missing-message"])
def test_conclusive_launchd_status_remains_known(status_environment, outcome, running, capsys):
    status_environment(outcome)
    assert gateway_cli._probe_launchd_service_running() is running
    snapshot = gateway_cli.get_gateway_runtime_snapshot()
    assert snapshot.service_status_known is True
    assert snapshot.service_running is running
    assert LaunchdServiceManager().is_running("gateway") is running
    assert snapshot.has_process_service_mismatch is (not running)

    gateway_cli.launchd_status()
    output = capsys.readouterr().out
    assert "Unable to verify" not in output
    if running:
        assert "supervised by launchd" in output
    elif outcome[0] != 0:
        assert "not loaded" in output
