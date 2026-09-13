"""PR #69118: a named profile served by the default multiplexer reports as running.

``hermes gateway status`` / ``gateway list`` / ``profile list`` keyed liveness
off the profile's own gateway.pid, so a satellite profile served by the default
multiplexer showed "not running" even though the multiplexer was its live
inbound process. All three now consult the same
``named_profile_served_by_running_multiplexer()`` lookup the start guard and
cron liveness use.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest


def _fake_multiplexer(monkeypatch, tmp_path, *, multiplex: bool):
    import hermes_constants
    import gateway.status as status

    (tmp_path / "profiles" / "beta").mkdir(parents=True)
    (tmp_path / "config.yaml").write_text(
        f"gateway:\n  multiplex_profiles: {'true' if multiplex else 'false'}\n"
    )
    (tmp_path / "gateway.pid").write_text(str(os.getpid()))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "beta"))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)


def _run_status():
    from hermes_cli import gateway as gw

    buf = io.StringIO()
    with redirect_stdout(buf):
        gw._gateway_command_inner(
            SimpleNamespace(gateway_command="status", deep=False, full=False, system=False)
        )
    return buf.getvalue().splitlines()[0]


def test_served_named_profile_reports_running(monkeypatch, tmp_path):
    from hermes_cli.profiles import list_profiles

    _fake_multiplexer(monkeypatch, tmp_path, multiplex=True)

    beta = next(p for p in list_profiles() if p.name == "beta")
    assert beta.gateway_running is True
    assert _run_status().startswith("✓ Gateway is running via the default-profile multiplexer")


def test_unserved_named_profile_still_reports_stopped(monkeypatch, tmp_path):
    from hermes_cli.profiles import list_profiles

    _fake_multiplexer(monkeypatch, tmp_path, multiplex=False)

    beta = next(p for p in list_profiles() if p.name == "beta")
    assert beta.gateway_running is False
    assert _run_status().startswith("✗ Gateway is not running")


def _fake_launchd_multiplexer(
    monkeypatch, tmp_path, *, multiplex: bool = True, gateway_state: str = "running", pid_alive: bool = True
):
    """A launch-service-managed default gateway: live process + runtime status record, no gateway.pid.

    The PID file is absent (a replace/cleanup path unlinks it while the process keeps serving); the
    process is the live multiplexer the ``gateway_state.json`` record points at.
    """
    import json

    import hermes_constants
    import gateway.status as status

    (tmp_path / "profiles" / "beta").mkdir(parents=True)
    (tmp_path / "config.yaml").write_text(
        f"gateway:\n  multiplex_profiles: {'true' if multiplex else 'false'}\n"
    )
    (tmp_path / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(),
        "kind": "hermes-gateway",
        "gateway_state": gateway_state,
        # Same call the production PID-reuse guard makes, so the guard compares like with like.
        "start_time": status._get_process_start_time(os.getpid()),
        "argv": ["hermes", "gateway", "run", "--replace"],
        "hermes_home": str(tmp_path),
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "beta"))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: pid_alive)
    monkeypatch.setattr(
        status, "_read_process_cmdline", lambda pid: "python -m hermes_cli.main gateway run --replace"
    )


def test_served_named_profile_reports_running_without_default_pid_file(monkeypatch, tmp_path):
    """A live multiplexer whose PID file is missing still serves the profile it ticks."""
    from hermes_cli.profiles import list_profiles

    _fake_launchd_multiplexer(monkeypatch, tmp_path)

    beta = next(p for p in list_profiles() if p.name == "beta")
    assert beta.gateway_running is True
    assert _run_status().startswith("✓ Gateway is running via the default-profile multiplexer")


@pytest.mark.parametrize(
    ("gateway_state", "pid_alive"),
    [("stopped", True), ("startup_failed", True), ("running", False)],
)
def test_not_live_multiplexer_without_default_pid_file_reports_stopped(
    monkeypatch, tmp_path, gateway_state, pid_alive
):
    """Fails closed: a stopped/failed state or a dead PID must never be reported as running."""
    from hermes_cli.profiles import list_profiles

    _fake_launchd_multiplexer(monkeypatch, tmp_path, gateway_state=gateway_state, pid_alive=pid_alive)

    beta = next(p for p in list_profiles() if p.name == "beta")
    assert beta.gateway_running is False
    assert _run_status().startswith("✗ Gateway is not running")
