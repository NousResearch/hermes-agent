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


def _fake_multiplexer(monkeypatch, tmp_path, *, multiplex: bool, pid_file: bool = True):
    """A live default gateway at ``tmp_path`` whose runtime record names this process; the process passes
    the identity check because its command line reads as a gateway's. ``pid_file=False`` models a
    launch-service gateway whose ``gateway.pid`` was unlinked while it kept serving."""
    import json

    import hermes_constants
    import gateway.status as status

    (tmp_path / "profiles" / "beta").mkdir(parents=True)
    # A profile dir needs an identity marker to be listed/served (bare dirs are side-effect shells).
    (tmp_path / "profiles" / "beta" / "config.yaml").write_text("{}\n")
    (tmp_path / "config.yaml").write_text(
        f"gateway:\n  multiplex_profiles: {'true' if multiplex else 'false'}\n"
    )
    if pid_file:
        (tmp_path / "gateway.pid").write_text(str(os.getpid()))
    (tmp_path / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(), "kind": "hermes-gateway", "gateway_state": "running",
        "start_time": status._get_process_start_time(os.getpid()), "hermes_home": str(tmp_path),
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "beta"))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(
        status, "_read_process_cmdline", lambda pid: "python -m hermes_cli.main gateway run --replace"
    )


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


def test_served_named_profile_reports_running_without_default_pid_file(monkeypatch, tmp_path):
    """A live multiplexer whose PID file is missing still serves the profile it ticks (#110166)."""
    from hermes_cli.profiles import list_profiles

    _fake_multiplexer(monkeypatch, tmp_path, multiplex=True, pid_file=False)

    beta = next(p for p in list_profiles() if p.name == "beta")
    assert beta.gateway_running is True
    assert _run_status().startswith("✓ Gateway is running via the default-profile multiplexer")


def _run_status_lines():
    from hermes_cli import gateway as gw

    buf = io.StringIO()
    with redirect_stdout(buf):
        gw._gateway_command_inner(
            SimpleNamespace(gateway_command="status", deep=False, full=False, system=False)
        )
    return buf.getvalue().splitlines()


def _fake_hosted_wrapper_home(monkeypatch, tmp_path, *, updated_at):
    """A default home whose runtime record is written by THIS (non-``gateway run``) process:
    the #116416 in-process deployment shape. No PID file, no lock — only the fresh record."""
    import json

    import hermes_constants
    import gateway.status as status

    (tmp_path / "config.yaml").write_text("gateway:\n  multiplex_profiles: false\n")
    (tmp_path / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(), "kind": "hermes-gateway", "gateway_state": "running",
        "argv": ["hermes", "dashboard", "--host", "127.0.0.1", "--no-open"],
        "start_time": status._get_process_start_time(os.getpid()),
        "hermes_home": str(tmp_path),
        "updated_at": updated_at,
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    # The hosting process is NOT a `gateway run` — the strict ladder must stay down.
    monkeypatch.setattr(
        status, "_read_process_cmdline", lambda pid: "hermes dashboard --host 127.0.0.1 --no-open"
    )

    from hermes_cli import gateway as gw

    monkeypatch.setattr(
        gw, "get_gateway_runtime_snapshot",
        lambda system=False: SimpleNamespace(
            manager="manual process", running=False, gateway_pids=(),
            has_process_service_mismatch=False),
    )
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda probe=None: None)


def test_hosted_in_process_loop_reports_live(monkeypatch, tmp_path):
    """#116416: a fresh-heartbeat runtime record written by a live NON-``gateway run`` process
    (the loop embedded in a dashboard/wrapper) reports as live instead of "not running" —
    a false "down" nudges operators into starting a second gateway that fights the live one."""
    from datetime import datetime, timezone

    _fake_hosted_wrapper_home(
        monkeypatch, tmp_path, updated_at=datetime.now(timezone.utc).isoformat())

    lines = _run_status_lines()
    assert any(line.startswith("✓ Gateway messaging loop is live") for line in lines)
    assert not any(line.startswith("✗ Gateway is not running") for line in lines)


def test_stale_hosted_record_still_reports_stopped(monkeypatch, tmp_path):
    """The display-only trust is bounded by the heartbeat TTL: an orphaned record whose writer
    died keeps the honest "not running" verdict."""
    _fake_hosted_wrapper_home(
        monkeypatch, tmp_path, updated_at="2020-01-01T00:00:00+00:00")

    assert _run_status_lines()[0].startswith("✗ Gateway is not running")
