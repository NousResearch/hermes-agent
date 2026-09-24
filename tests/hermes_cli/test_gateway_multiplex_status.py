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


def test_standalone_profile_status_reports_standalone_by_config(monkeypatch, tmp_path):
    """`hermes -p X gateway status` on a standalone X says so and never claims the multiplexer."""
    _fake_multiplexer(monkeypatch, tmp_path, multiplex=True)
    (tmp_path / "profiles" / "beta" / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
    from hermes_cli import gateway as gw

    buf = io.StringIO()
    with redirect_stdout(buf):
        gw._gateway_command_inner(
            SimpleNamespace(gateway_command="status", deep=False, full=False, system=False)
        )
    out = buf.getvalue()
    assert "standalone by config (gateway.standalone: true)" in out
    assert "via the default-profile multiplexer" not in out


