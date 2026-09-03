"""Multiplex secondary profiles must show as "running" in per-profile
liveness checks, not stuck on a stale/dead-PID gateway_state.json.

Regression coverage for two layered bugs:

1. ``_start_secondary_profile_adapters`` (gateway/run.py) never refreshed a
   SECONDARY profile's own ``gateway_state.json`` after startup — only the
   active/default profile's file got rewritten, so a secondary profile's
   file just rotted at whatever it last said (often a dead PID from before
   multiplexing, or before that profile was ever run standalone).

2. Even after refreshing it, ``_record_matches_live_gateway_pid`` (gateway/
   status.py) validates a *live* PID's command line against the profile's
   ``-p <name>``/``--profile <name>`` flag — which a multiplex gateway's
   single SHARED process never carries (it serves every profile from one
   bare command). Without the ``multiplex_secondary`` marker this check
   fails even for a freshly-written, genuinely-live record.
"""

from pathlib import Path

from gateway.status import (
    _record_matches_live_gateway_pid,
    get_runtime_status_running_pid,
    read_runtime_status,
    write_runtime_status,
)


def _write_and_read(tmp_path, monkeypatch, *, multiplex_secondary):
    """Write a runtime-status record scoped to tmp_path, as the gateway's
    secondary-profile refresh path would, then read it back."""
    write_runtime_status(
        gateway_state="running",
        multiplex_secondary=multiplex_secondary,
        path=tmp_path / "gateway_state.json",
    )
    return read_runtime_status(tmp_path / "gateway_state.json")


def test_write_runtime_status_stamps_multiplex_secondary_marker(tmp_path):
    record = _write_and_read(tmp_path, None, multiplex_secondary=True)
    assert record is not None
    assert record.get("multiplex_secondary") is True
    assert record.get("gateway_state") == "running"


def test_write_runtime_status_omits_marker_when_not_passed(tmp_path):
    write_runtime_status(
        gateway_state="running", path=tmp_path / "gateway_state.json"
    )
    record = read_runtime_status(tmp_path / "gateway_state.json")
    assert record is not None
    assert "multiplex_secondary" not in record


def test_multiplex_secondary_record_skips_per_profile_cmdline_check(
    tmp_path, monkeypatch
):
    """A multiplex-secondary record must match a live PID even when that
    PID's command line carries NO ``-p <profile>`` flag (the shared-process
    case) — this is exactly what a non-multiplex per-profile record would
    correctly REJECT (see the sibling assertion below)."""
    record = {"multiplex_secondary": True}
    monkeypatch.setattr(
        "gateway.status._read_process_cmdline",
        lambda pid: "/usr/bin/python -m hermes_cli.main gateway run",
    )
    monkeypatch.setattr(
        "gateway.status.looks_like_gateway_runtime_command_line",
        lambda cmdline: True,
    )
    assert _record_matches_live_gateway_pid(
        record, 12345, expected_home=Path("/home/hermes/.hermes/profiles/summer")
    )


def test_non_multiplex_record_still_requires_matching_profile_cmdline(
    tmp_path, monkeypatch
):
    """Sibling/contrast case: WITHOUT the marker, a shared-process command
    line (no ``-p <profile>``) must still be rejected for a named profile's
    expected_home — preserves the existing PID-reuse protection for the
    "one dedicated process per profile" deployment this check was built for.
    """
    record = {}  # no multiplex_secondary marker
    monkeypatch.setattr(
        "gateway.status._read_process_cmdline",
        lambda pid: "/usr/bin/python -m hermes_cli.main gateway run",
    )
    monkeypatch.setattr(
        "gateway.status.looks_like_gateway_runtime_command_line",
        lambda cmdline: True,
    )
    assert not _record_matches_live_gateway_pid(
        record, 12345, expected_home=Path("/home/hermes/.hermes/profiles/summer")
    )


def test_resolve_gateway_liveness_reports_running_for_refreshed_secondary_profile(
    tmp_path, monkeypatch
):
    """End-to-end: after the gateway's secondary-profile refresh writes a
    fresh, multiplex_secondary-stamped record naming THIS process's own
    PID, resolve_gateway_liveness (the exact function the desktop dashboard
    calls, scoped to that profile's home) must report it running — closing
    the loop on the actual reported symptom (a healthy secondary profile
    showing as unreachable in the desktop app)."""
    import os

    from gateway.status import resolve_gateway_liveness

    profile_home = tmp_path / "profiles" / "summer"
    profile_home.mkdir(parents=True)
    state_path = profile_home / "gateway_state.json"

    write_runtime_status(
        gateway_state="running",
        multiplex_secondary=True,
        path=state_path,
    )

    # write_runtime_status stamps pid/start_time from THIS test process —
    # a real live PID, so the (pid, start_time) reuse guard passes and only
    # the multiplex_secondary bypass under test is what makes the
    # per-profile command-line check pass too.
    monkeypatch.setattr(
        "gateway.status._read_process_cmdline",
        lambda pid: "/usr/bin/python -m hermes_cli.main gateway run",
    )
    monkeypatch.setattr(
        "gateway.status.looks_like_gateway_runtime_command_line",
        lambda cmdline: True,
    )

    liveness = resolve_gateway_liveness(
        profile_dir=profile_home,
        pid_probe=lambda *a, **k: None,  # no gateway.pid file for this profile
        health_probe=None,
    )
    assert liveness.running is True
    assert liveness.pid == os.getpid()


def test_platform_entries_are_restamped_to_the_live_process(tmp_path):
    """The gateway_state top-level refresh alone is NOT enough: each
    platform sub-entry (e.g. "slack") carries its OWN writer_pid/
    writer_start_time fingerprint, separate from the top-level pid.

    hermes_cli/web_server.py's cross-profile /api/status aggregation
    (``_owned_profile_platforms``) only includes a platform entry when its
    writer_pid/writer_start_time EXACTLY match the profile's live gateway
    process. If a secondary profile's platform entry is left stamped with
    a stale/dead writer identity (e.g. from before multiplexing took over),
    the profile's gateway_state.json can correctly say "running" while
    still showing ZERO connected platforms to any caller that reads the
    aggregation -- reproducing "can't reach this profile's chat in the
    desktop app" even after the top-level gateway_state fix lands.

    This asserts the fix: re-stamping a platform via
    write_runtime_status(platform=..., path=...) after the top-level
    refresh makes that entry pass the SAME ownership check
    _owned_profile_platforms uses.
    """
    import os

    state_path = tmp_path / "gateway_state.json"

    # Simulate the old, stale writer identity a secondary profile's platform
    # entry could be left with (a long-dead pre-multiplex PID/start_time).
    write_runtime_status(
        gateway_state="running",
        multiplex_secondary=True,
        path=state_path,
    )

    # This is the fix under test: re-stamp the platform entry via the live
    # process (this test process stands in for the live gateway).
    write_runtime_status(
        platform="slack",
        platform_state="connected",
        path=state_path,
    )

    record = read_runtime_status(state_path)
    slack_entry = record["platforms"]["slack"]

    # Mirrors hermes_cli.web_server._owned_profile_platforms's exact-match
    # ownership check against the live process's own identity.
    live_pid = os.getpid()
    from gateway.status import _get_process_start_time

    live_start = _get_process_start_time(live_pid)
    assert slack_entry.get("writer_pid") == live_pid
    assert slack_entry.get("writer_start_time") == live_start
    assert slack_entry.get("state") == "connected"
