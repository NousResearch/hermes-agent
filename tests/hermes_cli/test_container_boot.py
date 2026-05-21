"""Tests for hermes_cli.container_boot — the cont-init.d-time
reconciliation that recreates per-profile gateway s6 service slots
from the persistent profiles directory.

These tests run against a fake $HERMES_HOME under tmp_path; no real
s6 supervision tree is required. The in-container integration test
covering end-to-end "docker restart" survival lives in
tests/docker/test_container_restart.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.container_boot import (
    ReconcileAction,
    reconcile_profile_gateways,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_profile(
    hermes_home: Path,
    name: str,
    *,
    state: str | None,
    with_pid: bool = False,
    config: bool = True,
) -> Path:
    """Create a fake profile directory under hermes_home/profiles/<name>/."""
    p = hermes_home / "profiles" / name
    p.mkdir(parents=True)
    if config:
        # SOUL.md is what the reconciler keys on — it's always seeded by
        # `hermes profile create`. See container_boot._render_run_script.
        (p / "SOUL.md").write_text("# fake profile\n")
    if state is not None:
        (p / "gateway_state.json").write_text(json.dumps({
            "gateway_state": state, "timestamp": 1234567890,
        }))
    if with_pid:
        (p / "gateway.pid").write_text(json.dumps(
            {"pid": 99999, "host": "old-container"},
        ))
        (p / "processes.json").write_text("[]")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_running_profile_is_registered_and_autostarted(tmp_path: Path) -> None:
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions == [ReconcileAction(
        profile="coder", prior_state="running", action="started",
    )]
    svc = scandir / "gateway-coder"
    assert (svc / "run").exists()
    assert (svc / "run").stat().st_mode & 0o111  # executable
    assert (svc / "type").read_text().strip() == "longrun"
    # Auto-start means no down-marker.
    assert not (svc / "down").exists()


def test_stopped_profile_is_registered_but_not_started(tmp_path: Path) -> None:
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "writer", state="stopped")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions == [ReconcileAction(
        profile="writer", prior_state="stopped", action="registered",
    )]
    # down marker tells s6-svscan to NOT start the service.
    assert (scandir / "gateway-writer" / "down").exists()


def test_startup_failed_does_not_autostart(tmp_path: Path) -> None:
    """Avoid crash-loop on restart when the gateway was failing to boot."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "broken", state="startup_failed")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions[0].action == "registered"
    assert (scandir / "gateway-broken" / "down").exists()


def test_starting_state_does_not_autostart(tmp_path: Path) -> None:
    """`starting` means the gateway died mid-boot last time; treat as
    failed, not as a candidate for auto-restart."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "unlucky", state="starting")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions[0].action == "registered"


def test_stale_runtime_files_are_removed(tmp_path: Path) -> None:
    scandir = tmp_path / "run-service"; scandir.mkdir()
    profile = _make_profile(tmp_path, "coder", state="running", with_pid=True)
    assert (profile / "gateway.pid").exists()
    assert (profile / "processes.json").exists()

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert not (profile / "gateway.pid").exists()
    assert not (profile / "processes.json").exists()


def test_profile_without_state_file_is_registered_but_not_started(
    tmp_path: Path,
) -> None:
    """A freshly-created profile that's never been started: register
    its slot but don't auto-start."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "fresh", state=None)

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions == [ReconcileAction(
        profile="fresh", prior_state=None, action="registered",
    )]
    assert (scandir / "gateway-fresh" / "down").exists()


def test_directory_without_marker_file_is_skipped(tmp_path: Path) -> None:
    """A stray dir under profiles/ that isn't actually a profile (no
    SOUL.md — the marker the reconciler keys on) should be skipped."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    # Create a profile dir but without SOUL.md
    (tmp_path / "profiles" / "stray").mkdir(parents=True)

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions == []
    assert not (scandir / "gateway-stray").exists()


def test_corrupt_state_file_treated_as_no_prior_state(tmp_path: Path) -> None:
    """If gateway_state.json is malformed JSON, don't blow up the whole
    reconciliation — register the slot in the down state."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    profile = _make_profile(tmp_path, "junk", state="running")
    (profile / "gateway_state.json").write_text("{ not valid json")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions[0].action == "registered"  # not "started"
    assert (scandir / "gateway-junk" / "down").exists()


def test_reconcile_log_is_written(tmp_path: Path) -> None:
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "a", state="running")
    _make_profile(tmp_path, "b", state="stopped")

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    log = (tmp_path / "logs" / "container-boot.log").read_text()
    assert "profile=a" in log
    assert "action=started" in log
    assert "profile=b" in log
    assert "action=registered" in log


def test_dry_run_makes_no_filesystem_changes(tmp_path: Path) -> None:
    scandir = tmp_path / "run-service"; scandir.mkdir()
    profile = _make_profile(tmp_path, "coder", state="running", with_pid=True)

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=True,
    )

    # The action list is still produced...
    assert actions == [ReconcileAction(
        profile="coder", prior_state="running", action="started",
    )]
    # ...but nothing on disk was touched.
    assert (profile / "gateway.pid").exists()  # not removed under dry_run
    assert not (scandir / "gateway-coder").exists()
    assert not (tmp_path / "logs" / "container-boot.log").exists()


def test_missing_profiles_root_returns_empty(tmp_path: Path) -> None:
    """When $HERMES_HOME/profiles doesn't exist (fresh install), the
    reconciliation should return an empty list without raising."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )
    assert actions == []


def test_invalid_profile_name_in_directory_raises(tmp_path: Path) -> None:
    """A profile dir whose name doesn't match validate_profile_name's
    rules (uppercase, etc.) must surface as a hard error rather than
    silently produce an invalid s6 service dir."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "BadName", state="running")
    with pytest.raises(ValueError):
        reconcile_profile_gateways(
            hermes_home=tmp_path, scandir=scandir, dry_run=False,
        )
