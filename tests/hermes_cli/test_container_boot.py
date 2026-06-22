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


def _seed_default_root(
    hermes_home: Path,
    *,
    state: str | None = None,
    with_pid: bool = False,
) -> None:
    """Populate gateway_state.json / stale runtime files at the
    HERMES_HOME root (the implicit default profile)."""
    if state is not None:
        (hermes_home / "gateway_state.json").write_text(json.dumps({
            "gateway_state": state, "timestamp": 1234567890,
        }))
    if with_pid:
        (hermes_home / "gateway.pid").write_text(json.dumps(
            {"pid": 99999, "host": "old-container"},
        ))
        (hermes_home / "processes.json").write_text("[]")


def _named_actions(actions: list[ReconcileAction]) -> list[ReconcileAction]:
    """Drop the always-present default-profile action so tests that
    only care about named profiles can assert against a clean list."""
    return [a for a in actions if a.profile != "default"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_running_profile_is_registered_and_autostarted(tmp_path: Path) -> None:
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert _named_actions(actions) == [ReconcileAction(
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

    assert _named_actions(actions) == [ReconcileAction(
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

    named = _named_actions(actions)
    assert named[0].action == "registered"
    assert (scandir / "gateway-broken" / "down").exists()


def test_starting_state_does_not_autostart(tmp_path: Path) -> None:
    """`starting` means the gateway died mid-boot last time; treat as
    failed, not as a candidate for auto-restart."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "unlucky", state="starting")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    named = _named_actions(actions)
    assert named[0].action == "registered"


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

    assert _named_actions(actions) == [ReconcileAction(
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

    assert _named_actions(actions) == []
    assert not (scandir / "gateway-stray").exists()


def test_template_dir_with_marker_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A scaffold dir like '_template' carries a SOUL.md but its name is not a
    valid s6-slot/profile name. It must be skipped, NOT crash the reconcile —
    otherwise it aborts before later, valid profiles are registered."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    # '_template' sorts before lowercase names, so it is iterated first.
    _make_profile(tmp_path, "_template", state="running")
    _make_profile(tmp_path, "coder", state="running")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    # The invalid-named dir is skipped (no slot, no action) ...
    assert not (scandir / "gateway-_template").exists()
    # ... and the valid profile after it is still registered + started.
    assert _named_actions(actions) == [ReconcileAction(
        profile="coder", prior_state="running", action="started",
    )]
    assert (scandir / "gateway-coder" / "run").exists()


def test_corrupt_state_file_treated_as_no_prior_state(tmp_path: Path) -> None:
    """If gateway_state.json is malformed JSON, don't blow up the whole
    reconciliation — register the slot in the down state."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    profile = _make_profile(tmp_path, "junk", state="running")
    (profile / "gateway_state.json").write_text("{ not valid json")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    named = _named_actions(actions)
    assert named[0].action == "registered"  # not "started"
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


def test_reconcile_log_rotates_when_size_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When container-boot.log exceeds _LOG_ROTATE_BYTES, the existing
    file is rotated to .1 before the new entries are appended."""
    from hermes_cli import container_boot

    # Tighten the threshold so we don't have to write 256 KiB.
    monkeypatch.setattr(container_boot, "_LOG_ROTATE_BYTES", 200)

    log_path = tmp_path / "logs" / "container-boot.log"
    log_path.parent.mkdir()
    log_path.write_text("X" * 300)  # already over the threshold

    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    rotated = tmp_path / "logs" / "container-boot.log.1"
    assert rotated.exists(), "expected previous log to be rotated to .1"
    assert rotated.read_text().startswith("X" * 300)
    # The new entries land in a fresh container-boot.log (no leftover Xs).
    new_contents = log_path.read_text()
    assert "X" not in new_contents
    assert "profile=coder" in new_contents


def test_reconcile_log_does_not_rotate_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small existing log is appended to in place; no .1 is created."""
    from hermes_cli import container_boot
    monkeypatch.setattr(container_boot, "_LOG_ROTATE_BYTES", 10_000_000)

    log_path = tmp_path / "logs" / "container-boot.log"
    log_path.parent.mkdir()
    log_path.write_text("previous entry\n")

    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert not (tmp_path / "logs" / "container-boot.log.1").exists()
    contents = log_path.read_text()
    assert contents.startswith("previous entry\n")
    assert "profile=coder" in contents


def test_reconcile_log_rotation_overwrites_existing_dot1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rotating again replaces the prior .1 — we keep at most one
    rotated file (soft cap of ~2 × threshold)."""
    from hermes_cli import container_boot
    monkeypatch.setattr(container_boot, "_LOG_ROTATE_BYTES", 200)

    log_dir = tmp_path / "logs"; log_dir.mkdir()
    (log_dir / "container-boot.log.1").write_text("OLD ROTATION")
    (log_dir / "container-boot.log").write_text("Y" * 300)

    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    # .1 now contains the previous .log (Ys), not OLD ROTATION.
    rotated = (log_dir / "container-boot.log.1").read_text()
    assert "OLD ROTATION" not in rotated
    assert rotated.startswith("Y" * 300)


def test_dry_run_makes_no_filesystem_changes(tmp_path: Path) -> None:
    scandir = tmp_path / "run-service"; scandir.mkdir()
    profile = _make_profile(tmp_path, "coder", state="running", with_pid=True)

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=True,
    )

    # The action list is still produced...
    assert _named_actions(actions) == [ReconcileAction(
        profile="coder", prior_state="running", action="started",
    )]
    # ...but nothing on disk was touched.
    assert (profile / "gateway.pid").exists()  # not removed under dry_run
    assert not (scandir / "gateway-coder").exists()
    assert not (tmp_path / "logs" / "container-boot.log").exists()


def test_missing_profiles_root_still_registers_default_slot(
    tmp_path: Path,
) -> None:
    """When $HERMES_HOME/profiles doesn't exist (fresh install), the
    reconciliation should still register a gateway-default slot for
    the root profile and return without raising. Previously this
    returned an empty list; the default slot is now always present
    so `hermes gateway start` (no -p) has somewhere to land."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )
    assert actions == [ReconcileAction(
        profile="default", prior_state=None, action="registered",
    )]
    assert (scandir / "gateway-default").is_dir()
    assert (scandir / "gateway-default" / "down").exists()


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


def test_register_service_publishes_atomically(tmp_path: Path) -> None:
    """The reconciler should build the new service dir in a sibling
    tmp directory and rename it into place — never leaving a half-
    populated slot visible to a concurrent s6-svscan rescan.

    We verify the invariant indirectly: after a clean reconcile, the
    target directory exists with all required files, and no sibling
    .tmp leftovers remain. (Atomic publication is the only way to
    achieve both with mkdir + write.)
    """
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    # No leftover tmp dir.
    leftover = list(scandir.glob("*.tmp"))
    assert leftover == [], f"leftover tmp directories: {leftover}"

    # Target is fully populated.
    svc = scandir / "gateway-coder"
    assert (svc / "type").exists()
    assert (svc / "run").exists()
    assert (svc / "log" / "run").exists()


def test_register_service_overwrites_existing_slot(tmp_path: Path) -> None:
    """A second reconciliation pass cleanly replaces an existing
    slot (the tmp+rename publication overwrites the previous one)."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    profile = _make_profile(tmp_path, "coder", state="running")

    # First pass.
    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )
    first_run = (scandir / "gateway-coder" / "run").read_text()

    # Mutate the profile state so the run-script changes (extra_env
    # rendering would differ if we wired profile config through, but
    # for now just exercise the overwrite path).
    (profile / "gateway_state.json").write_text(
        '{"gateway_state": "stopped"}',
    )
    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    # Slot still exists, no .tmp remnants.
    assert (scandir / "gateway-coder" / "run").read_text() == first_run
    assert list(scandir.glob("*.tmp")) == []
    # Down marker now present (state went from running → stopped).
    assert (scandir / "gateway-coder" / "down").exists()


def test_register_service_cleans_up_stale_tmp_dir(tmp_path: Path) -> None:
    """If a previous interrupted run left a .tmp sibling directory,
    a fresh reconcile must clean it up rather than failing on mkdir."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    # Simulate a leftover from an interrupted run.
    stale_tmp = scandir / "gateway-coder.tmp"
    stale_tmp.mkdir()
    (stale_tmp / "stale-file").write_text("garbage")

    _make_profile(tmp_path, "coder", state="running")
    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert not stale_tmp.exists()
    assert (scandir / "gateway-coder" / "run").exists()


# ---------------------------------------------------------------------------
# Default-profile slot — always registered (PR #30136 review item I1)
# ---------------------------------------------------------------------------


def test_default_slot_always_registered_on_empty_home(tmp_path: Path) -> None:
    """Bare HERMES_HOME with nothing under it still produces a
    gateway-default slot (down state)."""
    scandir = tmp_path / "run-service"; scandir.mkdir()

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions == [ReconcileAction(
        profile="default", prior_state=None, action="registered",
    )]
    svc = scandir / "gateway-default"
    assert svc.is_dir()
    assert (svc / "run").exists()
    assert (svc / "down").exists()


def test_default_slot_run_script_omits_profile_flag(tmp_path: Path) -> None:
    """The default slot's run script must NOT pass `-p default` —
    that would resolve to $HERMES_HOME/profiles/default/ instead of
    the root profile. It must call `hermes gateway run` directly."""
    scandir = tmp_path / "run-service"; scandir.mkdir()

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    run = (scandir / "gateway-default" / "run").read_text()
    assert "hermes gateway run" in run
    assert "-p default" not in run
    assert "-p 'default'" not in run


def test_default_slot_autostarts_when_root_state_running(tmp_path: Path) -> None:
    """gateway_state.json at the HERMES_HOME root with state=running
    means the default slot auto-starts on container boot."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _seed_default_root(tmp_path, state="running")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    default_action = next(a for a in actions if a.profile == "default")
    assert default_action.prior_state == "running"
    assert default_action.action == "started"
    assert not (scandir / "gateway-default" / "down").exists()


def test_default_slot_does_not_autostart_when_root_state_stopped(
    tmp_path: Path,
) -> None:
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _seed_default_root(tmp_path, state="stopped")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    default_action = next(a for a in actions if a.profile == "default")
    assert default_action.action == "registered"
    assert (scandir / "gateway-default" / "down").exists()


def test_default_slot_does_not_autostart_when_root_state_startup_failed(
    tmp_path: Path,
) -> None:
    """Crash-loop guard applies to the default slot too."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _seed_default_root(tmp_path, state="startup_failed")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    default_action = next(a for a in actions if a.profile == "default")
    assert default_action.action == "registered"


def test_default_slot_cleans_up_stale_runtime_files_at_root(
    tmp_path: Path,
) -> None:
    """gateway.pid and processes.json at the HERMES_HOME root (left
    over from the previous container's default gateway) must be
    swept the same way as for named profiles."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _seed_default_root(tmp_path, state="running", with_pid=True)
    assert (tmp_path / "gateway.pid").exists()

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert not (tmp_path / "gateway.pid").exists()
    assert not (tmp_path / "processes.json").exists()


def test_default_slot_appears_before_named_profiles(tmp_path: Path) -> None:
    """The action list is ordered: default first, then named profiles
    in directory order. Operators and the boot-log reader rely on
    this ordering being stable."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "z-last-alphabetically", state="stopped")
    _make_profile(tmp_path, "a-first-alphabetically", state="stopped")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert [a.profile for a in actions] == [
        "default",
        "a-first-alphabetically",
        "z-last-alphabetically",
    ]


def test_profiles_default_subdir_is_skipped_with_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A user-created profiles/default/ collides with the reserved
    root-profile slot — the named entry is skipped (with a warning)
    so we don't double-register gateway-default."""
    import logging
    caplog.set_level(logging.WARNING)
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "default", state="running")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    # Only the root-profile default slot appears — not the colliding
    # named profile.
    default_actions = [a for a in actions if a.profile == "default"]
    assert len(default_actions) == 1
    # And the warning surfaces so operators know the named profile
    # was ignored.
    assert any(
        "profiles/default/" in record.message for record in caplog.records
    )
