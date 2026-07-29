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


@pytest.fixture(autouse=True)
def _hermetic_container_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ``_read_container_argv()`` to empty for the whole module.

    ``_read_container_argv()`` walks the entire ``/proc`` table looking for
    a process whose argv contains ``main-wrapper.sh`` (the s6-overlay v3
    fallback). On a host that is *also* running hermes containers, those
    containers' ``main-wrapper.sh`` processes are visible in the host's
    ``/proc`` (shared PID view), so the scan would pick up a foreign
    ``gateway run`` argv and make ``_maybe_migrate_legacy_gateway_run_state``
    synthesize ``running`` state — flaking any test that reconciles without
    injecting ``container_argv``. Inside the real container ``/proc`` is the
    container's own PID namespace, so production is unaffected; this fixture
    just makes the unit suite hermetic. Tests that need a specific argv
    either pass ``container_argv=`` to ``reconcile_profile_gateways`` or
    monkeypatch ``_read_container_argv`` themselves (both override this).
    """
    monkeypatch.setattr(
        "hermes_cli.container_boot._read_container_argv",
        lambda: (),
    )


def _make_profile(
    hermes_home: Path,
    name: str,
    *,
    state: str | None,
    desired_state: str | None = None,
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
    if state is not None or desired_state is not None:
        payload: dict[str, object] = {"timestamp": 1234567890}
        if state is not None:
            payload["gateway_state"] = state
        if desired_state is not None:
            payload["desired_state"] = desired_state
        (p / "gateway_state.json").write_text(json.dumps(payload))
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


def test_registered_profile_has_finish_script(tmp_path: Path) -> None:
    """The finish script must be written so s6 stops restarting on
    fatal config errors (exit 78 → exit 125).  See #51228."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    finish = scandir / "gateway-coder" / "finish"
    assert finish.exists()
    assert finish.stat().st_mode & 0o111  # executable
    text = finish.read_text()
    assert "78" in text
    assert "125" in text


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


def test_draining_default_root_autostarts(tmp_path: Path) -> None:
    """The hosted-agent path: the default (root) profile, not a named one.
    A managed Fly instance runs the root profile; a stranded `draining` there
    is exactly what wedged the relay-opted-in staging instance. Mirror the
    named-profile case for the default slot."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _seed_default_root(tmp_path, state="draining")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    default_action = next(a for a in actions if a.profile == "default")
    assert default_action.prior_state == "running"
    assert default_action.action == "started"
    assert not (scandir / "gateway-default" / "down").exists()


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

    # Slot still exists, no .tmp remnants (staging dir is dot-prefixed,
    # so match it explicitly — a leading-`*` glob won't catch dotfiles).
    assert (scandir / "gateway-coder" / "run").read_text() == first_run
    assert list(scandir.glob("*.tmp")) == []
    assert list(scandir.glob(".*.tmp")) == []
    # Down marker now present (state went from running → stopped).
    assert (scandir / "gateway-coder" / "down").exists()


def test_register_service_cleans_up_stale_tmp_dir(tmp_path: Path) -> None:
    """If a previous interrupted run left a staging sibling directory,
    a fresh reconcile must clean it up rather than failing on mkdir.

    The staging dir is dot-prefixed (``.gateway-<profile>.tmp``) so a
    concurrent s6-svscan rescan can't supervise it half-built; the
    cleanup must target that same dot-prefixed name.
    """
    scandir = tmp_path / "run-service"; scandir.mkdir()
    # Simulate a leftover from an interrupted run (current staging name).
    stale_tmp = scandir / ".gateway-coder.tmp"
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


def test_legacy_gateway_run_env_no_supervise_does_not_seed_s6_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env opt-out matches the CLI `--no-supervise` flag."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", "1")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path,
        scandir=scandir,
        dry_run=False,
        container_argv=("gateway", "run"),
    )

    default_action = next(a for a in actions if a.profile == "default")
    assert default_action.prior_state is None
    assert default_action.action == "registered"
    assert (scandir / "gateway-default" / "down").exists()
    assert not (tmp_path / "gateway_state.json").exists()


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


# ---------------------------------------------------------------------------
# Dashboard-container role detection (skip reconcile on the dashboard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "container_argv",
    [
        # Bare subcommand (docker run ... dashboard ...).
        ("dashboard",),
        ("dashboard", "--host", "127.0.0.1", "--no-open"),
        # Through s6 /init + the main-wrapper that re-execs `hermes`.
        ("/init", "/opt/hermes/docker/main-wrapper.sh", "dashboard"),
        (
            "/init",
            "/opt/hermes/docker/main-wrapper.sh",
            "dashboard",
            "--host",
            "127.0.0.1",
            "--no-open",
        ),
        # Wrapper that kept the explicit `hermes` argv0.
        ("/init", "/opt/hermes/docker/main-wrapper.sh", "hermes", "dashboard"),
        # s6-overlay v3: PID 1 is s6-svscan, so the role is read off the
        # rc.init-launched process whose argv is
        # `/bin/sh -e .../rc.init top .../main-wrapper.sh dashboard ...`.
        # This is the exact shape that regressed in issue #49196.
        (
            "/bin/sh",
            "-e",
            "/run/s6/basedir/scripts/rc.init",
            "top",
            "/opt/hermes/docker/main-wrapper.sh",
            "dashboard",
            "--host",
            "0.0.0.0",
            "--port",
            "9119",
            "--no-open",
            "--insecure",
        ),
    ],
)
def test_is_dashboard_container_true_for_dashboard_argv(
    container_argv: tuple[str, ...],
) -> None:
    """A dashboard command is detected across every wrapper prefix shape."""
    from hermes_cli.container_boot import _is_dashboard_container

    assert _is_dashboard_container(container_argv) is True


def test_main_skips_reconcile_in_dashboard_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() must NOT reconcile when PID 1 argv is the dashboard command.

    A running profile is seeded so that, if reconcile ran, it would create
    the gateway-<profile> slot. Asserting the slot is absent proves the
    skip is real, not just a log line.
    """
    from hermes_cli import container_boot

    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "worker", state="running")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("S6_PROFILE_GATEWAY_SCANDIR", str(scandir))
    monkeypatch.setattr(
        container_boot,
        "_read_container_argv",
        lambda: ("/init", "/opt/hermes/docker/main-wrapper.sh", "dashboard"),
    )

    rc = container_boot.main()

    assert rc == 0
    assert not (scandir / "gateway-worker").exists()
    assert not (scandir / "gateway-default").exists()
    assert "skipping (dashboard container" in capsys.readouterr().out


def test_main_skips_reconcile_in_dashboard_container_s6v3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dashboard skip must fire under the s6-overlay v3 argv shape.

    Regression test for issue #49196: under s6-overlay v3 the container
    command is read off the rc.init-launched process, whose argv is
    ``/bin/sh -e .../rc.init top .../main-wrapper.sh dashboard ...`` — not a
    bare ``/init`` prefix. Before the fix, the prefix-strip left ``/bin/sh``
    at args[0], so the role read as non-dashboard, the dashboard container
    reconciled, and it started its own gateway-default (dual Telegram
    getUpdates 409). Asserting the slot is absent proves the skip fires.
    """
    from hermes_cli import container_boot

    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "worker", state="running")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("S6_PROFILE_GATEWAY_SCANDIR", str(scandir))
    monkeypatch.setattr(
        container_boot,
        "_read_container_argv",
        lambda: (
            "/bin/sh",
            "-e",
            "/run/s6/basedir/scripts/rc.init",
            "top",
            "/opt/hermes/docker/main-wrapper.sh",
            "dashboard",
            "--host",
            "0.0.0.0",
            "--port",
            "9119",
            "--no-open",
            "--insecure",
        ),
    )

    rc = container_boot.main()

    assert rc == 0
    assert not (scandir / "gateway-worker").exists()
    assert not (scandir / "gateway-default").exists()
    assert "skipping (dashboard container" in capsys.readouterr().out


def test_main_ignores_removed_skip_reconcile_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy HERMES_SKIP_PROFILE_RECONCILE flag is gone: setting it on a
    gateway container must NOT suppress reconciliation. Role is decided by
    PID 1 argv alone, so a stale flag in someone's manifest is inert."""
    from hermes_cli import container_boot

    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "worker", state="running")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("S6_PROFILE_GATEWAY_SCANDIR", str(scandir))
    monkeypatch.setenv("HERMES_SKIP_PROFILE_RECONCILE", "1")
    monkeypatch.setattr(
        container_boot,
        "_read_container_argv",
        lambda: ("/init", "/opt/hermes/docker/main-wrapper.sh", "gateway", "run"),
    )

    rc = container_boot.main()

    assert rc == 0
    # Reconcile still ran despite the stale env var.
    assert (scandir / "gateway-worker").exists()


# ---------------------------------------------------------------------------
# prior_exit annotation (NS-608 — unclean-shutdown forensics)
# ---------------------------------------------------------------------------


def _write_lifecycle_sentinel(profile_dir: Path, payload: dict) -> None:
    state_dir = profile_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "gateway.lifecycle.json").write_text(json.dumps(payload))


def test_reconcile_log_prior_exit_unknown_without_sentinel(tmp_path: Path) -> None:
    """No lifecycle sentinel (fresh profile, pre-upgrade volume) →
    prior_exit=unknown, and reconciliation is unaffected."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "fresh", state="running")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    (fresh,) = [a for a in actions if a.profile == "fresh"]
    assert fresh.prior_exit == "unknown"
    assert fresh.action == "started"


