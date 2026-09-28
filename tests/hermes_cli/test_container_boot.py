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
import os
from pathlib import Path

import pytest

from hermes_cli.container_boot import (
    ReconcileAction,
    reconcile_profile_gateways,
)

pytestmark = pytest.mark.platforms("linux")


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
    # This fixture owns real files, but does not run as the image's service user.
    monkeypatch.setattr("hermes_cli.service_manager._HERMES_UID", os.getuid())
    monkeypatch.setattr("hermes_cli.service_manager._HERMES_GID", os.getgid())


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


def test_a_named_profile_slot_is_registered_but_never_autostarted(tmp_path: Path) -> None:
    """Multiplex-only: the container boots ONE gateway. A named slot that s6 starts on its own is
    a second gateway process, so the slot is registered DOWN even when its run intent says
    "running" -- that is how an image upgraded from the per-profile era converges with no manual
    step (the root gateway serves `coder` instead)."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")
    (tmp_path / "gateway_state.json").write_text('{"gateway_state": "running"}', encoding="utf-8")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert _named_actions(actions) == [ReconcileAction(
        profile="coder", prior_state="running", action="registered", folded_into_root=True,
    )]
    svc = scandir / "gateway-coder"
    assert (svc / "run").exists()
    assert (svc / "run").stat().st_mode & 0o111  # executable
    assert (svc / "type").read_text().strip() == "longrun"
    assert (svc / "down").exists(), "a named slot must not boot itself"
    # The root profile's slot is the ONE gateway and keeps its run intent.
    default = next(a for a in actions if a.profile == "default")
    assert default.action == "started"


@pytest.mark.parametrize("config_value,env_value", [
    pytest.param("false", None, id="config-false"),
    pytest.param("true", "false", id="env-false-overrides-config"),
])
def test_the_retired_opt_out_cannot_boot_a_second_gateway_in_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_value: str, env_value: str | None,
) -> None:
    """`gateway.multiplex_profiles: false` (and its env override) is retired as a topology switch.
    It used to decide whether s6 booted the per-profile slots; reading it here is what shipped the
    opt-out topology by accident on the UNSET default. The slot stays down either way."""
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")
    (tmp_path / "config.yaml").write_text(f"multiplex_profiles: {config_value}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if env_value is None:
        monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    else:
        monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", env_value)

    actions = reconcile_profile_gateways(hermes_home=tmp_path, scandir=scandir, dry_run=False)

    assert _named_actions(actions) == [ReconcileAction(
        profile="coder", prior_state="running", action="registered", folded_into_root=True)]
    assert (scandir / "gateway-coder" / "down").exists()


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


# ---------------------------------------------------------------------------
# Default-profile slot — always registered (PR #30136 review item I1)
# ---------------------------------------------------------------------------


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


def test_main_skips_reconcile_in_dashboard_container_s6v3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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


# ---------------------------------------------------------------------------
# Multiplex-only: the root slot inherits every named slot's autostart intent
# ---------------------------------------------------------------------------


def test_a_named_slots_autostart_intent_boots_the_root_slot(tmp_path: Path) -> None:
    """An image only ever driven as `hermes -p coder gateway start` has no root state at all.
    Named slots are now registered DOWN unconditionally (one gateway per container), so without
    the root slot inheriting their intent the container boots with ZERO gateways while every
    action reports "registered" — healthy-looking and completely dark."""
    hermes_home = tmp_path / "data"
    hermes_home.mkdir()
    _make_profile(hermes_home, "coder", state=None, desired_state="running")

    actions = reconcile_profile_gateways(
        hermes_home=hermes_home, scandir=tmp_path / "svc", dry_run=True, container_argv=())

    by_profile = {a.profile: a for a in actions}
    assert by_profile["default"].action == "started", "something must serve this container"
    assert by_profile["default"].folded_into_root is True
    assert by_profile["coder"].action == "registered" and by_profile["coder"].folded_into_root is True


def test_a_stopped_fleet_still_boots_nothing(tmp_path: Path) -> None:
    """Control: no autostart intent anywhere means no gateway is started — the crash-loop guard
    and the operator's `gateway stop` both keep working."""
    hermes_home = tmp_path / "data"
    hermes_home.mkdir()
    _seed_default_root(hermes_home, state="stopped")
    _make_profile(hermes_home, "coder", state=None, desired_state="stopped")
    _make_profile(hermes_home, "ops", state=None, desired_state="startup_failed")

    actions = reconcile_profile_gateways(
        hermes_home=hermes_home, scandir=tmp_path / "svc", dry_run=True, container_argv=())

    assert [a.action for a in actions] == ["registered"] * 3
    assert not any(a.folded_into_root for a in actions)
