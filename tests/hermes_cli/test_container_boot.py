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
    assert "skipping (non-gateway container" in capsys.readouterr().out


def test_main_skips_reconcile_in_serve_container_s6v3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A `serve` container must skip reconciliation like the dashboard.

    Regression test for issue #78810: `serve` (the desktop-app headless
    backend) boots the same server as `dashboard` and shares a bind-mounted
    HERMES_HOME with the gateway container. Before the fix it was not
    exempt, so it registered and auto-started its own `gateway-default`
    slot — a duplicate live gateway racing on the dispatcher lock and the
    s6-log lock files. Asserting the slot is absent proves the skip fires
    for the serve role under the s6-overlay v3 argv shape.
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
            "serve",
            "--host",
            "0.0.0.0",
        ),
    )

    rc = container_boot.main()

    assert rc == 0
    assert not (scandir / "gateway-worker").exists()
    assert not (scandir / "gateway-default").exists()
    assert "skipping (non-gateway container" in capsys.readouterr().out




# ---------------------------------------------------------------------------
# prior_exit annotation (NS-608 — unclean-shutdown forensics)
# ---------------------------------------------------------------------------


def _write_lifecycle_sentinel(profile_dir: Path, payload: dict) -> None:
    state_dir = profile_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "gateway.lifecycle.json").write_text(json.dumps(payload))




