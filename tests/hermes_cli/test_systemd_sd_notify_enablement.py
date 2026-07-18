"""Tests for the persistent sd_notify + WatchdogSec enablement path.

Reviewer feedback on #55018: the previous PR relied on direct unit-file
edits, but `refresh_systemd_unit_if_needed()` regenerates the unit on
every hermes update — so any direct edit gets overwritten. The
enablement path here is env-var driven at generation time so it
survives regeneration.

Contract under test:

  1. Off by default (no env var) → Type=simple, no WatchdogSec, no
     on-watchdog restart trigger. Byte-identical to prior gen.

  2. Enabled (HERMES_SD_NOTIFY_WATCHDOG_SEC=60) → Type=notify,
     WatchdogSec=60s emitted, both `Restart=always` and
     `Restart=on-watchdog` present (systemd unions them; on-watchdog
     fires when the WATCHDOG=1 heartbeat stops).

  3. Regeneration parity — since both invocations read the same env
     var, the unit text is byte-identical across two back-to-back
     `generate_systemd_unit()` calls. No drift.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_watchdog_env(monkeypatch):
    monkeypatch.delenv("HERMES_SD_NOTIFY_WATCHDOG_SEC", raising=False)


def test_generator_off_by_default_is_type_simple(monkeypatch, tmp_path):
    """Default generator output preserves the pre-PR Type=simple posture."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.gateway import generate_systemd_unit

    text = generate_systemd_unit(system=False)

    assert "Type=simple" in text
    assert "Type=notify" not in text
    assert "WatchdogSec" not in text
    assert "Restart=always" in text
    assert "Restart=on-watchdog" not in text


def test_generator_enables_type_notify_and_watchdog_from_env(monkeypatch, tmp_path):
    """HERMES_SD_NOTIFY_WATCHDOG_SEC=60 promotes to Type=notify + WatchdogSec=60s."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SD_NOTIFY_WATCHDOG_SEC", "60")
    from hermes_cli.gateway import generate_systemd_unit

    text = generate_systemd_unit(system=False)

    assert "Type=notify" in text
    assert "Type=simple" not in text
    assert "WatchdogSec=60s" in text
    # Restart semantics: always (for crash/exit) + on-watchdog (for hang).
    assert "Restart=always" in text
    assert "Restart=on-watchdog" in text


def test_generator_system_unit_also_honors_env(monkeypatch, tmp_path):
    """The system-unit template (root install) uses the same env var."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SD_NOTIFY_WATCHDOG_SEC", "30")
    from hermes_cli import gateway as hermes_gateway

    # System-unit path requires a real user identity — patch the helper
    # so we can exercise the branch without root and without a real
    # /etc/passwd lookup.
    monkeypatch.setattr(
        hermes_gateway,
        "_system_service_identity",
        lambda run_as_user=None: ("hermes-user", "hermes-group", "/home/hermes-user"),
    )
    monkeypatch.setattr(
        hermes_gateway,
        "_hermes_home_for_target_user",
        lambda home_dir: "/home/hermes-user/.hermes",
    )
    monkeypatch.setattr(
        hermes_gateway,
        "_profile_arg_for_target_user",
        lambda hermes_home, home_dir: "",
    )

    text = hermes_gateway.generate_systemd_unit(system=True)

    assert "Type=notify" in text
    assert "WatchdogSec=30s" in text
    assert "Restart=on-watchdog" in text
    # System-unit-specific field must survive.
    assert "User=hermes-user" in text


def test_generator_is_stable_across_regenerations(monkeypatch, tmp_path):
    """Two consecutive `generate_systemd_unit()` calls with the same env
    produce byte-identical output — proves `refresh_systemd_unit_if_needed()`
    won't perceive drift and try to rewrite the unit."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SD_NOTIFY_WATCHDOG_SEC", "60")
    from hermes_cli.gateway import generate_systemd_unit

    first = generate_systemd_unit(system=False)
    second = generate_systemd_unit(system=False)
    assert first == second, "regeneration must be stable — no drift under refresh"


def test_generator_falls_back_when_env_is_blank(monkeypatch, tmp_path):
    """Empty string is treated as unset; avoid emitting `WatchdogSec=s`
    (a malformed unit that systemctl would refuse to load)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SD_NOTIFY_WATCHDOG_SEC", "")
    from hermes_cli.gateway import generate_systemd_unit

    text = generate_systemd_unit(system=False)
    assert "Type=simple" in text
    assert "WatchdogSec" not in text
    assert "Restart=on-watchdog" not in text


def test_generator_falls_back_when_env_is_whitespace(monkeypatch, tmp_path):
    """Whitespace-only value (common env-file mistake) is also treated as unset."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SD_NOTIFY_WATCHDOG_SEC", "   ")
    from hermes_cli.gateway import generate_systemd_unit

    text = generate_systemd_unit(system=False)
    assert "Type=simple" in text
    assert "WatchdogSec" not in text
