"""#117953: freshly-restarted `hermes gateway run` warned about its own pending fleet
restart, because `_warn_pending_fleet_restart_on_startup()` used to fire at the very top of
`main()`, strictly before this gateway's own `write_runtime_status` stamp existed on disk.
`collect_fleet_versions()` then found no coverage row for this profile at all and the
false-positive fired on every single restart.

The fix moves the check out of `main()` for a `gateway run` invocation and into
`gateway/run_startup.py`, right after that gateway's own startup stamp succeeds. The unit-level
halves (argv classification, the underlying warn primitive) are covered in
tests/hermes_cli/test_update_fleet_restart_pending.py. These tests drive the REAL boot sequence
(`GatewayRunner.start()`) instead, so a regression that reorders the stamp and the deferred call
relative to each other -- without touching either isolated function -- still fails here.
"""

from __future__ import annotations

import os

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
import gateway.run_startup as run_startup
import gateway.status as gateway_status
from hermes_cli import update_cmd
import hermes_cli.update_cmd_fleet as update_cmd_fleet


def _disarm_prefilter(monkeypatch):
    """No pending platform connections: startup exits cleanly right after the code under
    test, mirroring tests/gateway/test_free_tier_gateway_boot.py."""

    async def fake_prefilter(self):
        return (False, 0, [], [])

    monkeypatch.setattr(run_startup.GatewayStartupMixin, "_start_prefilter_platforms", fake_prefilter)


def _boot_config(tmp_path):
    return GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=False)},
        sessions_dir=tmp_path / "sessions",
    )


def _arm_default_profile_marker(monkeypatch, disk_sha, *runtimes):
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: disk_sha)
    update_cmd._write_fleet_restart_pending_marker(
        expected_sha=disk_sha,
        runtimes=list(runtimes) or [{"kind": "gateway", "profile": "default"}],
    )


@pytest.mark.asyncio
async def test_gateway_boot_defers_fleet_restart_check_until_after_own_stamp(monkeypatch, tmp_path, capsys):
    """Regression for #117953. The fake fleet probe only "sees" this profile once its own
    `gateway_state.json` says `gateway_state == "starting"` -- exactly the condition the
    original bug missed. If the deferred call in gateway/run_startup.py ever moves back ahead
    of the `write_runtime_status` call, this reproduces the false positive."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    disk_sha = "e" * 40
    _arm_default_profile_marker(monkeypatch, disk_sha)

    def fake_collect_fleet_versions(**kwargs):
        record = gateway_status.read_runtime_status()
        if not record or record.get("gateway_state") != "starting":
            return []
        return [{
            "profile": "default", "pid": os.getpid(), "code_sha": disk_sha,
            "code_version": "0.21.0", "state": "current",
        }]

    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", fake_collect_fleet_versions)
    _disarm_prefilter(monkeypatch)

    ok = await GatewayRunner(_boot_config(tmp_path)).start()

    assert ok is True
    assert capsys.readouterr().err == ""
    assert not update_cmd_fleet._fleet_restart_obligation_armed()


@pytest.mark.asyncio
async def test_gateway_boot_still_warns_for_stale_sibling_in_mixed_fleet(monkeypatch, tmp_path, capsys):
    """Mixed fleet through the real wiring: this gateway's own stamp lands current, but a
    sibling profile named in the marker is genuinely stale. "My own row is now visible" must
    not read as blanket coverage for the whole fleet."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    disk_sha = "e" * 40
    _arm_default_profile_marker(
        monkeypatch, disk_sha,
        {"kind": "gateway", "profile": "default"}, {"kind": "gateway", "profile": "other"},
    )

    def fake_collect_fleet_versions(**kwargs):
        record = gateway_status.read_runtime_status()
        if not record or record.get("gateway_state") != "starting":
            return []
        return [
            {"profile": "default", "pid": os.getpid(), "code_sha": disk_sha,
             "code_version": "0.21.0", "state": "current"},
            {"profile": "other", "pid": 999999, "code_sha": "0" * 40,
             "code_version": "0.20.0", "state": "stale"},
        ]

    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", fake_collect_fleet_versions)
    _disarm_prefilter(monkeypatch)

    ok = await GatewayRunner(_boot_config(tmp_path)).start()

    assert ok is True
    assert "did not restart running gateways" in capsys.readouterr().err
    assert update_cmd_fleet._fleet_restart_obligation_armed()


@pytest.mark.asyncio
async def test_gateway_boot_stays_fail_closed_when_own_stamp_write_never_persists(monkeypatch, tmp_path, capsys):
    """Adversarial: `write_runtime_status` failing/timing out must not go silent. The deferred
    call is unconditional on `persisted` (wrapped only in `suppress(Exception)`), which is
    correct here -- with no stamp on disk there is genuinely no evidence this gateway is
    current, so the warning firing is the fix staying fail-closed, not a leftover race."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    disk_sha = "e" * 40
    _arm_default_profile_marker(monkeypatch, disk_sha)

    monkeypatch.setattr(gateway_status, "write_runtime_status", lambda **kwargs: False)
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", lambda **kwargs: [])
    _disarm_prefilter(monkeypatch)

    ok = await GatewayRunner(_boot_config(tmp_path)).start()

    assert ok is True
    assert "did not restart running gateways" in capsys.readouterr().err
    assert update_cmd_fleet._fleet_restart_obligation_armed()
