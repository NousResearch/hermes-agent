"""Regression contract for stale-systemd-unit repair guidance."""

from gateway.run_startup import _SYSTEMD_TIMING_REMEDIATION


def test_stale_unit_repair_guidance_preserves_service_scope():
    assert "user scope: `hermes gateway restart`" in _SYSTEMD_TIMING_REMEDIATION
    assert "system scope: `sudo hermes gateway restart --system`" in _SYSTEMD_TIMING_REMEDIATION
    assert "gateway install --force" not in _SYSTEMD_TIMING_REMEDIATION
