"""Ordinary CLI tests cannot stop or start Windows gateways during an update."""

from __future__ import annotations

import pytest


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("phase", ["pause", "resume"])
def test_unmarked_update_helpers_do_not_touch_gateways(monkeypatch, phase):
    from gateway import status
    from hermes_cli import gateway, main

    def forbidden(*_args, **_kwargs):
        pytest.fail("An ordinary test reached a live Windows gateway operation")

    for name in (
        "find_gateway_pids",
        "find_profile_gateway_processes",
        "find_windows_gateway_services",
        "launch_detached_profile_gateway_restart",
        "launch_detached_gateway_restart_by_cmdline",
    ):
        monkeypatch.setattr(gateway, name, forbidden)
    monkeypatch.setattr(status, "terminate_pid", forbidden)
    monkeypatch.setattr(main, "_refresh_windows_gateway_launchers", forbidden)
    monkeypatch.setattr(main, "_cold_start_windows_gateway_after_update", forbidden)
    monkeypatch.setattr(main.subprocess, "Popen", forbidden)

    if phase == "pause":
        assert main._pause_windows_gateways_for_update() is None
    else:
        assert main._resume_windows_gateways_after_update(
            {"resume_needed": True, "profiles": {"default": 12345}, "unmapped": []}
        ) is None
