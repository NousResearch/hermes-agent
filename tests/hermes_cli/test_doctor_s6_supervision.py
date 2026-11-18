"""``hermes doctor``'s s6 section probes the image's static s6-rc services in the root scandir and
lists profile gateway slots from the separate, unprivileged gateway scandir."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import doctor_platform, service_manager

ROOT_SCANDIR = Path("/run/service")


def test_static_services_and_gateway_slots_use_distinct_scandirs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    svstat_targets: list[Path] = []
    listed: list[Path] = []

    def _fake_s6_run(cmd: str, *args: str, **_kw):
        if cmd == "s6-svstat":
            svstat_targets.append(Path(args[-1]))
        return subprocess.CompletedProcess([cmd, *args], 0, "true\n", "")

    def _list(self) -> list[str]:
        listed.append(self.scandir)
        return []

    monkeypatch.setattr(service_manager, "detect_service_manager", lambda: "s6")
    monkeypatch.setattr(service_manager, "_s6_run", _fake_s6_run)
    monkeypatch.setattr(service_manager.S6ServiceManager, "list_profile_gateways", _list)
    monkeypatch.setattr("gateway.host_topology.host_gateway_topology", lambda: None)

    doctor_platform._check_s6_supervision([])

    assert ROOT_SCANDIR / "main-hermes" in svstat_targets
    assert ROOT_SCANDIR / "dashboard" in svstat_targets
    assert "main-hermes: up" in capsys.readouterr().out
    # Gateway slots are hermes-registered, so they must not live in the root-supervised scandir.
    assert len(listed) == 1 and listed[0] != ROOT_SCANDIR
    assert ROOT_SCANDIR not in listed[0].parents
