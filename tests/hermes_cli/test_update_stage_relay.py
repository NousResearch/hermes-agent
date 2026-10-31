"""The Desktop hand-off window names the update stage that is actually running."""
from __future__ import annotations

import json

import pytest


class _StopBeforeAnyGateway(BaseException):
    """Escapes the restart phase's recovery (it catches Exception) before any gateway is probed."""


def test_restart_after_a_failed_build_names_the_restart(tmp_path, monkeypatch):
    """A failed build does not end the update (the code is committed): the fleet restart still
    runs, and the window must say so instead of the build message it last showed."""
    from hermes_cli import update_cmd, update_cmd_fleet
    from hermes_cli.update_finish import finish_update
    from hermes_cli.update_stage import publish_stage

    status = tmp_path / "hermes-update-status.1"
    monkeypatch.setenv("HERMES_UPDATE_STATUS_FILE", str(status))

    def shown() -> str:
        return json.loads(status.read_text(encoding="utf-8"))["message"] if status.exists() else ""

    def failed_build(*, followups, **_kwargs):
        publish_stage("Building the desktop app")
        followups.append(("build", "desktop app build failed"))
        return True

    during_restart = []

    def restart_phase_starts():
        during_restart.append(shown())
        raise _StopBeforeAnyGateway

    monkeypatch.setattr(update_cmd, "_run_post_update_maintenance", failed_build)
    # The real restart phase runs up to its first step; nothing past it touches a gateway.
    monkeypatch.setattr(update_cmd_fleet, "_gateway_drain_budget", restart_phase_starts)

    with pytest.raises(_StopBeforeAnyGateway):
        finish_update(root=tmp_path, assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
                      had_desktop_app_before_update=False, pre_update_version=None, plan=None,
                      windows_resume=None)

    assert during_restart == ["Restarting Hermes services"]
