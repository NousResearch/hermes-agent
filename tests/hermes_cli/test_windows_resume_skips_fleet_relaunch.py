"""A Windows update must not relaunch a profile the fleet restart already armed (#126821).

The fleet path stops a gateway that appeared during the update and arms
``gateway run --replace``. The Windows resume path then relaunches every
profile the pre-update pause recorded. The second ``--replace`` kills the
first. Profiles already on the fleet outcome are not launched again and
stay off the resume path's relaunch result: the fleet outcome already
classifies them, so an externally supervised profile must not surface as
relaunched too.
"""

from types import SimpleNamespace
from unittest.mock import patch

import hermes_cli.update_cmd_windows as update_cmd_windows


def test_resume_does_not_replace_a_profile_the_fleet_already_relaunched(monkeypatch):
    calls = []

    def _launch(profile, pid):
        calls.append((profile, pid))
        return True

    def _verify_alive(token, launched, launched_unmapped):
        token["relaunched_profiles"] = dict(launched)

    monkeypatch.setattr(
        "hermes_cli.gateway.launch_detached_profile_gateway_restart", _launch)
    monkeypatch.setattr(update_cmd_windows, "_resume_windows_services", lambda token: None)
    monkeypatch.setattr(update_cmd_windows, "_verify_relaunched_gateways_alive", _verify_alive)
    monkeypatch.setattr(update_cmd_windows, "_cold_start_attested_profiles", lambda token: None)
    monkeypatch.setattr(
        "hermes_cli.update_cmd._m",
        lambda: SimpleNamespace(_refresh_windows_gateway_launchers=lambda: None))
    token = {
        "fleet_relaunched": ["default"],
        "profiles": {"default": 111, "other": 222},
        "unmapped": [],
    }
    update_cmd_windows._resume_paused_set(token)

    assert calls == [("other", 222)]
    assert token["relaunched_profiles"] == {"other": 222}
    assert token["resume_needed"] is False


def test_merge_records_fleet_profiles_before_windows_resume(monkeypatch):
    import hermes_cli.main as hm
    from hermes_cli import update_cmd

    seen = {}

    def _resume(token):
        seen["fleet"] = list(token.get("fleet_relaunched") or [])

    monkeypatch.setattr(hm, "_resume_windows_gateways_after_update", _resume)
    outcome = update_cmd._GatewayRestartOutcome(
        incomplete=False,
        phase_errors=[],
        pre_restart_gateway_pids=[],
        restarted_services=[],
        failed_or_stale_units=[],
        relaunched_profiles=["default"],
        externally_supervised_profiles=[],
        killed_pids=set(),
    )
    token = {"resume_needed": False, "profiles": {"default": 111, "other": 222}}
    with patch("hermes_cli.update_receipt.record_gateway_restart", lambda **_kw: None):
        update_cmd._resume_windows_gateways_and_merge_outcome(outcome, token, False)

    assert seen["fleet"] == ["default"]
    assert token["profiles"] == {"default": 111, "other": 222}


def test_merge_records_externally_supervised_profiles(monkeypatch):
    import hermes_cli.main as hm
    from hermes_cli import update_cmd

    seen = {}

    def _resume(token):
        seen["fleet"] = list(token.get("fleet_relaunched") or [])

    monkeypatch.setattr(hm, "_resume_windows_gateways_after_update", _resume)
    outcome = update_cmd._GatewayRestartOutcome(
        incomplete=False,
        phase_errors=[],
        pre_restart_gateway_pids=[],
        restarted_services=[],
        failed_or_stale_units=[],
        relaunched_profiles=[],
        externally_supervised_profiles=["default"],
        killed_pids=set(),
    )
    token = {"resume_needed": False, "profiles": {"default": 111, "other": 222}}
    with patch("hermes_cli.update_receipt.record_gateway_restart", lambda **_kw: None):
        update_cmd._resume_windows_gateways_and_merge_outcome(outcome, token, False)

    assert seen["fleet"] == ["default"]
    assert token["profiles"] == {"default": 111, "other": 222}
