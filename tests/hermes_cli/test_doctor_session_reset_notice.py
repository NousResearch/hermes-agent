"""``hermes doctor`` names the profile whose plugin set it checked and the surface that plugin covers.

#137010 follow-up (reporter, 2026-10-22): ``session_reset.mode: both is applied by the session reset
plugin`` reads as a verified reset. It is an enablement check in ONE profile's plugin set, and the
plugin hooks ``pre_gateway_dispatch``, which native local CLI/TUI/Desktop turns never fire.
"""
import pytest

from hermes_cli import config_effective, profiles, session_reset_retirement
from hermes_cli.doctor_config import _check_retired_session_reset


@pytest.fixture
def timed_policy_with_plugin(monkeypatch):
    """A retired timed policy plus an enabled plugin, in the ``data-gover`` profile."""
    monkeypatch.setattr(
        config_effective, "load_user_config_effective",
        lambda *args, **kwargs: {"session_reset": {"mode": "both", "idle_minutes": 1440}},
    )
    monkeypatch.setattr(session_reset_retirement, "reset_plugin_enabled", lambda: True)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "data-gover")


def test_doctor_names_the_profile_whose_plugin_set_was_checked(timed_policy_with_plugin, capsys):
    _check_retired_session_reset(False)

    out = capsys.readouterr().out
    assert "'data-gover'" in out
    assert session_reset_retirement.PLUGIN_NAME in out


def test_doctor_does_not_report_the_enablement_check_as_a_verified_reset(timed_policy_with_plugin, capsys):
    _check_retired_session_reset(False)

    out = capsys.readouterr().out
    assert "is applied by the session reset plugin" not in out
    assert "not a verified reset" in out
    # The surface the plugin does not reach, named by the reporter (`gateway/run_inbound_hooks.py`).
    assert "pre_gateway_dispatch" in out
    assert "CLI/TUI/Desktop" in out
