"""Optional judgments require a scoped, explicit per-task opt-in."""

from unittest.mock import patch

import pytest

from agent.jev_runtime import is_jev_enabled


@pytest.mark.parametrize("config", [{}, {"jev": None}, {"jev": {"title_generation": {}}},
                                    {"jev": {"title_generation": {"enabled": "true"}}},
                                    {"jev": {"title_generation": {"enabled": False}}}])
def test_missing_or_ambiguous_opt_in_stays_disabled(config):
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        assert not is_jev_enabled("title_generation")


def test_explicit_opt_in_applies_only_to_requested_task():
    config = {"jev": {"title_generation": {"enabled": True}}}
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        assert is_jev_enabled("title_generation")
        assert not is_jev_enabled("threat_screen")


def test_config_failure_stays_disabled():
    with patch("hermes_cli.config.load_config_readonly", side_effect=OSError):
        assert not is_jev_enabled("title_generation")
