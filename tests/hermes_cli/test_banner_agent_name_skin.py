"""Regression tests for #6768: the startup banner version label must carry the
active skin's ``agent_name`` branding instead of the hardcoded "Hermes Agent"."""
from unittest.mock import patch


def test_format_banner_version_label_uses_active_skin_agent_name():
    """A skin that renames the agent shows in the banner version label."""
    from hermes_cli import banner

    with (
        patch.object(banner, "get_git_banner_state", return_value=None),
        patch.object(banner, "_skin_branding", return_value="Ares Agent"),
    ):
        value = banner.format_banner_version_label()

    assert value.startswith("Ares Agent v")
    assert "Hermes Agent" not in value


def test_format_banner_version_label_falls_back_without_skin_engine():
    """A broken skin engine keeps the banner working with the default name."""
    from hermes_cli import banner

    with (
        patch.object(banner, "get_git_banner_state", return_value=None),
        patch.object(banner, "_active_skin", side_effect=RuntimeError("no skin engine")),
    ):
        value = banner.format_banner_version_label()

    assert value.startswith("Hermes Agent v")
