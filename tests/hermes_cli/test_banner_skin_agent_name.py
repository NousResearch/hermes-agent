"""Regression test for issue #6768.

The banner version label must honour the active skin's `branding.agent_name`
when one is configured; it must fall back to "Hermes Agent" when no skin
overrides it.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import banner
from hermes_cli.skin_engine import get_active_skin_name, set_active_skin

VERSION = "9.9.9"


@pytest.fixture
def pinned_label_inputs(monkeypatch):
    """Pin everything except the skin: only the agent name is under test here."""
    monkeypatch.setattr(banner, "get_version_info", lambda: SimpleNamespace(derived_version=VERSION))
    monkeypatch.setattr(banner, "get_git_banner_state", lambda: None)
    monkeypatch.setattr("hermes_cli.steward.read_install_stamp", lambda root: {})
    monkeypatch.setattr("hermes_cli.update_channel.resolve_update_channel", lambda *a, **k: "main")


def test_format_banner_version_label_uses_skin_agent_name(pinned_label_inputs):
    """A skin with branding.agent_name = 'My Agent' produces 'My Agent v...'."""
    with patch("hermes_cli.skin_engine.get_active_skin") as get_active_skin:
        get_active_skin.return_value.get_branding.return_value = "My Agent"
        value = banner.format_banner_version_label()

    assert value == f"My Agent v{VERSION} ({banner.RELEASE_DATE})"


def test_format_banner_version_label_falls_back_when_skin_unavailable(pinned_label_inputs):
    """Skin engine raises -> still produces a safe 'Hermes Agent v...' label."""
    with patch(
        "hermes_cli.skin_engine.get_active_skin",
        side_effect=RuntimeError("skin engine unavailable"),
    ):
        value = banner.format_banner_version_label()

    assert value == f"Hermes Agent v{VERSION} ({banner.RELEASE_DATE})"


def test_format_banner_version_label_falls_back_when_skin_branding_is_null(pinned_label_inputs):
    """A skin may set branding.agent_name explicitly to null/empty -> fallback, never 'None'."""
    for blank in (None, ""):
        with patch("hermes_cli.skin_engine.get_active_skin") as get_active_skin:
            get_active_skin.return_value.get_branding.return_value = blank
            value = banner.format_banner_version_label()

        assert value == f"Hermes Agent v{VERSION} ({banner.RELEASE_DATE})", blank


def test_real_skin_agent_name_reaches_the_label(pinned_label_inputs):
    """End-to-end through the real skin engine, no mocks: ares names itself."""
    previous = get_active_skin_name()
    try:
        set_active_skin("ares")
        assert banner.format_banner_version_label().startswith("Ares Agent v")
    finally:
        set_active_skin(previous)
