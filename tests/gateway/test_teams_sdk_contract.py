"""The parts of the pinned microsoft-teams-apps API the Teams adapter relies on, checked against the
real SDK when the ``teams`` extra is installed (``tests/gateway/test_teams.py`` runs on a mock)."""

import importlib.util
import inspect

import pytest


def _sdk_installed() -> bool:
    try:
        return importlib.util.find_spec("microsoft_teams.apps") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _sdk_installed(), reason="needs the teams extra (microsoft-teams-apps)")


def test_app_sends_through_send_with_a_service_url():
    from plugins.platforms.teams.adapter import _suppress_third_party_dotenv
    with _suppress_third_party_dotenv():  # the SDK loads a cwd .env at import (#62935)
        from microsoft_teams.apps import App
    assert not hasattr(App, "activity_sender")
    assert "service_url" in inspect.signature(App.send).parameters
