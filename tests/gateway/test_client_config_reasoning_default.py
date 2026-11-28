"""``config.get full`` resolves an unset ``show_reasoning`` the way a local turn does (#154628).

A local Ink/Desktop launch streams reasoning when ``display.show_reasoning`` is unset
(``launch_shows_reasoning``: default on). Ink hydrates its reasoning gate from this projection and reads
an absent key as off, so the projection must carry the same default or Ink drops what the turn streams.
"""
import pytest

from gateway.session_config import _client_config


@pytest.mark.parametrize("cfg", [{}, {"display": {}}, {"display": {"skin": "default"}}])
def test_unset_show_reasoning_projects_as_on(cfg):
    assert _client_config(cfg)["display"]["show_reasoning"] is True


@pytest.mark.parametrize("display, expected", [
    ({"show_reasoning": False}, False),
    ({"show_reasoning": True, "platforms": {"cli": {"show_reasoning": False}}}, False),
    ({"platforms": {"cli": {"show_reasoning": False}}}, False),
    ({"show_reasoning": False, "platforms": {"cli": {"show_reasoning": True}}}, True),
])
def test_a_saved_show_reasoning_still_wins(display, expected):
    assert _client_config({"display": display})["display"]["show_reasoning"] is expected
