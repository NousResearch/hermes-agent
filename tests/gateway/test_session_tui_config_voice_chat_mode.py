"""The shared-owner writer for ``voice.voice_chat_mode`` accepts what main's sidecar accepts.

Desktop on a multiplexed gateway writes the voice engine through ``gateway.session_tui_config``;
it must take the same values as ``tui_gateway`` (``chained``, ``gpt-live`` and an extension
engine id ``<plugin>:<engine>``) and refuse anything else without touching the config.
"""

import pytest

from gateway.session_tui_config import _set_profile_leaf
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.parametrize(("value", "stored"), [
    ("gpt-live", "gpt-live"),
    ("Chained ", "chained"),
    ("hermes-xai:grok", "hermes-xai:grok"),
    ("My_Plugin:Live-1", "my_plugin:live-1"),
])
def test_engine_values_land_in_the_voice_section(value, stored):
    cfg = {"voice": {"auto_tts": True}}
    _set_profile_leaf(cfg, "voice.voice_chat_mode", value)
    assert cfg == {"voice": {"auto_tts": True, "voice_chat_mode": stored}}


@pytest.mark.parametrize("value", ["loud", "", None, "hermes-xai:", "a:b:c", "x y:z"])
def test_other_values_are_refused_and_leave_the_config_alone(value):
    cfg = {"voice": {"voice_chat_mode": "chained"}}
    with pytest.raises(RuntimeStoreError):
        _set_profile_leaf(cfg, "voice.voice_chat_mode", value)
    assert cfg == {"voice": {"voice_chat_mode": "chained"}}
