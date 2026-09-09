"""API-server reasoning-effort request parsing accepts the full ladder.

_request_reasoning_config() previously whitelisted only none..xhigh, so an
API/browser client sending ``max`` or ``ultra`` (both valid /reasoning and
config.yaml levels) was silently ignored and the session ran at the default
effort. The server now accepts the full internal ladder; per-provider wire
clamping happens downstream in the transports/profiles via
agent.reasoning_effort (#78216's api_server observation).
"""

from gateway.platforms.api_server import _request_reasoning_config
from hermes_constants import VALID_REASONING_EFFORTS


class TestRequestReasoningFullLadder:
    def test_every_configurable_level_is_accepted(self):
        for level in VALID_REASONING_EFFORTS:
            out = _request_reasoning_config({"reasoning_effort": level})
            assert out == {"enabled": True, "effort": level}, level

    def test_none_disables(self):
        assert _request_reasoning_config({"reasoning_effort": "none"}) == {
            "enabled": False
        }

    def test_structured_object_shape(self):
        out = _request_reasoning_config(
            {"reasoning": {"enabled": True, "effort": "ultra"}}
        )
        assert out == {"enabled": True, "effort": "ultra"}

    def test_unknown_level_still_ignored(self):
        assert _request_reasoning_config({"reasoning_effort": "turbo"}) is None
