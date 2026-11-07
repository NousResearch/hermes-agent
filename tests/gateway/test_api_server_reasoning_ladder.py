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


class TestRequestTurnOptOuts:
    def test_request_agent_overrides_extracts_top_level_and_nested_opt_outs(self):
        from gateway.platforms.api_server import _request_agent_overrides

        overrides = _request_agent_overrides(
            {"skip_background_review": True, "model_options": {"skip_title": "true"}}
        )
        assert overrides["model_options"] == {
            "skip_background_review": True,
            "skip_title": "true",
        }

    def test_apply_request_turn_opt_outs_sets_agent_flags(self):
        from unittest.mock import MagicMock
        from gateway.platforms.api_server_request_options import _apply_request_turn_opt_outs

        captured_kwargs = {}

        def _fake_agent_cls(**kwargs):
            captured_kwargs.update(kwargs)
            m = MagicMock()
            m.skip_background_review = kwargs.get("skip_background_review", False)
            m.skip_title = False
            return m

        agent = _apply_request_turn_opt_outs(
            _fake_agent_cls,
            {"model": "test-model"},
            {"skip_background_review": True, "skip_title": True},
        )
        assert captured_kwargs.get("skip_background_review") is True
        assert agent.skip_background_review is True
        assert agent.skip_title is True

    def test_maybe_title_session_skips_when_skip_title_set(self, monkeypatch):
        from unittest.mock import MagicMock
        import agent.turn_context as tc
        import agent.title_generator as tg

        called = []
        monkeypatch.setattr(tg, "maybe_auto_title", lambda *a, **kw: called.append(True))
        fake_agent = MagicMock(
            _session_db=MagicMock(),
            session_id="sess-1",
            platform="api_server",
            skip_title=True,
            _session_db_created=True,
        )
        tc._maybe_title_session_at_turn_start(
            fake_agent, [{"role": "user", "content": "unattended batch extraction"}]
        )
        assert called == []
