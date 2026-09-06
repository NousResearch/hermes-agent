"""Regression coverage for explicit scoped-reasoning runtime state."""

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_state import SessionState


def _runner_with_override(value):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "-100123:188": ChannelOverride(reasoning_effort=value),
                },
            ),
        },
    )
    runner._peek_session_state = lambda _key: None
    runner._session_key_for_source = lambda _source: "agent:main:telegram:topic"
    runner._load_reasoning_config = lambda _model="": {
        "enabled": True,
        "effort": "low",
    }
    return runner


def _topic_source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="forum",
        thread_id="188",
        user_id="u1",
    )


def test_channel_reasoning_resolves_and_marks_turn_scoped():
    runner = _runner_with_override("high")
    source = _topic_source()

    assert runner._resolve_session_reasoning_config(source=source) == {
        "enabled": True,
        "effort": "high",
    }
    assert runner._has_scoped_reasoning_override(source=source) is True


def test_disabled_channel_reasoning_is_still_scoped():
    runner = _runner_with_override(False)
    source = _topic_source()

    assert runner._resolve_session_reasoning_config(source=source) == {
        "enabled": False,
    }
    assert runner._has_scoped_reasoning_override(source=source) is True


def test_session_reasoning_override_is_scoped_without_global_monkeypatching():
    runner = _runner_with_override(None)
    source = _topic_source()
    state = SessionState()
    state.conversation.reasoning_override = {
        "enabled": True,
        "effort": "minimal",
    }
    runner._peek_session_state = lambda _key: state

    assert runner._resolve_session_reasoning_config(source=source) == {
        "enabled": True,
        "effort": "minimal",
    }
    assert runner._has_scoped_reasoning_override(source=source) is True


def test_unconfigured_channel_keeps_model_reasoning_unscoped():
    runner = _runner_with_override(None)
    source = _topic_source()

    assert runner._resolve_session_reasoning_config(source=source) == {
        "enabled": True,
        "effort": "low",
    }
    assert runner._has_scoped_reasoning_override(source=source) is False
