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


def test_scoped_reasoning_reads_yaml_from_each_profile_without_cross_talk(tmp_path, monkeypatch):
    """The native YAML loader keeps topic values and disabled booleans profile-local."""
    import yaml
    from gateway.config import load_gateway_config

    profiles = []
    for label, effort in (("a", "high"), ("b", False)):
        home = tmp_path / label
        home.mkdir()
        (home / "config.yaml").write_text(yaml.safe_dump({
            "platforms": {"telegram": {"enabled": True, "channel_overrides": {
                "-100123": {"reasoning_effort": "low"},
                "-100123:188": {"reasoning_effort": effort},
            }}},
            "agent": {"reasoning_effort": "medium"},
        }))
        profiles.append((home, {"enabled": True, "effort": effort} if effort else {"enabled": False}))

    for home, expected in (profiles[0], profiles[1], profiles[0]):
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = _runner_with_override(None)
        runner.config = load_gateway_config()
        assert runner._resolve_session_reasoning_config(source=_topic_source()) == expected
        assert runner._has_scoped_reasoning_override(source=_topic_source()) is True
