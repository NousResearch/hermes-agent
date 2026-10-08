"""Regression coverage for scoped gateway reasoning across route changes."""

from types import SimpleNamespace

import hermes_constants

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.scoped_reasoning import ScopedReasoningConfig
from gateway.session import SessionSource
from gateway.session_state import SessionState


def _runtime_resolve(agent, cfg, model):
    """Mirror route-changing runtime helpers, which call the shared resolver."""
    return hermes_constants.resolve_reasoning_config(cfg, model)


def test_channel_reasoning_resolves_as_scoped_config():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "-100123:188": ChannelOverride(reasoning_effort="high"),
                },
            ),
        },
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="forum",
        thread_id="188",
        user_id="u1",
    )

    resolved = runner._resolve_session_reasoning_config(source=source)

    assert isinstance(resolved, ScopedReasoningConfig)
    assert resolved == {"enabled": True, "effort": "high"}


def test_session_reasoning_override_is_marked_scoped_at_storage_boundary():
    state = SessionState()

    state.conversation.reasoning_override = {
        "enabled": True,
        "effort": "minimal",
    }

    assert isinstance(state.conversation.reasoning_override, ScopedReasoningConfig)
    assert state.conversation.reasoning_override == {
        "enabled": True,
        "effort": "minimal",
    }


def test_scoped_reasoning_survives_runtime_reresolution():
    scoped = ScopedReasoningConfig({"enabled": True, "effort": "high"})
    agent = SimpleNamespace(reasoning_config=scoped)
    cfg = {
        "agent": {
            "reasoning_effort": "low",
            "reasoning_overrides": {"fallback-model": "minimal"},
        },
    }

    resolved = _runtime_resolve(agent, cfg, "fallback-model")

    assert resolved is scoped
    assert resolved == {"enabled": True, "effort": "high"}


def test_unscoped_reasoning_still_reresolves_for_new_model():
    agent = SimpleNamespace(
        reasoning_config={"enabled": True, "effort": "high"},
    )
    cfg = {
        "agent": {
            "reasoning_effort": "low",
            "reasoning_overrides": {"fallback-model": "minimal"},
        },
    }

    resolved = _runtime_resolve(agent, cfg, "fallback-model")

    assert type(resolved) is dict
    assert resolved == {"enabled": True, "effort": "minimal"}


def test_invalid_channel_reasoning_falls_back_without_scope_marker():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "-100123:188": ChannelOverride(reasoning_effort="bogus"),
                },
            ),
        },
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="forum",
        thread_id="188",
        user_id="u1",
    )
    runner._load_reasoning_config = lambda _model="": {
        "enabled": True,
        "effort": "low",
    }

    resolved = runner._resolve_session_reasoning_config(source=source)

    assert type(resolved) is dict
    assert resolved == {"enabled": True, "effort": "low"}
