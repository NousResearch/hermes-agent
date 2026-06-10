"""Shared fixtures for voice_call plugin tests."""

import pytest

from plugins.platforms.voice_call.config import (
    OutboundConfig,
    TimeoutsConfig,
    VoiceCallConfig,
)
from plugins.platforms.voice_call.manager import CallManager
from plugins.platforms.voice_call.providers.mock import MockProvider
from plugins.platforms.voice_call.store import CallStore


def _make_config(**overrides) -> VoiceCallConfig:
    """Mock-provider config with test-friendly (sub-second) timeouts."""
    cfg = VoiceCallConfig.from_extra({"provider": "mock", **overrides.pop("extra", {})})
    cfg.from_number = overrides.pop("from_number", "+15555550000")
    cfg.timeouts = overrides.pop(
        "timeouts",
        TimeoutsConfig(max_call_s=5, ring_s=0.2, silence_s=0, transcript_wait_s=0.5),
    )
    cfg.outbound = overrides.pop(
        "outbound", OutboundConfig(default_mode="conversation", notify_hangup_delay_s=0)
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def make_config():
    """Factory fixture (conftest isn't importable without package context)."""
    return _make_config


@pytest.fixture
def vc_config():
    return _make_config()


@pytest.fixture
def store(tmp_path):
    return CallStore(base_dir=tmp_path / "voice-calls")


@pytest.fixture
def provider(vc_config):
    return MockProvider(vc_config)


@pytest.fixture
def manager(vc_config, provider, store):
    mgr = CallManager(vc_config, provider, store)
    # The runtime normally wires this; tests connect it directly.
    provider.event_sink = mgr.process_event
    return mgr


@pytest.fixture(autouse=True)
def _reset_runtime_singleton():
    """Each test starts with a clean runtime singleton."""
    import plugins.platforms.voice_call.runtime as runtime_mod

    runtime_mod._runtime = None
    runtime_mod._runtime_lock = None
    yield
    runtime_mod._runtime = None
    runtime_mod._runtime_lock = None
