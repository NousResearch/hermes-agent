"""Registration and adapter-shape tests for the voice_call platform plugin.

The generic contract is also covered by
``tests/gateway/test_plugin_platform_interface.py`` (which auto-discovers
this plugin); these tests pin voice_call-specific registration details.
"""

from unittest.mock import MagicMock

import pytest

from plugins.platforms import voice_call as voice_call_plugin
from plugins.platforms.voice_call import events
from plugins.platforms.voice_call.events import CallState, is_valid_transition


@pytest.fixture
def clean_registry():
    from gateway.platform_registry import platform_registry

    original = dict(platform_registry._entries)
    platform_registry._entries.clear()
    yield platform_registry
    platform_registry._entries.clear()
    platform_registry._entries.update(original)


class _Ctx:
    """Mock PluginContext implementing only register_platform."""

    def __init__(self):
        self.entries = {}

    def register_platform(self, *, name, label, adapter_factory, check_fn, **kwargs):
        from gateway.platform_registry import PlatformEntry, platform_registry

        entry = PlatformEntry(
            name=name, label=label,
            adapter_factory=adapter_factory, check_fn=check_fn,
            **kwargs,
        )
        platform_registry.register(entry)
        self.entries[name] = entry


def _mock_platform_config(extra=None):
    cfg = MagicMock()
    cfg.extra = extra or {}
    cfg.enabled = True
    cfg.token = None
    cfg.api_key = None
    cfg.home_channel = None
    cfg.reply_to_mode = "first"
    return cfg


def test_registers_voice_call_entry(clean_registry):
    ctx = _Ctx()
    voice_call_plugin.register(ctx)
    entry = ctx.entries["voice_call"]
    assert entry.label == "Voice Calls"
    assert entry.pii_safe is True
    assert entry.emoji == "📞"
    assert entry.max_message_length == 1000
    assert "phone call" in entry.platform_hint
    assert entry.allowed_users_env == "VOICE_CALL_ALLOWED_NUMBERS"
    assert entry.cron_deliver_env_var == "VOICE_CALL_HOME_NUMBER"


def test_adapter_constructs_from_magicmock_config(clean_registry):
    """__init__ must be side-effect-free and tolerate synthetic configs."""
    ctx = _Ctx()
    voice_call_plugin.register(ctx)
    adapter = ctx.entries["voice_call"].adapter_factory(_mock_platform_config())
    assert adapter.platform.value == "voice_call"
    assert adapter.enforces_own_access_policy is True
    assert adapter.is_connected is False


def test_validate_config_returns_bool_not_errors(clean_registry):
    ctx = _Ctx()
    voice_call_plugin.register(ctx)
    entry = ctx.entries["voice_call"]
    # Unconfigured (default telnyx, no creds) → not valid, but a clean bool.
    assert entry.validate_config(_mock_platform_config()) is False
    # Mock provider needs no credentials.
    assert entry.validate_config(_mock_platform_config({"provider": "mock"})) is True


def test_env_enablement_requires_explicit_opt_in(clean_registry, monkeypatch):
    """Credential presence alone must not auto-enable a port-binding platform."""
    from plugins.platforms.voice_call.adapter import _env_enablement

    monkeypatch.delenv("VOICE_CALL_ENABLED", raising=False)
    monkeypatch.setenv("TELNYX_API_KEY", "k")
    assert _env_enablement() is None

    monkeypatch.setenv("VOICE_CALL_ENABLED", "true")
    monkeypatch.setenv("VOICE_CALL_PROVIDER", "mock")
    monkeypatch.setenv("VOICE_CALL_HOME_NUMBER", "+15555550001")
    seed = _env_enablement()
    assert seed is not None
    assert seed["provider"] == "mock"
    assert seed["home_channel"]["chat_id"] == "+15555550001"


@pytest.mark.asyncio
async def test_connect_fails_cleanly_on_invalid_config(clean_registry):
    """Invalid config → connect() returns False with a non-retryable error."""
    ctx = _Ctx()
    voice_call_plugin.register(ctx)
    adapter = ctx.entries["voice_call"].adapter_factory(
        _mock_platform_config({"provider": "doesnotexist"})
    )
    assert await adapter.connect() is False
    assert adapter.is_connected is False


@pytest.mark.asyncio
async def test_send_without_runtime_fails_cleanly(clean_registry):
    ctx = _Ctx()
    voice_call_plugin.register(ctx)
    adapter = ctx.entries["voice_call"].adapter_factory(
        _mock_platform_config({"provider": "mock"})
    )
    result = await adapter.send("+15555550001", "hello")
    assert result.success is False
    assert "runtime" in (result.error or "")


# -- events.py shape ---------------------------------------------------------


def test_state_transitions():
    assert is_valid_transition(CallState.INITIATED, CallState.RINGING)
    assert is_valid_transition(CallState.RINGING, CallState.ANSWERED)
    assert is_valid_transition(CallState.ANSWERED, CallState.ACTIVE)
    assert is_valid_transition(CallState.SPEAKING, CallState.LISTENING)
    assert is_valid_transition(CallState.LISTENING, CallState.SPEAKING)
    assert is_valid_transition(CallState.ACTIVE, CallState.COMPLETED)
    assert is_valid_transition(CallState.INITIATED, CallState.NO_ANSWER)
    # No backwards moves, no leaving terminal states, no self-loops.
    assert not is_valid_transition(CallState.ACTIVE, CallState.RINGING)
    assert not is_valid_transition(CallState.COMPLETED, CallState.ACTIVE)
    assert not is_valid_transition(CallState.COMPLETED, CallState.FAILED)
    assert not is_valid_transition(CallState.ACTIVE, CallState.ACTIVE)


def test_call_record_roundtrip():
    record = events.CallRecord(
        call_id=events.new_call_id(),
        provider="mock",
        direction="outbound",
        to_number="+15555550001",
        from_number="+15555550000",
        mode="notify",
    )
    record.transcript.append(
        events.TranscriptEntry(timestamp=1.0, speaker="bot", text="hi")
    )
    restored = events.CallRecord.from_dict(record.to_dict())
    assert restored == record


def test_call_record_from_dict_tolerates_garbage():
    restored = events.CallRecord.from_dict({"state": "made-up", "transcript": ["junk"]})
    assert restored.state == CallState.ERROR
    assert restored.transcript == []
    assert restored.call_id  # generated
