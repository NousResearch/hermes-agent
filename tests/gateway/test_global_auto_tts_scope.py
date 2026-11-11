"""Global reply scope from real config, with per-chat voice choices preserved (#100431)."""

import weakref
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Adapter(BasePlatformAdapter):
    """Real adapter policy/state with no network transport."""

    async def connect(self, *, is_reconnect=False):
        raise NotImplementedError

    async def disconnect(self):
        raise NotImplementedError

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        raise NotImplementedError


def _configured_runner(tmp_path, monkeypatch, mode="voice_only", *, enabled=True):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mode_line = "" if mode is None else f"  auto_tts_mode: {mode}\n"
    (tmp_path / "config.yaml").write_text(
        f"voice:\n  auto_tts: {str(enabled).lower()}\n{mode_line}", encoding="utf-8"
    )
    runner = object.__new__(GatewayRunner)
    runner._voice_mode = {}
    adapter = _Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._sync_voice_mode_state_to_adapter(adapter)
    return runner, adapter


def _event(kind, *, profile=None):
    return MessageEvent(
        text="hello",
        message_type=kind,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="user", profile=profile),
    )


def test_global_voice_only_config_controls_streamed_and_nonstreamed_replies(tmp_path, monkeypatch):
    runner, adapter = _configured_runner(tmp_path, monkeypatch)
    for already_sent in (False, True):
        assert runner._should_send_voice_reply(
            _event(MessageType.TEXT), "reply", [], already_sent=already_sent
        ) is False
    assert runner._should_send_voice_reply(
        _event(MessageType.VOICE), "reply", [], already_sent=True
    ) is True
    # Non-streamed voice audio remains adapter-owned, not synthesized twice by the runner.
    assert runner._should_send_voice_reply(_event(MessageType.VOICE), "reply", []) is False
    assert adapter._should_auto_tts_for_chat("123") is True


@pytest.mark.parametrize("mode", [None, "all", " voice_ONLY ", "sometimes", "null"])
def test_scope_defaults_normalization_and_invalid_fallback(tmp_path, monkeypatch, mode, caplog):
    runner, _adapter = _configured_runner(tmp_path, monkeypatch, mode)
    expects_text = mode != " voice_ONLY "
    assert runner._should_send_voice_reply(_event(MessageType.TEXT), "reply", []) is expects_text
    assert runner._should_send_voice_reply(_event(MessageType.VOICE), "reply", [], already_sent=True) is True
    if mode in {"sometimes", "null"}:
        assert "falling back to 'all'" in caplog.text


@pytest.mark.parametrize("scope", ["all", "voice_only"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("chat_mode, text_audio, voice_audio", [
    (None, None, None), ("off", False, False), ("voice_only", False, True), ("all", True, True),
])
def test_explicit_chat_choices_override_global_scope(tmp_path, monkeypatch, scope, enabled,
                                                    chat_mode, text_audio, voice_audio):
    runner, adapter = _configured_runner(tmp_path, monkeypatch, scope, enabled=enabled)
    if chat_mode is not None:
        runner._voice_mode["telegram:123"] = chat_mode
        runner._sync_voice_mode_state_to_adapter(adapter)
    else:
        text_audio, voice_audio = enabled and scope == "all", enabled
    assert runner._should_send_voice_reply(_event(MessageType.TEXT), "reply", []) is text_audio
    assert runner._should_send_voice_reply(_event(MessageType.VOICE), "reply", [], already_sent=True) is voice_audio
    assert adapter._should_auto_tts_for_chat("123") is voice_audio


def test_profile_configs_and_receiving_bot_keep_independent_scopes(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    runner, primary = _configured_runner(tmp_path, monkeypatch, "voice_only")
    secondary_home = tmp_path / "profiles" / "second"
    secondary_home.mkdir(parents=True)
    (secondary_home / "config.yaml").write_text(
        "voice:\n  auto_tts: true\n  auto_tts_mode: all\n", encoding="utf-8")
    secondary = _Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    secondary.set_owner_profile("second")
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._profile_adapters = {"second": {Platform.TELEGRAM: secondary}}
    with _profile_runtime_scope(secondary_home):
        runner._sync_voice_mode_state_to_adapter(secondary)
    runner._sync_voice_mode_state_to_adapter(primary)

    primary_event = _event(MessageType.TEXT)
    secondary_event = _event(MessageType.TEXT, profile="second")
    # A routed turn still answers through the bot that received it, not the runtime's bot.
    routed_event = _event(MessageType.TEXT, profile="second")
    routed_event.source._transport_adapter_ref = weakref.ref(primary)
    for event, expected in ((primary_event, False), (secondary_event, True), (routed_event, False)):
        assert runner._should_send_voice_reply(event, "reply", []) is expected

    runner._voice_mode = {"telegram:123": "all", "second:telegram:123": "off"}
    with _profile_runtime_scope(secondary_home):
        runner._sync_voice_mode_state_to_adapter(secondary)
    runner._sync_voice_mode_state_to_adapter(primary)
    for event, expected in ((primary_event, True), (secondary_event, False), (routed_event, True)):
        assert runner._should_send_voice_reply(event, "reply", []) is expected
    assert primary._auto_tts_enabled_chats == {"123"}
    assert secondary._auto_tts_disabled_chats == {"123"}
