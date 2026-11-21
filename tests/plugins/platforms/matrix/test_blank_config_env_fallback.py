"""A present-but-blank ``matrix:`` key in config.yaml means "unset": the env-var fallback must fire
exactly as it does when the key is absent (0.21.2 started seeding blank YAML values into
``config.extra``, which flipped the precedence and silently disabled free-response rooms)."""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key


@pytest.mark.parametrize("blank", ["", "  \t "])
def test_blank_yaml_values_fall_through_to_env(monkeypatch, blank):
    from plugins.platforms.matrix.adapter import MatrixAdapter, _extra_csv_set, _resolve_max_message_length

    monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", "!home:example.org")
    monkeypatch.setenv("MATRIX_MAX_MESSAGE_LENGTH", "9000")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    config = PlatformConfig(enabled=True, extra={
        "free_response_rooms": blank, "max_message_length": blank, "auto_thread": blank})

    assert _extra_csv_set(config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == {"!home:example.org"}
    assert _resolve_max_message_length(config) == 9000
    assert MatrixAdapter._extra_truthy(config, "auto_thread", "MATRIX_AUTO_THREAD", "true") is False


def test_explicit_env_beats_yaml_and_yaml_beats_default(monkeypatch):
    """Per-profile precedence: explicit scoped env → the profile's YAML → default. A blank env
    value is unset (it must not clobber YAML); an explicit empty list is a real "no rooms" value."""
    from plugins.platforms.matrix.adapter import MatrixAdapter, _extra_csv_set, _resolve_max_message_length

    yaml_config = PlatformConfig(enabled=True, extra={
        "free_response_rooms": ["!a:example.org", " !b:example.org "], "max_message_length": 4000,
        "auto_thread": False})

    monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", "!env:example.org")
    monkeypatch.setenv("MATRIX_MAX_MESSAGE_LENGTH", "9000")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "true")
    assert _extra_csv_set(yaml_config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == {"!env:example.org"}
    assert _resolve_max_message_length(yaml_config) == 9000
    assert MatrixAdapter._extra_truthy(yaml_config, "auto_thread", "MATRIX_AUTO_THREAD", "true") is True

    for name in ("MATRIX_FREE_RESPONSE_ROOMS", "MATRIX_MAX_MESSAGE_LENGTH", "MATRIX_AUTO_THREAD"):
        monkeypatch.setenv(name, "  ")
    assert _extra_csv_set(yaml_config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == {"!a:example.org", "!b:example.org"}
    assert _resolve_max_message_length(yaml_config) == 4000
    assert MatrixAdapter._extra_truthy(yaml_config, "auto_thread", "MATRIX_AUTO_THREAD", "true") is False

    monkeypatch.delenv("MATRIX_AUTO_THREAD")
    assert MatrixAdapter._extra_truthy(PlatformConfig(enabled=True, extra={}), "auto_thread", "MATRIX_AUTO_THREAD", "true") is True
    assert _extra_csv_set(PlatformConfig(enabled=True, extra={"free_response_rooms": []}),
                          "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == set()


@pytest.mark.asyncio
async def test_matrix_inchannel_continuable_matches_flat_room_and_dm_sessions(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from plugins.platforms.matrix.adapter import MatrixAdapter

    monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "true")
    monkeypatch.delenv("MATRIX_DM_AUTO_THREAD", raising=False)

    def configure(adapter, *, is_dm):
        adapter._resolve_room_identity = AsyncMock(
            return_value=SimpleNamespace(display_name="Room", room_topic=None, server_name=None))
        adapter._is_dm_room = AsyncMock(return_value=is_dm)
        adapter._get_display_name = AsyncMock(return_value="Alice")
        adapter._background_read_receipt = lambda *_args: None
        adapter._require_mention = False

    room_adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"session_scope": "room"}))
    configure(room_adapter, is_dm=False)
    room_ctx = await room_adapter._resolve_message_context(
        room_id="!room:example", sender="@alice:example", event_id="$event",
        body="hello", source_content={"body": "hello"}, relates_to={})

    assert room_adapter.supports_inchannel_continuable is True
    assert room_ctx is not None
    room_reply_source = room_ctx[-1]
    room_seed_source = SessionSource(
        platform=Platform.MATRIX, chat_id="!room:example", chat_type="group",
        user_id="@alice:example", thread_id=None)
    assert build_session_key(room_seed_source) == build_session_key(room_reply_source)

    monkeypatch.setenv("MATRIX_DM_AUTO_THREAD", "true")
    dm_adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"session_scope": "room"}))
    configure(dm_adapter, is_dm=True)
    dm_ctx = await dm_adapter._resolve_message_context(
        room_id="!dm:example", sender="@alice:example", event_id="$event",
        body="hello", source_content={"body": "hello"}, relates_to={})

    assert dm_adapter.supports_inchannel_continuable is False
    assert dm_ctx is not None
    dm_reply_source = dm_ctx[-1]
    dm_seed_source = SessionSource(
        platform=Platform.MATRIX, chat_id="!dm:example", chat_type="dm",
        user_id="@alice:example", thread_id=None)
    assert build_session_key(dm_seed_source) != build_session_key(dm_reply_source)
