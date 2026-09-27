"""Matrix reply references and thread fallbacks."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import Platform, PlatformConfig
from gateway.run import _profile_runtime_scope
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
async def test_reply_modes_control_split_chunks_without_losing_thread_relations():
    room_id = "!room:example.org"
    expected = {
        "off": [None, None, None],
        "first": [
            {"m.in_reply_to": {"event_id": "$request"}},
            None,
            None,
        ],
        "all": [
            {"m.in_reply_to": {"event_id": "$request"}},
            {"m.in_reply_to": {"event_id": "$request"}},
            {"m.in_reply_to": {"event_id": "$request"}},
        ],
    }
    expected_thread = {
        "off": [
            {
                "rel_type": "m.thread",
                "event_id": "$root",
                "m.in_reply_to": {"event_id": "$known"},
                "is_falling_back": True,
            },
        ]
        * 3,
        "first": [
            {
                "rel_type": "m.thread",
                "event_id": "$root",
                "m.in_reply_to": {"event_id": "$request"},
                "is_falling_back": False,
            },
            *[
                {
                    "rel_type": "m.thread",
                    "event_id": "$root",
                    "m.in_reply_to": {"event_id": "$known"},
                    "is_falling_back": True,
                },
            ]
            * 2,
        ],
        "all": [
            {
                "rel_type": "m.thread",
                "event_id": "$root",
                "m.in_reply_to": {"event_id": "$request"},
                "is_falling_back": False,
            },
        ]
        * 3,
    }

    for mode in ("off", "first", "all"):
        config = PlatformConfig(
            enabled=True,
            token="syt_test",
            reply_to_mode=mode,
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
        adapter = MatrixAdapter(config)
        client = MagicMock()
        client.send_message_event = AsyncMock(return_value="$sent")
        adapter._client = client

        with patch.object(
            adapter, "truncate_message", return_value=["one", "two", "three"]
        ):
            await adapter.send(room_id, "split answer", reply_to="$request")
            plain = [
                call.args[2].get("m.relates_to")
                for call in client.send_message_event.await_args_list
            ]

            client.send_message_event.reset_mock()
            await adapter.send(
                room_id,
                "split thread answer",
                reply_to="$request",
                metadata={
                    "thread_id": "$root",
                    "matrix_thread_fallback_event_id": "$known",
                },
            )
            threaded = [
                call.args[2]["m.relates_to"]
                for call in client.send_message_event.await_args_list
            ]

        assert (plain, threaded) == (expected[mode], expected_thread[mode])


def test_off_mode_keeps_a_valid_fallback_on_threaded_media():
    config = PlatformConfig(
        enabled=True,
        token="syt_test",
        reply_to_mode="off",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    content = {"msgtype": "m.image", "body": "photo"}

    adapter._apply_relation_metadata(
        "!room:example.org",
        content,
        reply_to="$specific",
        metadata={"thread_id": "$root"},
    )

    assert (adapter._reply_to_mode, content) == (
        "off",
        {
            "msgtype": "m.image",
            "body": "photo",
            "m.relates_to": {
                "rel_type": "m.thread",
                "event_id": "$root",
                "m.in_reply_to": {"event_id": "$root"},
                "is_falling_back": True,
            },
        },
    )


def test_reply_mode_follows_each_profile_config_in_multiplex(monkeypatch, tmp_path):
    root = tmp_path / "hermes"
    secondary = root / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "matrix:\n  enabled: true\n  reply_to_mode: all\n"
    )
    (secondary / "config.yaml").write_text(
        "matrix:\n  enabled: true\n  reply_to_mode: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("MATRIX_REPLY_TO_MODE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    set_multiplex_active(True)
    try:
        from gateway.config import load_gateway_config

        def current_mode():
            config = load_gateway_config().platforms[Platform.MATRIX]
            return config.reply_to_mode, MatrixAdapter(config)._reply_to_mode

        default_before = current_mode()
        with _profile_runtime_scope(secondary, prepared_secret_scope={}):
            served = current_mode()
        default_after = current_mode()
    finally:
        set_multiplex_active(False)

    assert (default_before, served, default_after) == (
        ("all", "all"),
        ("off", "off"),
        ("all", "all"),
    )


@pytest.mark.parametrize(("env_override", "expected"), [(None, "off"), ("all", "all")])
def test_reply_mode_uses_highest_priority_yaml_block(
    monkeypatch, tmp_path, env_override, expected
):
    from gateway.config import load_gateway_config

    (tmp_path / "config.yaml").write_text(
        "gateway:\n  platforms:\n    matrix:\n      reply_to_mode: all\n"
        "platforms:\n  matrix:\n    reply_to_mode: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if env_override is None:
        monkeypatch.delenv("MATRIX_REPLY_TO_MODE", raising=False)
    else:
        monkeypatch.setenv("MATRIX_REPLY_TO_MODE", env_override)

    config = load_gateway_config().platforms[Platform.MATRIX]

    assert (config.reply_to_mode, MatrixAdapter(config)._reply_to_mode) == (
        expected,
        expected,
    )
