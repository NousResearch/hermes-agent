"""Matrix outbound thread payloads and reply chaining."""
import asyncio
import types

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from tests.gateway.matrix_helpers import _make_adapter, make_fake_mautrix


class TestMatrixRenderingPayloads:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.mock_client = MagicMock()
        self.mock_client.send_message_event = AsyncMock(return_value="$evt")
        self.adapter._client = self.mock_client

    def _sent_contents(self):
        return [
            call.args[2] if len(call.args) > 2 else call.kwargs["content"]
            for call in self.mock_client.send_message_event.await_args_list
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reply_to", "is_falling_back"),
        [
            ("$other-thread", False),
            ("$root", True),
        ],
    )
    async def test_thread_payload_replies_to_any_event_except_the_root(
        self, reply_to, is_falling_back
    ):
        result = await self.adapter.send(
            "!room:example.org",
            "threaded reply",
            reply_to=reply_to,
            metadata={"thread_id": "$root"},
        )

        assert result.success is True
        assert self._sent_contents()[0]["m.relates_to"] == {
            "rel_type": "m.thread",
            "event_id": "$root",
            "m.in_reply_to": {"event_id": reply_to},
            "is_falling_back": is_falling_back,
        }

    @pytest.mark.asyncio
    async def test_split_threaded_reply_continues_after_the_first_chunk(self):
        self.adapter.max_message_length = 60
        self.mock_client.send_message_event = AsyncMock(
            side_effect=lambda *args: (
                f"$sent-{self.mock_client.send_message_event.await_count}"
            )
        )

        result = await self.adapter.send(
            "!room:example.org",
            "one two three four five " * 15,
            reply_to="$incoming",
            metadata={"thread_id": "$root"},
        )

        relations = [content["m.relates_to"] for content in self._sent_contents()]
        assert result.success is True
        assert len(relations) > 1
        assert relations == [
            {
                "rel_type": "m.thread",
                "event_id": "$root",
                "m.in_reply_to": {"event_id": "$incoming"},
                "is_falling_back": False,
            },
            *[
                {
                    "rel_type": "m.thread",
                    "event_id": "$root",
                    "m.in_reply_to": {"event_id": f"$sent-{index}"},
                    "is_falling_back": True,
                }
                for index in range(1, len(relations))
            ],
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("steps", "expected"),
        [
            pytest.param(
                [
                    ("inbound", "$q1"),
                    ("send", "$q1"),
                    ("inbound", "$q2"),
                    ("send", "$q2"),
                    ("send", "$q1"),
                ],
                [("$q1", False), ("$q2", False), ("$sent-2", True)],
                id="busy-ack-between-messages-of-a-response",
            ),
            pytest.param(
                [
                    ("inbound", "$a"),
                    ("inbound", "$b"),
                    ("send", "$a"),
                    ("send", "$b"),
                    ("send", "$a"),
                    ("send", "$b"),
                ],
                [("$a", False), ("$b", False), ("$sent-2", True), ("$sent-3", True)],
                id="interleaved-responses",
            ),
            pytest.param(
                [
                    ("inbound", "$incoming"),
                    ("send", "$incoming"),
                    ("inbound", "$other-user"),
                    ("send", "$sent-1"),
                ],
                [("$incoming", False), ("$other-user", True)],
                id="inbound-post-between-stream-chunks",
            ),
        ],
    )
    async def test_interleaved_thread_events_do_not_repeat_a_reply(
        self, steps, expected
    ):
        self.mock_client.send_message_event = AsyncMock(
            side_effect=lambda *args: (
                f"$sent-{self.mock_client.send_message_event.await_count}"
            )
        )

        for kind, event_id in steps:
            if kind == "inbound":
                self.adapter._thread_fallbacks.remember(
                    "!room:example.org", "$root", event_id
                )
                continue

            await self.adapter.send(
                "!room:example.org",
                "message",
                reply_to=event_id,
                metadata={"thread_id": "$root"},
            )

        assert [content["m.relates_to"] for content in self._sent_contents()] == [
            {
                "rel_type": "m.thread",
                "event_id": "$root",
                "m.in_reply_to": {"event_id": target},
                "is_falling_back": is_falling_back,
            }
            for target, is_falling_back in expected
        ]

    @pytest.mark.asyncio
    async def test_thread_payload_accepts_explicit_fallback_anchor(self):
        result = await self.adapter.send(
            "!room:example.org",
            "threaded fallback",
            metadata={
                "thread_id": "$root",
                "matrix_thread_fallback_event_id": "$latest",
            },
        )

        assert result.success is True
        assert self._sent_contents()[0]["m.relates_to"] == {
            "rel_type": "m.thread",
            "event_id": "$root",
            "m.in_reply_to": {"event_id": "$latest"},
            "is_falling_back": True,
        }

    @pytest.mark.asyncio
    async def test_thread_replies_chain_from_inbound_event_and_preserve_explicit_target(
        self,
    ):
        self.adapter._is_dm_room = AsyncMock(return_value=True)
        self.adapter._resolve_room_identity = AsyncMock(
            return_value=types.SimpleNamespace(
                display_name="Alice",
                room_topic="",
                server_name="example.org",
                members_digest=None,
            )
        )
        self.adapter._get_display_name = AsyncMock(return_value="Alice")
        self.adapter._background_read_receipt = MagicMock()
        self.adapter.max_message_length = 60
        self.mock_client.send_message_event = AsyncMock(
            side_effect=lambda *args: (
                f"$sent-{self.mock_client.send_message_event.await_count}"
            )
        )

        await self.adapter._resolve_message_context(
            "!room:example.org",
            "@alice:example.org",
            "$incoming",
            "continue",
            {"msgtype": "m.text", "body": "continue"},
            {"rel_type": "m.thread", "event_id": "$root"},
        )
        result = await self.adapter.send(
            "!room:example.org",
            "one two three four five " * 15,
            metadata={"thread_id": "$root"},
        )

        contents = self._sent_contents()
        assert result.success is True
        assert len(contents) > 1
        assert [
            content["m.relates_to"]["m.in_reply_to"]["event_id"] for content in contents
        ] == [
            "$incoming",
            *[f"$sent-{index}" for index in range(1, len(contents))],
        ]

        await self.adapter.send(
            "!room:example.org",
            "explicit",
            reply_to="$other-thread",
            metadata={"thread_id": "$root"},
        )
        explicit_relation = self._sent_contents()[-1]["m.relates_to"]
        assert explicit_relation == {
            "rel_type": "m.thread",
            "event_id": "$root",
            "m.in_reply_to": {"event_id": "$other-thread"},
            "is_falling_back": False,
        }

        await self.adapter.send(
            "!room:example.org", "after", metadata={"thread_id": "$root"}
        )
        assert self._sent_contents()[-1]["m.relates_to"] == {
            "rel_type": "m.thread",
            "event_id": "$root",
            "m.in_reply_to": {"event_id": f"$sent-{len(self._sent_contents()) - 1}"},
            "is_falling_back": True,
        }


class TestMatrixUploadAndSend:
    @pytest.mark.asyncio
    async def test_encrypted_and_plain_media_extend_thread_fallback_chain(
        self, tmp_path
    ):
        adapter = _make_adapter()
        adapter._encryption = True
        mock_client = MagicMock()
        mock_client.crypto = object()
        mock_client.state_store.is_encrypted = AsyncMock(
            side_effect=[True, False, False, False]
        )
        mock_client.upload_media = AsyncMock(
            side_effect=[
                "mxc://example.org/secret",
                "mxc://example.org/plain",
                "mxc://example.org/one",
                "mxc://example.org/two",
            ]
        )
        mock_client.send_message_event = AsyncMock(
            side_effect=[
                "$text",
                "$encrypted",
                "$plain",
                "$image-one",
                "$image-two",
            ]
        )
        adapter._client = mock_client
        metadata = {"thread_id": "$root"}
        first = tmp_path / "one.png"
        second = tmp_path / "two.png"
        first.write_bytes(b"one")
        second.write_bytes(b"two")

        with patch.dict("sys.modules", make_fake_mautrix()):
            text_result = await adapter.send(
                "!room:example.org", "start", metadata=metadata
            )
            encrypted_result = await adapter._upload_and_send(
                "!room:example.org",
                b"secret",
                "secret.png",
                "image/png",
                "m.image",
                metadata=metadata,
            )
            plain_result = await adapter._upload_and_send(
                "!room:example.org",
                b"plain",
                "plain.png",
                "image/png",
                "m.image",
                metadata=metadata,
            )
            image_result = await adapter.send_multiple_images(
                "!room:example.org",
                [(first.as_uri(), "one"), (second.as_uri(), "two")],
                metadata=metadata,
            )

        contents = [
            call.args[2] for call in mock_client.send_message_event.await_args_list
        ]
        assert (
            text_result.success,
            encrypted_result.success,
            plain_result.success,
            image_result.success,
        ) == (True, True, True, True)
        assert [
            content["m.relates_to"]["m.in_reply_to"]["event_id"] for content in contents
        ] == [
            "$root",
            "$text",
            "$encrypted",
            "$plain",
            "$image-one",
        ]
        assert ["file" in content for content in contents] == [
            False,
            True,
            False,
            False,
            False,
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["buffered", "live"])
async def test_streamed_threaded_reply_continues_after_the_first_message(delivery):
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
                "max_message_length": 1000,
            },
        )
    )
    client = MagicMock()
    client.send_message_event = AsyncMock(
        side_effect=[f"$sent-{index}" for index in range(20)]
    )
    adapter._client = client
    preview_sent = asyncio.Event()
    consumer = GatewayStreamConsumer(
        adapter,
        "!room:example.org",
        StreamConsumerConfig(edit_interval=0, cursor=""),
        metadata={"thread_id": "$root"},
        initial_reply_to_id="$incoming",
        on_new_message=preview_sent.set,
    )

    task = asyncio.create_task(consumer.run())
    try:
        if delivery == "live":
            consumer.on_delta("preview " * 25)
            await asyncio.wait_for(preview_sent.wait(), timeout=5)
        consumer.on_delta("answer " * 350)
        consumer.finish()
        await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()

    messages = [
        (f"$sent-{index}", call.args[2]["m.relates_to"])
        for index, call in enumerate(client.send_message_event.await_args_list)
        if call.args[2]["m.relates_to"].get("rel_type") != "m.replace"
    ]
    assert len(messages) > 2
    assert [relation for _, relation in messages] == [
        {
            "rel_type": "m.thread",
            "event_id": "$root",
            "m.in_reply_to": {"event_id": "$incoming"},
            "is_falling_back": False,
        },
        *[
            {
                "rel_type": "m.thread",
                "event_id": "$root",
                "m.in_reply_to": {"event_id": previous},
                "is_falling_back": True,
            }
            for previous, _ in messages[:-1]
        ],
    ]
