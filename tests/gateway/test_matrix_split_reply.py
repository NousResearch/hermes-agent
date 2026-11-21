"""Matrix split-send reply relations: ``reply_to_mode`` decides which chunks carry the
``m.in_reply_to`` fallback, while ``m.thread`` stays on every chunk."""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    return MatrixAdapter(config)


class TestMatrixSplitReplyRelations:
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
    @pytest.mark.parametrize("mode, fallback_chunks", [
        ('"off"', []), ("off", []), ("first", [0]), ("all", [0, 1]),
        ('" ALL "', [0, 1]), ("invalid", [0]),
    ])
    @pytest.mark.parametrize("thread_id", [None, "$root"])
    async def test_split_reply_modes(self, mode, fallback_chunks, thread_id, tmp_path):
        from gateway.config import PlatformConfig
        from plugins.platforms.matrix.adapter import MatrixAdapter

        # PyYAML is not a project dependency. These are the values a YAML 1.1
        # load of the parametrized source text produces, including unquoted off.
        parsed = {
            '"off"': "off",
            "off": False,
            "first": "first",
            "all": "all",
            '" ALL "': " ALL ",
            "invalid": "invalid",
        }[mode]
        config = PlatformConfig.from_dict({"reply_to_mode": parsed})
        adapter = MatrixAdapter(config)
        adapter._client = self.adapter._client
        text = "line\n" * (adapter.max_message_length // len("line\n") + 200)
        result = await adapter.send("!room:example.org", text, reply_to="$parent",
                                    metadata={"thread_id": thread_id})
        assert result.success
        contents = self._sent_contents()
        assert len(contents) == 2
        for i, content in enumerate(contents):
            relation = content.get("m.relates_to", {})
            fallback = i in fallback_chunks
            assert ("m.in_reply_to" in relation) == fallback
            assert ("is_falling_back" in relation) == bool(thread_id and fallback)
            if fallback:
                assert relation["m.in_reply_to"] == {"event_id": "$parent"}
            if thread_id:
                assert relation["event_id"] == thread_id
                assert relation["rel_type"] == "m.thread"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["off", "first", "all"])
    @pytest.mark.parametrize("thread_id", [None, "$root"])
    async def test_document_honors_reply_mode(self, mode, thread_id, tmp_path):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        config = self.adapter.config
        config.reply_to_mode = mode
        adapter = MatrixAdapter(config)
        adapter._client = self.mock_client
        adapter._encryption = False
        self.mock_client.upload_media = AsyncMock(return_value="mxc://example.org/file")
        document = tmp_path / "answer.txt"
        document.write_text("answer")

        result = await adapter.send_document(
            "!room:example.org", str(document), reply_to="$parent",
            metadata={"thread_id": thread_id},
        )

        assert result.success
        relation = self._sent_contents()[0].get("m.relates_to", {})
        if mode == "off":
            assert "m.in_reply_to" not in relation
            assert "is_falling_back" not in relation
        else:
            assert relation["m.in_reply_to"] == {"event_id": "$parent"}
        if thread_id:
            assert relation["rel_type"] == "m.thread"
            assert relation["event_id"] == thread_id

    @pytest.mark.parametrize("thread_id", [None, "$root"])
    def test_include_reply_fallback_false_suppresses_explicit_reply(self, thread_id):
        payload = {"body": "tail"}
        self.adapter._apply_relation_metadata(
            payload, reply_to="$parent", metadata={"thread_id": thread_id},
            include_reply_fallback=False,
        )
        relation = payload.get("m.relates_to", {})
        assert "m.in_reply_to" not in relation
        assert "is_falling_back" not in relation
        if thread_id:
            assert relation["rel_type"] == "m.thread"
            assert relation["event_id"] == thread_id
        else:
            assert "m.relates_to" not in payload

    @pytest.mark.asyncio
    @pytest.mark.parametrize("retry_chunk", [0, 1])
    async def test_split_reply_key_share_retry_preserves_chunk_relation(self, retry_chunk):
        self.adapter._encryption = True
        self.adapter._client.crypto = types.SimpleNamespace(share_keys=AsyncMock())
        metadata = {"thread_id": "$root"}
        attempts = []
        failed = False

        async def send_chunk(room_id, payload):
            nonlocal failed
            attempts.append(payload.copy())
            if len(attempts) == retry_chunk + 1 and not failed:
                failed = True
                raise RuntimeError("share keys first")
            return f"$sent-{len(attempts)}"

        # Use deterministic split boundaries; exercise send's real retry path.
        with patch.object(self.adapter, "truncate_message", return_value=["first", "tail"]), \
             patch.object(self.adapter, "_send_room_message", side_effect=send_chunk):
            result = await self.adapter.send(
                "!room:example.org", "answer", reply_to="$parent", metadata=metadata)

        assert result.success
        assert result.message_id == "$sent-3"
        assert len(attempts) == 3
        assert attempts[retry_chunk] == attempts[retry_chunk + 1]
        self.adapter._client.crypto.share_keys.assert_awaited_once()
        for payload in attempts:
            relation = payload["m.relates_to"]
            assert relation["rel_type"] == "m.thread"
            assert relation["event_id"] == "$root"
            if payload["body"] == "first":
                assert relation["m.in_reply_to"] == {"event_id": "$parent"}
                assert relation["is_falling_back"] is True
            else:
                assert "m.in_reply_to" not in relation
                assert "is_falling_back" not in relation
        assert metadata == {"thread_id": "$root"}

    @pytest.mark.asyncio
    async def test_long_response_split_replies_only_on_first_chunk(self):
        # Exceed configurable outbound chunk size (default 16k since #53026).
        repeats = (self.adapter.max_message_length // 5) + 200
        long_text = "line\n" * repeats

        result = await self.adapter.send(
            "!room:example.org",
            long_text,
            reply_to="$user-message",
            metadata={"thread_id": "$root"},
        )

        assert result.success is True
        contents = self._sent_contents()
        assert len(contents) > 1
        assert contents[0]["m.relates_to"]["m.in_reply_to"] == {
            "event_id": "$user-message"
        }
        for content in contents[1:]:
            assert content["m.relates_to"]["rel_type"] == "m.thread"
            assert content["m.relates_to"]["event_id"] == "$root"
            assert "m.in_reply_to" not in content["m.relates_to"]

    @pytest.mark.asyncio
    async def test_long_response_split_plain_reply_only_on_first_chunk(self):
        """Plain (non-thread) multi-chunk replies only quote the parent once."""
        repeats = (self.adapter.max_message_length // 5) + 200
        long_text = "line\n" * repeats

        result = await self.adapter.send(
            "!room:example.org",
            long_text,
            reply_to="$user-message",
        )

        assert result.success is True
        contents = self._sent_contents()
        assert len(contents) > 1
        assert contents[0]["m.relates_to"] == {
            "m.in_reply_to": {"event_id": "$user-message"}
        }
        for content in contents[1:]:
            assert "m.relates_to" not in content

    @pytest.mark.asyncio
    async def test_long_response_split_preserves_thread_context(self):
        # Build a payload guaranteed to exceed the adapter's outbound chunk
        # size (configurable since #53026) so send() must split it.
        repeats = (self.adapter.max_message_length // 15) + 200
        long_text = "Intro\n```python\n" + ("print('hello')\n" * repeats) + "```\nDone"

        result = await self.adapter.send(
            "!room:example.org",
            long_text,
            metadata={"thread_id": "$root"},
        )

        assert result.success is True
        contents = self._sent_contents()
        assert len(contents) > 1
        for index, content in enumerate(contents):
            assert content["m.relates_to"]["rel_type"] == "m.thread"
            assert content["m.relates_to"]["event_id"] == "$root"
            if index == 0:
                assert content["m.relates_to"]["m.in_reply_to"] == {"event_id": "$root"}
            else:
                assert "m.in_reply_to" not in content["m.relates_to"]
            assert content["body"].count("```") % 2 == 0
