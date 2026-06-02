import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.stream_consumer import _NEW_SEGMENT

from agent.memory_manager import sanitize_context
from gateway.run import _sanitize_gateway_final_response
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class TestSanitizeContextExtraSystemNotes:
    def test_strips_resume_note(self):
        text = (
            "before\n"
            "[System note: Your previous turn in this session was interrupted by a gateway shutdown. "
            "The conversation history below is intact. If it contains unfinished tool result(s), "
            "process them first and summarize what was accomplished, then address the user's new message below.]\n\n"
            "after"
        )
        out = sanitize_context(text)
        assert "Your previous turn in this session was interrupted" not in out
        assert "before" in out
        assert "after" in out


class TestGatewayFinalResponseSanitization:
    def test_gateway_final_response_strips_memory_context(self):
        text = (
            "before\n"
            "<memory-context>\n[System note: The following is recalled memory context, "
            "NOT new user input. Treat as authoritative reference data — this is the agent's persistent memory and should inform all responses.]\n\nsecret\n"
            "</memory-context>\nafter"
        )
        out = _sanitize_gateway_final_response("telegram", text)
        assert "memory-context" not in out.lower()
        assert "secret" not in out
        assert "before" in out
        assert "after" in out

    def test_unclosed_memory_context_strips_payload(self):
        text = "before <memory-context>\nsecret"
        out = _sanitize_gateway_final_response("telegram", text)
        assert "memory-context" not in out.lower()
        assert "secret" not in out
        assert out == "before"

    def test_reversed_memory_context_strips_payload(self):
        text = "before </memory-context>secret<memory-context> after"
        out = _sanitize_gateway_final_response("telegram", text)
        assert "memory-context" not in out.lower()
        assert "secret" not in out
        assert "before" in out
        assert "after" in out

    def test_orphan_close_memory_context_strips_payload(self):
        text = "before </ memory-context >secret after"
        out = _sanitize_gateway_final_response("telegram", text)
        assert "memory-context" not in out.lower()
        assert "secret" not in out
        assert out == "before"


class TestGatewayStreamConsumerMemoryContext:
    async def _run_consumer(self, deltas):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        )
        task = asyncio.create_task(consumer.run())
        for d in deltas:
            if d is _NEW_SEGMENT:
                consumer.on_segment_break()
            else:
                consumer.on_delta(d)
        consumer.finish()
        await task
        return adapter

    def test_streamed_memory_context_is_not_sent(self):
        deltas = [
            "hello ",
            "<memory-context>\n[System note: The following is recalled memory context, ",
            "NOT new user input. Treat as authoritative reference data — this is the agent's persistent memory and should inform all responses.]\n\nsecret\n",
            "</memory-context> world",
        ]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_inline_streamed_memory_context_is_not_sent(self):
        deltas = [
            "hello <memory-context>\nsecret\n",
            "</memory-context> world",
        ]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_unclosed_streamed_memory_context_is_not_sent(self):
        deltas = ["hello ", "<memory-context>\nsecret"]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible

    def test_streamed_memory_context_whitespace_variant_is_not_sent(self):
        deltas = [
            "hello < memory-",
            "context >\nsecret\n",
            "</ memory-context > world",
        ]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_unclosed_streamed_memory_context_whitespace_variant_is_not_sent(self):
        deltas = ["hello < memory-context >\nsecret"]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible

    def test_streamed_memory_context_long_whitespace_split_is_not_sent(self):
        spaces = " " * 50
        deltas = [
            f"hello <{spaces}",
            f"memory-context>secret</{spaces}memory-context> world",
        ]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_streamed_orphan_close_context_is_not_sent(self):
        deltas = ["hello </memory-context>secret world"]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" not in visible

    def test_streamed_repeated_orphan_close_context_stays_dropped(self):
        deltas = ["hello </memory-context>secret</memory-context> after"]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "after" not in visible
        assert "hello" in visible

    def test_streamed_memory_context_split_across_segment_break_is_not_sent(self):
        deltas = [
            "hello <",
            _NEW_SEGMENT,
            "memory-context>secret</memory-context> world",
        ]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_streamed_memory_context_whitespace_split_across_segment_break_is_not_sent(self):
        spaces = " " * 50
        deltas = [
            f"hello <{spaces}",
            _NEW_SEGMENT,
            f"memory-context>secret</{spaces}memory-context> world",
        ]

        adapter = asyncio.run(self._run_consumer(deltas))
        visible = "\n".join(
            [
                call.kwargs.get("content", "")
                for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
            ]
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_memory_context_split_across_commentary_boundary_is_not_sent(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        )

        async def _run():
            task = asyncio.create_task(consumer.run())
            consumer.on_delta("hello <")
            consumer.on_commentary("memory-context>secret</memory-context> world")
            consumer.finish()
            await task

        asyncio.run(_run())
        visible = "\n".join(call.kwargs.get("content", "") for call in adapter.send.await_args_list)
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_memory_context_whitespace_split_across_commentary_boundary_is_not_sent(self):
        spaces = " " * 50
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        )

        async def _run():
            task = asyncio.create_task(consumer.run())
            consumer.on_delta(f"hello <{spaces}")
            consumer.on_commentary(f"memory-context>secret</{spaces}memory-context> world")
            consumer.finish()
            await task

        asyncio.run(_run())
        visible = "\n".join(call.kwargs.get("content", "") for call in adapter.send.await_args_list)
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_memory_context_split_from_commentary_to_delta_is_not_sent(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        )

        async def _run():
            task = asyncio.create_task(consumer.run())
            consumer.on_commentary("hello <")
            consumer.on_delta("memory-context>secret</memory-context> world")
            consumer.finish()
            await task

        asyncio.run(_run())
        visible = "\n".join(
            call.kwargs.get("content", "")
            for call in adapter.send.await_args_list + adapter.edit_message.await_args_list
        )
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_memory_context_split_across_two_commentary_messages_is_not_sent(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        )

        async def _run():
            task = asyncio.create_task(consumer.run())
            consumer.on_commentary("hello <")
            consumer.on_commentary("memory-context>secret</memory-context> world")
            consumer.finish()
            await task

        asyncio.run(_run())
        visible = "\n".join(call.kwargs.get("content", "") for call in adapter.send.await_args_list)
        assert "memory-context" not in visible.lower()
        assert "secret" not in visible
        assert "hello" in visible
        assert "world" in visible

    def test_commentary_resume_note_is_not_sent(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        )

        async def _run():
            task = asyncio.create_task(consumer.run())
            consumer.on_commentary(
                "[System note: Your previous turn in this session was interrupted by a gateway shutdown. "
                "The conversation history below is intact. If it contains unfinished tool result(s), "
                "process them first and summarize what was accomplished, then address the user's new message below.]\n\nreal text"
            )
            consumer.finish()
            await task

        asyncio.run(_run())
        sent = "\n".join(call.kwargs.get("content", "") for call in adapter.send.await_args_list)
        assert "Your previous turn in this session was interrupted" not in sent
        assert "real text" in sent
