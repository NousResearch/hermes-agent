import asyncio
import json
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from tests.plugins.platforms.feishu_native_cards.support import make_adapter, add_turn, response, event


class CardTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_drafts_and_final_keep_one_message_and_fold_trace(self):
        adapter, _hub, ops = make_adapter()
        turn = add_turn(adapter, thread="topic")
        md = {"thread_id": "topic", "reply_to_message_id": "user_a"}
        turn.calls["call"] = {"name": "read_file", "status": "done", "round": 1, "duration": 0.2}
        self.assertTrue((await adapter.send_draft("chat_a", 10, "draft", md)).success)
        self.assertTrue((await adapter.send_draft("chat_a", 10, "draft more", md)).success)
        result = await adapter.send(
            "chat_a", "final", reply_to="user_a", metadata={**md, "notify": True}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "im_test")
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
        self.assertEqual(
            adapter._feishu_send_with_retry.await_args.kwargs["metadata"], {"thread_id": "topic"}
        )
        card = json.loads(ops[-1][1].request_body.card.data)
        self.assertFalse(card["config"]["streaming_mode"])
        self.assertEqual(card["body"]["elements"][1]["expanded"], False)
        before = len(ops)
        await adapter.send_draft("chat_a", 10, "late", md)
        self.assertEqual(len(ops), before)
        self.assertEqual(turn.text, "final")

    async def test_short_final_without_a_draft_is_a_completed_card(self):
        adapter, _hub, ops = make_adapter()
        add_turn(adapter)
        result = await adapter.send("chat_a", "好", reply_to="user_a", metadata={"notify": True})
        self.assertTrue(result.success)
        self.assertEqual([op for op, _ in ops], ["create", "update"])

    async def test_long_chinese_response_stays_one_bounded_card(self):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter, _hub, ops = make_adapter()
        add_turn(adapter)
        content = "开头" + "长文本" * 80_000 + "结尾"
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_a",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor="", transport="auto"),
            initial_reply_to_id="user_a",
        )
        consumer.on_delta(content)
        consumer.finish(final_text=content)
        await consumer.run()
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
        final_payload = ops[-1][1].request_body.card.data
        self.assertLess(len(final_payload.encode()), 200_000)
        self.assertIn("开头", final_payload)
        self.assertIn("结尾", final_payload)
        self.assertIn("省略", final_payload)

    async def test_edit_transport_preview_is_not_finalized_early(self):
        adapter, _hub, _ops = make_adapter()
        turn = add_turn(adapter)
        first = await adapter.send(
            "chat_a", "preview", reply_to="user_a", metadata={"expect_edits": True}
        )
        self.assertTrue(first.success)
        self.assertFalse(turn.final)
        await adapter.edit_message("chat_a", "im_test", "final", finalize=True)
        self.assertTrue(turn.final)

    async def test_explicit_cardkit_rejection_patches_same_reference(self):
        adapter, _hub, ops = make_adapter()
        turn = add_turn(adapter)
        adapter._client.cardkit.v1.card_element.content = lambda request: NS(
            success=lambda: False, code=300317, msg="sequence compare failed"
        )
        result = await adapter.send_draft("chat_a", 1, "draft", {"reply_to_message_id": "user_a"})
        self.assertTrue(result.success)
        self.assertTrue(turn.fallback)
        self.assertEqual(ops[-1][0], "patch")
        self.assertEqual(ops[-1][1].message_id, "im_test")
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
        self.assertTrue((await adapter.send("chat_a", "done", reply_to="user_a")).success)
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)

    async def test_unknown_reference_ack_refuses_second_message(self):
        adapter, _hub, _ops = make_adapter()
        turn = add_turn(adapter)
        adapter._feishu_send_with_retry.side_effect = TimeoutError(
            "simulated uncertain acknowledgement"
        )
        self.assertFalse(
            (
                await adapter.send_draft("chat_a", 1, "draft", {"reply_to_message_id": "user_a"})
            ).success
        )
        self.assertFalse((await adapter.send("chat_a", "final", reply_to="user_a")).success)
        self.assertTrue(turn.uncertain)
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)

    async def test_sdk_retry_reuses_uuid_sequence_and_reserves_unknown_write(self):
        adapter, _hub, ops = make_adapter()
        requests = []

        def mutate(request):
            requests.append(request)
            if len(requests) < 3:
                raise TimeoutError("simulated response loss")
            return response()

        adapter._client.cardkit.v1.card_element.content = mutate
        await adapter._stream_cardkit_content("c", "body")
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(req is requests[0] for req in requests))
        await adapter._set_cardkit_streaming_mode("c", False)
        self.assertGreater(ops[-1][1].request_body.sequence, requests[0].request_body.sequence)

    async def test_streaming_window_reopens_only_nonterminal_card(self):
        adapter, _hub, ops = make_adapter()
        requests = []

        def content(request):
            requests.append(request)
            return (
                NS(success=lambda: False, code=300309, msg="streaming mode is closed")
                if len(requests) == 1
                else response()
            )

        adapter._client.cardkit.v1.card_element.content = content
        await adapter._stream_cardkit_content_with_recovery("live", "body")
        self.assertEqual([name for name, _ in ops], ["settings"])
        adapter._cardkit_terminal.add("terminal")
        requests.clear()
        with self.assertRaises(RuntimeError):
            await adapter._stream_cardkit_content_with_recovery("terminal", "late")
        self.assertEqual(len(ops), 1)

    async def test_failed_compatibility_patch_never_creates_second_message(self):
        adapter, _hub, _ops = make_adapter()
        add_turn(adapter)
        adapter._client.cardkit.v1.card_element.content = lambda request: NS(
            success=lambda: False, code=300317, msg="compare failed"
        )
        adapter._client.im.v1.message.patch = lambda request: NS(
            success=lambda: False, code=1, msg="patch refused"
        )
        md = {"reply_to_message_id": "user_a"}
        self.assertFalse((await adapter.send_draft("chat_a", 1, "draft", md)).success)
        self.assertFalse((await adapter.send("chat_a", "final", reply_to="user_a")).success)
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)

    async def test_reference_retry_reuses_the_same_uuid(self):
        from unittest.mock import patch

        adapter, _hub, _ops = make_adapter()
        seen = []

        def reply(request):
            seen.append(request.request_body.uuid)
            if len(seen) == 1:
                raise TimeoutError("simulated lost acknowledgement")
            return response()

        adapter._client.im.v1.message.reply = reply
        with patch("plugins.platforms.feishu.native_cards.sending.asyncio.sleep", new=AsyncMock()):
            result = await type(adapter)._feishu_send_with_retry(
                adapter,
                chat_id="chat_a",
                msg_type="interactive",
                payload="{}",
                reply_to="user_a",
                metadata={},
            )
        self.assertTrue(result.success())
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1])

    async def test_actual_consumer_cancellation_settles_and_seals_without_duplicate(self):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter, _hub, _ops = make_adapter()
        turn = add_turn(adapter)
        entered, release = asyncio.Event(), asyncio.Event()

        async def send(**kwargs):
            entered.set()
            await release.wait()
            return response()

        adapter._feishu_send_with_retry.side_effect = send
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_a",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor="", transport="auto"),
            initial_reply_to_id="user_a",
        )
        consumer.on_delta("draft")
        task = asyncio.create_task(consumer.run())
        await entered.wait()
        task.cancel()
        release.set()
        await task
        self.assertTrue(turn.final)
        self.assertEqual(turn.message_id, "im_test")
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)

    async def test_disconnect_retires_open_card_and_clarify_controls(self):
        from tools import clarify_gateway as cg
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from unittest.mock import patch

        adapter, _hub, _ops = make_adapter()
        turn = add_turn(adapter)
        await adapter.send_draft("chat_a", 1, "draft", {"reply_to_message_id": "user_a"})
        cg.register("shutdown", "shutdown_session", "question", None)
        try:
            await adapter.send_clarify(
                "chat_a",
                "question",
                None,
                "shutdown",
                "shutdown_session",
                {"__hermes_prompt_user_ids": ("owner_a",)},
            )
            with patch.object(FeishuAdapter, "disconnect", new=AsyncMock()):
                await adapter.disconnect()
            self.assertTrue(turn.final)
            self.assertFalse(adapter._clarify_state)
            with patch.object(FeishuAdapter, "connect", new=AsyncMock(return_value=True)):
                self.assertTrue(await adapter.connect(is_reconnect=True))
            self.assertFalse(adapter._cards_closing)
        finally:
            cg.clear_session("shutdown_session")

    async def test_cancel_during_reference_send_keeps_identity_until_settled(self):
        adapter, _hub, _ops = make_adapter()
        turn = add_turn(adapter)
        entered, release = asyncio.Event(), asyncio.Event()

        async def send(**kwargs):
            entered.set()
            await release.wait()
            return response()

        adapter._feishu_send_with_retry.side_effect = send
        task = asyncio.create_task(
            adapter.send_draft("chat_a", 1, "draft", {"reply_to_message_id": "user_a"})
        )
        await entered.wait()
        task.cancel()
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(turn.message_id, "im_test")
        await adapter.abandon_open_draft("chat_a", "draft", {"reply_to_message_id": "user_a"})
        self.assertTrue(turn.final)
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)

    async def test_real_sdk_worker_cancellation_waits_for_mutation(self):
        adapter, _hub, _ops = make_adapter()
        started, release = threading.Event(), threading.Event()
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter._run_blocking = FeishuAdapter._run_blocking.__get__(adapter)

        def mutate(request):
            started.set()
            if not release.wait(5):
                raise TimeoutError("test barrier timed out")
            return response()

        adapter._client.cardkit.v1.card_element.content = mutate
        task = asyncio.create_task(adapter._stream_cardkit_content("c", "draft"))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            task.cancel()
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(adapter._cardkit_confirmed_sequences["c"], 1)
        finally:
            release.set()
            adapter._get_sdk_executor().shutdown(wait=True)

    async def test_corrected_final_edits_original_message(self):
        adapter, _hub, ops = make_adapter()
        add_turn(adapter)
        await adapter.send("chat_a", "first", reply_to="user_a")
        result = await adapter.edit_message("chat_a", "im_test", "corrected", finalize=True)
        self.assertTrue(result.success)
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
        self.assertIn("corrected", ops[-1][1].request_body.card.data)

    async def test_other_chat_thread_and_stale_draft_are_refused(self):
        adapter, _hub, _ops = make_adapter()
        add_turn(adapter, thread="t")
        self.assertFalse(
            (
                await adapter.send_draft(
                    "chat_b", 1, "x", {"thread_id": "t", "reply_to_message_id": "user_a"}
                )
            ).success
        )
        self.assertFalse(
            (await adapter.send_draft("chat_a", 1, "x", {"reply_to_message_id": "user_a"})).success
        )
        md = {"thread_id": "t", "reply_to_message_id": "user_a"}
        self.assertTrue((await adapter.send_draft("chat_a", 1, "x", md)).success)
        self.assertFalse((await adapter.send_draft("chat_a", 2, "stale", md)).success)

    async def test_actual_gateway_stream_consumer_finalizes_same_card(self):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter, _hub, _ops = make_adapter()
        add_turn(adapter)
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_a",
            StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor="", transport="auto"),
            initial_reply_to_id="user_a",
        )
        consumer.on_delta("first")
        consumer.on_segment_break()
        consumer.on_delta("second")
        consumer.finish(final_text="complete answer")
        await consumer.run()
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
        self.assertEqual(consumer.message_id, "im_test")
        self.assertTrue(consumer.delivered_final_matches("complete answer"))

    async def test_parallel_tool_hooks_are_bound_to_the_handler_context(self):
        adapter, hub, _ops = make_adapter()
        adapter.schedule_refresh = lambda turn: None
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(evt):
            if evt.message_id == "a":
                entered.set()
                await release.wait()
            await asyncio.to_thread(
                hub.pre_tool, "terminal", tool_call_id=evt.message_id, api_request_id="api"
            )
            await asyncio.to_thread(
                hub.post_tool,
                "terminal",
                tool_call_id=evt.message_id,
                status="success",
                duration_ms=12,
            )

        adapter.set_message_handler(handler)
        first = asyncio.create_task(adapter._message_handler(event(message_id="a")))
        await entered.wait()
        await adapter._message_handler(event(chat="other", message_id="b"))
        release.set()
        await first
        self.assertEqual(set(adapter._card_turns[("chat_a", "", "a")].calls), {"a"})
        self.assertEqual(set(adapter._card_turns[("other", "", "b")].calls), {"b"})
        hub.pre_tool("terminal", tool_call_id="unbound")
        self.assertNotIn("unbound", adapter._card_turns[("chat_a", "", "a")].calls)
