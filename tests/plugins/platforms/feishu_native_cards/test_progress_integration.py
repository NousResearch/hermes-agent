import asyncio
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from tests.plugins.platforms.feishu_native_cards.support import make_adapter, add_turn, event


class ProgressIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_reference_projects_degraded_progress_and_final_trace(self):
        adapter, hub, ops = make_adapter()
        adapter.schedule_refresh = lambda turn: None
        turn = add_turn(adapter)
        token = hub.current.set((adapter, turn))
        try:
            ref = dict(aux_task="moa_reference", api_request_id="failed", model="ref")
            hub.pre_auxiliary(**ref)
            hub.post_auxiliary(**ref, error="provider unavailable")
            hub.pre_auxiliary(aux_task="moa_aggregator", api_request_id="agg", model="fallback")
            await adapter._refresh_turn(turn)
            self.assertIn("正在降级综合", ops[-1][1].request_body.content)
            await adapter.send("chat_a", "回答", reply_to="user_a")
            self.assertIn("已降级综合", ops[-1][1].request_body.card.data)
            self.assertEqual(turn.orchestration["references"][0]["status"], "failed")
        finally:
            hub.current.reset(token)

    async def test_busy_cancel_waits_for_write_and_preserves_answer_and_card_identity(self):
        adapter, _, _ = make_adapter()
        turn = add_turn(adapter)
        await adapter.send_draft("chat_a", 1, "正文", {"reply_to_message_id": "user_a"})
        entered, release = asyncio.Event(), asyncio.Event()
        original = adapter._stream_cardkit_content_with_recovery

        async def delayed(card_id, content):
            entered.set()
            await release.wait()
            await original(card_id, content)

        adapter._stream_cardkit_content_with_recovery = delayed
        task = asyncio.create_task(adapter._send_busy_card(turn, "已排队"))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertTrue(turn.io_lock.locked())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(turn.io_lock.locked())
        self.assertEqual(turn.text, "正文")
        self.assertFalse(turn.final)
        self.assertEqual(turn.message_id, "im_test")
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)

    async def test_busy_failure_keeps_identity_and_does_not_send_an_extra_message(self):
        adapter, _, _ = make_adapter()
        turn = add_turn(adapter)
        await adapter.send_draft("chat_a", 1, "正文", {"reply_to_message_id": "user_a"})
        adapter._stream_cardkit_content_with_recovery = AsyncMock(
            side_effect=TimeoutError("lost ack")
        )

        async def handler(incoming, key):
            return await adapter.send("chat_a", "ack", reply_to=incoming.message_id)

        adapter.set_busy_session_handler(handler)
        result = await adapter._busy_session_handler(event(message_id="followup"), turn.session_key)
        self.assertFalse(result.success)
        self.assertEqual(result.message_id, "im_test")
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
        self.assertEqual(turn.text, "正文")
        self.assertIsNone(adapter._busy_card_scope.get())

    async def test_real_auxiliary_hooks_and_moa_worker_context_drive_one_card(self):
        from agent.auxiliary_hooks import run_with_aux_hooks
        from agent.moa_loop import _run_references_parallel

        adapter, hub, ops = make_adapter()
        turn = add_turn(adapter)
        callbacks = {
            "pre_auxiliary_call": hub.pre_auxiliary,
            "post_auxiliary_call": hub.post_auxiliary,
        }
        refreshed = []
        adapter.schedule_refresh = lambda value: refreshed.append(value.status())

        def call_model(*, task, messages, model, **kwargs):
            response = NS(
                choices=[NS(message=NS(content="advice", tool_calls=[]), finish_reason="stop")]
            )
            result = run_with_aux_hooks(
                lambda: response,
                aux_task=task,
                metadata={"api_request_id": model},
                client=NS(),
                kwargs={"messages": messages},
                provider="test",
                model=model,
                api_mode="chat_completions",
            )
            return result

        token = hub.current.set((adapter, turn))
        try:
            with (
                patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name in callbacks),
                patch(
                    "hermes_cli.lifecycle.invoke_hook",
                    side_effect=lambda name, **kw: callbacks[name](**kw),
                ),
                patch("agent.moa_loop.call_llm", side_effect=call_model),
                patch("agent.moa_loop._slot_runtime", side_effect=lambda slot: dict(slot)),
                patch(
                    "agent.moa_loop._trim_messages_for_reference",
                    side_effect=lambda messages, *a, **kw: messages,
                ),
                patch(
                    "agent.moa_loop._maybe_apply_moa_cache_control",
                    side_effect=lambda messages, *a, **kw: messages,
                ),
                patch(
                    "agent.moa_loop._price_reference_response",
                    return_value=(None, None, None, None),
                ),
            ):
                results = await asyncio.to_thread(
                    _run_references_parallel, [{"model": "a"}, {"model": "b"}], []
                )
            self.assertEqual(len(results), 2)
            self.assertEqual(turn.status()["orchestration"]["refs_done"], 2)
            self.assertTrue(any(s["orchestration"]["refs_done"] == 1 for s in refreshed))
            # 官方 MoA 在主请求 hook 前准备参考意见，主 hook 不得擦掉进度。
            hub.pre_api(model="moa/default", session_id="s", api_request_id="main")
            hub.pre_auxiliary(aux_task="moa_aggregator", api_request_id="agg", model="synthesis")
            hub.post_auxiliary(
                aux_task="moa_aggregator", api_request_id="agg", model="synthesis", streaming=True
            )
            self.assertEqual(turn.orchestration["phase"], "aggregating")
            self.assertEqual(turn.orchestration["refs_total"], 2)
            await adapter._refresh_turn(turn)
            self.assertIn("正在综合参考意见", ops[-1][1].request_body.content)
            await adapter.send("chat_a", "最终回答", reply_to="user_a")
            payload = json.loads(ops[-1][1].request_body.card.data)
            self.assertIn("moa_trace", json.dumps(payload))
            self.assertIn("MoA 协作已完成", ops[-1][1].request_body.card.data)
            self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
        finally:
            hub.current.reset(token)

    async def test_moa_retries_new_wave_foreign_late_and_terminal_events(self):
        adapter, hub, _ = make_adapter()
        adapter.schedule_refresh = lambda turn: None
        turn = add_turn(adapter)
        token = hub.current.set((adapter, turn))
        try:
            turn.agent_session_id = "main"
            ref = dict(aux_task="moa_reference", api_request_id="r", model="ref", session_id="main")
            hub.pre_auxiliary(**ref)
            hub.post_auxiliary(**ref, error="temporary")
            hub.pre_auxiliary(**ref, retry_count=1)
            hub.post_auxiliary(**ref, retry_count=1)
            hub.post_auxiliary(**ref, error="late old attempt")
            self.assertEqual(turn.orchestration["refs_done"], 1)
            self.assertEqual(turn.orchestration["references"][0]["status"], "done")
            hub.pre_auxiliary(aux_task="moa_reference", api_request_id="child", session_id="child")
            self.assertNotIn("child", turn.moa_requests)
            hub.pre_auxiliary(aux_task="moa_aggregator", api_request_id="a", model="agg")
            hub.pre_auxiliary(aux_task="moa_reference", api_request_id="next", model="ref2")
            hub.post_auxiliary(**ref)
            self.assertEqual(turn.orchestration["phase"], "references")
            self.assertEqual(list(turn.moa_requests), ["next"])
            turn.final = True
            before = turn.status()
            hub.post_auxiliary(aux_task="moa_reference", api_request_id="next")
            self.assertEqual(before, turn.status())
        finally:
            hub.current.reset(token)
        hub.pre_auxiliary(aux_task="moa_reference", api_request_id="unbound")
        self.assertEqual(before, turn.status())

    async def test_busy_official_reply_path_updates_original_card_without_finalizing(self):
        from gateway.run_busy import GatewayBusySessionMixin

        adapter, _, ops = make_adapter()
        turn = add_turn(adapter, thread="topic")
        await adapter.send_draft(
            "chat_a", 1, "正文", {"thread_id": "topic", "reply_to_message_id": "user_a"}
        )

        class Runner(GatewayBusySessionMixin):
            def _reply_anchor_for_event(self, event):
                return event.message_id

            def _thread_metadata_for_source(self, source, reply_anchor):
                return {"thread_id": source.thread_id} if source.thread_id else {}

        runner = Runner()

        async def handler(incoming, session_key):
            await runner._send_busy_ack_reply(incoming, adapter, "已排队，继续处理中")
            return True

        adapter.set_busy_session_handler(handler)
        incoming = event(thread="topic", message_id="followup")
        incoming.internal = False
        self.assertTrue(await adapter._busy_session_handler(incoming, turn.session_key))
        self.assertIn("已排队", ops[-1][1].request_body.content)
        self.assertFalse(turn.final)
        self.assertFalse(turn.fallback)
        self.assertEqual(turn.text, "正文")
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
        await adapter.send_draft(
            "chat_a", 1, "继续回答", {"thread_id": "topic", "reply_to_message_id": "user_a"}
        )
        self.assertNotIn("已排队", ops[-1][1].request_body.content)
        self.assertIsNone(adapter._busy_card_scope.get())

    async def test_busy_never_merges_wrong_session_owner_thread_command_or_ambiguous_turn(self):
        adapter, _, _ = make_adapter()
        turn = add_turn(adapter)
        adapter._send_busy_card = AsyncMock()
        # 未匹配的回执继续走官方发送，不调用外部网络。
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from gateway.platforms.base import SendResult

        async def handler(incoming, key):
            await adapter.send(
                incoming.source.chat_id,
                "ack",
                reply_to=incoming.message_id,
                metadata={"thread_id": incoming.source.thread_id},
            )
            return True

        adapter.set_busy_session_handler(handler)
        with patch.object(
            FeishuAdapter, "send", new=AsyncMock(return_value=SendResult(success=True))
        ):
            for incoming, key in (
                (event(message_id="followup"), "other-session"),
                (event(message_id="followup", owner="other"), turn.session_key),
                (event(message_id="followup", thread="other"), turn.session_key),
                (event(message_id="followup", text="/status"), turn.session_key),
            ):
                await adapter._busy_session_handler(incoming, key)
            other = add_turn(adapter, reply="other-active")
            await adapter._busy_session_handler(event(message_id="followup"), turn.session_key)
            other.final = True
            turn.final = True
            await adapter._busy_session_handler(event(message_id="followup"), turn.session_key)
        adapter._send_busy_card.assert_not_awaited()
        self.assertIsNone(adapter._busy_card_scope.get())
