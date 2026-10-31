import json
import queue
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from tests.plugins.platforms.feishu_native_cards.support import add_turn, event, make_adapter


class NoticeTests(unittest.IsolatedAsyncioTestCase):
    async def test_commands_keep_official_handler_and_send_static_cards_without_creating_turn(self):
        adapter, _, _ = make_adapter()
        received = []

        async def handler(incoming):
            received.append(incoming.text)
            return await adapter.send(
                incoming.source.chat_id,
                "✨ 会话已重置！\n\n**模型**：test",
                reply_to=incoming.message_id,
                metadata={"thread_id": incoming.source.thread_id},
            )

        adapter.set_message_handler(handler)
        for command in ["/new", "/status", "/help"]:
            result = await adapter._message_handler(event(text=command, thread="topic"))
            self.assertTrue(result.success)
            sent = adapter._feishu_send_with_retry.await_args.kwargs
            self.assertEqual(sent["msg_type"], "interactive")
            card = json.loads(sent["payload"])
            self.assertFalse(card["config"]["streaming_mode"])
            self.assertNotIn("header", card)
            self.assertIn("**模型**", card["elements"][0]["content"])
            self.assertEqual(sent["reply_to"], "user_a")
            self.assertEqual(sent["metadata"], {"thread_id": "topic"})
        self.assertEqual(received, ["/new", "/status", "/help"])
        self.assertEqual(len(adapter._card_turns), 0)

    async def test_receipt_edits_patch_same_card_and_unknown_delivery_does_not_fallback(self):
        adapter, _, operations = make_adapter()
        result = await adapter.send("chat_a", "处理中")
        self.assertTrue(result.success)
        edited = await adapter.edit_message("chat_a", result.message_id, "完成", finalize=True)
        self.assertTrue(edited.success)
        self.assertEqual(operations[-1][0], "patch")
        self.assertEqual(operations[-1][1].message_id, result.message_id)
        self.assertIn("完成", operations[-1][1].request_body.content)
        adapter._feishu_send_with_retry.side_effect = TimeoutError("unknown acknowledgement")
        self.assertFalse((await adapter.send("chat_a", "另一条回执")).success)
        self.assertEqual(adapter._feishu_send_with_retry.await_count, 2)

    async def test_disabling_legacy_progress_keeps_hook_driven_card_progress(self):
        from gateway.display_config import resolve_tool_progress
        from gateway.run_turn_progress import GatewayTurnProgressMixin

        adapter, hub, operations = make_adapter()
        turn = add_turn(adapter)
        mode, _ = resolve_tool_progress(
            {"display": {"platforms": {"feishu": {"tool_progress": "off"}}}}, "feishu"
        )
        ctx = NS(
            progress_queue=queue.Queue(),
            _native_slack_task_cards=False,
            log_queue=None,
            tool_progress_enabled=mode != "off",
            _run_still_current=lambda: True,
        )
        runner = NS(_ctx=ctx, _progress_live_status=lambda *a: None)
        with patch("gateway.run_turn_progress.local_viewer_turn", return_value=False):
            GatewayTurnProgressMixin.progress_callback(
                runner, "tool.started", "skill_view", "test-skill"
            )
        self.assertTrue(ctx.progress_queue.empty())
        token = hub.current.set((adapter, turn))
        try:
            hub.pre_tool("skill_view", tool_call_id="call", api_request_id="api")
            hub.post_tool("skill_view", tool_call_id="call", status="success", duration_ms=20)
            await adapter._refresh_turn(turn)
            self.assertIn("skill_view", operations[-1][1].request_body.content)
            self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
        finally:
            hub.current.reset(token)
            await adapter.disconnect()
