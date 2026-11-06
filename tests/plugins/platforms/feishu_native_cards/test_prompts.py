import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from tests.plugins.platforms.feishu_native_cards.support import make_adapter, add_turn, event
from tools import clarify_gateway as cg


def callback(
    adapter,
    *,
    action,
    value=None,
    owner="owner_a",
    chat="chat_a",
    message="im_test",
    cid="cid",
    form=None,
):
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt", "event_type": "card.action.trigger"},
        "event": {
            "operator": {"open_id": owner},
            "context": {"open_chat_id": chat, "open_message_id": message},
            "action": {
                "tag": "button",
                "value": {"hermes_clarify_action": action, "clarify_id": cid},
                "form_value": form if form is not None else {"selection": value},
            },
        },
    }
    return adapter._build_event_handler()._do_without_validation(json.dumps(payload).encode())


class PromptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        cg.clear_session("session_a")

    async def test_single_multi_other_and_skip_resolve_real_waiters(self):
        for multi, action, selection, expected in [
            (False, "submit", "0", "one"),
            (True, "submit", ["0", "1"], '["one", "two"]'),
            (False, "skip", None, cg.SKIPPED),
            (False, "other", None, "free answer"),
        ]:
            with self.subTest(action=action, multi=multi):
                adapter, hub, ops = make_adapter()
                turn = add_turn(adapter)
                entry = cg.register(
                    "cid", "session_a", "question", ["one", "two"], multi_select=multi
                )
                token = hub.current.set((adapter, turn))
                try:
                    result = await adapter.send_clarify(
                        "chat_a", "question", ["one", "two"], "cid", "session_a"
                    )
                finally:
                    hub.current.reset(token)
                self.assertTrue(result.success)
                task = adapter._clarify_tasks["cid"]
                callback(adapter, action=action, value=selection)
                if action == "other":
                    self.assertFalse(entry.event.is_set())
                    self.assertTrue(entry.awaiting_text)
                    self.assertTrue(
                        cg.resolve_text_response_for_session("session_a", "free answer")
                    )
                self.assertTrue(entry.event.is_set())
                self.assertEqual(entry.response, expected)
                await task
                self.assertNotIn("cid", adapter._clarify_state)
                self.assertEqual(ops[-1][0], "patch")
                self.assertEqual(ops[-1][1].message_id, "im_test")
                cg.clear_session("session_a")

    async def test_foreign_user_chat_message_and_stale_waiter_are_rejected(self):
        for bad in ("owner", "chat", "message", "entry", "answered"):
            with self.subTest(bad=bad):
                adapter, _hub, _ops = make_adapter()
                cg.register("cid", "session_a", "question", ["one"])
                await adapter.send_clarify(
                    "chat_a",
                    "question",
                    ["one"],
                    "cid",
                    "session_a",
                    {"__hermes_prompt_user_ids": ("owner_a",)},
                )
                task = adapter._clarify_tasks["cid"]
                if bad == "entry":
                    cg.clear_session("session_a")
                    cg.register("cid", "session_a", "new question", ["two"])
                if bad == "answered":
                    cg.resolve_gateway_clarify("cid", "already answered")
                result = callback(
                    adapter,
                    action="submit",
                    value="0",
                    owner="foreign" if bad == "owner" else "owner_a",
                    chat="foreign" if bad == "chat" else "chat_a",
                    message="foreign" if bad == "message" else "im_test",
                )
                self.assertIsNone(result.card)
                pending = cg.get_pending("cid")
                if bad != "answered":
                    self.assertIsNotNone(pending)
                    self.assertFalse(pending.event.is_set())
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                cg.clear_session("session_a")

    async def test_invalid_selection_and_replay_cannot_answer_twice(self):
        adapter, _hub, _ops = make_adapter()
        entry = cg.register("cid", "session_a", "question", ["one"])
        await adapter.send_clarify(
            "chat_a",
            "question",
            ["one"],
            "cid",
            "session_a",
            {"__hermes_prompt_user_ids": ("owner_a",)},
        )
        task = adapter._clarify_tasks["cid"]
        for value in ("-1", "1", ["0", "0"], None):
            callback(adapter, action="submit", value=value)
            self.assertFalse(entry.event.is_set())
        callback(adapter, action="submit", value="0")
        await task
        callback(adapter, action="skip")
        self.assertEqual(entry.response, "one")

    async def test_missing_identity_refuses_native_prompt(self):
        adapter, _hub, _ops = make_adapter()
        cg.register("cid", "session_a", "question", ["one"])
        result = await adapter.send_clarify("chat_a", "question", ["one"], "cid", "session_a")
        self.assertFalse(result.success)
        adapter._feishu_send_with_retry.assert_not_awaited()

    async def test_native_input_and_other_submit_inside_original_card(self):
        for choices in (None, ["one", "two"]):
            with self.subTest(choices=choices):
                adapter, _hub, ops = make_adapter()
                entry = cg.register("cid", "session_a", "question", choices)
                await adapter.send_clarify(
                    "chat_a",
                    "question",
                    choices,
                    "cid",
                    "session_a",
                    {"__hermes_prompt_user_ids": ("owner_a",)},
                )
                task = adapter._clarify_tasks["cid"]
                if choices:
                    rejected = callback(adapter, action="text_submit", form={"answer": "bypass"})
                    self.assertIsNone(rejected.card)
                    self.assertFalse(entry.event.is_set())
                    changed = callback(adapter, action="other")
                    self.assertIn('"input"', json.dumps(changed.card.data))
                else:
                    raw = adapter._feishu_send_with_retry.await_args.kwargs["payload"]
                    self.assertIn('"input"', raw)
                for answer in ("", "  ", "x" * 4001, ["not a string"]):
                    callback(adapter, action="text_submit", form={"answer": answer})
                    self.assertFalse(entry.event.is_set())
                callback(adapter, action="text_submit", form={"answer": "我的原生输入"})
                self.assertEqual(entry.response, "我的原生输入")
                await task
                self.assertEqual(adapter._feishu_send_with_retry.await_count, 1)
                self.assertEqual(ops[-1][1].message_id, "im_test")
                cg.clear_session("session_a")

    async def test_native_form_submits_have_named_components(self):
        from plugins.platforms.feishu.native_cards.clarify import text_question_card

        card = text_question_card("question", "cid")
        form = next(element for element in card["elements"] if element["tag"] == "form")
        self.assertEqual({el["name"] for el in form["elements"]}, {"answer", "submit_answer"})
        self.assertEqual(form["elements"][-1]["action_type"], "form_submit")

    async def test_still_pending_clarify_is_not_expired_by_callback_dedup_ttl(self):
        adapter, _hub, _ops = make_adapter()
        entry = cg.register("cid", "session_a", "question", ["one"])
        await adapter.send_clarify(
            "chat_a",
            "question",
            ["one"],
            "cid",
            "session_a",
            {"__hermes_prompt_user_ids": ("owner_a",)},
        )
        task = adapter._clarify_tasks["cid"]
        adapter._clarify_state["cid"]["created_at"] -= 7200
        callback(adapter, action="submit", value="0")
        await task
        self.assertEqual(entry.response, "one")

    async def test_reset_and_text_answer_retire_native_controls(self):
        for reset in (True, False):
            with self.subTest(reset=reset):
                adapter, _hub, ops = make_adapter()
                cg.register("cid", "session_a", "question", None)
                await adapter.send_clarify(
                    "chat_a",
                    "question",
                    None,
                    "cid",
                    "session_a",
                    {"__hermes_prompt_user_ids": ("owner_a",)},
                )
                task = adapter._clarify_tasks["cid"]
                if reset:
                    cg.clear_session("session_a")
                else:
                    cg.resolve_text_response_for_session("session_a", "text answer")
                await task
                self.assertNotIn("cid", adapter._clarify_state)
                card = json.loads(ops[-1][1].request_body.content)
                self.assertNotIn('"button"', json.dumps(card))

    async def test_foreign_text_is_blocked_before_official_clarify_consumption(self):
        adapter, _hub, _ops = make_adapter()
        evt = event(owner="intruder")
        session = adapter._event_session_key(evt)
        adapter._clarify_state["cid"] = {
            "session_key": session,
            "chat_id": "chat_a",
            "owner_ids": ("owner_a",),
        }
        from plugins.platforms.feishu.adapter import FeishuAdapter

        with patch.object(FeishuAdapter, "handle_message", new=AsyncMock()) as parent:
            await adapter.handle_message(evt)
            parent.assert_not_awaited()

    async def test_approval_processing_then_actual_truth_and_replay(self):
        import tools.approval as approvals

        for outcome in (1, 0, RuntimeError("simulated resolver failure")):
            with self.subTest(outcome=outcome):
                adapter, _hub, ops = make_adapter()
                adapter._approval_state[1] = {
                    "session_key": "session_a",
                    "chat_id": "chat_a",
                    "message_id": "im_test",
                    "created_at": time.time(),
                    "request_id": "req",
                    "owner_ids": ("owner_a",),
                }
                scheduled = []
                adapter._submit_on_loop = lambda loop, coro: scheduled.append(coro) or True

                def resolve(*args, **kwargs):
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome

                payload = {
                    "schema": "2.0",
                    "header": {"event_id": "approve", "event_type": "card.action.trigger"},
                    "event": {
                        "operator": {"open_id": "owner_a"},
                        "context": {"open_chat_id": "chat_a", "open_message_id": "im_test"},
                        "action": {"value": {"hermes_action": "approve_once", "approval_id": 1}},
                    },
                }
                with (
                    patch.object(
                        approvals, "list_gateway_approvals", return_value=[{"request_id": "req"}]
                    ),
                    patch.object(
                        approvals, "resolve_gateway_approval", side_effect=resolve
                    ) as resolver,
                ):
                    result = adapter._build_event_handler()._do_without_validation(
                        json.dumps(payload).encode()
                    )
                    self.assertIsNone(result.card)
                    self.assertIn("处理中", result.toast.content)
                    adapter._build_event_handler()._do_without_validation(
                        json.dumps(payload).encode()
                    )
                    self.assertEqual(len(scheduled), 1)
                    await scheduled.pop()
                    resolver.assert_called_once_with("session_a", "once", request_id="req")
                self.assertEqual([op for op, _ in ops], ["patch", "patch"])
                final = json.loads(ops[-1][1].request_body.content)
                approved = adapter._build_resolved_approval_card(choice="once", user_name="owner_a")
                self.assertEqual(final == approved, outcome == 1)

    async def test_approval_rejects_foreign_identity_context_expiry_and_request(self):
        for bad in ("owner", "chat", "message", "expiry", "request"):
            with self.subTest(bad=bad):
                adapter, _hub, ops = make_adapter()
                adapter._approval_state[1] = {
                    "session_key": "session_a",
                    "chat_id": "chat_a",
                    "message_id": "im_test",
                    "created_at": time.time() - (3600 if bad == "expiry" else 0),
                    "request_id": "req",
                    "owner_ids": ("owner_a",),
                }
                payload = {
                    "schema": "2.0",
                    "header": {"event_id": "approval", "event_type": "card.action.trigger"},
                    "event": {
                        "operator": {"open_id": "foreign" if bad == "owner" else "owner_a"},
                        "context": {
                            "open_chat_id": "foreign" if bad == "chat" else "chat_a",
                            "open_message_id": "foreign" if bad == "message" else "im_test",
                        },
                        "action": {"value": {"hermes_action": "approve_once", "approval_id": 1}},
                    },
                }
                adapter._submit_on_loop = lambda loop, coro: self.fail("无效审批不能调度执行")
                with patch(
                    "tools.approval.list_gateway_approvals",
                    return_value=[] if bad == "request" else [{"request_id": "req"}],
                ):
                    result = adapter._build_event_handler()._do_without_validation(
                        json.dumps(payload).encode()
                    )
                self.assertIsNone(result.card)
                self.assertEqual(ops, [])

    async def test_clarify_timeout_retires_controls(self):
        adapter, _hub, ops = make_adapter()
        cg.register("cid", "session_a", "question", None)
        await adapter.send_clarify(
            "chat_a",
            "question",
            None,
            "cid",
            "session_a",
            {"__hermes_prompt_user_ids": ("owner_a",)},
        )
        task = adapter._clarify_tasks["cid"]
        self.assertIsNone(await asyncio.to_thread(cg.wait_for_response, "cid", 0.001))
        await task
        self.assertNotIn("cid", adapter._clarify_state)
        self.assertNotIn('"button"', ops[-1][1].request_body.content)

    async def test_ambiguous_approval_request_never_selects_a_candidate(self):
        adapter, hub, _ops = make_adapter()
        turn = add_turn(adapter)
        token = hub.current.set((adapter, turn))
        try:
            pending = [
                {"command": "same", "request_id": "a"},
                {"command": "same", "request_id": "b"},
            ]
            with patch("tools.approval.list_gateway_approvals", return_value=pending):
                await asyncio.to_thread(hub.pre_approval, session_key="session_a", command="same")
                result = await adapter.send_exec_approval("chat_a", "same", session_key="session_a")
            self.assertFalse(result.success)
            adapter._feishu_send_with_retry.assert_not_awaited()
        finally:
            hub.current.reset(token)

    async def test_hook_binds_exact_pending_approval_identity(self):
        adapter, hub, _ops = make_adapter()
        turn = add_turn(adapter)
        token = hub.current.set((adapter, turn))
        try:
            with patch(
                "tools.approval.list_gateway_approvals",
                return_value=[{"command": "echo unit", "request_id": "req"}],
            ):
                await asyncio.to_thread(
                    hub.pre_approval, session_key="session_a", command="echo unit"
                )
                result = await adapter.send_exec_approval(
                    "chat_a", "echo unit", session_key="session_a"
                )
            self.assertTrue(result.success)
            state = next(iter(adapter._approval_state.values()))
            self.assertEqual(state["request_id"], "req")
            self.assertEqual(state["owner_ids"], ("owner_a",))
        finally:
            hub.current.reset(token)
