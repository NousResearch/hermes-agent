"""Native clarify cards bind callbacks directly to the existing gateway waiter."""

import asyncio
import json
import logging
import time

logger = logging.getLogger(__name__)


def text_question_card(question, clarify_id):
    from .cards import card, button

    result = card("需要确认", "blue", question)
    submit = button(
        "提交回答", "primary", {"hermes_clarify_action": "text_submit", "clarify_id": clarify_id}
    )
    submit["action_type"] = "form_submit"
    submit["name"] = "submit_answer"
    result["elements"].append(
        {
            "tag": "form",
            "name": "clarify_text_form",
            "elements": [
                {
                    "tag": "input",
                    "name": "answer",
                    "input_type": "multiline_text",
                    "required": True,
                    "max_length": 4000,
                    "placeholder": {"tag": "plain_text", "content": "在这里填写回答"},
                },
                submit,
            ],
        }
    )
    result["elements"].append(
        {
            "tag": "action",
            "actions": [
                button(
                    "跳过", "default", {"hermes_clarify_action": "skip", "clarify_id": clarify_id}
                )
            ],
        }
    )
    return result


class FeishuClarifyMixin:
    def can_answer_clarify(self, clarify_id, source):
        with self._card_action_lock:
            state = self._clarify_state.get(clarify_id)
            if state is None:
                return True  # the existing plain-text fallback has no native binding
            return (
                source.chat_id == state["chat_id"]
                and str(source.user_id or "") in set(state["owner_ids"]) | self._admins
            )

    async def send_clarify(
        self, chat_id, question, choices, clarify_id, session_key, metadata=None
    ):
        from .cardkit_mutations import await_settled

        return await await_settled(
            self._send_clarify_card(chat_id, question, choices, clarify_id, session_key, metadata)
        )

    async def _send_clarify_card(
        self, chat_id, question, choices, clarify_id, session_key, metadata
    ):
        from .cards import card as _card, button as _card_button
        from tools import clarify_gateway as cg
        from gateway.platforms.base import SendResult

        meta = metadata or {}
        owners = meta.get("__hermes_prompt_user_ids") or (
            (meta["user_id"],) if meta.get("user_id") else ()
        )
        if not owners:
            # The generic runner can render its text fallback; never publish
            # an actionable card that an arbitrary group member can answer.
            return SendResult(
                success=False, error="Clarify owner identity missing; use text fallback"
            )
        entry = cg.get_pending_for_session(session_key, include_choice_prompts=True)
        if entry is None or entry.clarify_id != clarify_id or entry.event.is_set():
            return SendResult(success=False, error="Clarify request expired")

        def button(label, action, kind="default"):
            return _card_button(
                label, kind, {"hermes_clarify_action": action, "clarify_id": clarify_id}
            )

        controls = []
        if choices:
            controls.append(
                {
                    "tag": "select_static" if not entry.multi_select else "multi_select_static",
                    "name": "selection",
                    "placeholder": {"tag": "plain_text", "content": "请选择"},
                    "options": [
                        {"text": {"tag": "plain_text", "content": str(label)}, "value": str(i)}
                        for i, label in enumerate(choices)
                    ],
                }
            )
            submit = button("提交", "submit", "primary")
            submit["action_type"] = "form_submit"
            submit["name"] = "submit_selection"
            controls.append(submit)
        else:
            cg.mark_awaiting_text(clarify_id)
        card = _card("需要确认", "blue", question)
        if controls:
            card["elements"].append({"tag": "form", "name": "clarify_form", "elements": controls})
        card["elements"].append(
            {
                "tag": "action",
                "actions": [button("其他（文字回答）", "other"), button("跳过", "skip")],
            }
        )
        if not choices:
            card = text_question_card(question, clarify_id)
        response = await self._feishu_send_with_retry(
            chat_id=chat_id,
            msg_type="interactive",
            payload=json.dumps(card, ensure_ascii=False),
            reply_to=None,
            metadata=metadata,
        )
        result = self._finalize_send_result(response, "clarify card send failed")
        if result.success and result.message_id:
            self._clarify_state[clarify_id] = {
                "session_key": session_key,
                "chat_id": chat_id,
                "message_id": result.message_id,
                "created_at": time.time(),
                "owner_ids": owners,
                "chat_type": meta.get("__hermes_prompt_chat_type", "dm"),
                "entry": entry,
                "metadata": metadata,
                "question": question,
                "text_mode": not bool(choices),
            }
            task = asyncio.create_task(self._retire_clarify(clarify_id, entry))
            self._clarify_tasks[clarify_id] = task

            def finished(task):
                self._clarify_tasks.pop(clarify_id, None)
                if not task.cancelled():
                    self._log_background_failure(task)

            task.add_done_callback(finished)
        return result

    async def _retire_clarify(self, clarify_id, entry):
        from tools import clarify_gateway as cg

        # The real waiter is authoritative for text answers, reset, cancellation,
        # and timeout. No new gateway turn is injected by this UI lifecycle.
        while not entry.event.is_set():
            pending = cg.get_pending_for_session(entry.session_key, include_choice_prompts=True)
            if pending is not entry:
                break
            await asyncio.sleep(0.2)
        response = entry.response
        label = (
            "已跳过"
            if response == cg.SKIPPED
            else "已取消"
            if response == cg.CANCELLED
            else "已过期"
            if response is None
            else "已回答"
        )
        await self.retire_clarify_card(clarify_id, label)

    async def retire_clarify_card(self, clarify_id, notice):
        """The existing gateway text/timeout retirement surface; claim once."""
        from .cards import card as _card

        with self._card_action_lock:
            state = self._clarify_state.pop(clarify_id, None)
        if state:
            payload = json.dumps(_card("确认结束", "grey", notice), ensure_ascii=False)
            request = self._build_patch_message_request(
                message_id=state["message_id"],
                request_body=self._build_patch_message_body(content=payload),
            )
            result = await self._run_blocking(self._client.im.v1.message.patch, request)
            if not self._response_succeeded(result):
                logger.warning(
                    "[Feishu] Clarify retirement patch failed: %s",
                    getattr(result, "msg", "unknown"),
                )

    def _handle_clarify_card_action(self, event, value):
        from .cards import card as _card
        from tools import clarify_gateway as cg

        clarify_id = value.get("clarify_id")
        state = self._clarify_state.get(clarify_id)
        if not state:
            return self._card_response()
        entry = cg.get_pending_for_session(state["session_key"], include_choice_prompts=True)
        if entry is not state["entry"] or entry.event.is_set():
            return self._card_response()
        if (
            self._validate_card_action(event=event, state=state, label="clarify", ident=clarify_id)
            is None
        ):
            return self._card_response()
        action = value.get("hermes_clarify_action")
        if action == "other":
            cg.mark_awaiting_text(clarify_id)
            state["text_mode"] = True
            return self._card_response(text_question_card(state["question"], clarify_id))
        if action == "skip":
            answer = cg.SKIPPED
        elif action == "text_submit":
            answer = (getattr(event.action, "form_value", None) or {}).get("answer")
            if (
                not state.get("text_mode")
                or not isinstance(answer, str)
                or not answer.strip()
                or len(answer) > 4000
            ):
                return self._card_response(toast_message="请输入 1 至 4000 字的回答。")
            answer = answer.strip()
        elif action == "submit":
            values = (getattr(event.action, "form_value", None) or {}).get("selection")
            values = values if isinstance(values, list) else [values]
            if not values or (not entry.multi_select and len(values) != 1):
                return self._card_response()
            if any(
                not isinstance(v, str) or not v.isdigit() or int(v) >= len(entry.choices or [])
                for v in values
            ):
                return self._card_response()
            labels = list(dict.fromkeys(entry.choices[int(v)] for v in values))
            answer = json.dumps(labels, ensure_ascii=False) if entry.multi_select else labels[0]
        else:
            return self._card_response()
        if not cg.resolve_gateway_clarify(clarify_id, answer):
            return self._card_response()
        return self._card_response(
            _card("确认结束", "grey", "已跳过" if answer == cg.SKIPPED else "已回答")
        )
