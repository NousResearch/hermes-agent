"""原卡片的请求绑定、回放拒绝和真实审批结果更新。"""

from __future__ import annotations
import json
import logging
import time
from types import SimpleNamespace
import uuid
from typing import Any, Dict, Optional
from agent.i18n import t
from gateway.platforms.base import SendResult, MessageType
from lark_oapi.event.callback.model.p2_card_action_trigger import (
    CallBackCard,
    CallBackToast,
    P2CardActionTriggerResponse,
)
from lark_oapi.api.im.v1.model import PatchMessageRequest, PatchMessageRequestBody
from .cards import card as _card

logger = logging.getLogger(__name__)
_FEISHU_CARD_ACTION_DEDUP_TTL_SECONDS = 15 * 60
_APPROVAL_CHOICE_MAP = {
    "approve_once": "once",
    "approve_session": "session",
    "approve_always": "always",
    "deny": "deny",
}


class PromptMixin:
    @staticmethod
    def _build_patch_message_body(*, content):
        return PatchMessageRequestBody.builder().content(content).build()

    @staticmethod
    def _build_patch_message_request(*, message_id, request_body):
        return (
            PatchMessageRequest.builder().message_id(message_id).request_body(request_body).build()
        )

    async def _send_interactive_card(
        self,
        chat_id: str,
        card: dict[str, Any],
        metadata: Optional[dict[str, Any]],
        failure_message: str,
        *,
        state_map: dict[int, dict[str, Any]],
        state_id: int,
        session_key: str,
    ) -> SendResult:
        """Send a button card and, on success, remember where it went so a click can be validated."""
        response = await self._feishu_send_with_retry(
            chat_id=chat_id,
            msg_type="interactive",
            payload=json.dumps(card, ensure_ascii=False),
            reply_to=None,
            metadata=metadata,
        )
        result = self._finalize_send_result(response, failure_message)
        if result.success:
            state_map[state_id] = {
                "session_key": session_key,
                "message_id": result.message_id or "",
                "chat_id": chat_id,
                "created_at": time.time(),
                "request_id": (metadata or {}).get("__hermes_approval_request_id", ""),
                "owner_ids": (metadata or {}).get("__hermes_prompt_user_ids", ()),
                "chat_type": (metadata or {}).get("__hermes_prompt_chat_type", "dm"),
            }
        return result

    def _on_card_action_trigger(self, data: Any) -> Any:
        # SDK invokes this from its WS or webhook worker, not the gateway loop.
        with self._card_action_lock:
            return self._route_card_action_trigger(data)

    def _route_card_action_trigger(self, data: Any) -> Any:
        """Synchronous SDK card-action callback.

        Approval/update-prompt buttons return the resolved card inline (the only reliable way
        to sync all clients) and schedule the async resolution; other clicks are routed as
        synthetic commands via ``_handle_card_action_event``.
        """
        loop = self._loop
        if not self._loop_accepts_callbacks(loop):
            logger.warning("[Feishu] Dropping card action before adapter loop is ready")
            return self._card_response()
        event = getattr(data, "event", None)
        action = getattr(event, "action", None)
        action_value = getattr(action, "value", {}) or {}
        if isinstance(action_value, dict):
            if action_value.get("hermes_clarify_action"):
                return self._handle_clarify_card_action(event, action_value)
            if action_value.get("hermes_action"):
                return self._handle_approval_card_action(
                    event=event, action_value=action_value, loop=loop
                )
            if action_value.get("hermes_update_prompt_action"):
                return self._handle_update_prompt_card_action(
                    event=event, action_value=action_value, loop=loop
                )
        self._submit_on_loop(loop, self._handle_card_action_event(data))
        return self._card_response()

    @staticmethod
    def _card_response(
        card_data: Optional[dict[str, Any]] = None, *, toast_message: Optional[str] = None
    ) -> Any:
        """Synchronous card-callback response; ``card_data`` updates the card inline."""
        if P2CardActionTriggerResponse is None:
            return None
        response = P2CardActionTriggerResponse()
        if toast_message is not None and CallBackToast is not None:
            response.toast = CallBackToast({"type": "info", "content": toast_message})
        if card_data is not None and CallBackCard is not None:
            card = CallBackCard()
            card.type = "raw"
            card.data = card_data
            response.card = card
        return response

    def _validate_card_action(
        self,
        *,
        event: Any,
        state: dict[str, Any],
        label: str,
        ident: Any,
    ) -> Optional[tuple[str, str, str]]:
        """Shared operator/chat checks for approval + update-prompt clicks.

        Returns ``(open_id, callback_chat_id, user_name)`` or None (already logged).
        """
        if state.get("claimed"):
            return None
        # Clarify validity is the bound gateway waiter's lifetime (checked by
        # _handle_clarify_card_action), including unlimited waits. The callback
        # replay-cache TTL must not silently expire a still-live prompt.
        if (
            label != "clarify"
            and "created_at" in state
            and time.time() - state["created_at"] > _FEISHU_CARD_ACTION_DEDUP_TTL_SECONDS
        ):
            return None
        context = getattr(event, "context", None)
        if "created_at" in state and (
            str(getattr(context, "open_message_id", "") or "") != state.get("message_id")
            or str(getattr(context, "open_chat_id", "") or "") != state.get("chat_id")
        ):
            return None
        request_id = state.get("request_id")
        if request_id and label == "approval":
            from tools.approval import list_gateway_approvals

            if not any(
                item.get("request_id") == request_id
                for item in list_gateway_approvals(state["session_key"])
            ):
                return None
        operator = getattr(event, "operator", None)
        open_id = str(getattr(operator, "open_id", "") or "")
        operator_ids = {
            str(getattr(operator, name, "") or "") for name in ("open_id", "user_id", "union_id")
        } - {""}
        owners = set(state.get("owner_ids") or ())
        if owners and not operator_ids.intersection(owners | self._admins):
            return None
        if self._authorization_check is not None and not any(
            self._is_sender_authorized(ident, state.get("chat_type", "dm"), state.get("chat_id"))
            for ident in operator_ids
        ):
            return None
        if not self._is_interactive_operator_authorized(open_id):
            logger.warning("[Feishu] Unauthorized %s click by %s", label, open_id or "<unknown>")
            return None
        callback_chat_id = str(getattr(getattr(event, "context", None), "open_chat_id", "") or "")
        expected_chat_id = str(state.get("chat_id", "") or "")
        if callback_chat_id and expected_chat_id and callback_chat_id != expected_chat_id:
            logger.warning(
                "[Feishu] %s callback chat mismatch for %s (expected=%s, got=%s)",
                label.capitalize(),
                ident,
                expected_chat_id,
                callback_chat_id,
            )
            return None
        return open_id, callback_chat_id, self._get_cached_sender_name(open_id) or open_id

    def _handle_approval_card_action(
        self, *, event: Any, action_value: dict[str, Any], loop: Any
    ) -> Any:
        """Schedule approval resolution and build the synchronous callback response."""
        approval_id = action_value.get("approval_id")
        if approval_id is None:
            logger.debug("[Feishu] Card action missing approval_id, ignoring")
            return self._card_response()
        state = self._approval_state.get(approval_id)
        if not state:
            logger.debug("[Feishu] Approval %s already resolved or unknown", approval_id)
            return self._card_response()
        choice = _APPROVAL_CHOICE_MAP.get(action_value.get("hermes_action"))
        if choice is None:
            return self._card_response()
        checked = self._validate_card_action(
            event=event, state=state, label="approval", ident=approval_id
        )
        if checked is None:
            return self._card_response()
        open_id, chat_id, user_name = checked
        coro = self._resolve_approval(
            approval_id=approval_id,
            choice=choice,
            user_name=user_name,
            open_id=open_id,
            chat_id=chat_id,
        )
        state["claimed"] = True
        if not self._submit_on_loop(loop, coro):
            state.pop("claimed", None)
            return self._card_response()
        # A toast cannot overwrite a terminal PATCH if the transport ACK is late.
        return self._card_response(toast_message=f"处理中，请等待实际审批结果。操作人：{user_name}")

    def _handle_update_prompt_card_action(
        self, *, event: Any, action_value: dict[str, Any], loop: Any
    ) -> Any:
        """Schedule update prompt resolution and build the synchronous callback response."""
        prompt_id = action_value.get("update_prompt_id")
        if prompt_id is None:
            logger.debug("[Feishu] Card action missing update_prompt_id, ignoring")
            return self._card_response()
        state = self._update_prompt_state.get(prompt_id)
        if not state:
            logger.debug("[Feishu] Update prompt %s already resolved or unknown", prompt_id)
            return self._card_response()
        answer = str(action_value.get("hermes_update_prompt_action", "") or "").strip().lower()
        if answer not in {"y", "n"}:
            logger.debug("[Feishu] Card action has invalid update prompt answer=%r", answer)
            return self._card_response()
        checked = self._validate_card_action(
            event=event, state=state, label="update prompt", ident=prompt_id
        )
        if checked is None:
            return self._card_response()
        open_id, chat_id, user_name = checked
        coro = self._resolve_update_prompt(
            prompt_id, answer, user_name, open_id=open_id, chat_id=chat_id
        )
        state["claimed"] = True
        if not self._submit_on_loop(loop, coro):
            state.pop("claimed", None)
            return self._card_response()
        return self._card_response(
            self._build_resolved_update_prompt_card(answer=answer, user_name=user_name)
        )

    def _pop_validated_prompt_state(
        self,
        *,
        states: dict[int, dict[str, Any]],
        ident: Any,
        label: str,
        open_id: str,
        chat_id: str,
        unauthorized_fmt: str,
        operator_repr: str,
    ) -> Optional[dict[str, str]]:
        """Re-validate on the loop thread (state may have changed since the callback) and pop."""
        state = states.get(ident)
        if not state:
            logger.debug("[Feishu] %s %s already resolved or unknown", label, ident)
            return None
        if not self._is_interactive_operator_authorized(open_id):
            logger.warning(unauthorized_fmt, operator_repr, ident)
            return None
        expected_chat_id = str(state.get("chat_id", "") or "")
        if expected_chat_id and chat_id and expected_chat_id != chat_id:
            logger.warning(
                "[Feishu] %s %s chat mismatch (expected=%s, got=%s)",
                label,
                ident,
                expected_chat_id,
                chat_id,
            )
            return None
        state = states.pop(ident, None)
        if not state:
            logger.debug("[Feishu] %s %s already resolved while validating callback", label, ident)
        return state

    async def _resolve_approval(
        self,
        approval_id: Any,
        choice: str,
        user_name: str,
        *,
        open_id: str = "",
        chat_id: str = "",
    ) -> None:
        """Pop approval state and unblock the waiting agent thread."""
        state = self._pop_validated_prompt_state(
            states=self._approval_state,
            ident=approval_id,
            label="Approval",
            open_id=open_id,
            chat_id=chat_id,
            unauthorized_fmt="[Feishu] Unauthorized approval click by %s for approval %s",
            operator_repr=open_id or "<unknown>",
        )
        if not state:
            return
        try:
            processing = _card(
                "审批处理中", "blue", f"处理中，请等待实际审批结果。\n操作人：{user_name}"
            )
            request = self._build_patch_message_request(
                message_id=state["message_id"],
                request_body=self._build_patch_message_body(
                    content=json.dumps(processing, ensure_ascii=False)
                ),
            )
            response = await self._run_blocking(self._client.im.v1.message.patch, request)
            if not self._response_succeeded(response):
                logger.warning("[Feishu] Approval processing patch failed")
        except Exception:
            logger.warning("[Feishu] Approval processing patch failed", exc_info=True)
        try:
            from tools.approval import resolve_gateway_approval

            request_id = state.get("request_id")
            count = resolve_gateway_approval(
                state["session_key"], choice, **({"request_id": request_id} if request_id else {})
            )
            logger.info(
                "Feishu button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                count,
                state["session_key"],
                choice,
                user_name,
            )
            card = (
                self._build_resolved_approval_card(choice=choice, user_name=user_name)
                if count
                else _card("审批已失效", "grey", t("platform.shared.approval_expired"))
            )
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Feishu button: %s", exc, exc_info=True)
            card = _card("审批失败", "red", "审批未成功，请重新发起请求。")
        try:
            request = self._build_patch_message_request(
                message_id=state["message_id"],
                request_body=self._build_patch_message_body(
                    content=json.dumps(card, ensure_ascii=False)
                ),
            )
            response = await self._run_blocking(self._client.im.v1.message.patch, request)
            if not self._response_succeeded(response):
                logger.warning(
                    "[Feishu] Approval result same-card patch failed: %s",
                    getattr(response, "msg", "unknown"),
                )
        except Exception:
            logger.warning("[Feishu] Approval result same-card patch failed", exc_info=True)

    async def _handle_card_action_event(self, data: Any) -> None:
        """Route Feishu interactive card button clicks as synthetic COMMAND events."""
        event = getattr(data, "event", None)
        token = str(
            getattr(getattr(data, "header", None), "event_id", "")
            or getattr(event, "token", "")
            or ""
        )
        if token and self._is_card_action_duplicate(token):
            logger.debug("[Feishu] Dropping duplicate card action token: %s", token)
            return
        context = getattr(event, "context", None)
        chat_id = str(getattr(context, "open_chat_id", "") or "")
        operator = getattr(event, "operator", None)
        open_id = str(getattr(operator, "open_id", "") or "")
        if not chat_id or not open_id:
            logger.debug("[Feishu] Card action missing chat_id or operator open_id, dropping")
            return
        action = getattr(event, "action", None)
        action_tag = str(getattr(action, "tag", "") or "button")
        action_value = getattr(action, "value", {}) or {}
        synthetic_text = f"/card {action_tag}"
        if action_value:
            try:
                synthetic_text += f" {json.dumps(action_value, ensure_ascii=False)}"
            except (TypeError, ValueError):
                logger.warning("Invalid Feishu action value", exc_info=True)
        logger.info(
            "[Feishu] Routing card action %r from %s in %s as synthetic command",
            action_tag,
            open_id,
            chat_id,
        )
        await self._dispatch_synthetic_event(
            text=synthetic_text,
            message_type=MessageType.COMMAND,
            chat_id=chat_id,
            sender_id=SimpleNamespace(open_id=open_id, user_id=None, union_id=None),
            event_chat_type="group",
            raw_message=data,
            message_id=token or str(uuid.uuid4()),
        )
