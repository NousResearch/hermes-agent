"""Approval card flow — Hermes business module routed off cardAction events.

Design rule: this module does not import FeishuAdapter at runtime. All
dependencies flow in through the ``adapter`` parameter, avoiding circular
imports.

Public API (delegate targets for FeishuAdapter instance methods):
    async def _send_exec_approval_impl(adapter, chat_id, ...) -> SendResult
    async def _resolve_approval_via_sdk_impl(adapter, action, hermes_action) -> None

Module-internal helpers:
    - _build_resolved_approval_card(*, choice, user_name) -> dict

Module-level constants:
    - _APPROVAL_CHOICE_MAP / _APPROVAL_LABEL_MAP
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from gateway.platforms.base import SendResult
from hermes_constants import get_hermes_home

if TYPE_CHECKING:
    # Type hint only; not imported at runtime to avoid a cycle.
    from gateway.platforms.feishu.adapter import FeishuAdapter
    from lark_oapi.channel import CardActionEvent

logger = logging.getLogger(__name__)


_APPROVAL_CHOICE_MAP: Dict[str, str] = {
    "approve_once": "once",
    "approve_session": "session",
    "approve_always": "always",
    "deny": "deny",
}
_APPROVAL_LABEL_MAP: Dict[str, str] = {
    "once": "Approved once",
    "session": "Approved for session",
    "always": "Approved permanently",
    "deny": "Denied",
}


def _target_and_opts(
    adapter: "FeishuAdapter",
    chat_id: str,
    metadata: Optional[Dict[str, Any]],
) -> tuple[str, Optional[Dict[str, Any]]]:
    resolver = getattr(adapter, "_target_and_send_opts", None)
    if callable(resolver):
        return resolver(chat_id, None, metadata)
    return chat_id, None


def _build_resolved_approval_card(*, choice: str, user_name: str) -> Dict[str, Any]:
    """Build raw card JSON for a resolved approval action."""
    icon = "❌" if choice == "deny" else "✅"
    label = _APPROVAL_LABEL_MAP.get(choice, "Resolved")
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"content": f"{icon} {label}", "tag": "plain_text"},
            "template": "red" if choice == "deny" else "green",
        },
        "elements": [
            {
                "tag": "markdown",
                "content": f"{icon} **{label}** by {user_name}",
            },
        ],
    }


async def _send_exec_approval_impl(
    adapter: "FeishuAdapter",
    chat_id: str,
    command: str,
    session_key: str,
    description: str = "dangerous command",
    metadata: Optional[Dict[str, Any]] = None,
) -> SendResult:
    """Send an interactive card with approval buttons.

    The buttons carry ``hermes_action`` in their value dict so that the
    cardAction handler can intercept them and call
    ``resolve_gateway_approval()`` to unblock the waiting agent thread.
    Sends go via ``channel.send`` (interactive).
    """
    if not adapter._channel:
        return SendResult(success=False, error="Not connected")

    try:
        approval_id = next(adapter._approval_counter)
        cmd_preview = command[:3000] + "..." if len(command) > 3000 else command

        def _btn(label: str, action_name: str, btn_type: str = "default") -> dict:
            return {
                "tag": "button",
                "text": {"tag": "plain_text", "content": label},
                "type": btn_type,
                "value": {"hermes_action": action_name, "approval_id": approval_id},
            }

        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"content": "⚠️ Command Approval Required", "tag": "plain_text"},
                "template": "orange",
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": f"```\n{cmd_preview}\n```\n**Reason:** {description}",
                },
                {
                    "tag": "action",
                    "actions": [
                        _btn("✅ Allow Once", "approve_once", "primary"),
                        _btn("✅ Session", "approve_session"),
                        _btn("✅ Always", "approve_always"),
                        _btn("❌ Deny", "deny", "danger"),
                    ],
                },
            ],
        }

        target_id, send_opts = _target_and_opts(adapter, chat_id, metadata)
        sdk_result = await adapter._channel.send(
            target_id,
            {"card": card},
            opts=send_opts,
        )
        if sdk_result.success:
            adapter._approval_state[approval_id] = {
                "session_key": session_key,
                "message_id": sdk_result.message_id or "",
                "chat_id": chat_id,
            }
            return SendResult(success=True, message_id=sdk_result.message_id)
        return SendResult(
            success=False,
            error=str(sdk_result.error) if sdk_result.error else "send_exec_approval failed",
        )
    except Exception as exc:
        logger.warning("[Feishu] send_exec_approval failed: %s", exc)
        return SendResult(success=False, error=str(exc))


async def _resolve_approval_via_sdk_impl(
    adapter: "FeishuAdapter",
    action: "CardActionEvent",
    hermes_action: str,
) -> None:
    """Approval-button branch of _on_sdk_card_action.

    Looks up _approval_state, calls tools.approval.resolve_gateway_approval,
    then refreshes the original card via channel.update_card. The resolved
    card is pushed asynchronously via update_card; the SDK self-manages the
    synchronous P2CardActionTriggerResponse for the dispatcher.
    """
    payload = action.action if action.action else None
    value = payload.value if payload is not None else {}
    approval_id = value.get("approval_id") if isinstance(value, dict) else None
    if approval_id is None:
        logger.debug("[Feishu] Card action missing approval_id, ignoring")
        return

    choice = _APPROVAL_CHOICE_MAP.get(hermes_action, "deny")
    operator = action.operator
    operator_open_id = operator.open_id if operator else ""
    user_name = (operator.name if operator else "") or operator_open_id

    state = adapter._approval_state.get(approval_id)
    if not state:
        logger.debug("[Feishu] Approval %s already resolved or unknown", approval_id)
        return

    # 1) Unblock the waiting agent thread. Pop state only on success so that
    # transient resolve failures leave the state available for retry rather
    # than stranding the user with a stuck card and the agent blocked.
    try:
        from tools.approval import resolve_gateway_approval
        count = resolve_gateway_approval(state["session_key"], choice)
        logger.info(
            "Feishu button resolved %d approval(s) for session %s (choice=%s, user=%s)",
            count, state["session_key"], choice, user_name,
        )
    except Exception as exc:
        logger.error("Failed to resolve gateway approval from Feishu button: %s", exc)
        return
    adapter._approval_state.pop(approval_id, None)

    # 2) Refresh the original card asynchronously.
    try:
        message_id = state.get("message_id") or action.message_id
        if message_id and adapter._channel is not None:
            resolved_card = _build_resolved_approval_card(
                choice=choice, user_name=user_name,
            )
            await adapter._channel.update_card(message_id, resolved_card)
    except Exception as exc:
        logger.warning(
            "[Feishu] update_card for resolved approval %s failed: %s",
            approval_id, exc,
        )


def _build_update_prompt_card(*, prompt: str, default: str, prompt_id: int) -> Dict[str, Any]:
    default_hint = f"\n\nDefault: `{default}`" if default else ""

    def _btn(label: str, answer: str, btn_type: str) -> dict:
        return {
            "tag": "button",
            "text": {"tag": "plain_text", "content": label},
            "type": btn_type,
            "value": {
                "hermes_update_prompt_action": answer,
                "update_prompt_id": prompt_id,
            },
        }

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"content": "⚕ Update Needs Your Input", "tag": "plain_text"},
            "template": "orange",
        },
        "elements": [
            {"tag": "markdown", "content": f"{prompt}{default_hint}"},
            {
                "tag": "action",
                "actions": [
                    _btn("✓ Yes", "y", "primary"),
                    _btn("✗ No", "n", "danger"),
                ],
            },
        ],
    }


def _build_resolved_update_prompt_card(*, answer: str, user_name: str) -> Dict[str, Any]:
    yes = answer == "y"
    label = "Yes" if yes else "No"
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"content": f"{'✅' if yes else '❌'} Update prompt answered: {label}", "tag": "plain_text"},
            "template": "green" if yes else "red",
        },
        "elements": [
            {"tag": "markdown", "content": f"Answered by **{user_name}**"},
        ],
    }


def _write_update_prompt_response(answer: str) -> None:
    response_path = get_hermes_home() / ".update_response"
    tmp_path = response_path.with_suffix(".tmp")
    tmp_path.write_text(answer)
    tmp_path.replace(response_path)


def _is_interactive_operator_authorized(adapter: "FeishuAdapter", open_id: str) -> bool:
    normalized = str(open_id or "").strip()
    if not normalized:
        return False
    allowed_ids = set(getattr(adapter, "_admins", set())) | set(
        getattr(adapter, "_allowed_group_users", set())
    )
    if not allowed_ids:
        return True
    return "*" in allowed_ids or normalized in allowed_ids


async def _send_update_prompt_impl(
    adapter: "FeishuAdapter",
    chat_id: str,
    prompt: str,
    default: str = "",
    session_key: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> SendResult:
    """Send an interactive update prompt with Yes/No buttons."""
    if not adapter._channel:
        return SendResult(success=False, error="Not connected")

    try:
        prompt_id = next(adapter._update_prompt_counter)
        card = _build_update_prompt_card(
            prompt=prompt, default=default, prompt_id=prompt_id,
        )
        target_id, send_opts = _target_and_opts(adapter, chat_id, metadata)
        sdk_result = await adapter._channel.send(
            target_id,
            {"card": card},
            opts=send_opts,
        )
        if sdk_result.success:
            adapter._update_prompt_state[prompt_id] = {
                "session_key": session_key,
                "message_id": sdk_result.message_id or "",
                "chat_id": chat_id,
            }
            return SendResult(success=True, message_id=sdk_result.message_id)
        return SendResult(
            success=False,
            error=str(sdk_result.error) if sdk_result.error else "send_update_prompt failed",
        )
    except Exception as exc:
        logger.warning("[Feishu] send_update_prompt failed: %s", exc)
        return SendResult(success=False, error=str(exc))


async def _resolve_update_prompt_via_sdk_impl(
    adapter: "FeishuAdapter",
    action: "CardActionEvent",
    answer: str,
) -> None:
    """Persist update-prompt answers and refresh the original card."""
    answer = str(answer or "").strip().lower()
    if answer not in {"y", "n"}:
        logger.debug("[Feishu] Card action has invalid update prompt answer=%r", answer)
        return

    payload = action.action if action.action else None
    value = payload.value if payload is not None else {}
    prompt_id = value.get("update_prompt_id") if isinstance(value, dict) else None
    if prompt_id is None:
        logger.debug("[Feishu] Card action missing update_prompt_id, ignoring")
        return

    operator = action.operator
    operator_open_id = operator.open_id if operator else ""
    if not _is_interactive_operator_authorized(adapter, operator_open_id):
        logger.warning(
            "[Feishu] Unauthorized update prompt click by %s",
            operator_open_id or "<unknown>",
        )
        return

    user_name = (operator.name if operator else "") or operator_open_id
    state = adapter._update_prompt_state.pop(prompt_id, None)
    if not state:
        logger.debug("[Feishu] Update prompt %s already resolved or unknown", prompt_id)
        return

    try:
        _write_update_prompt_response(answer)
        logger.info(
            "Feishu update prompt resolved for session %s (answer=%s, user=%s)",
            state["session_key"], answer, user_name,
        )
    except Exception as exc:
        logger.error("Failed to resolve Feishu update prompt: %s", exc)
        return

    try:
        message_id = state.get("message_id") or action.message_id
        if message_id and adapter._channel is not None:
            await adapter._channel.update_card(
                message_id,
                _build_resolved_update_prompt_card(answer=answer, user_name=user_name),
            )
    except Exception as exc:
        logger.warning(
            "[Feishu] update_card for resolved update prompt %s failed: %s",
            prompt_id, exc,
        )
