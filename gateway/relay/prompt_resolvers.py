"""Relay prompt-response resolvers: one per prompt kind (exec approval, slash confirm, clarify).

Each runs on the transport read loop with the adapter, the popped prompt state, the chosen option,
the chat id and the ack metadata; acks go through ``adapter._send_lifecycle_ack`` (fire-and-forget).
"""

from __future__ import annotations


async def resolve_exec_approval(adapter, state, option_id, chat_id, ack_meta) -> None:
    from gateway.relay import adapter as _relay
    from tools.approval import resolve_gateway_approval

    choice = option_id if option_id in _relay._EXEC_APPROVAL_LABELS else "deny"
    count = resolve_gateway_approval(str(state.get("session_key") or ""), choice)
    label = _relay._EXEC_APPROVAL_LABELS[choice] if count else "⌛ Approval expired — no command was waiting."
    # In-channel ack preserves the audit trail the native edit gives (the
    # connector's prompt message can't be edited cross-platform yet).
    adapter._send_lifecycle_ack(chat_id, label, ack_meta)
    if count:
        adapter.resume_typing_for_chat(chat_id)


async def resolve_slash_confirm(adapter, state, option_id, chat_id, ack_meta) -> None:
    from gateway.relay import adapter as _relay
    from tools import slash_confirm as slash_confirm_mod

    choice = option_id if option_id in _relay._SLASH_CONFIRM_LABELS else "cancel"
    result_text = await slash_confirm_mod.resolve(
        str(state.get("session_key") or ""), str(state.get("confirm_id") or ""), choice
    )
    adapter._send_lifecycle_ack(chat_id, _relay._SLASH_CONFIRM_LABELS[choice], ack_meta)
    if result_text:
        adapter._send_lifecycle_ack(chat_id, str(result_text), ack_meta)


async def resolve_clarify(adapter, state, option_id, chat_id, ack_meta) -> None:
    from tools.clarify_gateway import mark_awaiting_text, resolve_gateway_clarify

    clarify_id = str(state.get("clarify_id") or "")
    if option_id == "other":
        mark_awaiting_text(clarify_id)
        adapter._send_lifecycle_ack(chat_id, "✏️ Type your answer:", ack_meta)
        return
    choices = state.get("choices") or []
    try:
        idx = int(option_id[1:]) if option_id.startswith("c") else -1
    except ValueError:
        idx = -1
    if 0 <= idx < len(choices):
        resolve_gateway_clarify(clarify_id, str(choices[idx]))
        adapter._send_lifecycle_ack(chat_id, f"✅ {choices[idx]}", ack_meta)
    else:
        # Unmappable option: flip to text capture (never dead-end a clarify).
        mark_awaiting_text(clarify_id)


# prompt kind -> resolver (order-independent: kinds are distinct keys).
PROMPT_RESOLVERS = {
    "exec_approval": resolve_exec_approval,
    "slash_confirm": resolve_slash_confirm,
    "clarify": resolve_clarify,
}
