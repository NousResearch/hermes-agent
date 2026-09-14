"""Responses replay identity safety.

Provider-issued assistant ``message`` IDs can depend on a preceding native
``reasoning`` item identity. Hermes intentionally strips reasoning IDs when
replaying stateless Responses history (``store=False``), so a dependent
``msg_*`` ID must not survive on the outgoing request by itself.
"""

from __future__ import annotations

from typing import Any


def enforce_responses_replay_identity_closure(
    api_kwargs: Any, *, drop_all_message_ids: bool = False,
) -> Any:
    """Return request kwargs with unsafe Responses ``message.id`` values removed.

    The function is deliberately a final wire guard:

    * after a ``reasoning`` item, typed assistant ``message`` IDs are removed
      until a real output-group boundary is reached;
    * a ``function_call`` does not close the group because commentary/final
      messages may still belong to the same reasoning output group;
    * ``function_call_output`` and ordinary role messages close the group;
    * when reasoning replay has been disabled for the session, all typed
      assistant message IDs are removed because their original dependency can
      no longer be proven from the remaining history.

    Only the native identity is removed. Content, status, phase, tool pairing,
    and independent message IDs are preserved. The input object is not mutated.
    """
    if not isinstance(api_kwargs, dict):
        return api_kwargs
    raw_items = api_kwargs.get("input")
    if not isinstance(raw_items, list):
        return api_kwargs

    pending_reasoning_identity = False
    changed = False
    normalized_items: list[Any] = []

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            normalized_items.append(raw_item)
            continue

        item_type = raw_item.get("type")
        if item_type == "reasoning":
            # Preflight intentionally removes reasoning.id for store=False.
            # Any native message identity in this output group must degrade
            # symmetrically or the server sees an orphaned msg_* item.
            pending_reasoning_identity = True

        item = raw_item
        if item_type == "message" and (
            pending_reasoning_identity or drop_all_message_ids
        ) and "id" in raw_item:
            item = dict(raw_item)
            item.pop("id", None)
            changed = True

        normalized_items.append(item)

        # Tool execution completes the reasoning/tool output group. Untyped
        # role messages are also explicit conversation boundaries. A bare
        # function_call is intentionally NOT a boundary.
        if item_type == "function_call_output" or (
            item_type is None and raw_item.get("role") in {"user", "assistant"}
        ):
            pending_reasoning_identity = False

    if not changed:
        return api_kwargs
    sanitized = dict(api_kwargs)
    sanitized["input"] = normalized_items
    return sanitized
