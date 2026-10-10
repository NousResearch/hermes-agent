"""Request-local tool identities for lossless Chat Completions history replay."""

from collections import deque
from typing import Any

from agent.message_sanitization import (
    coalesce_tool_call_id,
    tool_call_id_variants,
    tool_result_id_variants,
)


class _ReplayIds:
    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.suffixes: dict[str, int] = {}
        self.pending: dict[str, deque[tuple[int, str]]] = {}
        self.answered: set[int] = set()
        self.next_call = 0

    def for_call(self, call: dict) -> str:
        base = coalesce_tool_call_id(call)
        if not base:
            return ""
        wire_id = base
        suffix = self.suffixes.get(base, 2)
        while wire_id in self.seen:
            wire_id = f"{base}_d{suffix}"
            suffix += 1
        self.suffixes[base] = suffix
        self.seen.add(wire_id)
        occurrence = (self.next_call, wire_id)
        self.next_call += 1
        for alias in tool_call_id_variants(call):
            self.pending.setdefault(alias, deque()).append(occurrence)
        return wire_id

    def for_result(self, raw_id: Any) -> Any:
        candidates = []
        for alias in tool_result_id_variants(raw_id):
            queue = self.pending.get(alias)
            while queue and queue[0][0] in self.answered:
                queue.popleft()
            if queue:
                candidates.append(queue[0])
        if not candidates:
            return raw_id
        index, wire_id = min(candidates)
        self.answered.add(index)
        return wire_id


def _rewrite_message(message: dict, ids: _ReplayIds) -> dict:
    role = message.get("role")
    if role == "tool":
        raw_id = message.get("tool_call_id")
        wire_id = ids.for_result(raw_id)
        return {**message, "tool_call_id": wire_id} if wire_id != raw_id else message
    if role != "assistant" or not isinstance(message.get("tool_calls"), list):
        return message
    calls = message["tool_calls"]
    rewritten = []
    changed = False
    for call in calls:
        wire_id = ids.for_call(call) if isinstance(call, dict) else ""
        if wire_id and wire_id != call.get("id"):
            call = {**call, "id": wire_id}
            changed = True
        rewritten.append(call)
    return {**message, "tool_calls": rewritten} if changed else message


def normalize_replayed_tool_call_ids(messages: list[dict]) -> list[dict]:
    """Give every call occurrence a unique wire ID and preserve its matching result.

    Allocation only reads the prefix already visited: appending a future raw ID
    that collides with a generated suffix cannot change an earlier request prefix.
    Responses bridge aliases are matched before the transport strips sidecars.
    Durable history and clean message objects stay untouched.
    """
    ids = _ReplayIds()
    rewritten = [_rewrite_message(row, ids) if isinstance(row, dict) else row for row in messages]
    return rewritten if any(old is not new for old, new in zip(messages, rewritten)) else messages
