"""Plugin transformation contract for built-in context compaction input."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

COMPACTION_HOOK_PROVENANCE_KEY = "compaction_hook_provenance"
_VALID_ACTIONS = frozenset({"keep", "drop", "shorten"})


@dataclass(frozen=True)
class CompactionHookResult:
    """Transformed summary blocks plus durable, model-hidden decision provenance."""

    messages: list[dict[str, Any]]
    provenance: dict[str, Any] | None


def compaction_input_hook_enabled() -> bool:
    """Whether a plugin owns pre-summary block selection for this process."""
    try:
        from hermes_cli.lifecycle import has_hook

        return has_hook("transform_compaction_input")
    except Exception:
        return False


def _tool_call_field(tool_call: Any, field: str) -> Any:
    if isinstance(tool_call, dict):
        if field in tool_call:
            return tool_call[field]
        function = tool_call.get("function")
    else:
        value = getattr(tool_call, field, None)
        if value is not None:
            return value
        function = getattr(tool_call, "function", None)
    if field != "name":
        return None
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


def _block_payloads(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    call_names: dict[str, str] = {}
    for message in messages:
        for tool_call in message.get("tool_calls") or ():
            call_id = str(_tool_call_field(tool_call, "id") or "")
            name = str(_tool_call_field(tool_call, "name") or "")
            if call_id and name:
                call_names[call_id] = name

    all_names: list[str] = []
    blocks: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        names = [
            str(_tool_call_field(tool_call, "name") or "")
            for tool_call in message.get("tool_calls") or ()
        ]
        names = [name for name in names if name]
        tool_call_id = str(message.get("tool_call_id") or "")
        result_name = str(message.get("name") or call_names.get(tool_call_id) or "")
        if result_name and result_name not in names:
            names.append(result_name)
        for name in names:
            if name not in all_names:
                all_names.append(name)
        blocks.append({
            "block_index": index,
            "role": message.get("role"),
            "content": copy.deepcopy(message.get("content")),
            "tool_names": names,
            "tool_call_id": tool_call_id,
            "message": copy.deepcopy(message),
        })
    return blocks, all_names


def _validated_decisions(value: Any, block_count: int) -> list[dict[str, Any]] | None:
    if not isinstance(value, dict) or not isinstance(value.get("decisions"), list):
        return None
    decisions: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in value["decisions"]:
        if not isinstance(raw, dict):
            return None
        index = raw.get("block_index")
        action = raw.get("action")
        if type(index) is not int or index < 0 or index >= block_count or index in seen or action not in _VALID_ACTIONS:
            return None
        if action == "shorten" and not isinstance(raw.get("content"), str):
            return None
        seen.add(index)
        decision = {"block_index": index, "action": action}
        if action == "shorten":
            decision["content"] = raw["content"]
        decisions.append(decision)
    return decisions


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_chars(part.get("text") if isinstance(part, dict) else part) for part in content)
    return 0


def _apply_decisions(
    messages: list[dict[str, Any]], decisions: Iterable[dict[str, Any]], *,
    blocks: list[dict[str, Any]], task_source: dict[str, Any] | None,
) -> CompactionHookResult:
    by_index = {decision["block_index"]: decision for decision in decisions}
    transformed: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        decision = by_index.get(index)
        if decision is None:
            transformed.append(message)
            continue
        action = decision["action"]
        original_chars = _content_chars(message.get("content"))
        result_chars = original_chars
        if action == "drop":
            result_chars = 0
        else:
            new_message = copy.deepcopy(message)
            if action == "shorten":
                new_message["content"] = decision["content"]
                result_chars = len(decision["content"])
            transformed.append(new_message)
        records.append({
            "block_index": index,
            "action": action,
            "tool_names": list(blocks[index]["tool_names"]),
            "original_chars": original_chars,
            "result_chars": result_chars,
        })
    provenance = {"task_source": task_source, "decisions": records} if records else None
    return CompactionHookResult(transformed, provenance)


def transform_compaction_input(
    messages: list[dict[str, Any]], *, task_text: str, task_message_index: int | None,
    task_id: str = "", session_id: str = "",
) -> CompactionHookResult:
    """Apply the first valid ``transform_compaction_input`` plugin result.

    Plugins receive immutable-by-convention deep copies. Invalid results and hook failures are
    fail-open, preserving the original summary input. The host owns task provenance so a plugin
    cannot attribute its decision to a different instruction.
    """
    try:
        from hermes_cli.lifecycle import invoke_hook

        if not compaction_input_hook_enabled():
            return CompactionHookResult(messages, None)
        blocks, tool_names = _block_payloads(messages)
        task_source = (
            {"message_index": task_message_index, "content": task_text, "task_id": task_id}
            if task_message_index is not None else None
        )
        results = invoke_hook(
            "transform_compaction_input",
            blocks=copy.deepcopy(blocks),
            task_text=task_text,
            task_source=copy.deepcopy(task_source),
            tool_names=list(tool_names),
            task_id=task_id,
            session_id=session_id,
        )
    except Exception as exc:
        logger.debug("transform_compaction_input hook error: %s", exc)
        return CompactionHookResult(messages, None)

    for value in results:
        decisions = _validated_decisions(value, len(messages))
        if decisions is not None:
            return _apply_decisions(messages, decisions, blocks=blocks, task_source=task_source)
    return CompactionHookResult(messages, None)
