"""Trajectory conversion, JSONL persistence and scratchpad normalization."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Whether content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    return bool(content) and "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


_TRAJECTORY_SYSTEM_PROMPT = (
    "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
    "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
    "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
    "into functions. After calling & executing the functions, you will be provided with function results within "
    "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
    "<tools>\n{tools}\n</tools>\n"
    "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
    "{{'title': 'FunctionCall', 'type': 'object', 'properties': {{'name': {{'title': 'Name', 'type': 'string'}}, "
    "'arguments': {{'title': 'Arguments', 'type': 'object'}}}}, 'required': ['name', 'arguments']}}\n"
    "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
    "Example:\n<tool_call>\n{{'name': <function-name>,'arguments': <args-dict>}}\n</tool_call>"
)


def _trajectory_gpt_prefix(msg: Dict[str, Any]) -> str:
    """Leading ``<think>`` block from native reasoning tokens, if any."""
    if msg.get("reasoning") and msg["reasoning"].strip():
        return f"<think>\n{msg['reasoning']}\n</think>\n"
    return ""


def _with_think_block(content: str) -> str:
    """Every gpt turn gets a <think> block (empty if none) for a consistent training format."""
    return content if "<think>" in content else "<think>\n</think>\n" + content


def _trajectory_tool_call_turn(msg: Dict[str, Any]) -> str:
    content = _trajectory_gpt_prefix(msg)
    if msg.get("content") and msg["content"].strip():
        # <REASONING_SCRATCHPAD> -> <think> (model reasons via XML when native thinking is off)
        content += convert_scratchpad_to_think(msg["content"]) + "\n"
    for tool_call in msg["tool_calls"]:
        if not tool_call or not isinstance(tool_call, dict):
            continue
        raw_args = tool_call["function"]["arguments"]
        # Arguments were validated during conversation; degrade to {} rather than abort.
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            logger.warning("Unexpected invalid JSON in trajectory conversion: %s", raw_args[:100])
            arguments = {}
        tool_call_json = {"name": tool_call["function"]["name"], "arguments": arguments}
        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
    return _with_think_block(content).rstrip()


def _trajectory_tool_responses(msg: Dict[str, Any], messages: List[Dict[str, Any]], start: int) -> Tuple[List[str], int]:
    """Collect the ``<tool_response>`` blocks for the tool run starting at ``start``; returns ``(blocks, next_index)``."""
    tool_responses = []
    j = start
    while j < len(messages) and messages[j]["role"] == "tool":
        tool_msg = messages[j]
        tool_content = tool_msg["content"]
        try:  # pretty-print tool content if it looks like JSON
            if tool_content.strip().startswith(("{", "[")):
                tool_content = json.loads(tool_content)
        except (json.JSONDecodeError, AttributeError):
            pass
        tool_index = len(tool_responses)
        tool_name = (
            msg["tool_calls"][tool_index]["function"]["name"]
            if tool_index < len(msg["tool_calls"])
            else "unknown"
        )
        payload = json.dumps(
            {"tool_call_id": tool_msg.get("tool_call_id", ""), "name": tool_name, "content": tool_content},
            ensure_ascii=False,
        )
        tool_responses.append(f"<tool_response>\n{payload}\n</tool_response>")
        j += 1
    return tool_responses, j


def convert_to_trajectory_format(agent, messages: List[Dict[str, Any]], user_query: Optional[str], completed: bool) -> List[Dict[str, Any]]:
    """Export history when ``user_query`` is None; otherwise retain the canonical dataset prompt."""
    from agent.codex_responses_adapter import _summarize_user_message_for_log
    from agent.tool_dispatch_helpers import _trajectory_normalize_msg

    # Trajectories are text-only: swap image-bearing tool messages for their text_summary so ~1MB
    # base64 blobs are not embedded.
    normalized = [_trajectory_normalize_msg(m) for m in messages]
    if user_query is None:
        for index, message in enumerate(messages):
            if message["role"] == "user":
                # Preserve the first-user summary before image parts become screenshots.
                normalized[index] = {**normalized[index], "content": _summarize_user_message_for_log(message.get("content"))}
                break
    messages = normalized
    trajectory = [
        {"from": "system", "value": _TRAJECTORY_SYSTEM_PROMPT.format(tools=agent._format_tools_for_system_message())},
    ]
    i = 0
    if user_query is not None:
        trajectory.append({"from": "human", "value": user_query})
        i = 1  # Dataset/sample callers supply the canonical replacement for the first row.
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant":
            if msg.get("tool_calls"):
                trajectory.append({"from": "gpt", "value": _trajectory_tool_call_turn(msg)})
                tool_responses, j = _trajectory_tool_responses(msg, messages, i + 1)
                if tool_responses:
                    trajectory.append({"from": "tool", "value": "\n".join(tool_responses)})
                    i = j - 1  # skip the tool messages just processed
            else:
                content = _trajectory_gpt_prefix(msg) + convert_scratchpad_to_think(msg["content"] or "")
                trajectory.append({"from": "gpt", "value": _with_think_block(content).strip()})
        elif msg["role"] == "user":
            trajectory.append({"from": "human", "value": msg["content"]})
        i += 1
    return trajectory


def _lock_append_handle(f, acquire: bool) -> None:
    """Exclusive whole-file lock on an append handle: ``flock`` on POSIX, a 1-byte
    ``msvcrt.locking`` range at offset 0 on Windows (append position is restored by the OS)."""
    if os.name == "nt":
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if acquire else msvcrt.LK_UNLCK, 1)
        f.seek(0, os.SEEK_END)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if acquire else fcntl.LOCK_UN)


def save_trajectory(trajectory: List[Dict[str, Any]], model: str, completed: bool, filename: str = None):
    """Append a ShareGPT-format entry to a JSONL file (default trajectory_samples.jsonl / failed_trajectories.jsonl by ``completed``)."""
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"
    entry = {"conversations": trajectory, "timestamp": datetime.now().isoformat(), "model": model, "completed": completed}
    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"  # serialize before taking the lock
        with open(filename, "a", encoding="utf-8") as f:
            # Gateway sessions and batch workers append to the SAME default file; without an
            # exclusive lock around write+flush, entries larger than one write() interleave and the
            # JSONL stops parsing (#12684).
            _lock_append_handle(f, True)
            try:
                f.write(line)
                f.flush()
            finally:
                _lock_append_handle(f, False)
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
