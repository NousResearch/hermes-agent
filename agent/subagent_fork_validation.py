"""Host transcript invariants for plugin-formatted child history."""
from copy import deepcopy
import json


_METADATA = {"_compressed_summary", "_compressed_summary_has_user_turn"}


def validate_fork_history(history):
    """Reject malformed/tool-incomplete transcripts and reset persistence identity."""
    if not isinstance(history, list):
        raise ValueError("Fork formatter must return a list of message dictionaries.")
    result, pending, seen = [], set(), set()
    for message in history:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError("Fork formatter returned an invalid message role.")
        if any(not isinstance(key, str) for key in message):
            raise ValueError("Fork message keys must be strings.")
        row = deepcopy({k: v for k, v in message.items() if not k.startswith("_") or k in _METADATA})
        if row.get("content") is not None and not isinstance(row["content"], (str, list)):
            raise ValueError("Fork message content must be text, content parts, or null.")
        if row["role"] == "tool":
            call_id = row.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("Fork history contains an unmatched tool result.")
            pending.remove(call_id)
        else:
            if pending:
                raise ValueError("Fork history contains unfinished tool calls.")
            calls = row.get("tool_calls", [])
            if not isinstance(calls, list) or (calls and row["role"] != "assistant"):
                raise ValueError("Fork tool_calls must be an assistant's list.")
            for call in calls:
                call_id = call.get("id") if isinstance(call, dict) else None
                function = call.get("function") if isinstance(call, dict) else None
                if (not isinstance(call_id, str) or not call_id or call_id in seen
                        or not isinstance(function, dict) or not isinstance(function.get("name"), str)
                        or not isinstance(function.get("arguments"), str)):
                    raise ValueError("Fork history contains an invalid or duplicate tool call.")
                pending.add(call_id)
                seen.add(call_id)
        result.append(row)
    if pending:
        raise ValueError("Fork history contains unfinished tool calls.")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fork history must contain JSON-compatible message data.") from exc
    return result
