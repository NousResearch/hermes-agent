# AgentResult Fields Reference

`AgentResult` is defined in `environments/agent_loop.py` as a dataclass.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `List[Dict[str, Any]]` | Full conversation history in OpenAI message format |
| `managed_state` | `Optional[Dict]` | ManagedServer.get_state() if Phase 2, else None |
| `turns_used` | `int` | Number of LLM calls made during the loop |
| `finished_naturally` | `bool` | True if model stopped calling tools on its own |
| `reasoning_per_turn` | `List[Optional[str]]` | Extracted reasoning content per turn |
| `tool_errors` | `List[ToolError]` | Tool errors encountered during the loop |

## ToolError Fields

| Field | Type | Description |
|-------|------|-------------|
| `turn` | `int` | Which turn the error occurred |
| `tool_name` | `str` | Name of the tool that failed |
| `arguments` | `str` | Arguments passed to the tool |
| `error` | `str` | Error message |
| `tool_result` | `str` | The result returned to the model |

## Extracting Data from Messages

Messages follow OpenAI format. Common patterns:

```python
# Get final assistant response
for msg in reversed(result.messages):
    if msg.get("role") == "assistant" and msg.get("content"):
        final_response = msg["content"]
        break

# Get all tool names used
tools = []
for msg in result.messages:
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            tools.append(fn.get("name", ""))

# Get tool results
for msg in result.messages:
    if msg.get("role") == "tool":
        tool_output = msg.get("content", "")
        call_id = msg.get("tool_call_id", "")
```

## Fields that DO NOT EXIST

These are common mistakes — AgentResult does NOT have:
- `final_response` — extract from messages
- `tool_calls` — extract from messages  
- `tools_used` — extract from messages
- `output` — extract from messages
- `response` — extract from messages
