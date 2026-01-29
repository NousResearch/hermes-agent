# Message Format & Trajectories

Hermes Agent uses two message formats: the **API format** for LLM calls and the **trajectory format** for training data export.

## API Message Format

Standard OpenAI chat format used during execution:

```python
messages = [
    # System prompt
    {"role": "system", "content": "You are a helpful assistant with tools..."},
    
    # User query
    {"role": "user", "content": "Search for Python tutorials"},
    
    # Assistant with tool call
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": "{\"query\": \"Python tutorials\"}"
            }
        }]
    },
    
    # Tool result
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": "{\"results\": [...]}"
    },
    
    # Final response
    {"role": "assistant", "content": "Here's what I found..."}
]
```

## Trajectory Format (ShareGPT)

Exported for training in ShareGPT format:

```json
{
    "conversations": [
        {"from": "system", "value": "You are a helpful assistant..."},
        {"from": "human", "value": "Search for Python tutorials"},
        {"from": "gpt", "value": "<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"Python tutorials\"}}\n</tool_call>"},
        {"from": "tool", "value": "<tool_response>\n{\"results\": [...]}\n</tool_response>"},
        {"from": "gpt", "value": "Here's what I found..."}
    ],
    "tools": "[{\"type\": \"function\", \"function\": {...}}]",
    "source": "hermes-agent"
}
```

## Reasoning Content

For models that output reasoning/chain-of-thought:

**During execution** (API format):
```python
# Stored internally but not sent back to model in content
assistant_msg = {
    "role": "assistant",
    "content": "Here's what I found...",
    "reasoning": "Let me think about this step by step..."  # Internal only
}
```

**In trajectory export** (reasoning wrapped in tags):
```json
{
    "from": "gpt",
    "value": "<think>\nLet me think about this step by step...\n</think>\nHere's what I found..."
}
```

## Conversion Flow

```
API Response → Internal Storage → Trajectory Export
     ↓              ↓                    ↓
tool_calls    reasoning field      <tool_call> tags
reasoning_content                  <think> tags
```

The conversion happens in `_convert_to_trajectory_format()` in `run_agent.py`.

## Ephemeral System Prompts

Batch processing supports ephemeral system prompts that guide behavior during execution but are NOT saved to trajectories:

```python
# During execution: full system prompt + ephemeral guidance
messages = [
    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + ephemeral_prompt},
    ...
]

# In saved trajectory: only the base system prompt
trajectory = {
    "conversations": [
        {"from": "system", "value": SYSTEM_PROMPT},  # No ephemeral
        ...
    ]
}
```

## Trajectory Compression

Long trajectories can be compressed for training using `trajectory_compressor.py`:

- Protects first/last N turns
- Summarizes middle turns with LLM
- Targets specific token budget
- See `configs/trajectory_compression.yaml` for settings
