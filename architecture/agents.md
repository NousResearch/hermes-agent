# Agents

The agent is the core loop that orchestrates LLM calls and tool execution.

## AIAgent Class

The main agent is implemented in `run_agent.py`:

```python
class AIAgent:
    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_turns: int = 20,
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        verbose_logging: bool = False,
    ):
        # Initialize OpenAI client, load tools based on toolsets
        ...
    
    def chat(self, user_message: str, task_id: str = None) -> str:
        # Main entry point - runs the agent loop
        ...
```

## Agent Loop

The core loop in `_run_agent_loop()`:

```
1. Add user message to conversation
2. Call LLM with tools
3. If LLM returns tool calls:
   - Execute each tool
   - Add tool results to conversation
   - Go to step 2
4. If LLM returns text response:
   - Return response to user
```

```python
while turns < max_turns:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tool_schemas,
    )
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = await execute_tool(tool_call)
            messages.append(tool_result_message(result))
        turns += 1
    else:
        return response.content
```

## Conversation Management

Messages are stored as a list of dicts following OpenAI format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Search for Python tutorials"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "Here's what I found..."},
]
```

## Reasoning Context

For models that support reasoning (chain-of-thought), the agent:
1. Extracts `reasoning_content` from API responses
2. Stores it in `assistant_msg["reasoning"]` for trajectory export
3. Passes it back via `reasoning_content` field on subsequent turns

## Trajectory Export

Conversations can be exported for training:

```python
agent = AIAgent(save_trajectories=True)
agent.chat("Do something")
# Saves to trajectories/*.jsonl in ShareGPT format
```

## Batch Processing

For processing multiple prompts, use `batch_runner.py`:

```bash
python batch_runner.py \
    --dataset_file=prompts.jsonl \
    --batch_size=20 \
    --num_workers=4 \
    --run_name=my_run
```

See `batch_runner.py` for parallel execution with checkpointing.
