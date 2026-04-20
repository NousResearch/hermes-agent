---
name: hermes-atropos-environments
description: Build, test, and debug Hermes Agent RL environments for Atropos training. Covers the HermesAgentBaseEnv interface, reward functions, agent loop integration, evaluation with tools, wandb logging, and the three CLI modes (serve/process/evaluate). Use when creating, reviewing, or fixing RL environments in the hermes-agent repo.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [atropos, rl, environments, training, reinforcement-learning, reward-functions]
    related_skills: [axolotl, fine-tuning-with-trl, lm-evaluation-harness]
---

# Hermes Agent Atropos Environments

Guide for building RL environments in the hermes-agent repo that integrate with the Atropos training framework.

## Architecture Overview

```
Atropos BaseEnv (atroposlib/envs/base.py)
    └── HermesAgentBaseEnv (environments/hermes_base_env.py)
            ├── Handles agent loop orchestration
            ├── Handles tool resolution per group
            ├── Handles ToolContext for reward verification
            └── YOUR ENVIRONMENT (environments/your_env.py)
                    Only implements: setup, get_next_item, format_prompt,
                                    compute_reward, evaluate, wandb_log
```

Hermes environments are special because they run a **multi-turn agent loop with tool calling** — not just single-turn completions. The base env handles the loop; you implement the task and scoring.

## File Locations

| File | Purpose |
|------|---------|
| `environments/hermes_base_env.py` | Base class with agent loop + tool resolution |
| `environments/agent_loop.py` | `HermesAgentLoop` + `AgentResult` dataclass |
| `environments/tool_context.py` | `ToolContext` for reward verification |
| `environments/tool_call_parsers.py` | Phase 2 tool call parsers (hermes, mistral, etc.) |
| `environments/your_env.py` | Your environment implementation |

## Inference Setup — Ask the User First

**IMPORTANT:** Before running any test, evaluation, or data generation command, always ask the user how they want to handle inference. Do NOT assume OpenRouter or any specific endpoint. Present these options:

1. **OpenRouter** — Ask which model they want to use (e.g., `anthropic/claude-sonnet-4.5`, `google/gemini-2.5-pro`, `meta-llama/llama-3.3-70b-instruct`, etc.). Requires `OPENROUTER_API_KEY` in environment.
2. **Self-hosted VLLM endpoint** — Ask for their base URL (e.g., `http://localhost:8000/v1`) and model name. Set `--openai.server_type vllm`.
3. **Other OpenAI-compatible API** — Ask for the base URL, model name, and any required API key. Set `--openai.server_type openai` and `--openai.health_check false`.
4. **Local Atropos training server** — For `serve` mode with a live training loop. Default `http://localhost:8000/v1`.

Once the user tells you their setup, use those values in all CLI commands for that session. Example prompts:

> "Before I run this, how would you like to handle inference?
> 1. OpenRouter (I'll need your preferred model, e.g. claude-sonnet-4.5)
> 2. A self-hosted VLLM endpoint (give me the URL and model name)
> 3. Another OpenAI-compatible API (give me the URL, model, and any auth details)
> 4. Local Atropos training server (serve mode)"

### Key flags by provider:

| Provider | `--openai.server_type` | `--openai.health_check` | `--openai.api_key` |
|----------|----------------------|------------------------|-------------------|
| OpenRouter | `openai` | `false` | `$OPENROUTER_API_KEY` |
| VLLM (self-hosted) | `vllm` | (default) | (not needed) |
| Other OpenAI-compatible | `openai` | `false` | As needed |
| Local Atropos | (default) | (default) | (not needed) |

## Required Methods

### 1. `setup()` — Load dataset and initialize state

```python
async def setup(self) -> None:
    """Called once at startup. Load datasets, initialize state."""
    # Try HuggingFace first, fallback to built-in samples
    try:
        from datasets import load_dataset
        ds = load_dataset("your/dataset", split="test")
        self._items = [...]
    except Exception:
        self._items = BUILTIN_SAMPLES

    # Always split into train/eval
    random.shuffle(self._items)
    eval_size = max(20, int(len(self._items) * 0.1))
    self._eval_items = self._items[:eval_size]
    self._items = self._items[eval_size:]
```

### 2. `get_next_item()` — Return next training item

```python
async def get_next_item(self) -> dict:
    """Return next item, cycling through dataset."""
    item = self._items[self._index % len(self._items)]
    self._index += 1
    return item
```

### 3. `format_prompt(item)` — Convert item to user message

```python
def format_prompt(self, item: dict) -> str:
    """Convert a dataset item into the user-facing prompt."""
    return f"Research this question: {item['question']}"
```

### 4. `compute_reward(item, result, ctx)` — Score the rollout

**CRITICAL**: `result` is an `AgentResult`, NOT a dict. It has these attributes:
- `result.messages` — List of message dicts (OpenAI format)
- `result.turns_used` — Number of LLM calls made
- `result.finished_naturally` — True if model stopped voluntarily
- `result.tool_errors` — List of ToolError objects

**AgentResult does NOT have**: `final_response`, `tool_calls`, `tools_used`.
You must extract these from `result.messages`:

```python
async def compute_reward(self, item, result: AgentResult, ctx: ToolContext) -> float:
    # Extract final response (last assistant message with content)
    final_response = ""
    tools_used = []
    for msg in reversed(result.messages):
        if msg.get("role") == "assistant" and msg.get("content") and not final_response:
            final_response = msg["content"]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                if name:
                    tools_used.append(name)

    # Score using LLM judge, heuristic, or ToolContext verification
    correctness = await self._llm_judge(item, final_response)
    return correctness
```

`ctx` (ToolContext) gives you terminal/file access to the agent's sandbox for verification:
```python
# Run tests in the agent's sandbox
result = ctx.terminal("pytest /workspace/test.py")
return 1.0 if result["exit_code"] == 0 else 0.0
```

### 5. `evaluate()` — Periodic evaluation with full agent loop

**MUST use the full agent loop with tools**, not single-turn chat_completion.
The whole point of hermes-agent environments is agentic evaluation:

```python
async def evaluate(self, *args, **kwargs) -> None:
    import time, uuid
    from environments.agent_loop import HermesAgentLoop
    from environments.tool_context import ToolContext

    start_time = time.time()
    tools, valid_names = self._resolve_tools_for_group()
    samples = []

    for item in self._eval_items[:self.config.eval_size]:
        task_id = str(uuid.uuid4())
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": self.format_prompt(item)})

        agent = HermesAgentLoop(
            server=self.server,
            tool_schemas=tools,
            valid_tool_names=valid_names,
            max_turns=self.config.max_agent_turns,
            task_id=task_id,
            temperature=0.0,  # Deterministic for eval
            max_tokens=self.config.max_token_length,
            extra_body=self.config.extra_body,
        )
        result = await agent.run(messages)

        ctx = ToolContext(task_id)
        try:
            reward = await self.compute_reward(item, result, ctx)
        finally:
            ctx.cleanup()

        samples.append({"prompt": ..., "response": ..., "reward": reward})

    eval_metrics = {"eval/mean_reward": ...}
    await self.evaluate_log(metrics=eval_metrics, samples=samples,
                            start_time=start_time, end_time=time.time())
```

### 6. `wandb_log()` — Custom metrics logging

Always call `super().wandb_log()` at the end:

```python
async def wandb_log(self, wandb_metrics=None):
    if wandb_metrics is None:
        wandb_metrics = {}
    if self._reward_buffer:
        n = len(self._reward_buffer)
        wandb_metrics["train/mean_reward"] = sum(self._reward_buffer) / n
        self._reward_buffer.clear()
    await super().wandb_log(wandb_metrics)  # MUST call super
```

**Pitfall**: `compute_reward` appends to metric buffers. During eval, this pollutes training metrics. Roll back buffer entries added during eval.

## Config Class

Always create a custom config subclass with Pydantic Field descriptors. Key inherited fields you can tune: `enabled_toolsets`, `max_agent_turns`, `agent_temperature`, `system_prompt`, `terminal_backend`, `group_size`, `steps_per_eval`, `total_steps`.

## config_init() — Default Configuration

Classmethod returning `(YourEnvConfig, [APIServerConfig(...)])`. Set server_type to "openai" for OpenRouter/external APIs. Load API key from environment variable.

## Three CLI Modes

```bash
# SERVE — Full training loop (connects to Atropos API server)
python environments/my_env.py serve --openai.base_url http://localhost:8000/v1

# PROCESS — Offline data generation (saves JSONL)
python environments/my_env.py process --env.total_steps 10 --env.group_size 1 \
    --env.use_wandb false --env.data_path_to_save_groups output.jsonl \
    --openai.base_url "<USER_BASE_URL>" \
    --openai.model_name "<USER_MODEL>" \
    --openai.server_type <USER_SERVER_TYPE> --openai.health_check false

# EVALUATE — Standalone eval (runs setup + evaluate only)
python environments/my_env.py evaluate --env.eval_size 20 \
    --env.data_dir_to_save_evals /tmp/eval_results \
    --openai.base_url "<USER_BASE_URL>" \
    --openai.model_name "<USER_MODEL>" \
    --openai.server_type <USER_SERVER_TYPE> --openai.health_check false
```

Config priority: CLI args > YAML file > config_init() defaults.

## Common Pitfalls

1. **AgentResult has .messages, not .final_response** — Extract the final response by iterating reversed(result.messages) looking for the last assistant message with content.

2. **evaluate() must use HermesAgentLoop, not chat_completion** — Single-turn chat_completion has no tools. The whole point of hermes-agent benchmarks is agentic evaluation with tool use.

3. **Don't call _llm_judge twice** — If compute_reward already calls it, extract the score from the buffer instead of calling judge separately in evaluate().

4. **Eval pollutes training buffers** — compute_reward appends to metric buffers. During eval, roll back buffer entries to keep training metrics clean.

5. **Always set health_check=false for OpenRouter** — OpenRouter has no /health endpoint.

6. **Set data_dir_to_save_evals in evaluate mode** — Without it, results aren't saved.

7. **default_toolsets class variable vs enabled_toolsets config** — The class variable is a hint; the config field is what actually controls tool resolution.

8. **Tool call parsing in messages** — Tool calls are dicts with `{"function": {"name": ..., "arguments": ...}}`. Always check `isinstance(tc, dict)`.

9. **ToolContext.cleanup()** — Always call in a finally block to release sandbox resources.

10. **server_type must be "openai" for external APIs** — Without it, Atropos assumes a local VLLM server.

11. **Always ask the user for their inference setup** — Never hardcode or assume a specific provider/model. See the "Inference Setup" section above.

## Reward Function Patterns

### LLM Judge (for open-ended tasks)
Use `self.server.chat_completion()` with a scoring prompt. Parse JSON response for score float. Always include a heuristic fallback (keyword overlap) for when the judge call fails.

### Binary Verification (for code/terminal tasks)
Use `ctx.terminal("pytest test.py -q")` to run tests in the agent's sandbox. Return 1.0 for pass, 0.0 for fail.

### Multi-Signal (combine multiple indicators)
Weight correctness (0.6) + tool usage (0.2) + efficiency (0.2) + optional bonuses. Clamp to [0, 1].

## Testing Your Environment

1. **Import test**: `python -c "from environments.my_env import MyEnv; print('OK')"`
2. **Ask the user for inference setup** (see "Inference Setup" section above)
3. **Process mode** (1 item): Verify JSONL output has valid tokens, masks, scores
4. **Evaluate mode**: Verify full agent loop runs with tools, metrics logged correctly
5. **Check reward range**: Scores should be in [0, 1], not all identical

## Minimum Implementation Checklist

```python
class MyEnv(HermesAgentBaseEnv):
    name = "my-env"
    env_config_cls = MyEnvConfig

    @classmethod
    def config_init(cls): ...          # Default server + env config
    async def setup(self): ...         # Load dataset + train/eval split
    async def get_next_item(self): ... # Cycle through training items
    def format_prompt(self, item): ... # Item → user message string
    async def compute_reward(self, item, result, ctx): ...  # Score rollout
    async def evaluate(self, *args, **kwargs): ...  # Full agent loop eval
    async def wandb_log(self, metrics=None): ...    # Custom metrics + super()

if __name__ == "__main__":
    MyEnv.cli()
```
