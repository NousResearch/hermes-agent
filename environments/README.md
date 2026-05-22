# Hermes-Agent Atropos Environments

This directory contains the integration layer between **hermes-agent's** tool-calling capabilities and the **Atropos** RL training framework. It provides everything needed to run agentic LLMs through multi-turn tool-calling loops, score their output with arbitrary reward functions, and feed results into Atropos for training or evaluation.

## Architecture Overview

```
                        Atropos Framework
                    ┌───────────────────────┐
                    │       BaseEnv          │  (atroposlib)
                    │  - Server management   │
                    │  - Worker scheduling   │
                    │  - Wandb logging       │
                    │  - CLI (serve/process/ │
                    │    evaluate)           │
                    └───────────┬───────────┘
                                │ inherits
                    ┌───────────┴───────────┐
                    │  HermesAgentBaseEnv    │  hermes_base_env.py
                    │  - Terminal backend    │
                    │  - Tool resolution     │
                    │  - Agent loop          │
                    │  - ToolContext          │
                    │  - Async patches       │
                    └───────────┬───────────┘
                                │ inherits
              ┌─────────────────┼─────────────────┐
              │                 │                  │
     TerminalTestEnv     HermesSweEnv    TerminalBench2EvalEnv
     (stack testing)     (SWE training)   (TB2 benchmark eval)
```

### Inheritance Chain

**BaseEnv** (from `atroposlib`) is the Atropos base class. It provides:
- Server management (OpenAI-compatible API servers, VLLM, SGLang)
- Worker scheduling for parallel rollouts
- Wandb integration for metrics and rollout logging
- CLI interface with three subcommands: `serve`, `process`, `evaluate`
- `evaluate_log()` for saving eval results to JSON + samples.jsonl

**HermesAgentBaseEnv** (`hermes_base_env.py`) extends BaseEnv with hermes-agent specifics:
- Sets `os.environ["TERMINAL_ENV"]` to configure the terminal backend (local, docker, ssh, singularity, modal, daytona, vercel_sandbox)
- Resolves hermes-agent toolsets via `_resolve_tools_for_group()` (calls `get_tool_definitions()` which queries `tools/registry.py`)
- Implements `collect_trajectory()` which runs the full agent loop and computes rewards
- Supports two-phase operation (Phase 1: OpenAI server, Phase 2: VLLM ManagedServer)
- Applies monkey patches for async-safe tool operation at import time

Concrete environments inherit from `HermesAgentBaseEnv` and implement:
- `setup()` -- Load dataset, initialize state
- `get_next_item()` -- Return the next item for rollout
- `format_prompt()` -- Convert a dataset item into the user message
- `compute_reward()` -- Score the rollout using ToolContext
- `evaluate()` -- Periodic evaluation logic

## Core Components

### Agent Loop (`agent_loop.py`)

`HermesAgentLoop` is the reusable multi-turn agent engine. It runs the same pattern as hermes-agent's `run_agent.py`:

1. Send messages + tools to the API via `server.chat_completion()`
2. If the response contains `tool_calls`, execute each one via `handle_function_call()` (which delegates to `tools/registry.py`'s `dispatch()`)
3. Append tool results to the conversation and go back to step 1
4. If the response has no tool_calls, the agent is done

Tool calls are executed in a thread pool (`run_in_executor`) so backends that use `asyncio.run()` internally (Modal, Docker) don't deadlock inside Atropos's event loop.

Returns an `AgentResult` containing the full conversation history, turn count, reasoning content per turn, tool errors, and optional ManagedServer state (for Phase 2).

### Tool Context (`tool_context.py`)

`ToolContext` is a per-rollout handle that gives reward/verification functions direct access to **all** hermes-agent tools, scoped to the rollout's `task_id`. The same `task_id` means the terminal/browser session is the SAME one the model used during its rollout -- all state (files, processes, browser tabs) is preserved.

```python
async def compute_reward(self, item, result, ctx: ToolContext):
    # Run tests in the model's terminal sandbox
    test = ctx.terminal("pytest -v")
    if test["exit_code"] == 0:
        return 1.0

    # Check if a file was created
    content = ctx.read_file("/workspace/solution.py")
    if content.get("content"):
        return 0.5

    # Download files locally for verification (binary-safe)
    ctx.download_file("/remote/output.bin", "/local/output.bin")

    return 0.0
```

Available methods:
- **Terminal**: `terminal(command, timeout)` -- run shell commands
- **Files**: `read_file(path)`, `write_file(path, content)`, `search(query, path)`
- **Transfers**: `upload_file()`, `upload_dir()`, `download_file()`, `download_dir()` -- binary-safe file transfers between host and sandbox
- **Web**: `web_search(query)`, `web_extract(urls)`
- **Browser**: `browser_navigate(url)`, `browser_snapshot()`
- **Generic**: `call_tool(name, args)` -- call any hermes-agent tool by name
- **Cleanup**: `cleanup()` -- release all resources (called automatically after `compute_reward`)

### Patches (`patches.py`)

**Problem**: Some hermes-agent tools use `asyncio.run()` internally (e.g., the Modal backend). This crashes when called from inside Atropos's event loop because `asyncio.run()` cannot be nested.

**Solution**: `ModalEnvironment` uses a dedicated `_AsyncWorker` background thread with its own event loop. The calling code sees a sync interface, but internally all async Modal SDK calls happen on the worker thread so they don't conflict with Atropos's loop. This is built directly into `tools/environments/modal.py` — no monkey-patching required.

`patches.py` is now a no-op (kept for backward compatibility with imports).

### Tool Call Parsers (`tool_call_parsers/`)

Client-side parsers that extract structured `tool_calls` from raw model output text. Used in **Phase 2** (VLLM server type) where ManagedServer's `/generate` endpoint returns raw text without tool call parsing.

Each parser is a standalone reimplementation of the corresponding VLLM parser's `extract_tool_calls()` logic. No VLLM dependency -- only standard library (`re`, `json`, `uuid`) and `openai` types.

Available parsers:
- `hermes` -- Hermes/ChatML `<tool_call>` XML format
- `mistral` -- Mistral `[TOOL_CALLS]` format
- `llama3_json` -- Llama 3 JSON tool calling
- `qwen` -- Qwen tool calling format
- `qwen3_coder` -- Qwen3 Coder format
- `deepseek_v3` -- DeepSeek V3 format
- `deepseek_v3_1` -- DeepSeek V3.1 format
- `kimi_k2` -- Kimi K2 format
- `longcat` -- Longcat format
- `glm45` / `glm47` -- GLM model formats

Usage:
```python
from environments.tool_call_parsers import get_parser

parser = get_parser("hermes")
content, tool_calls = parser.parse(raw_model_output)
```

In Phase 1 (OpenAI server type), these parsers are not needed -- the server handles tool call parsing natively.

## Two-Phase Operation

### Phase 1: OpenAI Server (Evaluation / SFT Data Generation)

Uses `server.chat_completion()` with `tools=` parameter. The server (VLLM, SGLang, OpenRouter, OpenAI) handles tool call parsing natively. Returns `ChatCompletion` objects with structured `tool_calls`.

- Good for: evaluation, SFT data generation, testing
- Run with: `serve` (with `run-api`), `process`, or `evaluate` subcommands
- Placeholder tokens are created for the Atropos pipeline

### Phase 2: VLLM ManagedServer (Full RL Training)

Uses ManagedServer for exact token IDs + logprobs via `/generate`. Client-side tool call parser (from `tool_call_parsers/`) reconstructs structured `tool_calls` from raw output.

- Good for: full RL training with GRPO/PPO
- Run with: `serve` subcommand
- Real tokens, masks, and logprobs flow through the pipeline

## Directory Structure

```
environments/
├── README.md                     # This file
├── __init__.py                   # Package exports
├── hermes_base_env.py            # Abstract base (HermesAgentBaseEnv)
├── agent_loop.py                 # Multi-turn agent engine (HermesAgentLoop)
├── tool_context.py               # Per-rollout tool access for reward functions
├── patches.py                    # Async-safety patches for Modal backend
│
├── tool_call_parsers/            # Phase 2 client-side parsers
│   ├── __init__.py               # Registry + base class
│   ├── hermes_parser.py
│   ├── mistral_parser.py
│   ├── llama_parser.py
│   ├── qwen_parser.py
│   ├── qwen3_coder_parser.py
│   ├── deepseek_v3_parser.py
│   ├── deepseek_v3_1_parser.py
│   ├── kimi_k2_parser.py
│   ├── longcat_parser.py
│   ├── glm45_parser.py
│   └── glm47_parser.py
│
├── terminal_test_env/            # Stack validation environment
│   └── terminal_test_env.py
│
├── hermes_swe_env/               # SWE-bench style training environment
│   └── hermes_swe_env.py
│
└── benchmarks/                   # Evaluation benchmarks
    ├── terminalbench_2/          # 89 terminal tasks, Modal sandboxes
    │   └── terminalbench2_env.py
    ├── tblite/                   # 100 calibrated tasks (fast TB2 proxy)
    │   └── tblite_env.py
    └── yc_bench/                 # Long-horizon strategic benchmark
        └── yc_bench_env.py
```

## Concrete Environments

### TerminalTestEnv (`terminal_test_env/`)

A self-contained environment with inline tasks (no external dataset needed) for validating the full stack end-to-end. Each task asks the model to create a file at a known path, and the verifier checks the content matches.

```bash
# Serve mode (needs run-api)
run-api
python environments/terminal_test_env/terminal_test_env.py serve

# Process mode (no run-api, saves to JSONL)
python environments/terminal_test_env/terminal_test_env.py process \
    --env.data_path_to_save_groups terminal_test_output.jsonl
```

### HermesSweEnv (`hermes_swe_env/`)

SWE-bench style training environment. The model gets a coding task, uses terminal + file + web tools to solve it, and the reward function runs tests in the same Modal sandbox.

```bash
python environments/hermes_swe_env/hermes_swe_env.py serve \
    --openai.model_name YourModel \
    --env.dataset_name bigcode/humanevalpack \
    --env.terminal_backend modal
```

### TerminalBench2EvalEnv (`benchmarks/terminalbench_2/`)

**Eval-only** environment for the Terminal-Bench 2.0 benchmark (89 tasks). Each task gets a pre-built Docker Hub image, a natural language instruction, and a test suite. The agent uses terminal + file tools to solve the task, then the test suite verifies correctness.

Follows the standard Atropos eval pattern (like GPQA, MMLU, etc.):
- Run via `evaluate` subcommand (no `run-api` needed)
- `setup()` loads the dataset, `evaluate()` runs all tasks
- `rollout_and_score_eval()` handles per-task agent loop + test verification
- Downloads verifier output locally for reliable reward checking (Harbor pattern)

```bash
# Run full benchmark
python environments/benchmarks/terminalbench_2/terminalbench2_env.py evaluate \
    --openai.model_name anthropic/claude-opus-4.6

# Run subset of tasks
python environments/benchmarks/terminalbench_2/terminalbench2_env.py evaluate \
    --openai.model_name anthropic/claude-opus-4.6 \
    --env.task_filter fix-git,git-multibranch

# Skip specific tasks
python environments/benchmarks/terminalbench_2/terminalbench2_env.py evaluate \
    --openai.model_name anthropic/claude-opus-4.6 \
    --env.skip_tasks heavy-task,slow-task
```

## Creating a New Environment

### Training Environment

1. Create a new directory under `environments/`
2. Create your env file inheriting from `HermesAgentBaseEnv`
3. Implement the four abstract methods + `evaluate()`

```python
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig

class MyEnvConfig(HermesAgentEnvConfig):
    pass  # Add custom fields as needed

class MyEnv(HermesAgentBaseEnv):
    name = "my-env"
    env_config_cls = MyEnvConfig

    @classmethod
    def config_init(cls):
        env_config = MyEnvConfig(
            enabled_toolsets=["terminal", "file"],
            terminal_backend="modal",
            # ... other config
        )
        server_configs = [APIServerConfig(...)]
        return env_config, server_configs

    async def setup(self):
        self.dataset = load_dataset(...)
        self.iter = 0

    async def get_next_item(self):
        item = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1
        return item

    def format_prompt(self, item):
        return item["instruction"]

    async def compute_reward(self, item, result, ctx):
        # ctx gives you full tool access to the rollout's sandbox
        test = ctx.terminal("pytest -v")
        return 1.0 if test["exit_code"] == 0 else 0.0

    async def evaluate(self, *args, **kwargs):
        # Periodic evaluation logic
        ...

if __name__ == "__main__":
    MyEnv.cli()
```

### Eval-Only Environment (Benchmark)

For eval benchmarks, follow the pattern in `terminalbench2_env.py`:
1. Create under `environments/benchmarks/your-benchmark/`
2. Inherit from `HermesAgentBaseEnv`
3. Set eval-only config: `eval_handling=STOP_TRAIN`, `steps_per_eval=1`, `total_steps=1`
4. Stub the training methods (`collect_trajectories`, `score`)
5. Implement `rollout_and_score_eval()` and `evaluate()`
6. Run with `evaluate` subcommand

## Key Config Fields

| Field | Description | Default |
|-------|-------------|---------|
| `enabled_toolsets` | Which hermes toolsets to enable | `None` (all) |
| `disabled_toolsets` | Toolsets to disable | `None` |
| `distribution` | Probabilistic toolset distribution name | `None` |
| `max_agent_turns` | Max LLM calls per rollout | `30` |
| `agent_temperature` | Sampling temperature | `1.0` |
| `terminal_backend` | `local`, `docker`, `modal`, `daytona`, `ssh`, `singularity` | `local` |
| `system_prompt` | System message for the agent | `None` |
| `tool_call_parser` | Parser name for Phase 2 | `hermes` |
| `eval_handling` | `STOP_TRAIN`, `LIMIT_TRAIN`, `NONE` | `STOP_TRAIN` |
