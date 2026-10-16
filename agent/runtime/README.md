# Tier-1 scaffold: `agent/runtime/`

This is a self-contained, provider-agnostic step machine intended to drop into
[`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent) as a
new package `agent/runtime/`. See the design rationale in
`../2026-05-15-smolagents-runtime-upgrade.md`.

**What this is:** a typed loop + memory + callback system modeled on
smolagents' `MultiStepAgent`, but with **no smolagents dependency**.

**What this is not:** a replacement for `AIAgent`. It is the substrate
`AIAgent.run()` should *delegate to* once integrated.

## Layout

```
agent/runtime/
├── __init__.py        # public surface
├── steps.py           # MemoryStep, ActionStep, PlanningStep, FinalAnswerStep, ToolCall, ToolOutput
├── result.py          # RunResult, TokenUsage
├── memory.py          # AgentMemory — append-only typed history; replay(); to/from JSONL
├── callbacks.py       # StepCallback Protocol + CallbackRegistry (per-step-type dispatch)
├── interfaces.py      # ModelProtocol, ToolHandlerProtocol, ModelOutput
└── loop.py            # MultiStepLoop — provider-agnostic orchestration
tests/runtime/
├── test_steps.py
├── test_memory.py
├── test_callbacks.py
└── test_loop.py
```

## How to drop into hermes-agent

1. **Copy `agent/runtime/` into the hermes-agent repo root** (under the existing
   `agent/` package).
2. **Copy `tests/runtime/` into the hermes-agent `tests/` tree.**
3. **Make hermes adapters satisfy `interfaces.ModelProtocol`** by adding a
   thin shim in each adapter (`anthropic_adapter`, `gemini_native_adapter`, …)
   that exposes:

   ```python
   def generate(self, messages, tools=None, **kwargs) -> ModelOutput: ...
   ```

   Internally it can call the existing provider-specific code paths.
4. **Make `model_tools.handle_function_call` satisfy `ToolHandlerProtocol`** by
   adding a wrapper that takes `list[ToolCall]` and returns `list[ToolOutput]`.
   The existing parallel-execution helpers in `run_agent.py`
   (`_should_parallelize_tool_batch`, `_extract_parallel_scope_path`,
   `_paths_overlap`) move into this wrapper.
5. **In `run_agent.AIAgent`, replace the inner `while` loop** with a
   `MultiStepLoop` instance. The big extracted helpers (`IterationBudget`,
   `_should_parallelize_tool_batch`, `_repair_tool_call_arguments`, surrogate
   sanitizers) become `step_callbacks` rather than inline branches.
6. **Run the existing test suite.** Behavior must not change. Add new tests
   under `tests/runtime/` for any extracted callback.

## Running the tests locally (in this workspace)

The scaffold uses only the Python standard library. From this directory:

```bash
cd docs/proposals/scaffold
python3 -m pytest tests/runtime -v
```

(or `python3 -m unittest discover -s tests/runtime -v` if pytest isn't available)

## Design constraints

- **No external runtime deps.** Standard library only.
- **Python 3.11+.** Matches hermes-agent's installer requirement.
- **Append-only memory.** `AgentMemory` never mutates past steps — this is
  what makes replay deterministic.
- **Step types are frozen dataclasses.** Step records are immutable once
  appended.
- **No I/O in the loop.** All side effects go through the `ModelProtocol` and
  `ToolHandlerProtocol` injected at construction. The loop never reads
  config, never opens files, never logs to disk. Logging/telemetry is a
  callback concern.
- **Interruption is cooperative.** `loop.interrupt()` flips a flag checked
  between steps. It does not raise from inside the model call — the
  injected model owns that.

## What this scaffold deliberately leaves out

- Provider adapters (Anthropic / Gemini / Bedrock / …) — they stay in the
  existing `agent/*_adapter.py` files.
- Compression, credential pooling, rate guarding — these become
  `step_callbacks`, but the implementations stay where they are.
- Streaming — Tier 1 surfaces the same final result the existing loop does.
  Streaming is a Tier-1.5 follow-up: the loop accepts an `event_sink`
  callable that downstream code (`cli.py`'s SSE handler) can subscribe to.
- CodeAgent / `PythonExecutor` — that's Tier 2 in the parent design doc.
