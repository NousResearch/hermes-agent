# `agent/runtime/` — typed, fail-closed agent loop kernel

A self-contained, provider-agnostic step machine. Standard library only; no
existing Hermes module imports it yet, so landing it changes no runtime
behavior.

**What this is:** a typed loop + memory + governance gate + replay system,
modeled on smolagents' `MultiStepAgent` but with **no smolagents dependency**.

**What this is not:** a replacement for `AIAgent`. It is the substrate
`AIAgent.run()` can *delegate to* once a provider is wired in (see
[Integration plan](#integration-plan)).

## Layout

```
agent/runtime/
├── __init__.py      # public surface — import from agent.runtime, not submodules
├── steps.py         # TaskStep, ActionStep, PlanningStep, FinalAnswerStep, ToolCall, ToolOutput, StepFailure
├── result.py        # RunResult, TokenUsage
├── interfaces.py    # ModelProtocol, ToolHandlerProtocol, GovernanceProtocol, Clock, IdSource
├── transitions.py   # TransitionGuard — rejects illegal step orderings
├── governance.py    # DenyAllGovernance (default), AllowAllGovernance, AllowListGovernance, GovernanceGate
├── state.py         # RunState — every mutation requires a reason; frozen after the run
├── callbacks.py     # CallbackRegistry — per-step-type dispatch
├── memory.py        # AgentMemory — append-only typed history; to/from JSONL
├── loop.py          # MultiStepLoop — model → governance → tools → termination
└── replay.py        # RecordedModel, RecordedToolHandler, ScriptedGovernance, run_from_trace
```

## Guarantees (each pinned by `tests/runtime/`)

- Every step is a frozen dataclass; failures carry a typed `FailureKind`.
- Every step is transition-validated before it lands in memory.
- Every tool call gets exactly one governance decision on record. With no
  policy supplied the loop uses `DenyAllGovernance`.
- `continue_on_error=False` by default: model errors, tool errors, rejected
  transitions and limit breaches terminate with `completed=False`.
- Every `RunState` mutation carries a non-empty reason.
- `run_from_trace(memory.to_jsonl())` reproduces the trace byte-for-byte
  (inject `FrozenClock` + `SequentialIdSource` for determinism).
- `MultiStepLoop` instances are single-use; a second `run()` raises
  `RuntimeError`.

## Tests

```bash
scripts/run_tests.sh tests/runtime/
# or, with the project venv active:
pytest tests/runtime/ -q
```

## Integration plan

Each step is a separate PR; the default path stays on the legacy loop until
step 5.

1. Land `agent/runtime/` + `tests/runtime/` (this package).
2. Add a `generate(messages, tools=None, **kwargs) -> ModelOutput` shim to one
   provider adapter so it satisfies `ModelProtocol`.
3. Wrap `model_tools.handle_function_call` as a `ToolHandlerProtocol`
   (`list[ToolCall] -> list[ToolOutput]`).
4. Delegate `AIAgent.run()` to `MultiStepLoop` behind an
   `agent.runtime: kernel | legacy` config flag (default `legacy`); diff
   trajectories of both paths on a reference task.
5. Move `IterationBudget`, tool-batch parallelization, compression and
   redaction into step callbacks one at a time, each with a regression test;
   flip the default once one provider reaches parity.

## Design constraints

- **No external runtime deps.** Standard library only. Python 3.11+.
- **Append-only memory.** `AgentMemory` never mutates past steps — this is
  what makes replay deterministic.
- **No I/O in the loop.** All side effects go through the injected
  `ModelProtocol`, `ToolHandlerProtocol` and `GovernanceProtocol`. The loop
  never reads config, opens files, or logs to disk; telemetry is a callback
  concern.
- **Interruption is cooperative.** `loop.interrupt()` flips a flag checked
  between steps; it does not raise from inside the model call.

## Out of scope

- Provider adapters, compression, credential pooling, rate guarding — they
  stay where they are and become callbacks during integration.
- Streaming — a follow-up via an `event_sink` callback.
