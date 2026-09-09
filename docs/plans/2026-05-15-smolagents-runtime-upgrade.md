# Upstream proposal: harden the Hermes Agent core runtime with smolagents

**Target repo:** [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (Python). This document is staged in `hermes-desktop` because the desktop UI is the workspace where the work was scoped, but **no desktop code changes** are implied.

**Status:** Design proposal — not implemented. Owner: TBD. Date: 2026-05-15.

**Inputs:**
- smolagents @ main (https://github.com/huggingface/smolagents)
- Local install at `~/.hermes/hermes-agent/` (commit reflected by the runtime; not pinned here)

---

## 1. Why touch this

The hermes-agent runtime works, but the core is a **single 15,883-line `AIAgent` class in `run_agent.py`** (821 KB file). Its inner loop is implicit — there is no typed step record, no extension point that isn't a monkey-patch, and replaying or visualizing a run requires re-reading raw JSONL. `agent/trajectory.py` is **56 lines** and produces an untyped list of dicts.

smolagents has solved the structural side of this problem:

| Concept | smolagents | hermes-agent today |
|---|---|---|
| Loop control | `MultiStepAgent.run() / step()` (~few hundred LOC) | embedded in `AIAgent` (15.8k LOC) |
| Step records | typed `ActionStep`, `PlanningStep`, `FinalAnswerStep` | untyped list of dicts (`trajectory.py`) |
| Tool calls | `ToolCallingAgent.process_tool_calls()` (generator) | `model_tools.handle_function_call()` + 1k LOC of arg coercion in `run_agent.py` |
| Code actions | `CodeAgent` writes Python; `PythonExecutor` runs it | none — code is only available as a tool (`tools/code_execution_tool.py`) the model *calls*, not the action format itself |
| Sandbox | `LocalPythonExecutor` (banned imports, dunder block, op caps, 30s timeout) | `tools/environments/{local,docker,ssh,modal,daytona,vercel_sandbox,singularity}` — these are shell backends for one tool, not the agent's action executor |
| Extensions | `step_callbacks` + `final_answer_checks` first-class | callsite-specific hooks scattered through `AIAgent` |
| Planning | `planning_interval: int` | implicit; relies on context compression to reshape history |
| Serialization | `from_dict / from_folder / from_hub` + `to_dict` | not present |
| Run result | typed `RunResult(output, state, steps, token_usage, timing)` | ad-hoc dicts |

The biggest robustness wins are typed step records and a clean loop boundary — **independent of whether we adopt smolagents code at all**.

## 2. What hermes-agent has that smolagents lacks

Adopting smolagents wholesale is **not** desirable. hermes-agent already does things smolagents doesn't:

- **Provider depth** — bespoke adapters for Anthropic (`agent/anthropic_adapter.py`, 89 KB), Bedrock, Gemini Cloud Code, Gemini Native, Codex Responses, OpenAI proxy. Each handles prompt-caching, reasoning blocks, ID quirks, rate-limit recovery.
- **Credential pooling** (`agent/credential_pool.py`, 68 KB) — multi-key rotation, per-account usage tracking (`agent/account_usage.py`).
- **Context compression** (`agent/context_compressor.py`, 74 KB) — far more sophisticated than truncate-and-summarize.
- **Tool ecosystem** — 153 KB browser tool, 140 KB MCP tool, 118 KB delegate tool, 71 KB code execution tool, 42 KB kanban, 41 KB image gen, …
- **Approval/safety gating** (`tools/approval.py`, 59 KB) + `agent/tool_guardrails.py`.
- **Skill curation loop** (`agent/curator.py`, 75 KB).
- **MCP OAuth** (`tools/mcp_oauth_manager.py`, 26 KB).
- **~17,000 tests** covering all of the above.

A wholesale port would regress most of that. The right move is to **adopt smolagents' structure, not its code path**.

## 3. Recommended approach: three tiers

### Tier 1 — Adopt patterns (no dependency added)

**Goal:** extract the loop from `AIAgent` into a thin typed step machine. No external dependency. Keep all provider adapters, compression, credentials, tools as-is.

**Concretely (new modules, all under `agent/runtime/`):**

```
agent/runtime/
├── __init__.py
├── steps.py           # ActionStep, PlanningStep, FinalAnswerStep, MemoryStep (dataclasses, frozen)
├── result.py          # RunResult(output, state, steps, token_usage, timing)
├── callbacks.py       # StepCallback protocol + registry
├── memory.py          # AgentMemory — append-only, typed, supports replay()
└── loop.py            # MultiStepLoop: pure orchestration, no provider/tool knowledge
```

**`AIAgent` shrinks from 15.8k to ~3k LOC** by delegating loop control to `MultiStepLoop`. Provider adapters and tool handling stay where they are. The loop calls them through narrow interfaces.

**Wins:**
- Every step becomes a typed object you can serialize, replay, diff against a known-good trajectory.
- `step_callbacks` give us a single extension point for: telemetry, redaction, budget enforcement (today scattered as `_should_parallelize_tool_batch`, `_extract_parallel_scope_path`, `IterationBudget`, etc. — all visible in `run_agent.py` lines 287–671).
- `replay()` and `visualize()` become trivial.
- Test surface: today most tests have to construct `AIAgent` with mock providers. After Tier 1, the loop is testable with stub callbacks.

**Risk:** medium. Touches the hottest path. Requires backfilling tests against the new `MultiStepLoop` while not changing observable behavior of `AIAgent.run()`.

**Effort:** ~3–4 weeks for one engineer.

### Tier 2 — Add smolagents as an optional dependency for the CodeAgent path

**Goal:** expose the **CodeAgent** pattern (write Python, execute, observe) as an opt-in mode alongside the existing tool-calling flow.

**Concretely:**

- Add `smolagents` to `pyproject.toml` under an extra: `[project.optional-dependencies] codeagent = ["smolagents>=...]`
- New module `agent/runtime/code_mode.py` defines:
  - `HermesModel(smolagents.Model)` — wraps existing hermes provider adapters so smolagents calls Anthropic/Gemini/etc. through hermes's credential pool, rate guard, and compression.
  - `HermesCodeAgent(smolagents.CodeAgent)` — overrides `create_python_executor()` to plug in our existing `tools/environments/` backends as smolagents `PythonExecutor` implementations.
- New config knob: `agent.mode: tool | code | hybrid` (default `tool`).
- `hybrid` mode: ToolCallingAgent by default; switches to CodeAgent when the model emits a code block, or for prompts with `agent: code` front-matter.

**Why this is safe:**
- Zero impact on the default code path. CodeAgent is gated behind a config setting.
- All hermes-specific behavior (credential pool, rate guard, compression) is preserved by injecting `HermesModel`.
- Our existing sandboxes (docker, modal, daytona, vercel_sandbox, singularity) become smolagents `PythonExecutor` implementations — a clean adapter, ~150 LOC per backend.

**Wins:**
- Per smolagents' benchmarks, CodeAgent uses ~30% fewer steps on difficult tasks.
- Users who already trust our Docker / Modal / Daytona sandboxes get the CodeAgent UX immediately.
- Lets us evaluate smolagents in production without committing to it.

**Risk:** low-medium. New code, gated. Main risk is `HermesModel` adapter completeness — smolagents' `Model` protocol differs from each provider's native shape.

**Effort:** ~2 weeks after Tier 1 lands.

### Tier 3 — Migrate the whole loop onto smolagents primitives

**Goal:** retire `MultiStepLoop` from Tier 1 in favor of a real `smolagents.MultiStepAgent` subclass.

**Concretely:** `AIAgent` becomes a subclass of `MultiStepAgent`. Compression, credential pool, rate guard become `step_callbacks`. Approval/guardrails become `final_answer_checks` + tool wrappers. The provider adapters become `Model` subclasses.

**Why this is risky:** smolagents' `Model` abstraction is intentionally thin. Things like Anthropic prompt-caching, Gemini native function-call format, and the OpenAI Responses API quirks have all required custom adapter code in hermes. Pushing them through `smolagents.Model` either bloats `HermesModel` until it's not "smolagents-style" anymore, or loses behavior. The ~17k test suite would need to be reviewed end-to-end.

**Recommendation:** **defer**. Revisit only if Tier 1 + Tier 2 prove the abstraction holds.

**Effort:** 6+ months, multi-engineer.

## 4. Sequencing & exit criteria

| Tier | Land in | Exit criteria |
|---|---|---|
| 1 | `agent/runtime/` new package; `run_agent.py` shrinks | All existing tests pass; new tests for `MultiStepLoop` give >90% line coverage; trajectory replay matches byte-for-byte for ≥3 reference sessions |
| 2 | `agent/runtime/code_mode.py` + `pyproject.toml` extra | `agent.mode: code` runs ≥10 reference tasks in CI against the local executor + Docker executor; CodeAgent benchmark within 1.2× ToolCallingAgent step count on hermes's task set |
| 3 | Decision deferred | Tier 1 + Tier 2 in production for ≥2 quarters with no regression rollback |

## 5. Things that change in the upstream repo

| File | Change | Why |
|---|---|---|
| `agent/runtime/` | NEW package (6 files, ~600 LOC total) | Tier 1 — loop extraction |
| `agent/trajectory.py` | Becomes a thin compatibility shim around `agent/runtime/memory.py::AgentMemory.to_jsonl()` | Preserves `batch_runner.py` consumers; deprecate the file in a later cycle |
| `run_agent.py` | `AIAgent` shrinks; delegates `_main_loop_iteration` and `_should_parallelize_tool_batch` etc. to `MultiStepLoop` | Tier 1 |
| `agent/anthropic_adapter.py`, `agent/gemini_*_adapter.py`, `agent/bedrock_adapter.py`, `agent/codex_responses_adapter.py` | UNCHANGED in Tier 1 | They are called through a narrow interface from the new loop |
| `pyproject.toml` | Add `[project.optional-dependencies] codeagent = ["smolagents>=..."]` | Tier 2 |
| `agent/runtime/code_mode.py` | NEW — `HermesModel`, `HermesCodeAgent`, executor adapters | Tier 2 |
| `tools/environments/{docker,modal,daytona,vercel_sandbox,singularity}.py` | Each gets a `as_python_executor()` adapter method | Tier 2 |
| `hermes_constants.py` / config schema | Add `agent.mode` setting | Tier 2 |
| `cli.py` (HermesCLI) | Plumb `agent.mode` to runtime constructor | Tier 2 |
| `tests/runtime/` | NEW directory mirroring Tier 1 / Tier 2 layout | Both tiers |

## 6. Things that explicitly do **not** change

- `gateway/` and all platform adapters — no change.
- `plugins/` system — no change. Plugins continue to register at the existing extension points.
- `agent/credential_pool.py`, `agent/nous_rate_guard.py`, `agent/account_usage.py` — unchanged code paths; called as before from the new loop.
- `agent/context_compressor.py` — invoked via a `step_callback` on `ActionStep` rather than inline; same behavior.
- `tools/registry.py` — unchanged. The tool registration mechanism is good as-is and orthogonal to the loop.
- MCP, browser, kanban, delegate, memory tools — unchanged.
- Skill curation (`agent/curator.py`) — unchanged.
- The HTTP / SSE protocol that `hermes-desktop` consumes at `127.0.0.1:8642` — **unchanged**. The desktop app does not need any modifications.

## 7. Open questions

1. **Step replay format.** smolagents serializes via `to_dict()`; hermes ships ShareGPT-formatted JSONL via `trajectory.py`. Do we keep both, or migrate `batch_runner.py` consumers to the new format and deprecate the old one over a release?
2. **Streaming.** smolagents' `stream_outputs=True` flag streams from `agent.run(stream=True)`. hermes's existing SSE flow lives in `cli.py` HTTP handlers. Tier 1 needs the loop to support both event-emission shapes — either via callback or via the same generator pattern smolagents uses.
3. **Interrupts.** smolagents has `agent.interrupt()`; hermes has `tools/interrupt.py` + signal handling. Decide whether `MultiStepLoop` owns interrupt semantics or delegates to the existing mechanism.
4. **Authorized imports for CodeAgent.** What's the default `additional_authorized_imports` list for hermes? Reuse the list from `tools/code_execution_tool.py`, or define a narrower default?
5. **smolagents pin.** smolagents iterates fast. Pin to a tested commit (not `>=`) and update on a deliberate cadence.

## 8. Recommended first PR

A single PR that lands Tier 1 in scope:

- New `agent/runtime/` package.
- `AIAgent.run()` delegates to `MultiStepLoop.run()`.
- Backward-compatible `agent/trajectory.py` shim.
- New `tests/runtime/` covering `MultiStepLoop` independently of `AIAgent`.
- Existing test suite green; no behavior change.

Estimated ~1,500 LOC added, ~3,000 LOC moved out of `run_agent.py`.

---

**Out of scope for this proposal:** any desktop changes, gateway changes, plugin changes, MCP changes. This document deliberately constrains the blast radius to the agent loop.
