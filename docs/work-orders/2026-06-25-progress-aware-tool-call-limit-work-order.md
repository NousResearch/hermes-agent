---
title: "work-order: Progress-aware agent iteration policy for perceived tool-call limits"
status: draft
date: 2026-06-25
type: work-order
target_repo: hermes-agent
parent_plan: ../plans/2026-06-25-progress-aware-tool-call-limit-plan.md
related_review: ../reviews/2026-06-25-progress-aware-tool-call-limit-plan-review.md
related_step_breakdown: ./2026-06-25-progress-aware-tool-call-limit-step-breakdown.md
---

# work-order: Progress-aware agent iteration policy for perceived tool-call limits

## Scope

Implement the parent plan in small TDD slices. This work order owns execution detail: W-units, RED/GREEN expectations, file targets, commands, evidence, and exit criteria.

Parent plan: `../plans/2026-06-25-progress-aware-tool-call-limit-plan.md`

Step breakdown: `./2026-06-25-progress-aware-tool-call-limit-step-breakdown.md`

## Global constraints

- Preserve legacy behavior unless `agent.iteration_policy.mode: progress_aware` is enabled.
- Preserve prompt-cache invariants and message role alternation.
- Do not introduce a new model-facing core tool.
- Do not change unrelated desktop/i18n files already dirty in the local checkout.
- Keep stop reasons structured enough for logs/tests and concise enough for user-facing finalization.
- Run focused tests before broader tests.

## Likely file targets

```text
agent/conversation_loop.py                 # loop condition, policy decision point, stop reason
agent/turn_finalizer.py                    # budget exhaustion vs policy stop finalization
agent/turn_context.py                      # per-turn IterationBudget initialization
agent/iteration_budget.py                  # inspect; preserve consume/refund semantics
agent/tool_executor.py                     # sequential/concurrent tool result observation seam
agent/tool_guardrails.py                   # reuse existing signatures/repeated-failure/no-progress state
agent/iteration_policy.py                  # new candidate: pure decision model
agent/progress_tracker.py                  # new candidate: bounded event window, no raw large outputs
agent/agent_init.py                        # runtime policy/config initialization
agent/chat_completion_helpers.py           # inspect only unless completion path requires it
cli.py                                     # current DEFAULT_CONFIG and CLI max_turns resolution
hermes_cli/config.py                       # authoritative load_config/cfg_get behavior if config defaults move
cron/scheduler.py                          # cron-specific conservative policy resolution
gateway/run.py                             # main messaging gateway AIAgent creation/cache path
gateway/platforms/api_server.py            # env-driven max_iterations surface
tui_gateway/server.py                      # TUI/gateway AIAgent creation surface
tools/delegate_tool.py                     # child-agent max_iterations/profile resolution
tests/agent/test_iteration_policy.py       # new candidate
tests/agent/test_progress_tracker.py       # new candidate
tests/agent/test_tool_guardrails.py        # existing guardrail reuse/regression
tests/run_agent/test_tool_call_guardrail_runtime.py
tests/run_agent/test_turn_completion_explainer.py
tests/run_agent/test_iteration_budget_race.py
website/docs/user-guide/configuration.md   # docs slice if config is user-facing
website/docs/developer-guide/prompt-assembly.md or new developer doc if loop docs fit there
```

## Current code anchors

Use these anchors before making production edits:

- `agent/conversation_loop.py`: current loop condition is `(api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call`; any progress-aware change must preserve legacy mode exactly.
- `agent/turn_context.py`: resets `agent.iteration_budget = IterationBudget(agent.max_iterations)` for each turn.
- `agent/turn_finalizer.py`: calls `_handle_max_iterations()` when `final_response is None` and either `api_call_count >= agent.max_iterations` or `iteration_budget.remaining <= 0`.
- `agent/tool_executor.py`: both sequential and concurrent paths call `_append_guardrail_observation()` after a tool returns and before appending the tool result to the message list.
- `agent/tool_guardrails.py`: already owns canonical args hashing, result hashing, exact failure counts, same-tool failure counts, and idempotent no-progress counts.
- `run_agent.py`: `_toolguard_controlled_halt_response()`, `_append_guardrail_observation()`, `_guardrail_block_result()`, and `_format_turn_completion_explanation()` are existing final-response/explainer seams.
- `run_agent.py`: `_append_guardrail_observation()` currently returns only the possibly augmented tool-result string; policy/tracker consumption needs either a public guardrail observation API or a shared bounded event object.
- `agent/conversation_loop.py`: the near-limit error guard at the post-error path compares against `agent.max_iterations - 1`; progress-aware mode must re-anchor it to the effective hard budget or policy decision.
- `agent/turn_finalizer.py`: `completed` currently requires `api_call_count < agent.max_iterations`; progress-aware success after the soft threshold must not be recorded as a failed/incomplete turn.
- `agent/conversation_loop.py`: provider/runtime paths that return before the normal loop, such as app-server runtimes, need explicit out-of-scope documentation or a separate policy adapter.
- `cli.py`: current default config contains `agent.max_turns`; config display and CLI resolution already use this name.
- `cli.py` and `hermes_cli/config.py`: both contain default/config loading surfaces; nested `agent.iteration_policy` must deep-merge rather than replace sibling defaults.
- `cron/scheduler.py`, `gateway/run.py`, `gateway/platforms/api_server.py`, `tui_gateway/server.py`, and `tools/delegate_tool.py`: separate AIAgent creation surfaces must not accidentally inherit an unbounded interactive policy.

## W0 — Re-establish current behavior and terminology

### Goal

Confirm the current limit surface and reproduce the user-visible failure class without changing production behavior.

### RED / evidence

- Inspect current loop and max iteration handling at `conversation_loop.py`, `turn_context.py`, `turn_finalizer.py`, and `iteration_budget.py`.
- Add or identify a focused test that proves legacy behavior stops on the fixed limit even when later progress would be possible.
- Add or identify a test for repeated identical failure/no-progress behavior through existing `ToolCallGuardrailController` and runtime halt wiring.
- Record how `_budget_grace_call`, `_handle_max_iterations()`, `completed`, and `_format_turn_completion_explanation()` behave today before changing policy code.
- Record how the near-limit error guard behaves today when `api_call_count >= agent.max_iterations - 1`.
- Record whether provider/runtime paths bypass the normal conversation loop and whether they are in scope for this feature.
- Confirm `execute_code`-only tool rounds still refund the iteration budget.

### Commands

```bash
python -m pytest tests/run_agent/test_run_agent.py -q -o 'addopts='
python -m pytest tests/agent -q -o 'addopts='
```

### Exit criteria

- Current behavior is documented in test output or review notes.
- The code path for `agent.max_turns` / `max_iterations` is identified.
- No production behavior changed unless a failing RED test was intentionally added in a branch/slice.

## W1 — Add policy data model and pure decision tests

### Goal

Create a pure, import-light iteration policy that can decide continue/stop without touching provider code.

### RED

Add tests for:

- below soft limit → continue;
- hard max reached → hard stop;
- soft limit reached + recent progress → continue;
- soft limit reached + no progress → graceful stop;
- guardrail hard stop → stop regardless of progress;
- policy reason names consistently use `iterations`, e.g. `hard_max_iterations_reached`, not the old turns-based variant.

### GREEN

Add `agent/iteration_policy.py` with a small decision object and policy function/class. The model must include at least:

- mode: `legacy` or `progress_aware`;
- `soft_max_iterations` and `hard_max_iterations`;
- progress-window summary input, not raw tool output;
- guardrail decision input;
- structured reason codes that can be mapped by `turn_finalizer` / `_format_turn_completion_explanation()`.

Do not wire it into `conversation_loop.py` in this slice.

### Exit criteria

- Pure policy tests pass.
- No conversation loop integration yet.

## W2 — Add progress tracker and fingerprint tests

### Goal

Track bounded recent progress/no-progress signals without relying on model self-reporting.

### RED

Add tests for:

- distinct tool result fingerprints;
- identical tool/args/failure repeat counting;
- file/artifact/test/git/process/todo event kinds recorded as progress;
- bounded window behavior.

### GREEN

Add `agent/progress_tracker.py` or extend the smallest existing helper if an equivalent structure already exists.

Implementation direction:

- Reuse `ToolCallSignature`, canonical args hashing, and result hashing behavior from `agent/tool_guardrails.py` where possible.
- If existing guardrail internals do not expose the required metadata, add a public observation/decision API or shared bounded event object instead of reading private dictionaries from outside the controller.
- Feed observations from `agent/tool_executor.py` in both sequential and concurrent paths after `_append_guardrail_observation()` has classified failures/no-progress.
- Store bounded metadata only: event kind, tool name, redacted/keyed args fingerprint, redacted/keyed result fingerprint/status, duration bucket, blocked/error flags, and small evidence labels.
- Do not store raw large tool output, raw secrets, full command output, or durable unsalted fingerprints of low-entropy secret-looking values in the tracker.
- In the concurrent tool path, update the tracker only after deterministic result collection or make the tracker explicitly thread-safe and order-aware.
- Treat cancelled, interrupted, or skipped tool calls as non-progress.

### Exit criteria

- Tracker tests pass.
- Tracker does not store large raw tool outputs unbounded.

## W3 — Wire legacy-compatible config defaults

### Goal

Expose policy config without changing default runtime behavior.

### RED

Add config/default tests proving missing config resolves to `legacy` mode and existing `max_turns` behavior remains unchanged.

### GREEN

Add config defaults/schema parsing for:

```yaml
agent:
  iteration_policy:
    mode: legacy
    soft_max_iterations: null
    hard_max_iterations: 300
    progress_window: 12
    require_progress_after_soft_limit: true
```

Adjust field names if current config style requires a user-facing `turns` alias, but keep the internal policy terminology on `iterations`. Update `cli.py`, `hermes_cli/config.py`, config display/check/migration behavior, and the AIAgent init-time resolver together.

Config validation must cover invalid modes, non-positive numbers, `hard_max_iterations <= soft_max_iterations`, missing nested defaults after partial user config, and context profiles that would make cron/delegation/gateway unbounded.

### Exit criteria

- Existing config tests pass.
- New defaults appear in `hermes config` output or the relevant default config source.

## W4 — Integrate policy with conversation loop behind legacy mode

### Goal

Replace raw fixed-count loop control with policy decisions while preserving legacy semantics by default.

### RED

Add a loop-level test proving legacy mode behaves exactly like the old fixed count.

### GREEN

Modify the loop control so it asks the policy for a decision at stable iteration boundaries.

Implementation requirements:

- In `legacy` mode, keep the current `while` condition and finalizer behavior functionally identical.
- In `progress_aware` mode, do not let `agent.max_iterations` alone remain the loop hard stop; introduce a separate effective hard budget while preserving `agent.max_iterations` as the soft threshold/reporting value.
- Update `turn_context.py` / `IterationBudget` initialization and `turn_finalizer.py` together so soft-limit continuation is not immediately converted back into `max_iterations_reached(...)`.
- Update post-error near-limit handling so `api_call_count >= agent.max_iterations - 1` does not remain an implicit hard stop in progress-aware mode.
- Explicitly exclude or adapt runtime paths that bypass the normal tool loop.
- Preserve `_budget_grace_call` and `execute_code` refund semantics.

### Exit criteria

- Legacy loop tests pass.
- Existing run-agent tests do not regress.
- Stop reason is available to finalization code.

## W5 — Enable progress-aware mode in loop-level tests

### Goal

Prove productive work can pass the soft threshold and no-progress loops stop.

### RED

Add loop-level tests for:

- progress event inside window permits another iteration after soft limit;
- repeated identical tool failure stops before hard cap;
- hard max always stops;
- user interrupt remains immediate.

### GREEN

Connect tracker events from tool results, validation/finalization points, or existing guardrail state.

Implementation requirements:

- Wire both `execute_tool_calls_sequential()` and `execute_tool_calls_concurrent()`; they must produce equivalent tracker observations for the same tool outcome.
- Treat `_tool_guardrail_halt_decision` as a terminal policy input, not as ordinary progress.
- Count a new result fingerprint from read/search/test commands as possible informational progress, but count repeated identical idempotent results as no-progress after the soft threshold.
- Count landed file mutations/evidence artifacts as strong progress, but repeated identical writes or failed patches as no-progress/failure.
- Do not count interrupted/cancelled/skipped tool results as progress, and always let user interrupt win over policy continuation.
- Include a concurrent mixed-result test proving deterministic observation order and equivalent classification to the sequential path.

### Exit criteria

- Progress-aware loop tests pass.
- Guardrail stops remain observable and user-safe.

## W6 — Finalization and user-visible stop reason

### Goal

Make graceful/hard stop reasons explicit in final responses and logs without fabricating completion.

### RED

Add tests that a policy stop produces the expected finalization path and does not claim task success.

Cover:

- `policy_soft_no_progress_stop` maps to an explicit non-success response;
- `policy_hard_max_iterations_reached` differs from legacy `max_iterations_reached(...)` where appropriate;
- progress-aware mode with `api_call_count >= soft_max_iterations` but below the hard cap and a valid final response records successful completion where the caller should treat the turn as complete;
- existing `guardrail_halt` remains handled by its current controlled halt response;
- `interrupted_by_user` remains immediate and is not rewritten as a policy stop.

### GREEN

Update `agent/turn_finalizer.py` or the loop finalization seam to map structured reasons to user-visible messages.

### Exit criteria

- Stop messages distinguish no-progress guardrail, hard cap, and user interrupt.
- Cron, delegation, gateway, and worker callers do not treat a productive soft-limit crossing as legacy budget exhaustion.
- No same-role message alternation regression.

## W7 — Context profiles for gateway/cron/delegation/worker

### Goal

Prevent one permissive interactive policy from leaking into contexts that need stronger caps.

### RED

Add focused tests or config resolution tests for context-specific policy selection.

### GREEN

Resolve context-specific defaults from existing runtime context/config seams.

Initial resolution rules to test before changing defaults:

- CLI/local interactive: soft threshold defaults to existing `agent.max_turns`; hard cap is separately bounded.
- Gateway: both `gateway/run.py` and API-server paths must use bounded policy resolution or explicitly remain legacy; platform delivery and inactivity timeouts still win.
- Cron: conservative hard cap and no unbounded progress extension by default.
- Delegation: child hard cap is bounded by delegation config and must not silently exceed the parent/session hard-budget policy.
- Kanban/worker: policy stop integrates with task failure accounting and claim timeout behavior.

### Exit criteria

- Cron remains conservative.
- Delegation hard cap is lower than or bounded by parent policy.
- Gateway retains delivery timeout boundaries.
- API-server env/backcompat behavior remains explicit.

## W8 — Documentation and migration notes

### Goal

Document the distinction between perceived tool-call limit and agent iteration policy.

### RED / review

Docs must explain:

- `agent.max_turns` / `max_iterations` terminology;
- `legacy` vs `progress_aware`;
- hard cap vs soft threshold;
- context-specific behavior;
- troubleshooting steps when the agent stops due to no progress.

### GREEN

Patch user/developer docs in the smallest appropriate locations.

Also update the superseded source artifact notice so it points to the active split documents in the current repository/worktree, not to a stale checkout path.

### Exit criteria

- Docs links resolve.
- Config example is valid YAML.

## W9 — Review, verification, and PR packet

### Goal

Close the work with evidence, independent review, and a clean PR lifecycle.

### Required checks

```bash
python -m pytest tests/agent/test_iteration_policy.py tests/agent/test_progress_tracker.py -q -o 'addopts='
python -m pytest tests/run_agent/test_run_agent.py -q -o 'addopts='
python -m pytest tests/run_agent/test_tool_call_guardrail_runtime.py tests/run_agent/test_turn_completion_explainer.py tests/run_agent/test_iteration_budget_race.py -q -o 'addopts='
python -m pytest tests/hermes_cli -q -o 'addopts='   # if config defaults changed
python -m pytest tests/gateway tests/cron -q -o 'addopts='  # if context profiles touch those surfaces
python -m pytest tests/tui_gateway tests/tools/test_delegate_tool.py -q -o 'addopts='  # if TUI/delegation surfaces touched and tests exist
git diff --check
```

### Evidence packet

```text
W-unit:
Gate profile:
Touched files:
Focused tests:
Broader tests:
Stop-reason behavior evidence:
Legacy compatibility evidence:
Soft-limit success/completed evidence:
Context profile evidence:
Superseded source-artifact notice evidence:
Deferred checks and reason:
Independent review verdict:
```

### Exit criteria

- Focused tests pass.
- No new failures versus baseline.
- Independent review has Critical 0 / High 0 equivalent finding level.
- Branch is clean after commit.
- If user approves lifecycle: push → PR → CI observe → merge → main CI observe.
