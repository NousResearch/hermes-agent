---
title: "plan: Progress-aware agent iteration policy for perceived tool-call limits"
status: review-ready
date: 2026-06-25
type: plan
target_repo: hermes-agent
source_artifact: /Users/cube-mac/dreampia-mvp-evidence/latest/hermes-progress-aware-tool-call-limit-plan-2026-06-25.md
related_work_order: ../work-orders/2026-06-25-progress-aware-tool-call-limit-work-order.md
related_review: ../reviews/2026-06-25-progress-aware-tool-call-limit-plan-review.md
---

# plan: Progress-aware agent iteration policy for perceived tool-call limits

## Summary

Hermes currently has a user-visible failure mode where a long but productive task can appear to stop because the foreground agent loop reaches its configured iteration/turn threshold. Users experience this as a “tool-call limit,” but the root control surface is broader than individual tool calls: the agent loop counts LLM/API iterations, which may include tool-call rounds, tool results, retries, and finalization attempts.

This plan proposes a progress-aware iteration policy that keeps the existing safety cap but changes the normal stop condition from “fixed count reached” to “fixed soft threshold reached with no recent progress.” Productive work should continue; repeated non-progress loops should stop with an explicit reason.

## Problem

The existing loop policy can stop normal work when the model is still making measurable progress, especially during long coding, verification, PR/CI, or multi-artifact tasks. Raising `agent.max_turns` alone is only a temporary workaround: it increases room for legitimate work but also increases room for broken loops.

The desired behavior is:

- normal progress continues past the soft threshold;
- repeated identical failures stop quickly;
- a large hard cap remains as a final safety boundary;
- the final user-facing message explains why the turn stopped;
- gateway, cron, delegation, and worker contexts can apply different policy profiles.

## Goals

- Treat `agent.max_turns` / `agent.max_iterations` as a soft threshold when progress-aware mode is enabled.
- Add a separate hard maximum iteration cap as a fail-safe.
- Detect recent progress using bounded signals from tool results, file state, process state, validation output, artifacts, and lifecycle state.
- Detect repeated no-progress loops, including identical tool failures and idempotent no-op calls.
- Preserve legacy behavior by default until the new policy is proven safe.
- Make stop reasons explicit for logs, observability, and final responses.

## Non-goals

- Do not remove hard limits entirely.
- Do not make Telegram/gateway delivery timeouts unbounded.
- Do not let cron or delegated subagents inherit unrestricted interactive-session limits.
- Do not introduce a new model-facing core tool.
- Do not rely on model self-reporting as the only progress signal.
- Do not mutate conversation history or break message role alternation to extend a turn.

## Design principles

1. **Progress beats raw count, but only inside a hard cap.** A soft limit can be crossed only when recent objective progress exists.
2. **No-progress loops fail closed.** Repeated same-tool/same-args/same-result failures should stop before consuming a large budget.
3. **Provider/model independence.** The policy should sit around the agent loop and tool dispatch results, not depend on provider-specific messages.
4. **Prompt-cache safety.** The policy must not rebuild the system prompt or mutate prior conversation content mid-turn.
5. **Profile-specific limits.** Interactive chat can be more permissive than cron/subagent/gateway contexts.
6. **Evidence-first finalization.** A stop reason should be stored and surfaced in a useful, non-ambiguous user message.

## Proposed architecture

Introduce two small internal modules:

- `agent/progress_tracker.py` — records bounded progress/no-progress signals and fingerprints recent tool/API outcomes.
- `agent/iteration_policy.py` — decides whether the loop should continue, warn, gracefully stop, or hard-stop based on config, iteration count, progress window, and guardrail state.

The conversation loop should ask the policy for a decision at a stable point in each iteration. The policy returns a structured decision with a reason such as `below_soft_limit`, `soft_limit_reached_but_progress_detected`, `soft_limit_reached_no_progress`, `guardrail_exact_failure`, or `hard_max_iterations_reached`.

## Current implementation reconnaissance

Implementation must be based on these existing seams, not on a fresh loop abstraction:

- Main per-turn loop: `agent/conversation_loop.py` currently gates the loop with `api_call_count < agent.max_iterations` and `agent.iteration_budget.remaining > 0`, plus the existing `_budget_grace_call` escape hatch.
- Budget object: `agent/iteration_budget.py` is a thread-safe consume/refund counter. `execute_code`-only tool rounds are currently refunded, so progress-aware mode must preserve that behavior.
- Turn setup: `agent/turn_context.py` resets `agent.iteration_budget = IterationBudget(agent.max_iterations)` per turn. A progress-aware hard cap cannot be added only by changing `agent.max_iterations`, because `agent.max_iterations` is also used for legacy reporting and finalization semantics.
- Finalization: `agent/turn_finalizer.py` currently treats `api_call_count >= agent.max_iterations` or `iteration_budget.remaining <= 0` as budget exhaustion and asks the model for a toolless summary through `_handle_max_iterations()`.
- Tool execution: actual tool observations are emitted from `agent/tool_executor.py`, in both sequential and concurrent paths, after the underlying tool call returns and before the tool result is appended to `messages`.
- Existing loop guardrails: `agent/tool_guardrails.py` already computes canonical tool-call signatures, repeated exact-failure counts, same-tool failure counts, and idempotent no-progress counts. The new policy should reuse or consume these decisions rather than duplicating a second fingerprinting implementation.
- Existing guardrail halt path: `conversation_loop.py` already maps `_tool_guardrail_halt_decision` to `_turn_exit_reason = "guardrail_halt"` and a controlled final response. Progress-aware policy must not regress this behavior.
- Config/default source: default CLI config is currently in `cli.py`; runtime entrypoints pass `max_iterations` into `AIAgent`. A user-facing config addition must update the authoritative default/config display path and the AIAgent runtime resolution path.

## Implementation constraints created by current code

- Legacy mode must leave the current `max_iterations` hard-stop behavior unchanged, including the existing one extra toolless summary attempt on exhaustion.
- Progress-aware mode needs two limits: existing `agent.max_turns` / `max_iterations` as the soft threshold, and a separate effective hard budget for the per-turn loop.
- The loop condition, `IterationBudget` initialization, and `turn_finalizer` exhaustion check must change together. Changing only one of them will either preserve the old premature stop or accidentally turn the soft threshold into a hard cap again.
- Success/failure finalization must change with the loop: a progress-aware turn that crosses the soft threshold and later produces a valid final response should not be marked incomplete merely because `api_call_count >= agent.max_iterations`.
- Existing near-limit error guards that compare against `agent.max_iterations - 1` must be re-anchored to the effective hard budget or a policy decision; otherwise they silently preserve the old hard-stop behavior.
- Code paths that bypass the normal tool-calling loop, such as app-server or provider-specific runtimes, must be explicitly out of scope or given their own policy adapter.
- Progress tracking should be event-fed from `tool_executor` and existing guardrail observations; it should not scan or mutate the conversation transcript after the fact.
- Progress signals must remain bounded and secret-safe: store event kind, tool name, hashed args/result summaries, status, and small metadata only; never store raw large tool outputs or secrets.

## Configuration strategy

Add an opt-in policy under `agent.iteration_policy` and keep legacy semantics as the default initial rollout.

Recommended initial shape:

```yaml
agent:
  iteration_policy:
    mode: legacy                # legacy | progress_aware
    soft_max_iterations: null   # null means use existing agent.max_turns / max_iterations
    hard_max_iterations: 300
    progress_window: 12
    require_progress_after_soft_limit: true
```

Guardrail thresholds should remain separate from policy mode but feed into the policy decision where available. Config rollout must update both current default/config surfaces (`cli.py` and `hermes_cli/config.py`) or route through one authoritative resolver, and nested `agent.iteration_policy` values must be deep-merged so partial user config does not erase defaults.

Validation rules should reject or safely fall back on invalid modes, non-positive limits, `hard_max_iterations <= soft_max_iterations` in progress-aware mode, and context profiles that would give cron/delegation an unbounded interactive policy.

For user-facing docs, `turns` can be described as the historical CLI term, but the internal policy should prefer `iterations` because the counted unit is an LLM/API loop iteration, not an individual tool call.

## Progress signal strategy

Progress should be recognized from objective bounded state changes, for example:

- a file write/patch result changed repository or filesystem state;
- test/validation output changed in a way that provides new information;
- a background process produced new output or changed state;
- git/PR/CI lifecycle state changed;
- a new evidence artifact was created;
- todo/progress state changed;
- a user steer changed the active path;
- tool args/result fingerprints changed in a meaningful way.

A signal should not be considered progress merely because a model says it is making progress.

The initial implementation should prioritize signals already observable without broad tool rewrites:

1. existing `ToolCallGuardrailDecision` values from `agent/tool_guardrails.py`;
2. tool completion events from `agent/tool_executor.py` with tool name, args hash, result hash/status, duration, and blocked/error flags;
3. file mutation verifier outcomes already recorded by `_record_file_mutation_result()`;
4. `todo` state changes and explicit user steer events;
5. lifecycle state changes from git/PR/CI/process tools when visible in structured or hashed tool results.

Signals that require tool-specific semantic parsing beyond these should be deferred until after the core policy is safe.

Fingerprinting must be secret-safe. Raw args/results should be redacted or summarized before fingerprinting, low-entropy secret-looking values should not be logged even as plain hashes, and any persisted/observable fingerprint should be bounded to a turn/session scope where practical rather than becoming a durable correlation oracle.

Concurrent tool execution should feed progress observations deterministically after result collection, or the tracker must be explicitly thread-safe. Cancelled, interrupted, or skipped tool calls are not progress and must not extend the turn after a soft limit.

## No-progress and loop guard strategy

The policy should classify the following as no-progress risks:

- identical tool + identical args + identical failure repeated;
- identical shell command returning identical exit/output repeatedly;
- identical failed `patch` old_string search repeated;
- same content written repeatedly;
- invalid tool-call shape repeated;
- read/search loops that do not change the task state after the soft threshold.

These should produce warning or stop decisions depending on configured thresholds.

The current `ToolCallGuardrailController` already covers the first three classes for many tools. The policy should treat an existing hard guardrail decision as an immediate stop and should treat repeated guardrail warnings as strong no-progress evidence after the soft threshold.

The policy should also distinguish:

- **same failing call**: same tool + same canonical args + failed result;
- **same idempotent no-op**: same read/search call + same successful result;
- **same tool failure path**: same tool failing with varying args;
- **productive mutation**: a mutating tool reports a landed change or creates new evidence;
- **informational novelty**: a read/search/test command returns a new result fingerprint that changes the known state.

## Context-specific policy profiles

| Context | Policy direction |
|---|---|
| CLI/local interactive chat | progress-aware opt-in, existing `agent.max_turns` as soft threshold, separate bounded hard cap candidate such as 300 |
| Telegram/gateway | progress-aware opt-in only after the main `gateway/run.py` agent creation path and API-server env path both resolve bounded policy; platform delivery/inactivity timeout remains authoritative |
| Cron | conservative by default; no progress extension beyond a small hard cap unless a job explicitly opts into a bounded profile |
| delegate_task | child hard cap must be bounded by delegation config and must not exceed the parent/session hard-budget policy by accident |
| Kanban/worker | integrate with task timeout/claim timeout and failure accounting; avoid infinite worker claims |

## Rollout plan

1. Add isolated tests around the new progress tracker and iteration policy.
2. Wire policy into the conversation loop behind `mode: legacy` default.
3. Add focused integration tests proving legacy mode is unchanged.
4. Add progress-aware tests for productive soft-limit crossing and no-progress stopping.
5. Add docs for config behavior and stop reasons.
6. Enable opt-in locally for validation before considering default changes.

## Acceptance criteria

- Legacy mode preserves the current hard-count behavior.
- Progress-aware mode allows productive work past the soft threshold until the hard cap.
- Progress-aware mode stops repeated identical failures before the hard cap.
- Stop reasons are explicit and test-covered.
- A productive progress-aware turn that crosses the soft threshold and then returns a final answer is recorded as successful where appropriate, not as legacy max-iteration exhaustion.
- Interactive/gateway/cron/delegation profiles do not accidentally share one unbounded behavior.
- Gateway, cron, delegation, worker, and API-server entrypoints all use bounded policy resolution or explicitly remain in legacy mode.
- Existing prompt caching and message role alternation invariants are preserved.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Infinite loop because weak signals count as progress | Require bounded objective signals and keep hard cap |
| Legitimate long read/search phase misclassified as no progress | Use a progress window and allow new result fingerprints/new evidence to count |
| Cron jobs run too long | Keep cron policy conservative regardless of interactive defaults |
| Gateway user sees silence for too long | Keep gateway delivery/task timeout independent from iteration policy |
| Config confusion between `max_turns`, `max_iterations`, and policy fields | Document terminology and preserve legacy default |
| Prompt-cache breakage | Policy must not mutate prior messages/system prompt mid-conversation |

## Open questions

- Should `soft_max_iterations: null` always mean reuse the existing `agent.max_turns` value, or should context profiles be allowed to override the soft threshold independently?
- What exact conservative hard-cap values should gateway, cron, delegation, API-server, and worker contexts use during the first opt-in rollout?
- Should the existing guardrail controller expose a public observation/metadata API for policy consumption, or should `tool_executor` feed both guardrail and progress tracker from the same bounded event object?
- Should stop decisions be exposed only in logs/final response, or also in structured observability hooks such as gateway run events, cron logs, delegation summaries, and activity metadata?

## Links

- Work order: `../work-orders/2026-06-25-progress-aware-tool-call-limit-work-order.md`
- Review: `../reviews/2026-06-25-progress-aware-tool-call-limit-plan-review.md`
- Source mixed artifact: `/Users/cube-mac/dreampia-mvp-evidence/latest/hermes-progress-aware-tool-call-limit-plan-2026-06-25.md`
