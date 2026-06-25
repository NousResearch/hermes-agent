---
title: "step-breakdown: Progress-aware agent iteration policy for perceived tool-call limits"
status: review-ready
date: 2026-06-25
type: step-breakdown
target_repo: hermes-agent
parent_plan: ../plans/2026-06-25-progress-aware-tool-call-limit-plan.md
parent_work_order: ./2026-06-25-progress-aware-tool-call-limit-work-order.md
related_review: ../reviews/2026-06-25-progress-aware-tool-call-limit-plan-review.md
---

# step-breakdown: Progress-aware agent iteration policy for perceived tool-call limits

## Purpose

This document decomposes the parent work order into handoff-sized implementation steps. It does **not** replace the parent work order. The parent work order remains the source for W-unit scope, RED/GREEN requirements, likely files, and final evidence packet shape.

Parent plan: `../plans/2026-06-25-progress-aware-tool-call-limit-plan.md`

Parent work order: `./2026-06-25-progress-aware-tool-call-limit-work-order.md`

Related review: `../reviews/2026-06-25-progress-aware-tool-call-limit-plan-review.md`

## Execution rules

- Execute W-units in order unless a blocker requires returning to a prior W-unit.
- Each step should be small enough to hand off to a fresh agent with no hidden context.
- Follow RED → GREEN → verify for code-producing steps.
- Preserve legacy runtime behavior unless `agent.iteration_policy.mode: progress_aware` is explicitly enabled.
- Do not perform broad conversation-loop rewrites before W0 evidence and W1 pure policy tests exist.
- Do not store raw tool outputs, secrets, credentials, or durable unsalted secret-like fingerprints in progress-tracking artifacts.
- Treat interrupted, cancelled, and skipped tool calls as non-progress.
- Verification wording pins: Delegation, Kanban/worker, Soft-limit success/completed evidence, redacted/keyed, and "cancelled, interrupted, and skipped tool calls are non-progress" are required coverage terms for automated review.
- If a step touches runtime behavior, run focused tests before moving to broader gates.

## Recommended first slice

Start with **W0.S1 → W0.S7** only. W0 is an evidence/RED slice: it should make the current behavior measurable without changing production behavior. Do not begin W1 policy code until W0 has captured legacy loop, guardrail, finalizer, near-limit, and bypass-path facts.

---

## W0 — Re-establish current behavior and terminology

Goal: Confirm the current limit surface and reproduce the user-visible failure class without changing production behavior.

### W0.S1 — Inventory current loop and budget seams

- Inspect `agent/conversation_loop.py`, `agent/turn_context.py`, `agent/turn_finalizer.py`, and `agent/iteration_budget.py`.
- Record the exact loop condition, `IterationBudget(agent.max_iterations)` initialization, consume/refund behavior, and finalizer budget-exhaustion predicate.
- Evidence: note line references and the observed relationship between `agent.max_turns`, `agent.max_iterations`, and `IterationBudget.max_total`.
- Exit gate: current loop/budget/finalizer facts are written to the W0 evidence packet.

### W0.S2 — Pin legacy max-iteration stop behavior

- Add or identify a focused baseline-pinning test proving legacy mode stops on the fixed limit even when later progress would be possible.
- The test should run in default/legacy behavior, should not require new policy modules, and should pass against the current implementation once the observed legacy behavior is correctly encoded.
- If no such test exists, first add the missing baseline-pinning test; the future progress-aware behavior RED belongs to W5, not W0.
- Exit gate: test either exists with evidence or a new baseline-pinning test demonstrates the current stop behavior.

### W0.S3 — Pin existing guardrail halt behavior

- Inspect `agent/tool_guardrails.py`, `run_agent.py`, and `agent/tool_executor.py` guardrail seams.
- Prove repeated exact failure/no-progress behavior still maps to the current controlled halt flow when hard-stop guardrails are enabled.
- Include evidence for `_tool_guardrail_halt_decision`, `_toolguard_controlled_halt_response()`, and `guardrail_halt` turn reason.
- Exit gate: current guardrail halt behavior is documented and covered by an existing or new focused test.

### W0.S4 — Capture `_budget_grace_call` and toolless-summary behavior

- Inspect how `_budget_grace_call`, `_handle_max_iterations()`, and the finalizer interact.
- Record whether the extra toolless summary attempt mutates messages and how role alternation is preserved.
- Evidence should include `_format_turn_completion_explanation()` behavior for budget exhaustion.
- Exit gate: legacy finalization semantics are documented before any policy change.

### W0.S5 — Capture `execute_code` refund behavior

- Inspect the tool-call branch where `execute_code`-only iterations refund `IterationBudget`.
- Add or identify a focused test that proves the refund behavior remains intact.
- Exit gate: refund behavior is pinned so W4/W5 cannot accidentally consume budget for programmatic tool-calling rounds.

### W0.S6 — Capture near-limit error guard behavior

- Inspect the post-error branch in `agent/conversation_loop.py` that compares `api_call_count` with `agent.max_iterations - 1`.
- Add or identify a focused test or evidence note showing how it behaves today.
- Exit gate: future W4 can re-anchor this hidden hard-stop behavior to the effective hard budget or policy decision.

### W0.S7 — Classify loop-bypass runtime paths

- Inspect app-server/provider-specific paths that can return before the normal tool-calling loop.
- Decide whether each path is out of scope for progress-aware policy or requires an adapter.
- Minimum surfaces: `conversation_loop.py` app-server path, provider-specific runtimes, and any path that bypasses `agent._execute_tool_calls()`.
- Exit gate: bypass paths are explicitly listed in the W0 evidence packet.

### W0 exit gate

- Current behavior is measurable and documented.
- No production behavior changed except intentional RED tests/evidence files.
- W0 evidence references exact code paths and test commands.

---

## W1 — Add policy data model and pure decision tests

Goal: Create a pure, import-light iteration policy that can decide continue/stop without touching provider code.

### W1.S1 — Define policy vocabulary and reason constants in tests first

- Write tests for the policy reason names before implementing the module.
- Required names include `below_soft_limit`, `soft_limit_reached_but_progress_detected`, `soft_limit_reached_no_progress`, `guardrail_exact_failure`, and `hard_max_iterations_reached`.
- Ensure no turns-based reason variant appears in active tests.
- Exit gate: RED test fails because `agent/iteration_policy.py` does not exist or lacks the reason model.

### W1.S2 — Add the minimal policy dataclasses/enums

- Create `agent/iteration_policy.py` with a small decision object and config object.
- Keep imports minimal and avoid provider, CLI, gateway, or tool executor imports.
- Include `mode`, `soft_max_iterations`, `hard_max_iterations`, `progress_window`, and `require_progress_after_soft_limit`.
- Exit gate: policy object can be imported in isolation.

### W1.S3 — Implement below-soft-limit and hard-cap decisions

- Implement pure decisions for below soft threshold and hard max reached.
- Do not inspect messages or tool outputs.
- Exit gate: tests for below soft limit and hard max pass.

### W1.S4 — Implement soft-limit progress/no-progress decisions

- Add tests for soft threshold + recent progress → continue.
- Add tests for soft threshold + no recent progress → graceful stop.
- Use a compact progress-window summary input, not raw tool result text.
- Exit gate: soft-limit decisions pass without loop integration.

### W1.S5 — Implement guardrail decision override

- Add tests proving guardrail hard stop wins over progress.
- Use a minimal guardrail decision input shape or adapter protocol rather than importing runtime-heavy objects.
- Exit gate: guardrail stop test passes and policy remains pure/import-light.

### W1.S6 — Verify W1 isolation

- Run focused policy tests.
- Confirm `conversation_loop.py`, `turn_context.py`, `turn_finalizer.py`, and `tool_executor.py` are unchanged in this slice.
- Exit gate: W1 creates only pure policy code/tests and does not change runtime behavior.

---

## W2 — Add progress tracker and fingerprint tests

Goal: Track bounded recent progress/no-progress signals without relying on model self-reporting.

### W2.S1 — Define tracker event schema tests

- Write RED tests for a bounded event schema: event kind, tool name, status/error flags, duration bucket, and small labels.
- Include tests rejecting raw large output storage.
- Exit gate: tests fail before tracker implementation.

### W2.S2 — Define secret-safe fingerprint behavior

- Write tests proving args/results are redacted or summarized before fingerprinting.
- Require keyed or per-turn/session-bounded fingerprints where persisted or observable.
- Add adversarial inputs with API-key-like strings, tokens, passwords, and low-entropy values.
- Exit gate: tests fail until tracker avoids raw/durable secret-like fingerprints.

### W2.S3 — Reuse or expose guardrail observation metadata

- Inspect whether `ToolCallGuardrailController` exposes enough metadata for policy/tracker consumption.
- If not, add a public observation/decision API or shared bounded event object.
- Do not read private controller dictionaries from outside the controller.
- Exit gate: tracker can consume guardrail-relevant metadata through a public seam.

### W2.S4 — Implement bounded progress tracker window

- Create `agent/progress_tracker.py` or extend an existing minimal helper.
- Implement bounded window behavior and event summary production.
- Do not scan or mutate conversation history.
- Exit gate: bounded-window tests pass.

### W2.S5 — Implement progress/no-progress classification basics

- Add classification for landed mutations, new evidence artifacts, changed test output, lifecycle state changes, and new informational fingerprints.
- Add no-progress classification for identical failed calls and repeated idempotent no-op reads/searches.
- Exit gate: classification tests pass for positive and negative cases.

### W2.S6 — Handle concurrency and cancellation semantics

- Add tests for deterministic concurrent result observation or explicit thread-safe/order-aware tracker updates.
- Add tests proving interrupted, cancelled, and skipped tool calls are non-progress.
- Exit gate: tracker cannot extend a turn because a cancellation produced a new result string.

### W2.S7 — Verify W2 no-runtime-integration boundary

- Run tracker and guardrail-focused tests.
- Confirm full loop behavior is still unchanged unless the chosen public guardrail API requires small pure changes.
- Keep W2 responsible for tracker-level schema, classification, fingerprinting, bounded window behavior, and the public guardrail metadata seam only.
- Keep runtime `tool_executor` sequential/concurrent observation wiring owned by W5.S3/W5.S4.
- Exit gate: tracker is ready for W5 wiring but not yet wired into the loop.

---

## W3 — Wire legacy-compatible config defaults

Goal: Expose policy config without changing default runtime behavior.

### W3.S1 — Inventory config/default surfaces

- Inspect `cli.py`, `hermes_cli/config.py`, AIAgent initialization, gateway runtime env bridging, cron config reads, API server env handling, and delegation config.
- Record current precedence and merge behavior.
- Exit gate: config surface map exists in the W3 evidence packet.

### W3.S2 — Add config tests for legacy default

- Add tests proving missing `agent.iteration_policy` resolves to `legacy` mode.
- Prove existing `agent.max_turns` behavior remains unchanged in legacy mode.
- Exit gate: RED tests capture default compatibility expectations.

### W3.S3 — Add nested config defaults with deep merge

- Add `agent.iteration_policy` defaults in all authoritative default/config surfaces or route them through one shared resolver.
- Ensure partial user config does not erase sibling defaults.
- Exit gate: config tests show deep-merged defaults.

### W3.S4 — Add config validation tests

- Test invalid mode, non-positive limits, hard cap less than or equal to soft threshold, and missing/unsafe context profile values.
- Define whether invalid values fail closed, warn and fall back, or refuse startup.
- Exit gate: invalid config behavior is deterministic and documented.

### W3.S5 — Add AIAgent runtime resolver seam

- Add a minimal resolver or initialization field for iteration policy while preserving constructor/backcompat behavior.
- Do not yet alter the main loop stop behavior.
- Exit gate: an agent can report resolved legacy policy without runtime behavior changes.

### W3.S6 — Verify config display/check/migration behavior

- Run focused `tests/hermes_cli` checks if config surfaces changed.
- Verify `hermes config`/config display paths expose or preserve new defaults as intended.
- Exit gate: config UX does not regress and legacy users see no behavior change.

---

## W4 — Integrate policy with conversation loop behind legacy mode

Goal: Replace raw fixed-count loop control with policy decisions while preserving legacy semantics by default.

### W4.S1 — Add loop-level legacy equivalence tests

- Add tests proving `legacy` mode behaves like the old fixed-count loop.
- Include `_budget_grace_call`, `_handle_max_iterations()`, and `execute_code` refund cases from W0 evidence.
- Exit gate: RED tests define exact compatibility before integration.

### W4.S2 — Introduce effective hard-budget computation

- Add a small helper or resolver that separates soft threshold from effective hard budget in progress-aware mode.
- Preserve `agent.max_iterations` as legacy/reporting soft threshold where appropriate.
- Exit gate: helper tests pass without changing loop behavior yet.

### W4.S3 — Update per-turn `IterationBudget` initialization

- Adjust `turn_context.py` initialization only as needed so progress-aware mode can use an effective hard budget.
- Keep legacy initialization equivalent to `IterationBudget(agent.max_iterations)`.
- Exit gate: legacy budget tests remain green; progress-aware budget uses the hard cap.

### W4.S4 — Insert policy decision point in loop

- Ask the policy for a decision at stable iteration boundaries.
- Preserve interrupt-first behavior and role alternation.
- Exit gate: legacy loop tests pass after integration.

### W4.S5 — Re-anchor near-limit error guard

- Replace the old `agent.max_iterations - 1` near-limit check with effective hard-budget/policy-aware logic.
- Add tests for both legacy and progress-aware modes.
- Exit gate: hidden hard-stop behavior no longer defeats soft-limit continuation.

### W4.S6 — Classify or adapt loop-bypass runtimes

- For each W0 bypass path, either document out-of-scope behavior in code/tests or add an adapter that preserves safety.
- Do not silently let bypass paths inherit inconsistent semantics.
- Exit gate: bypass behavior is explicit and test-covered or intentionally deferred.

### W4.S7 — Verify W4 compatibility

- Run focused run-agent loop tests and existing budget/guardrail tests.
- Exit gate: legacy mode is behavior-neutral and progress-aware mode can be wired in W5.

---

## W5 — Enable progress-aware mode in loop-level tests

Goal: Prove productive work can pass the soft threshold and no-progress loops stop.

### W5.S1 — Add productive soft-limit crossing RED test

- Build a loop-level test where recent objective progress inside the window permits another iteration after the soft threshold.
- Do not rely on model self-reporting as progress.
- Exit gate: RED test fails before tracker wiring.

### W5.S2 — Add repeated no-progress stop RED test

- Add a loop-level test where repeated identical failure or idempotent no-progress stops before the hard cap.
- Include a guardrail-derived signal where possible.
- Exit gate: RED test proves no-progress stop is required.

### W5.S3 — Wire tracker events from sequential tool execution

- Feed bounded progress observations after sequential tool results are classified and before final loop decision consumes the summary.
- Preserve existing tool result appending and budget enforcement order.
- Exit gate: sequential progress-aware tests pass.

### W5.S4 — Wire tracker events from concurrent tool execution

- Feed observations after deterministic concurrent result collection or through a thread-safe/order-aware tracker.
- Prove mixed success/failure batches produce equivalent classifications to sequential execution.
- Exit gate: concurrent mixed-result test passes.

### W5.S5 — Preserve guardrail halt semantics

- Treat `_tool_guardrail_halt_decision` as terminal policy input.
- Prove controlled guardrail halt response remains unchanged where expected.
- Exit gate: guardrail runtime tests pass.

### W5.S6 — Preserve interrupt priority

- Add or run tests showing user interrupt wins over policy continuation.
- Ensure interrupted, cancelled, and skipped tool outputs do not count as progress.
- Exit gate: interrupt tests pass and no cancellation extends the turn.

### W5.S7 — Verify W5 behavior matrix

- Run progress-aware loop tests, guardrail runtime tests, and focused interrupt/budget tests.
- Exit gate: productive progress crosses soft threshold, no-progress stops, hard cap always stops.

---

## W6 — Finalization and user-visible stop reason

Goal: Make graceful/hard stop reasons explicit in final responses and logs without fabricating completion.

### W6.S1 — Add finalizer tests for policy stop reasons

- Add tests for `policy_soft_no_progress_stop` and `policy_hard_max_iterations_reached` finalization.
- Ensure no-progress stop does not claim task success.
- Exit gate: RED tests fail before finalizer changes.

### W6.S2 — Add soft-limit success completion test

- Add a test where progress-aware mode crosses the soft threshold, remains below hard cap, and returns a valid final response.
- Prove the result is treated as complete where callers should consider the turn successful.
- Exit gate: current `api_call_count < agent.max_iterations` completion predicate fails this RED case.

### W6.S3 — Update finalizer completion semantics

- Modify `agent/turn_finalizer.py` or its caller contract so policy-aware completion uses the correct soft/hard distinction.
- Preserve legacy completion behavior in legacy mode.
- Exit gate: soft-limit success test passes without breaking legacy budget exhaustion tests.

### W6.S4 — Add user-visible stop messages

- Map structured policy reasons to concise final responses and logs.
- Distinguish no-progress guardrail stop, hard cap, and user interrupt.
- Exit gate: `test_turn_completion_explainer.py` or equivalent focused tests pass.

### W6.S5 — Verify caller semantics for cron/delegation/gateway/worker

- Add focused tests or evidence showing productive soft-limit crossing is not treated as legacy budget exhaustion by cron, delegation, gateway, or worker callers.
- Exit gate: caller-facing result metadata is unambiguous.

### W6.S6 — Verify role alternation and session persistence

- Confirm policy stop and controlled final responses do not create same-role adjacency or invalid resume history.
- Exit gate: message-sequence/role alternation tests pass.

---

## W7 — Context profiles for gateway/cron/delegation/worker

Goal: Prevent one permissive interactive policy from leaking into contexts that need stronger caps.

### W7.S1 — Add context profile resolver tests

- Add tests for CLI/local, gateway, API-server, cron, delegation, and worker profile resolution.
- Include partial config and invalid config cases.
- Exit gate: RED tests define context precedence before implementation.

### W7.S2 — Implement CLI/local profile resolution

- Resolve soft threshold from existing `agent.max_turns` by default.
- Resolve a separate bounded hard cap only in progress-aware mode.
- Exit gate: CLI/local resolver tests pass and legacy mode remains unchanged.

### W7.S3 — Implement gateway profile resolution

- Cover both `gateway/run.py` and `gateway/platforms/api_server.py`.
- Ensure platform delivery/inactivity timeout remains authoritative.
- Exit gate: gateway profile tests pass and API-server env/backcompat behavior is explicit.

### W7.S4 — Implement cron conservative profile

- Keep cron conservative by default and avoid unbounded progress extension.
- Integrate policy stops with cron failure/timeout accounting as appropriate.
- Exit gate: cron tests prove conservative hard cap behavior.

### W7.S5 — Implement delegation profile resolution

- Bound child hard cap by delegation config and parent/session hard-budget policy.
- Cover sync, batch, and background delegation where relevant.
- Exit gate: delegation tests prove children cannot silently exceed intended bounds.

### W7.S6 — Implement worker/Kanban profile behavior

- Integrate policy stop with task failure accounting and claim timeout behavior.
- Ensure no infinite worker claim/retry loop is introduced.
- Exit gate: worker/Kanban tests or explicit evidence cover policy stop behavior.

### W7.S7 — Verify context matrix

- Run gateway, cron, delegation, and worker-focused checks touched by this slice.
- Exit gate: context-specific profile evidence is ready for W9 packet.

---

## W8 — Documentation and migration notes

Goal: Document the distinction between perceived tool-call limit and agent iteration policy.

### W8.S1 — Update user-facing configuration docs

- Explain `agent.max_turns` / `max_iterations` terminology.
- Explain `legacy` vs `progress_aware` mode and soft threshold vs hard cap.
- Exit gate: config example is valid YAML and links resolve.

### W8.S2 — Update developer loop/policy docs

- Document where the policy sits relative to conversation loop, tool executor, finalizer, and guardrails.
- Include invariants: prompt-cache safety, role alternation, interrupt priority, secret-safe progress events.
- Exit gate: developer docs match implemented code paths.

### W8.S3 — Add troubleshooting copy for policy stops

- Document what users should do when the agent stops due to no progress.
- Keep copy concise and avoid exposing raw internal reason strings as product language.
- Exit gate: stop-reason copy is user-safe and test-covered where applicable.

### W8.S4 — Update superseded source artifact notice

- Use the `source_artifact` path from the parent plan/review front matter: `/Users/cube-mac/dreampia-mvp-evidence/latest/hermes-progress-aware-tool-call-limit-plan-2026-06-25.md`.
- Verify the mixed source artifact points to the active split docs in the current repo/worktree or repository-relative paths.
- Remove stale checkout-specific active-doc paths.
- Exit gate: stale-path search returns zero hits.

### W8.S5 — Verify docs and links

- Run link/path structural checks for plan, work order, step breakdown, review, and source artifact notice.
- Exit gate: docs validation passes and no plan/work-order boundary regression is introduced.

---

## W9 — Review, verification, and PR packet

Goal: Close the work with evidence, independent review, and a clean PR lifecycle.

### W9.S1 — Assemble W-unit evidence packets

- For each implemented W-unit, collect touched files, focused tests, broader tests, stop-reason evidence, legacy compatibility evidence, soft-limit success/completed evidence, context profile evidence, and deferred checks.
- Exit gate: evidence packet is complete for every implemented W-unit.

### W9.S2 — Run focused policy/tracker gates

- Run policy and tracker tests.
- Include secret-safety, bounded-window, and no-progress cases.
- Exit gate: focused policy/tracker gates pass.

### W9.S3 — Run loop/finalizer/guardrail gates

- Run run-agent, guardrail runtime, turn completion explainer, and iteration budget race tests.
- Exit gate: loop/finalizer/guardrail gates pass with no new failures.

### W9.S4 — Run context surface gates

- Run relevant config, gateway, cron, API-server, TUI, delegation, and worker tests depending on touched surfaces.
- Record any deferred checks and why.
- Exit gate: context profile evidence is complete or deferrals are explicit and justified.

### W9.S5 — Run static hygiene checks

- Run `git diff --check` and any project lint/static gates relevant to touched files.
- Check for hardcoded secrets or unsafe logging of raw tool outputs/fingerprints.
- Exit gate: no formatting/security hygiene failures.

### W9.S6 — Independent review

- Send the final diff and evidence packet to an independent reviewer.
- Require Critical 0 / High 0 equivalent before merge lifecycle.
- Exit gate: independent review has no blocking findings, or findings are fixed and re-reviewed.

### W9.S7 — PR lifecycle if approved

- If the user approves lifecycle, push branch, open PR, observe CI, merge only after checks pass, then observe main CI.
- Exit gate: branch/PR/main status is reported with concrete URLs/IDs and CI status.

---

## Step-breakdown verification checklist

- Parent work order links to this step-breakdown file.
- This file links back to the parent plan, parent work order, and related review.
- W-unit IDs match the parent exactly: W0 through W9, in order.
- Every W-unit has at least one `Wn.Sm` step.
- W0 remains an evidence/RED-first slice and is the recommended first slice.
- No production code changes are implied before W0 evidence is complete.
- Context profiles include CLI/local, gateway, API-server, cron, delegation, and worker/Kanban surfaces.
- Verification coverage includes focused policy/tracker, loop/finalizer/guardrail, config, context, hygiene, and independent review gates.
