---
title: "review: Progress-aware tool-call limit plan separation"
status: passed-after-deep-review-closure
date: 2026-06-25
type: review
target_repo: hermes-agent
reviewed_plan: ../plans/2026-06-25-progress-aware-tool-call-limit-plan.md
reviewed_work_order: ../work-orders/2026-06-25-progress-aware-tool-call-limit-work-order.md
reviewed_step_breakdown: ../work-orders/2026-06-25-progress-aware-tool-call-limit-step-breakdown.md
source_artifact: /Users/cube-mac/dreampia-mvp-evidence/latest/hermes-progress-aware-tool-call-limit-plan-2026-06-25.md
---

# review: Progress-aware tool-call limit plan separation

## Verdict

PASS with notes. The original mixed artifact has been split into a pure strategic plan and a separate execution work order. The new plan is review-ready for implementation planning, while the work order carries W-unit execution detail, RED/GREEN expectations, commands, evidence packet shape, and exit criteria.

## Reviewed files

| File | Class | Verdict |
|---|---|---|
| `../plans/2026-06-25-progress-aware-tool-call-limit-plan.md` | Plan | PASS |
| `../work-orders/2026-06-25-progress-aware-tool-call-limit-work-order.md` | Work Order | PASS |
| `/Users/cube-mac/dreampia-mvp-evidence/latest/hermes-progress-aware-tool-call-limit-plan-2026-06-25.md` | Source artifact | Superseded notice added |

## Plan purity review

The plan now owns:

- problem statement;
- goals and non-goals;
- design principles;
- architecture direction;
- configuration strategy;
- progress/no-progress strategy;
- context-specific policy direction;
- rollout strategy;
- acceptance criteria;
- risks and open questions.

The plan no longer owns:

- W-unit implementation bodies;
- RED/GREEN execution detail;
- exact test command matrix;
- per-slice file-edit instructions;
- PR evidence packet body.

## Work order review

The work order now owns:

- W0-W9 execution slices;
- likely file targets;
- RED/GREEN and evidence expectations;
- test commands;
- verification packet shape;
- implementation-specific constraints.

This is the correct place for the material that had been mixed into the original plan sections 8-12.

## Technical review notes

### N1 — Terminology clarified

The plan correctly distinguishes the user-visible “tool-call limit” from the implementation-level agent iteration/API-loop limit. This matters because fixing a perceived tool-call cap by adding a model-facing tool or simply increasing max turns would address the symptom poorly.

### N2 — Legacy default is correct

The plan keeps `legacy` as the default rollout mode. This is important for Hermes Agent because agent-loop changes can affect CLI, gateway, cron, delegation, and workers at once.

### N3 — Hard cap retained

The plan does not remove hard caps. This preserves a fail-safe for broken loops and is safer than unbounded progress-aware continuation.

### N4 — Context profiles need implementation attention

The work order correctly separates interactive, gateway, cron, delegation, and worker contexts. During implementation, this should be treated as a real acceptance gate, not a docs-only note.

### N5 — Prompt-cache and role alternation invariants included

The plan explicitly protects prompt caching and message role alternation. This aligns with Hermes Agent `AGENTS.md` guidance.

## Blockers

None remaining after the 2026-06-25 code-recon and deep-review closure patches below. Before those patches, the plan was directionally correct but not sufficiently implementation-ready because it did not identify several existing Hermes loop/tool seams and post-soft-limit finalization/context-profile pitfalls.

## Implementation sufficiency audit — 2026-06-25

### Question

Can an implementer build the requested progress-aware tool-call/iteration policy from the plan as written?

### Pre-patch verdict

**Not safely enough.** The original split plan had the right product intent, but it was missing concrete current-code anchors. An implementer could have duplicated existing guardrail logic, patched only the loop condition, or accidentally broken finalization semantics.

### Code facts checked

| Area | Current code fact | Implementation impact |
|---|---|---|
| Main loop | `agent/conversation_loop.py` gates on `api_call_count < agent.max_iterations` plus `agent.iteration_budget.remaining > 0` | Soft/hard limit split must change loop condition and budget together |
| Turn budget | `agent/turn_context.py` resets `IterationBudget(agent.max_iterations)` per turn | A hard cap cannot be introduced by only changing config docs |
| Finalization | `agent/turn_finalizer.py` treats `api_call_count >= agent.max_iterations` as exhaustion and calls `_handle_max_iterations()` | Soft-limit continuation requires finalizer-aware stop reasons |
| Tool results | `agent/tool_executor.py` has separate sequential/concurrent post-call paths | Progress observations must be wired in both paths |
| Guardrails | `agent/tool_guardrails.py` already owns canonical args/result hashes and repeated failure/no-progress counts | New tracker/policy should reuse existing decisions rather than duplicate fingerprinting |
| Existing halt | `conversation_loop.py` already turns `_tool_guardrail_halt_decision` into `guardrail_halt` and a controlled response | Progress policy must not regress current guardrail halt behavior |
| Config | default CLI config lives in `cli.py`; entrypoints pass `max_iterations` into `AIAgent` | Config work must cover default display/resolution and runtime init |
| Contexts | cron/gateway/TUI/delegation construct agents through separate surfaces | Context profile work is a real implementation gate, not a doc-only note |

### Gaps found and closed in docs

- Added current implementation reconnaissance to the plan.
- Added implementation constraints for `conversation_loop.py`, `turn_context.py`, `turn_finalizer.py`, `IterationBudget`, `tool_executor.py`, and `tool_guardrails.py`.
- Replaced ambiguous turns-based internal policy names with `soft_max_iterations` / `hard_max_iterations`, while preserving user-facing explanation of historical `turns` terminology.
- Added explicit requirement to reuse existing `ToolCallGuardrailController` fingerprint/failure/no-progress data.
- Expanded work-order file targets to include `agent/tool_executor.py`, `agent/tool_guardrails.py`, `agent/agent_init.py`, `agent/turn_context.py`, `cli.py`, cron/gateway/TUI/delegation surfaces, and existing tests.
- Strengthened W0/W1/W2/W4/W5/W6 so implementation starts from evidence, preserves legacy behavior, and covers both sequential/concurrent tool paths.

### Post-patch verdict

**Sufficient for W0/W1 implementation planning.** The plan/work-order now identify the critical seams and the failure modes that could make a naive implementation unsafe. Full implementation should still proceed W-unit by W-unit, beginning with W0 evidence and RED tests; do not jump directly to broad loop rewrites.

## Deep-review closure — 2026-06-25

### Review method

The augmented docs were re-reviewed from three angles:

1. architecture/current-code alignment;
2. document separation, plan purity, and implementation handoff quality;
3. operational/test/safety risk across CLI, gateway, cron, delegation, workers, and API-server surfaces.

### Findings and closure ledger

| Severity | Finding | Closure applied |
|---|---|---|
| P1 | Soft-limit continuation could still be marked incomplete because `turn_finalizer` currently requires `api_call_count < agent.max_iterations` for `completed`. | Plan acceptance and Work Order W6 now require successful post-soft-limit finalization and caller behavior evidence. |
| P1 | Main `gateway/run.py` AIAgent creation/cache path was missing from file targets/context-profile gates. | Work Order file targets, anchors, W7 profile rules, and W9 matrix now include `gateway/run.py` plus API-server behavior. |
| P1 | Existing guardrail data is not directly consumable because `_append_guardrail_observation()` returns only a string. | Plan open question and Work Order W2 now require a public observation/decision API or shared bounded event object. |
| P1 | Post-error near-limit guard still compares against `agent.max_iterations - 1`. | Plan constraints and Work Order W0/W4 now require measuring and re-anchoring this guard. |
| P1 | Provider/runtime paths that bypass the normal loop, such as app-server runtimes, were not classified. | Plan constraints and Work Order W0/W4 now require explicit exclusion or adapter coverage. |
| P1 | Nested config rollout could diverge across `cli.py`, `hermes_cli/config.py`, gateway, cron, API-server, and delegation surfaces. | Plan config strategy and Work Order W3/W7 now require authoritative/deep-merged config resolution and validation. |
| P1 | Context profile defaults were direction-only. | Plan context table and Work Order W7 now define initial resolution rules and evidence requirements. |
| P1 | Source artifact notice pointed at a stale checkout path. | Work Order W8 and evidence packet now require source-artifact notice verification; the notice was patched to the active isolated worktree paths. |
| P2 | `turns`/`iterations` reason naming was inconsistent. | Plan and Work Order now standardize on `hard_max_iterations_reached` internally. |
| P2 | Tracker secret-safety was too weak if implemented as plain hashes of raw args/results. | Plan progress strategy and Work Order W2 now require redacted/keyed bounded fingerprints and no durable hash of low-entropy secret-looking values. |
| P2 | Concurrent tool observation order/thread-safety was not explicit. | Plan and Work Order W2/W5 now require deterministic post-collection observation or thread-safe/order-aware tracker behavior. |
| P2 | Interrupted/cancelled/skipped tools could be misread as progress. | Plan and Work Order W2/W5 now state they are non-progress and interrupt wins over policy. |
| P2 | W9 verification matrix was narrower than touched surfaces. | Work Order W9 now includes guardrail runtime, turn completion explainer, iteration budget race, TUI/delegation conditional checks, context evidence, and soft-limit completion evidence. |

### Post-deep-review verdict

**No remaining document-level P0/P1 blockers for W0/W1 implementation planning.** The docs are still not a license for a broad one-shot loop rewrite; they are ready for the W0 evidence slice followed by W1 pure policy modeling.

## Follow-up before implementation

- Re-establish current code path in `agent/conversation_loop.py` and `agent/turn_finalizer.py` before editing.
- Confirm whether an existing guardrail/fingerprint helper already exists to avoid duplicate infrastructure.
- Keep W0 as a read/test reproduction slice before adding policy modules.
- Do not touch unrelated desktop/i18n dirty files in the current checkout.

## Step-breakdown independent review — 2026-06-25

### Reviewed step-breakdown file

- `../work-orders/2026-06-25-progress-aware-tool-call-limit-step-breakdown.md`

### Independent review verdict

**GO for W0/W1 implementation planning.** The step breakdown decomposes parent W0-W9 into contiguous `Wn.Sm` handoff steps, keeps W0 evidence-first, keeps W1 pure/import-light, and does not authorize broad conversation-loop rewrites before W0/W1 gates are complete.

### Step-breakdown P2 closure ledger

| Severity | Finding | Closure applied |
|---|---|---|
| P2 | W0.S2 mixed baseline-pinning language with RED wording, which could make implementers confuse current-behavior pinning with future desired progress-aware REDs. | W0.S2 now explicitly says the baseline-pinning test should pass once current behavior is encoded, and future progress-aware RED belongs to W5. |
| P2 | W2/W5 responsibility boundary around `tool_executor` wiring could be read as ambiguous. | W2.S7 now states W2 owns tracker schema/classification/fingerprints/public guardrail metadata seam, while runtime sequential/concurrent `tool_executor` observation wiring is owned by W5.S3/W5.S4. |
| P2 | W8.S4 did not repeat the concrete superseded source artifact path for handoff convenience. | W8.S4 now references the `source_artifact` path from the parent plan/review front matter and spells out the current artifact path. |

### Post-step-breakdown verdict

**No remaining document-level P0/P1 blockers for W0/W1 implementation planning.** Remaining caution: proceed W-unit by W-unit; do not treat this as approval for a broad one-shot loop rewrite.

## Closure evidence

This review should be paired with a structural validation run that checks:

- plan links to work order and review;
- work order links back to plan and review;
- work order links to the step breakdown;
- step breakdown links back to plan, work order, and review;
- plan has no W-unit/RED/GREEN/command bodies;
- superseded mixed artifact points to active split docs;
- new docs are present at the expected paths.
