# /ouro-intake upstream Ouroboros Interview/Seed parity matrix

Authority: Kanban BO-061 (`t_4530f72b`)

Goal: port/adapt the necessary upstream `Q00/ouroboros` Interview/Seed behavior into Hermes `/ouro-intake` so it works in the Hermes/Kanban authority model. This is not an “Ouroboros-inspired” checklist.

## Rule

Upstream Interview/Seed behavior is the default. Any difference must be documented as an intentional divergence. Unapproved divergence is a failure.

## Matrix

| ID | Upstream source | Required upstream behavior | Hermes `/ouro-intake` target | Status | Proof |
|---|---|---|---|---|---|
| INT-001 | `skills/interview/SKILL.md` lines 106-119 | Preserve role split: question generator/state, main-session answer router/refiner, human judgment path. | Local controller keeps interview state and explicit refined-answer records; execution authority remains Kanban. | ported | `gateway/ouro_intake.py`, `test_upstream_compound_scope_answer_is_refined_not_collapsed` |
| INT-002 | `skills/interview/SKILL.md` lines 278-330 | Do not compress meaningful answers; preserve structured payload with source prefix semantics. | Store `raw_answer`, `[from-user][refined]`, Decision, Reasoning, Constraints, Out of scope, Codebase context. | ported | `test_upstream_compound_scope_answer_is_refined_not_collapsed` |
| INT-003 | `skills/interview/SKILL.md` lines 331-403 | Free-text answers carrying scope/constraints/decisions require a Refine confirmation gate before being absorbed as ground truth. | Add `refine_pending`; structured payload is confirmed with `승인`/send-as-is before advancing. | ported | `test_upstream_refine_gate_requires_confirmation_for_scope_decision` |
| INT-004 | `skills/interview/SKILL.md` lines 405-410 | Maintain ambiguity/decision ledger across independent tracks. | Keep deterministic ledger and avoid same broad question after responsive answer. | ported | `test_upstream_same_question_not_repeated_after_responsive_answer` |
| INT-005 | `skills/interview/SKILL.md` lines 516-532 | Dialectic Rhythm Guard: avoid too many consecutive non-user answers. | Record `dialectic.non_user_streak`; current Discord intake mostly uses user judgment so streak remains bounded. | ported | `test_intentional_divergences_are_documented` |
| INT-006 | `skills/interview/SKILL.md` lines 536-553 | MCP recoverable retry semantics. | Gateway-local mode has no live MCP call; document as intentional divergence. | intentional_divergence | `test_intentional_divergences_are_documented` |
| CLOSE-001 | `src/ouroboros/agents/seed-closer.md` lines 11-17 | Low ambiguity is permission to audit closure, not permission to close. | Add Seed Closer audit; material gaps block restate/seed even if score is low. | ported | `test_upstream_seed_closer_blocks_score_only_seed_ready` |
| CLOSE-002 | `src/ouroboros/agents/seed-closer.md` lines 26-30 | For brownfield/system work, check ownership/SSOT, API/protocol contract, lifecycle/recovery, migration/cross-client impact, verification. | Add `seed_closer` blockers/questions for system/brownfield domains. | ported | `test_upstream_seed_closer_blocks_score_only_seed_ready` |
| RESTATE-001 | `skills/interview/SKILL.md` lines 425-508 | Restate before seed; restate corrections must go through refine and closure again. | Add `restate_correction_refine_pending`; correction is not merged directly. | ported | `test_upstream_restate_correction_never_bypasses_refine` |
| SEED-001 | `src/ouroboros/core/seed.py` lines 156-253 | Seed is an immutable workflow constitution with goal, task_type, brownfield_context, constraints, acceptance_criteria, ontology_schema, evaluation_principles, exit_conditions, metadata. | Project these upstream fields into `upstream_seed` inside the Kanban admission Seed Contract. | ported | `test_seed_projection_preserves_upstream_seed_fields` |
| SEED-002 | `src/ouroboros/core/seed_contract.py` lines 41-77 | Runtime SeedContract interprets Seed without changing the frozen Seed. | Hermes keeps upstream seed projection separate from Hermes/Kanban authority fields. | ported | `test_seed_projection_preserves_upstream_seed_fields` |
| SEED-003 | `src/ouroboros/bigbang/seed_generator.py` lines 133-168 | Gate seed generation on ambiguity and required initial context. | Gate on ambiguity plus Seed Closer audit; initial context maps to captured goal/context/refinements. | ported | `test_upstream_seed_closer_blocks_score_only_seed_ready` |
| SEED-004 | `src/ouroboros/auto/seed_reviewer.py`; `src/ouroboros/auto/seed_repairer.py` | Review/repair seeds with bounded deterministic repairs and unresolved findings. | Keep deterministic `seed_qa`, add upstream-shaped review/repair fields, bounded repair metadata. | ported | `test_seed_projection_preserves_upstream_seed_fields` |
| AUTH-001 | Hermes/Kanban user requirement | Seed must not grant execution authority in Hermes. | `/ouro-intake admit` creates blocked/proposed Kanban admission only; no task runs/worker dispatch. | intentional_divergence | `test_admission_never_dispatches_executor` |
| CANCEL-001 | Hermes Discord UX requirement plus upstream cancel concept | User must be able to leave capture. | `/ouro-intake cancel` and plain escape replies expire origin binding. | ported | `test_cancel_escape_expires_plain_reply_capture` |

## Intentional divergences

### DIV-001 — No live upstream MCP call in gateway path

- Upstream: `ouroboros_interview` MCP is the preferred question generator/state store.
- Hermes: `/ouro-intake` runs in the gateway and must not introduce execution authority confusion or external MCP availability as a runtime requirement.
- Adaptation: port the behavioral contracts locally and preserve `[from-user][refined]` semantics in durable session state.
- Approval: implicit in Chris's instruction to make it work in our Hermes/Kanban situation while keeping authority local.

### DIV-002 — Seed projected as Kanban admission source material, not execution authority

- Upstream: Seed is the immutable constitution for execution/evaluation.
- Hermes: `upstream_seed` is preserved inside a Kanban admission Seed Contract, while `authority`, `side_effect_boundary`, `initial_routing`, and `approval_required_for` keep execution blocked.
- Approval: explicit user requirement.

### DIV-003 — No gateway restart/live Discord smoke in implementation lane

- Upstream: plugin/runtime sessions see their own changes after restart/session refresh.
- Hermes: source is live-checkout applied only when approved; gateway restart/live smoke are separate gates.
- Approval: standing Hermes gateway boundary.
