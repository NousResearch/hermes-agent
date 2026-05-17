# Planning/readiness arbitration for crypto_bot

Use this reference when a session asks to “review planning files” or decide the next crypto_bot step from a mixture of `plan.json`, strategic summaries, Hermes project descriptors, native Kanban, and readiness outputs.

## Durable lesson

Do not let a stale or internally inconsistent planning field override live control-plane readiness. The correct next step is the first unresolved lifecycle blocker, not necessarily the `next_recommended_session_id` in `plan.json`.

Priority order when sources disagree:

1. Managed-project descriptor and current autonomy/readiness gates.
2. Native Kanban import/audit state, including dependency links and blocked/review columns.
3. PR/CI audit and remote lifecycle state for the exact branch/head.
4. Completion-gate and PR-evidence artifacts for the exact task/base/branch/head.
5. Strategic plan session list and `remaining_*` notes.
6. Strategic plan top-level summary fields and human-readable gap summaries.

## Session-derived example

A planning review found:

- `plan.json` top-level fields recommended `S017A` RunPod automation architecture gap mapping.
- The same plan’s session list still marked `S007A` as `planned_next` and `remaining_stream_a_screens` still required Daemon Audit read-only implementation / Stream A closeout.
- Native Kanban had 90 cards: 89 `blocked`, 1 `review_required`; `S007A` was blocked with no worker runs.
- Readiness and PR/CI audit showed S006 remote lifecycle still blocked next-task dispatch.
- The live PR existed at a newer/different HEAD than some evidence artifacts, creating completion-gate/PR-evidence source-head mismatch.
- Control-plane self-check also reported source/runtime asset parity drift for the `crypto-bot-pm` skill.

Correct decision: do not start `S007A` or `S017A`. First repair Hermes control-plane parity, rerun readiness, then reconcile S006 PR/evidence/CI for the actual PR HEAD. Only after the remote lifecycle is closed should product task selection resume.

## Reporting pattern

When reporting next steps:

- State which sources were reviewed.
- Separate verified machine state from plan-intent text.
- Name the exact blocker(s) that prevent product-task selection.
- Give an ordered sequence: control-plane repair → readiness rerun → exact-head evidence reconciliation → CI/check evidence read → merge-readiness dry-run if eligible → only then next product task.
- Avoid asking for broad approval when readiness does not permit the action yet.
