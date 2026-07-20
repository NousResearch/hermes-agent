# Q1 — Independent architecture/specification QA

Date: 2026-07-17
Task: t_cff7e226
Reviewer: automation-operator (fresh worker)
Plan: `/Users/hermes/.hermes/hermes-agent/.hermes/plans/2026-07-17_143520-truth-ledger-option-2.md`
Inputs reviewed:
- `docs/truth-ledger/discovery/gate-0.md`
- `docs/truth-ledger/discovery/repository-conventions.md`
- `docs/truth-ledger/discovery/runtime-contract.md`
- `docs/truth-ledger/design/storage-and-recovery.md`
- `docs/truth-ledger/design/extraction-evaluation.md`

## Verdict

PASS (Q1 complete; M2 may open)

Q1 acceptance requires implementation-ready architecture with no critical/important gaps across eligibility/identity, immutable semantics, recovery, privacy, evaluation, and repository-path conformity. Independent rerun checks now show prior C1/C2/C3 blockers are closed.

## Independent reproduction evidence (fresh-worker checks)

### Source/runtime contract probes re-run
- `agent/turn_finalizer.py:365-402` confirms `post_llm_call` still fires under `if final_response and not interrupted`, and now carries additive eligibility/origin/identity kwargs:
  - eligibility: `completed`, `failed`, `interrupted`, `turn_exit_reason`
  - origin: `delegate_depth`, `is_subagent`, `parent_session_id`, `kanban_task_id`
  - identity/lane: `speaker_id`, `conversation_id`, `chat_id`, `thread_id`, `chat_type`
- `agent/turn_context.py:216-218` confirms turn identity construction remains stable for idempotency design assumptions.
- `cli.py:12341-12346` and `cli.py:16162-16165` confirm CLI task-id path variability remains documented (interactive/session path explicit; fully quiet one-shot path implicit).
- `gateway/run.py:18171-18200` and `gateway/run.py:18777-18790` confirm gateway agent identity fields + explicit `task_id=session_id` handoff.
- `cron/scheduler.py:3044-3074` and `cron/scheduler.py:3104` confirm cron remains `platform="cron"`, `skip_memory=True`, `run_conversation(prompt)`.
- `tools/delegate_tool.py:1318` and `tools/delegate_tool.py:1921-1924` confirm subagent platform tagging and explicit child `task_id`.
- `hermes_cli/kanban_db.py:7711` and `hermes_cli/kanban_db.py:7794-7797` confirm worker env `HERMES_KANBAN_TASK` + non-quiet `chat -q` spawn path.

Automated spot-check run:
- command: string-contract verifier over 14 expected runtime markers
- result: `checked 14 patterns`, `missing 0`

### Regression test rerun (independent)
Command:
- `scripts/run_tests.sh tests/agent/test_turn_finalizer_post_llm_call_metadata.py tests/agent/test_turn_finalizer_interrupt_alternation.py tests/agent/test_turn_finalizer_final_response_persistence.py tests/agent/test_turn_finalizer_cleanup_guard.py -q`
Result:
- `18 passed, 0 failed`

### Artifact/path and threshold checks
- Plan path requirement verified at `...truth-ledger-option-2.md:541`: T4 output must be `docs/truth-ledger/design/extraction-evaluation.md`.
- Canonical T4 artifact exists at that exact path: `docs/truth-ledger/design/extraction-evaluation.md`.
- Repository content search for legacy name `admission-and-evaluation.md`: `0 matches`.
- `docs/truth-ledger/design/storage-and-recovery.md` confirms:
  - `## Frozen operational thresholds` section present
  - `## Closed Q1 decisions (C3)` section present
  - no remaining `Open implementation decisions` header
  - acceptance coverage includes `TL-STOR-001..TL-STOR-013`

## Compliance findings (spec conformance)

### C1 — Eligibility + identity hook contract
Severity: Critical (previous)
Status: Closed

Evidence:
- Required additive metadata now exists in `post_llm_call` emission path (`agent/turn_finalizer.py:389-401`).
- Runtime contract artifact records GO with fail-closed unknown identity semantics (`docs/truth-ledger/discovery/runtime-contract.md:171-182`).
- Focused regression suite passes (`18 passed`).

Conclusion:
- User-scoped admission can now gate on explicit eligibility fields and explicit identity presence, rather than inferred success/origin.

### C2 — T4 path canonicalization
Severity: Important (previous)
Status: Closed

Evidence:
- Plan requires `docs/truth-ledger/design/extraction-evaluation.md` (`plan:541`).
- Artifact is present at that exact path.
- Legacy filename search returns no live references.

Conclusion:
- Path drift risk resolved; downstream traceability is canonical.

### C3 — T3 operational thresholds freeze
Severity: Important (previous)
Status: Closed

Evidence:
- Storage design includes frozen defaults and acceptance mappings (`storage-and-recovery.md:198-213,307-321,323-325`).
- Prior unresolved-open-decisions section is removed.

Conclusion:
- T8 implementation now has explicit baseline constants and acceptance hooks; ad hoc drift risk is materially reduced.

## Design-quality findings (non-blocking)

### D1 — Canonical-vs-derived state boundary remains strong
Evidence: immutable ledger and disposable index/projection semantics are explicit (`storage-and-recovery.md:58-63`).

### D2 — Admission/evaluation precision-first posture is preserved
Evidence: `NONE` abstention semantics, leakage-zero requirement, and precision/no-fact gates are explicit (`extraction-evaluation.md:74-79,184-197,226-240`).

### D3 — Runtime path variability is documented as contract, not assumption
Evidence: T2 freezes per-runtime behavior and unknown-identity handling (`runtime-contract.md:58-169`).

## Requirement traceability summary

- Repository/plugin convention conformity (T1): PASS
- Identity safety and eligibility evidence (T2 + R1): PASS
- Immutable semantics + idempotency architecture (T3): PASS
- Recovery/concurrency architecture (T3): PASS
- Privacy/security boundaries (T3/T4): PASS
- Evaluation protocol and scoring gates (T4): PASS
- Path/traceability hygiene (T4 + R2): PASS

## Residual implementation watchpoints (non-blocking)

1) Validate frozen thresholds and lock/recovery behavior empirically in Q2 once T8 code exists (already mapped to TL-STOR-001..013).
2) Maintain fail-closed behavior whenever identity fields are null/unknown in production extraction paths.

## Final QA decision

Q1 = PASS.
No remaining critical or important architecture/specification gaps are open from this review scope; M2 may proceed under normal downstream gate controls.