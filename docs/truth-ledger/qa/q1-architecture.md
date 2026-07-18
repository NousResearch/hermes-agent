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

REQUEST_CHANGES (BLOCK M2)

Q1 acceptance requires implementation-ready architecture with **no critical/important gaps** across eligibility/identity, immutable semantics, recovery, privacy, evaluation, and repository-path conformity. This review found **1 critical** and **2 important** gaps that must be resolved before opening M2.

## Independent reproduction evidence (fresh-worker checks)

### Source/runtime contract probes re-run
- `agent/turn_finalizer.py:365-378` confirms `post_llm_call` fires on `if final_response and not interrupted` and currently passes only:
  `session_id, task_id, turn_id, user_message, assistant_response, conversation_history, model, platform`.
- `agent/turn_context.py:216-218` confirms `effective_task_id = task_id or uuid4()` and `turn_id` includes random suffix (unique, not deterministic).
- `cli.py:12341-12346` confirms CLI `chat()` path passes `task_id=self.session_id`.
- `cli.py:16162-16165` confirms fully quiet single-query path calls `run_conversation(...)` without explicit `task_id`.
- `gateway/run.py:18171-18200` confirms gateway agent receives user/chat/thread fields.
- `gateway/run.py:18777-18790` confirms gateway turn call passes `task_id=session_id`.
- `cron/scheduler.py:3044-3074` + `cron/scheduler.py:3104` confirms cron uses `platform="cron"`, `skip_memory=True`, and `run_conversation(prompt)` without explicit task_id.
- `tools/delegate_tool.py:1318` + `tools/delegate_tool.py:1921-1924` confirms delegated child runs on `platform="subagent"` with explicit `task_id=child_task_id`.
- `hermes_cli/kanban_db.py:7794-7797` + `hermes_cli/kanban_db.py:7711` confirms Kanban workers spawn via `chat -q` and set `HERMES_KANBAN_TASK` in env.

Automated spot-check run:
- command: pattern-presence verifier over 14 expected strings across the files above
- result: `checked 14 patterns`, `missing 0`

## Compliance findings (spec conformance)

### C1 — CRITICAL — Hook contract cannot yet prove eligibility + identity for active user-scoped facts
Severity: Critical
Status: Open
Owner: Core runtime integration (T5/T11 boundary)

Evidence:
- Plan requires Q1 approval of either: (a) existing hook provides success/top-level/speaker evidence, or (b) minimal backward-compatible metadata enrichment (`plan:74-79`).
- Current hook payload does **not** include explicit completion/failed/exit reason/depth/speaker/chat/thread fields (`agent/turn_finalizer.py:365-378`).
- Runtime-contract discovery reached NO-GO under current kwargs-only contract for user identity/preference activation (`docs/truth-ledger/discovery/runtime-contract.md:169-198`).

Impact:
- User-scoped truth admission cannot be safely activated without risk of false attribution or ineligible-turn admission.

Required remediation:
- Add narrow, backward-compatible `post_llm_call` metadata enrichment covering at least:
  - eligibility: `completed`, `failed`, `interrupted`, `turn_exit_reason`
  - origin: `delegate_depth`/`is_subagent` (or equivalent), parent linkage, kanban id when present
  - identity: stable `speaker_id`, `conversation_id`, `thread_id`, `chat_type` when available
- Add regression tests across CLI/gateway/cron/subagent/kanban proving stable semantics.

### C2 — IMPORTANT — Gate-T4 artifact filename mismatch versus approved plan
Severity: Important
Status: Open
Owner: Orchestrator/task decomposition consistency

Evidence:
- Plan specifies T4 output path as `docs/truth-ledger/design/extraction-evaluation.md` (`plan:541`).
- Produced artifact is `docs/truth-ledger/design/extraction-evaluation.md`.

Impact:
- Traceability and downstream task references can drift; automation/reviewers may target the wrong file.

Required remediation:
- Either rename/mirror artifact to planned path or formally update downstream cards to canonicalize the adopted filename.

### C3 — IMPORTANT — T3 leaves implementation-critical thresholds unspecified
Severity: Important
Status: Open
Owner: T3/T8 design handoff

Evidence:
- T3 explicitly leaves unresolved decisions for queue/envelope caps, projection update strategy, and ledger ambiguity probe depth (`docs/truth-ledger/design/storage-and-recovery.md:294-299`).

Impact:
- T8 implementation can diverge and invalidate Q2 comparability/security posture if these are decided ad hoc.

Required remediation:
- Freeze concrete defaults (with rationale) before T8 coding starts; include explicit values and acceptance tests.

## Design-quality findings (non-blocking quality/risk)

### D1 — Good separation of canonical vs derived state
Evidence: T3 clearly enforces immutable ledger + disposable index/current view (`storage-and-recovery.md:58-63`).
Assessment: Strong; aligns with Option-2 architecture and rebuildability.

### D2 — Strong abstention and leakage posture in admission design
Evidence: `NONE` first-class abstention and zero leakage gates are explicit (`extraction-evaluation.md:74-79,184-197,226-240`).
Assessment: Strong; supports precision-first target.

### D3 — Runtime variability documented clearly, but needs closure via hook enrichment
Evidence: T2 distinguishes call-path differences (CLI quiet/non-quiet, cron, subagent, kanban) and freezes what is/is not provable (`runtime-contract.md:86-168`).
Assessment: High-quality discovery; currently blocked only by unresolved C1 contract gap.

## Requirement traceability summary

- Repository/plugin convention conformity (T1): PASS with documented citations.
- Identity safety: FAIL (C1 critical).
- Immutable semantics + idempotency architecture: PASS conceptually; IMPORTANT open implementation constants (C3).
- Recovery/concurrency architecture: PASS conceptually; IMPORTANT open implementation constants (C3).
- Privacy/security boundaries: PASS at design level.
- Evaluation protocol and thresholds: PASS at design level.
- Path/traceability hygiene: IMPORTANT mismatch (C2).

## Enumerated remediation cards (to open before M2)

1) `R1 — Hook contract enrichment for eligibility/origin/identity`
- Severity: Critical
- Suggested assignee: automation-operator (implementation), independent reviewer for sign-off
- Deliverable: design note + minimal code+tests proving backward-compatible kwargs enrichment across runtimes.

2) `R2 — Resolve T4 artifact canonical path drift`
- Severity: Important
- Suggested assignee: default (orchestrator) or automation-operator
- Deliverable: canonical path decision + file rename/mirror or parent-card reference update.

3) `R3 — Freeze T3 unresolved operational thresholds`
- Severity: Important
- Suggested assignee: automation-operator
- Deliverable: amended storage design section with concrete caps/probe strategy + Q2-test hooks.

## Final QA decision

Q1 = REQUEST_CHANGES.
Do not open M2 until R1-R3 are resolved and independently re-verified.
