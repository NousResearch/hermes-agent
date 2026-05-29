# 2026-05-29 Goal: Hermes-Core Governed Memory Context Injection

## Goal Statement

Implement and verify Hermes-core governed memory context injection so automatic pre-turn memory recall applies explicit query normalization, scope filtering, stale/supersession suppression, secret redaction/exclusion, and relevance/reranking before memory context is injected into the model.

This goal turns the clean-holographic production-recall benchmark result into production-path Hermes behavior. The implementation target is Hermes core, especially `agent/memory_manager.py` and adjacent provider contracts/tests, not the clean-holographic storage backend alone.

## Source PRD / Plan

Read first:

- `docs/prds/2026-05-29-governed-memory-context-injection-prd.md`
- `agent/memory_manager.py`
- `agent/memory_provider.py`
- `run_agent.py` memory pre-turn call sites
- `~/.hermes/skills/devops/clean-holographic-live-repair/references/production-recall-benchmarking.md`
- `~/.hermes/plugins/clean-holographic/scripts/evaluate_production_recall.py`
- `~/.hermes/plugins/clean-holographic/tests/fixtures/production_recall_fixture.json`

## Implementation Boundary

### In scope

- Add Hermes-core tests that exercise the actual automatic memory context injection seam, not only raw backend search.
- Add query normalization for natural-language memory prefetch.
- Add governed filtering/reranking/redaction before `build_memory_context_block(...)` receives context.
- Preserve provider compatibility and failure isolation.
- Add bounded observability for governance decisions without logging sensitive content.
- Update docs/skills if implementation discovers reusable pitfalls.

### Out of scope

- Live default-profile memory DB writes, repairs, migrations, or synthetic fixture insertion.
- Broad model/provider/fallback config changes.
- Clean-holographic storage schema changes unless a strict provider-contract gap blocks the Hermes-core goal.
- Gateway restart or production rollout without explicit Zev approval.
- PHI-specific clinical memory behavior.

## Safety Guardrails

- Use synthetic fixtures and temporary `HERMES_HOME` only.
- If running clean-holographic harnesses, set `HERMES_FORBID_LIVE_DB=1` and use `~/.hermes/hermes-agent/venv/bin/python` on Calliope.
- Do not touch `~/.hermes/memory_store.db`, `memory_store.db-wal`, or `memory_store.db-shm` except for read-only fingerprinting if needed.
- Do not expose API keys, tokens, or secrets. Synthetic secrets may be used only as leak sentinels.
- Keep implementation update-friendly: prefer small core modules/helpers and tests over broad rewrites.
- Do not restart the live gateway from inside a Telegram turn.

## Required Deliverables

1. **Discovery note in final report:** precise injection seam and provider contract decision.
2. **Failing-then-passing Hermes-core tests:** cover current-vs-stale, profile/scope isolation, secret exclusion, query normalization, provider compatibility, and provider failure isolation.
3. **Governance implementation:** pure/testable helper or module integrated at the memory-manager prefetch seam.
4. **Verification evidence:** focused test commands and outputs, plus `git diff --check`.
5. **Documentation/skill updates:** patch relevant skill/reference docs if new pitfalls or commands are discovered.
6. **No-live-DB proof:** state how tests were isolated from the live memory DB.

## Quantitative Gates

- Governed injection strict pass rate: **100%** on the Hermes-core sandbox fixture.
- Secret leak count: **0** synthetic secret strings in injected context.
- Profile/scope isolation: **true** for all fixture cases.
- Supersession/stale suppression: **true** for all fixture cases with current replacements.
- Existing provider compatibility: focused existing memory-manager/provider tests pass.
- Workspace hygiene: `git diff --check` passes for changed files.

## Suggested Task Breakdown

1. Verify branch, working tree, and current memory-manager/provider code.
2. Inspect existing tests and choose the smallest injection-path test seam.
3. Write failing tests with synthetic provider/candidates.
4. Implement pure governance helpers: tokenize/query variants, stale/scope checks, secret redaction/exclusion, relevance scoring.
5. Integrate helpers into `MemoryManager.prefetch_all(...)` or a narrower adjacent method without breaking legacy providers.
6. Run focused tests; fix regressions.
7. Run `git diff --check`; optionally run clean-holographic sandbox benchmark if provider contracts changed.
8. Update docs/skills if needed.
9. Report evidence and remaining limitations.

## Acceptance Definition

The goal is complete when Hermes core has a tested governed memory context injection path with 100% strict pass on the sandbox governed fixture, 0 synthetic secret leaks, verified profile/scope isolation and stale suppression, no live DB mutation, and a concise final report with exact files changed and test evidence.

## Pause / Approval Condition

This goal is documented and ready, but implementation should remain paused until Zev explicitly says to proceed with coding. Once Zev approves, resume this goal and execute the task breakdown above.
