# Governed Memory Context Injection PRD

## 1. Product Overview

**Summary:** Add a Hermes-core governance layer for automatic memory context injection so production recall uses the same safety and relevance rules proven in the clean-holographic production-recall benchmark.

**Problem:** The clean-holographic backend now has a sandbox benchmark showing raw/native recall and governed recall separately. Governed recall passes the production-style fixture, but Hermes core still calls `MemoryProvider.prefetch(...)` through `agent/memory_manager.py` and injects provider text directly after basic fencing/scrubbing. That means production context injection can still depend on backend-specific raw behavior instead of explicit Hermes-level governance.

**Proposed solution:** Implement a provider-agnostic governance layer in Hermes core, centered around `agent/memory_manager.py`, that normalizes prefetch queries and filters/reranks/redacts candidate memory context before it is wrapped by `build_memory_context_block(...)` and added to the model context.

**Primary user:** Zev using Hermes over Telegram/CLI with persistent memory enabled.

**MVP boundary:** Govern automatic pre-turn memory context injection. Do not redesign memory storage, live DB schema, memory write/extraction, or the `fact_store` interactive tool API in this pass.

## 2. Goals and Non-Goals

### Goals

- **G1:** Automatic memory context injection should prefer current, profile-appropriate, non-secret, relevant memories.
  - Success metric: A Hermes-core sandbox test modeled on `clean-holographic/scripts/evaluate_production_recall.py` passes all governed context-injection cases.
- **G2:** Keep governance provider-agnostic and backward-compatible.
  - Success metric: built-in memory and one external provider can still register, prefetch, and sync without API breakage.
- **G3:** Preserve live-memory safety.
  - Success metric: tests run under temporary `HERMES_HOME` or `HERMES_FORBID_LIVE_DB=1`; no test touches `~/.hermes/memory_store.db`.
- **G4:** Make production behavior auditable.
  - Success metric: debug/report data can distinguish raw provider output from governed injected context without exposing secrets.

### Non-goals

- **NG1:** No live default-profile memory repair or migration.
- **NG2:** No changes to clean-holographic storage schema unless a test exposes a strict provider-contract gap.
- **NG3:** No broad provider/model/config changes.
- **NG4:** No claim of end-to-end memory quality without a Hermes-core injection-path test.
- **NG5:** No PHI-specific clinical memory logic in this pass.

## 3. Personas and Permissions

- **Zev / Hermes user:** Wants remembered preferences and environment facts to appear reliably without stale or unsafe context. Can approve production-impacting behavior.
- **Hermes core implementer:** Can modify source and tests in `~/.hermes/hermes-agent`; must keep changes isolated, tested, and update-friendly.
- **Memory provider/plugin:** Supplies raw context candidates or legacy prefetch text. Should not need live DB changes for MVP.

## 4. User Stories

- **US-1:** As Zev, I want Hermes to recall the current relevant fact instead of a stale superseded fact, so that answers do not regress to old preferences.
  - Maps to: FR-1, FR-3, FR-5.
- **US-2:** As Zev, I want topic/profile-specific memories isolated, so that one profile or context does not leak into another.
  - Maps to: FR-2, FR-5.
- **US-3:** As Zev, I want secrets and credential-like text excluded from injected memory context, so that recall cannot surface sensitive data.
  - Maps to: FR-4, FR-6.
- **US-4:** As a Hermes maintainer, I want governance tested at the actual memory-manager injection seam, so that benchmark wins are not confused with production behavior.
  - Maps to: FR-5, FR-7.

## 5. Functional Requirements

- **FR-1: Query normalization.** Hermes core must derive safe lexical query variants from the incoming user message for memory prefetch, including stopword/filler stripping similar to the clean-holographic benchmark/backend fix.
  - Priority: Must.
  - Acceptance: natural-language benchmark queries retrieve the intended current facts in a sandbox injection-path test.

- **FR-2: Scope filtering.** Governed injection must support profile/session scope metadata where available and must not inject candidate memory explicitly marked for a different profile/scope.
  - Priority: Must.
  - Acceptance: cross-profile distractor facts are absent from injected context in tests.

- **FR-3: Stale/supersession suppression.** Governed injection must suppress candidate facts marked stale/superseded/deprecated where metadata or structured text makes that state available.
  - Priority: Must.
  - Acceptance: stale facts are absent from top injected context when current replacements exist.

- **FR-4: Secret redaction/exclusion.** Governed injection must exclude or redact credential-like text before context wrapping.
  - Priority: Must.
  - Acceptance: injected context contains no synthetic API keys/tokens/secrets from fixtures.

- **FR-5: Injection-path test harness.** Add Hermes-core tests that exercise `MemoryManager.prefetch_all(...)` or the nearest real pre-turn seam, not only raw provider calls.
  - Priority: Must.
  - Acceptance: focused tests fail before governance and pass after implementation.

- **FR-6: Provider compatibility.** Existing provider interfaces must keep working. If richer candidate metadata is needed, add it as optional capability detection rather than a required breaking method.
  - Priority: Must.
  - Acceptance: existing memory-manager tests and provider registration tests pass.

- **FR-7: Observability.** Add bounded diagnostics for governed injection decisions without logging sensitive content.
  - Priority: Should.
  - Acceptance: tests or log assertions prove counts/reasons can be emitted without raw secrets.

## 6. User Experience

**Entry points:** normal Telegram/CLI messages that trigger memory prefetch.

**Core flow:**
1. User sends a message.
2. Hermes core normalizes the query and asks providers for candidate memory context.
3. Governance layer filters/reranks/redacts candidates.
4. `build_memory_context_block(...)` wraps only governed context.
5. Model receives current, scoped, non-secret background memory.

**Edge cases:**
- Provider returns only legacy text with no metadata: preserve backward compatibility and apply best-effort redaction/fence scrubbing.
- Provider returns pre-wrapped `<memory-context>`: existing sanitizer strips it.
- All candidates are filtered: inject no memory block rather than injecting unsafe/stale context.
- Governance errors: log non-fatal warning/debug and fall back safely, preferably to no injected external memory rather than unsafe raw text.

## 7. Narrative Scenario

Zev asks in Telegram about a current Hermes configuration preference. The memory backend has both an old stale fact and a current replacement, plus a distractor from another profile and a synthetic secret fixture. Hermes core receives raw candidates, suppresses the stale and cross-profile entries, redacts/excludes the secret-bearing entry, and injects only the current relevant fact into the fenced memory block.

## 8. Success Metrics

- **Hermes-core governed injection strict pass rate:** Baseline TBD; target 100% on the sandbox fixture adapted from clean-holographic governed cases.
- **Secret leakage:** Target 0 leaked synthetic secrets in injected context.
- **Profile isolation:** Target true for all scope-isolation cases.
- **Supersession suppression:** Target true for all stale/current replacement cases.
- **Regression suite:** Focused memory-manager/provider tests pass; `git diff --check` clean.

## 9. Technical Considerations

**Likely files to inspect first:**
- `agent/memory_manager.py`
- `agent/memory_provider.py`
- `run_agent.py` memory pre-turn call sites
- existing `tests/agent/*memory*` tests
- clean-holographic benchmark files for fixture shape and scoring semantics

**Architecture options:**
- MVP-compatible option: add a small `agent/memory_governance.py` module with pure functions/classes and call it from `MemoryManager.prefetch_all(...)`.
- Optional richer option: add a provider capability such as `prefetch_candidates(...)` returning structured candidates while keeping legacy `prefetch(...)` as fallback.

**Data/privacy/security:**
- Use synthetic fixtures only.
- Do not mutate live default memory DB.
- Do not print or log secrets; use synthetic secret sentinels for tests.

**Risks:**
- Legacy provider text may not expose enough metadata for perfect stale/profile filtering.
  - Mitigation: support optional structured candidates, document fallback limitations, and test both paths.
- Over-filtering could hide useful memories.
  - Mitigation: keep filtering rules explicit, tested, and conservative.
- Adding governance inside provider plugins would fragment behavior.
  - Mitigation: keep governance in Hermes core and provider-agnostic.

## 10. Short Implementation Plan

### Phase 0 — Discovery and contract

- Inspect `agent/memory_manager.py`, `agent/memory_provider.py`, `run_agent.py`, and existing memory tests.
- Identify the narrowest injection seam for governance.
- Decide whether MVP can parse legacy text or needs an optional structured-candidate provider capability.
- Document the final contract in code comments or a short developer doc if the contract differs from this PRD.

### Phase 1 — Failing Hermes-core tests

- Add a synthetic provider/test fixture that returns current, stale, cross-profile, distractor, and secret-bearing memory candidates.
- Add tests around `MemoryManager.prefetch_all(...)` or the closest real pre-turn injection path.
- Require strict checks for current recall, stale suppression, profile isolation, and secret exclusion.

### Phase 2 — Governance implementation

- Add a pure governance module or helper functions.
- Implement query normalization, scope checks, stale/supersession checks, redaction/exclusion, and simple relevance/reranking.
- Integrate at the memory-manager seam while preserving provider failure isolation.

### Phase 3 — Compatibility and regression

- Run focused memory tests and adjacent provider/agent tests.
- Run `git diff --check`.
- If clean-holographic plugin tests are relevant, run its production recall harness separately under `HERMES_FORBID_LIVE_DB=1`.

### Phase 4 — Report and compounding

- Report raw-vs-governed behavior precisely.
- Update the `hermes-agent` skill or clean-holographic reference if implementation reveals reusable pitfalls.
- Do not claim production memory is fixed until the actual injection-path test passes.

## 11. QA / Acceptance Plan

### Acceptance tests

- **AT-1:** Given current and stale facts for the same entity, when `MemoryManager.prefetch_all(...)` runs on a natural-language query, then injected context contains the current fact and excludes the stale one.
- **AT-2:** Given a cross-profile distractor, when governed injection runs for the default profile, then the cross-profile fact is absent.
- **AT-3:** Given a secret-bearing candidate, when governed injection runs, then no synthetic secret string appears in returned context.
- **AT-4:** Given a legacy provider that only returns plain text, when governed injection runs, then existing provider compatibility is preserved and fence/secret scrubbing still applies.
- **AT-5:** Given provider prefetch failure, when governed injection runs, then the failure remains non-fatal.

### Regression risks

- Provider API breakage.
- Memory block formatting changes that leak context to the visible UI.
- Excess context bloat from expanded query variants or diagnostics.

## 12. Implementation Gate

Coding should not start until:

- [ ] Zev explicitly says to proceed with implementation after reviewing this PRD/plan.
- [ ] The implementation agent verifies the current working tree and branch.
- [ ] The implementation agent confirms tests use temporary `HERMES_HOME`/sandbox fixtures only.
- [ ] The implementation agent verifies no live DB mutation is part of the plan.

## 13. Agent Handoff Contract

**Next agent should read:**
- This PRD: `docs/prds/2026-05-29-governed-memory-context-injection-prd.md`
- Goal doc: `docs/goals/2026-05-29-governed-memory-context-injection-goal.md`
- `agent/memory_manager.py`
- `agent/memory_provider.py`
- clean-holographic reference: `~/.hermes/skills/devops/clean-holographic-live-repair/references/production-recall-benchmarking.md`
- clean-holographic benchmark: `~/.hermes/plugins/clean-holographic/scripts/evaluate_production_recall.py`

**Next agent should build:**
- Hermes-core governed pre-turn memory context injection at the memory-manager seam.

**Next agent should avoid:**
- Live memory DB writes.
- Provider/model/config changes.
- Claiming end-to-end recall quality without injection-path evidence.

**Next agent should verify:**
- Focused Hermes-core memory tests.
- `git diff --check`.
- Optional clean-holographic sandbox benchmark if the implementation touches provider contracts.

**Next agent should report back:**
- Files changed.
- Raw-vs-governed behavior evidence.
- Test outputs.
- Any provider-contract limitations or follow-up skill/docs updates.

## 14. Assumptions and Open Questions

**Assumptions:**
- MVP can start with a pure Hermes-core governance layer and a synthetic provider fixture.
- Structured provider candidates are preferable but should remain optional unless discovery proves text-only governance is too weak.
- The first implementation should be local/source-tree only and not immediately restarted into the live Telegram gateway.

**Open questions:**
- Should the MVP add a formal optional `prefetch_candidates(...)` provider capability, or parse/score legacy provider text first?
- What exact profile/scope metadata fields are available from built-in memory and clean-holographic today?
- Should governance diagnostics be exposed through a debug command later, or remain logs/tests only for MVP?
