# Memory selection policy hardening

## Goal

Improve Hermes memory quality with a minimal-risk slice: make the memory layer surface deterministic policy warnings for likely-stale or imperative entries, and tighten agent-facing guidance so future agents prefer stable facts, session recovery, and skills over diary-style memory.

## Non-goals

- No graph-memory backend or vector retrieval in this slice.
- No blocking of existing memory writes unless they are already security-blocked.
- No schema/database migrations.
- No changes to provider/plugin contracts.
- No automatic deletion or rewriting of user memory files.

## Current facts

- Built-in memory is file-backed and compact (`tools/memory_tool.py`).
- Selection policy currently lives mostly in prompt/tool descriptions and background-review prompts.
- Background self-improvement review already exists in `run_agent.py` and can write memory/skills.
- Session recall is handled separately via `session_search`; temporary progress should stay there, not in memory.

## Proposed implementation slices

1. **Policy warning primitive**
   - Add a small pure helper near `_scan_memory_content()` that detects entries likely to be bad memory:
     - stale artifacts (`PR #...`, commit SHA, issue numbers, "submitted PR", "fixed bug", "phase done");
     - explicit short-horizon markers (`today`, `tomorrow`, `this week`, dates that look like task logs);
     - imperative entries (`Always ...`, `Never ...`, `Do not ...`, `Must ...`) that should be rewritten as declarative facts.
   - Return warning strings only; do not block writes.

2. **Attach warnings to write responses**
   - On successful `add` and `replace`, include `policy_warnings` if the accepted content triggered the helper.
   - Keep exact duplicate / failed write behavior unchanged.

3. **Agent-facing guidance update**
   - Extend `MEMORY_SCHEMA` with explicit reminders:
     - facts → memory;
     - progress/state → session_search/artifacts;
     - workflows → skills;
     - write declarative entries, not commands.

4. **Tests-first validation**
   - Add tests for policy-warning helper.
   - Add tests proving warnings are non-blocking and appear on add/replace responses.
   - Add schema guidance tests for the routing rule.

## Premortem

**Frame:** 6 months from now this plan failed.

**Most likely failure:** the slice only adds more prompt text, so agents still pollute memory because nothing at the tool boundary gives feedback.

**Most dangerous failure:** the tool starts rejecting legitimate user preferences or operational facts, silently reducing memory usefulness and frustrating agents/users.

**Hidden assumption:** better memory quality can be improved incrementally without a full GBrain/LCM architecture, as long as the first hard boundary is low-risk and observable.

**Plan changes from premortem:**
- Make the first code change non-blocking `policy_warnings`, not hard validation.
- Test warnings at the tool boundary, not only prompt strings.
- Keep graph/retrieval/session-snapshot work as later explicit slices.

**Pre-execution checklist:**
- [x] Inspect current memory tool and review prompts.
- [x] Create an isolated git branch.
- [x] Add failing tests before implementation.
- [x] Implement smallest non-blocking warning path.
- [x] Run targeted tests.

## Later safe follow-ups

- Add a `hermes memory audit` command that scores current entries without editing them. **Implemented in this PR.**
- Add explicit session snapshot manifests for active long-running tasks.
- Add retrieval/ranking hooks for external memory providers, behind opt-in config.
- Add telemetry for background-review actions: saved memory vs warnings vs nothing-to-save.
