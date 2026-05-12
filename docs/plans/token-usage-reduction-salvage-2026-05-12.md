# Token usage reduction salvage — 2026-05-12

## Decision

Use a fresh branch from `origin/main` and transplant only the measurement-only Phase 1 work from stale branch `feat/token-usage-reduction`.

The stale local branch was based on a local `main` that was 293 commits behind `origin/main`; rebasing the working tree would have mixed unrelated dirty files and untracked artifacts into the salvage effort. A clean worktree/branch keeps the result reviewable.

## Preserved

Cherry-picked/transplanted commit `18e5890b5` onto final salvage branch `salvage/token-usage-reduction-measurement` (current `HEAD` is the artifact to review/use):

- `agent/token_accounting.py` — rough request token bucket estimation.
- `hermes_state.py` — `session_token_events` schema and append/read helpers.
- `run_agent.py` — fail-open recording of per-API-call token events when usage data is available.
- Focused tests for token accounting and token event persistence.

## Deferred / not preserved

- `agent/tool_compaction.py` from the stale worktree was **not** wired or preserved in this salvage commit. It is deterministic but too early to enable without measured baseline data and approval.
- RTK/native compaction is **not** installed or registered.
- `hermes insights --tokens` remains a follow-up; the measurement table needs real rows before designing summaries.
- Unrelated dirty files and untracked artifacts in the original worktree were left untouched.

## Final artifact note

The reviewer follow-up found an older worktree/branch mismatch. The final cleaned artifact is the `salvage/token-usage-reduction-measurement` branch in `/home/joji/.hermes/hermes-agent-token-salvage-t_f9a80d98`; use the Kanban completion metadata / Project Update Hub completion entry for the exact final commit hash.

## Recommended next steps

1. Review and merge the measurement-only branch `salvage/token-usage-reduction-measurement` after focused test review.
2. Run a pilot session long enough to populate `session_token_events`.
3. Add an `insights --tokens` view over measured rows.
4. Only after measured evidence, design opt-in tool-result/schema compaction behind config gates.

## Verification

Focused verification run during salvage:

```bash
python -m pytest tests/agent/test_token_accounting.py tests/test_hermes_state_token_events.py tests/run_agent/test_token_accounting_events.py -q -o 'addopts='
python -m compileall agent/token_accounting.py hermes_state.py run_agent.py -q
```
