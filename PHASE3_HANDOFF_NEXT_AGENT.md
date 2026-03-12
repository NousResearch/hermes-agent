# Phase 3 Handoff (Clean Path for Next Agent)

Last updated: 2026-03-12

## 1) What is already done

Phase 2 must-do cleanup + Phase 3 core implementation are landed in working tree:

- Session-scoped tool resolution fix (global cache removed)
  - `model_tools.py`
- MCP coroutine cleanup fix (no unawaited coroutine leaks)
  - `tools/mcp_tool.py`
- Self-correction module and live-retry policy
  - `agent/self_correction.py`
  - `run_agent.py`
- Regression tests for above
  - `tests/test_model_tools.py`
  - `tests/tools/test_mcp_tool.py`
  - `tests/agent/test_self_correction.py`
  - `tests/test_run_agent.py`

Feature flags/policy now in `run_agent.py`:
- `HERMES_SELF_CORRECTION_MODE=off|shadow|live` (default `shadow`)
- `HERMES_SELF_CORRECTION_CONFIDENCE` (default `0.72`)
- `HERMES_SELF_CORRECTION_MAX_RETRIES` (default `2`)
- `HERMES_SELF_CORRECTION_SESSION_BUDGET` (default `20`)

## 2) Validation baseline

Recent full-suite result from current branch state:
- 2926 passed, 5 skipped, 23 deselected

Important local constraint:
- On this machine, full suite can hit EMFILE with low fd limits.
- Use: `ulimit -n 1024` before full test runs.

## 3) Current repository state (for awareness)

There are many modified/untracked files in the tree (not branch-clean yet), including:
- Core files touched by this work: `run_agent.py`, `model_tools.py`, `tools/mcp_tool.py`, tests
- Additional in-progress artifacts: `evals/`, `tools/safety_taxonomy.py`, gateway/tool-event test files, temp probe scripts (`tmp_honcho_*`), etc.

Do not assume all diffs are Phase 3-specific. Filter by file list in section 1.

## 4) Clean next objective (recommended)

Complete Phase 3 handoff evidence artifact so Phase 4 can start from measured baseline.

Target deliverable:
- `Retry Outcome Report v1` generated from current event data and/or deterministic eval runs
- Must include:
  1. first-attempt vs post-retry success delta
  2. retry cost/latency overhead
  3. failure-signature recurrence summary
  4. high-risk retry-policy compliance evidence (should be 100% blocked without re-approval)

## 5) Suggested execution sequence for next agent

1. Preflight
   - `source .venv/bin/activate`
   - `ulimit -n 1024`
2. Verify Phase 3 core tests
   - `python -m pytest tests/agent/test_self_correction.py -q`
   - `python -m pytest tests/test_run_agent.py -q`
   - `python -m pytest tests/test_model_tools.py tests/tools/test_mcp_tool.py -q`
3. Implement/report path
   - Add report generator (new module under `evals/` or extend `evals/harness.py`)
   - Consume self_correction events (`shadow_critique`, `live_retry_attempt`, `live_retry_skipped`)
   - Emit JSON artifact in `evals/reports/`
4. Add tests for report correctness and schema stability
5. Run targeted + full suite
6. Prepare commit split:
   - commit A: report generator + tests
   - commit B: docs/handoff updates

## 6) Guardrails to preserve

- Prompt-caching invariants must remain intact.
- No process-global tool-resolution cache reintroduction.
- High-risk/side-effect tools must not auto-retry.
- Keep `execute_code` session-scoped tool availability behavior.
- Do not write tests that touch real `~/.hermes/`.

## 7) Ready-to-paste kickoff prompt for next agent

"Continue from `PHASE3_HANDOFF_NEXT_AGENT.md`. Treat Phase 3 core as implemented. Your goal is to deliver Retry Outcome Report v1 and get to a commit-ready checkpoint for Phase 4. Preserve prompt-cache invariants, keep retries bounded/flagged, and add tests proving report accuracy and high-risk retry compliance. Use `.venv`, set `ulimit -n 1024` before full-suite runs, and finish with explicit pass/fail against Phase 3 exit criteria."

## 8) Companion docs added for clean relay

- `START_HERE_NEXT_SESSION.md` — one-screen command block + kickoff text
- `SESSION_CLOSE_CHECKLIST.md` — minimal closeout routine for reliable handoffs