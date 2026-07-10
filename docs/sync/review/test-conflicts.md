# Test conflict notes - 2026-07-10

Test conflicts were resolved toward the merged fleet contract when the hunk encoded fork-specific production behavior.

- Kept fork-side assertions for fallback/route announcements, LCM/calibrated compaction, restart-drain recovery, message-sequence duplicate-result handling, retry-after policy, and TUI/gateway protocol behavior.
- Kept fork deletion of `tests/run_agent/test_run_agent.py`; upstream modified a test file the fork had deleted. This avoids resurrecting stale monolithic contracts that no longer match the fork's split runtime tests.
- Upstream-only tests outside conflicting hunks remain in the tree.

Review flag: because several large test files were side-resolved to the fork side, the orchestrator should rely on the normal pytest gates to identify any upstream bugfix tests that need to be reintroduced against the merged implementation.
