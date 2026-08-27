# Live-system remediation with subagents — controller-owned runtime state

Use this note when executing a broad system hardening plan with subagents while the current agent process remains alive.

## Lesson

Subagents are good for independent source/test changes, but the controller must own live process/config mutation. A subagent can edit config or rebuild binaries, while the already-running controller/gateway may keep cached clients, cached MCP args, or already-instantiated plugins.

## Pattern

1. Dispatch subagents for independent source changes/tests.
2. Controller snapshots live process/config state before runtime changes.
3. Controller performs live process kills/restarts/config writes directly, with receipts.
4. After config changes, distinguish:
   - on-disk config proof,
   - fresh-process proof,
   - current-process proof.
5. If the user prefers no restart, do not claim the current process adopted new config. Say the boundary explicitly and verify with a fresh process or next reload instead.
6. Avoid using a tool path that depends on stale in-memory config after changing that same tool's config; it can respawn old-arg children. Use a lower-level verified path for final writes if needed.

## Example boundary language

- “On-disk config is fixed.”
- “Fresh process loads the new plugin.”
- “This already-running controller may still have cached old state until MCP reload/new process.”

## Verification

For each changed subsystem, save receipts for:

- pre-change snapshot,
- source tests,
- fresh-process smoke,
- live process status after cleanup,
- final residual-risk boundary.
