# Rebase Journal

This document records every rebase of the fork against upstream, including conflicts encountered and how they were resolved. It serves as:
1. Proof the thin-fork model works (or documentation of when it doesn't)
2. A guide for future rebase sessions
3. Input for improving the CI/CD rebase system

---

## Rebase 1: 2026-05-26 — Initial Rebase to Current Upstream

**Fork state:** 33 commits (reduced to 27 after cleanup), diverged since `4cc18877c69b`
**Upstream state:** 410 new commits on `upstream/main`
**Operator:** Claude (with Juniper)

### Pre-rebase cleanup

Before rebasing against upstream, we cleaned our own history:
- **Dropped 3 commits:** Intermediate `docs: regenerate patches` commits (we regenerate at the end)
- **Dropped 3 commits:** Debug logging cycle (add debug → add debug → remove debug = no-op)
- **Result:** 33 → 27 commits, no conflicts

### Rebase execution

```
git rebase upstream/main
```

**Commits 1-2/27: Memory scoping (context_id) — 2 CONFLICTS**

File: `tools/memory_tool.py`

Upstream added:
- Promptware/injection defense — `_sanitize_entries_for_snapshot()` scans entries at load time
- External drift detection — `_detect_external_drift()` with backup on mutation
- `_reload_target()` now returns `Optional[str]` (backup path) instead of `None`

Our patch adds:
- `context_id` parameter for per-channel memory isolation
- `_path_for()` method routing writes to `contexts/{context_id}/` subdirectory
- Merge-on-read semantic (global + scoped entries deduplicated)
- `_global_entry_count` tracking for mutation guards

**Conflict 1 (docstring):** Both added documentation to `load_from_disk()`. Resolution: concatenate both — they describe orthogonal features.

**Conflict 2 (`_reload_target`):** Upstream refactored the signature and added drift detection. Our patch added scoping logic. Resolution: keep upstream's drift detection (`_detect_external_drift`, backup path return), integrate our scoping logic (global + scoped merge, `_global_entry_count` update) inside the same method.

**Key insight:** The security features and scoping features are orthogonal — they modify different aspects of the same methods. The drift detection operates on `self._path_for(target)` which our code correctly routes, so the features compose well.

**Commits 3-13/27: ALL CLEAN** (no conflicts)

These include:
- swarm_map_policy plugin (new files — no upstream equivalent)
- CI/CD workflows (new files)
- Patches documentation (new files)
- boot.md plugin (new files)
- lifecycle-notify hook (new files)
- Mattermost mention gating + system posts (upstream hadn't changed these specific methods)
- Signal UUID allowlisting, group invite policy, profile name (upstream had 0 changes to signal.py)

**Commit 14/27: Mattermost channel join/leave gating — 1 CONFLICT**

File: `plugins/platforms/mattermost/adapter.py` (was `gateway/platforms/mattermost.py`)

Upstream moved the Mattermost adapter from `gateway/platforms/` to `plugins/platforms/mattermost/`. Our patch added instance methods (`_handle_channel_join`, `_leave_channel`) to the adapter class.

**Conflict:** Our methods needed to go inside the class body, but the conflict markers mixed class methods with a module-level function (`_standalone_send`) added by upstream.

Resolution: Place our `_leave_channel` method inside the class (after `_handle_channel_join`), then the upstream `_standalone_send` function remains as a module-level function after the class.

**Commits 15-27/27: ALL CLEAN** (no conflicts)

These include:
- observe_only field + handler (new field on MessageEvent)
- Signal observe_only + voice memo + SSE reconnect (signal.py untouched by upstream)
- Mattermost observe_only (upstream's file move was already resolved)
- Personal deployment configs (new files)
- faster-whisper Docker install (Dockerfile addition)
- boot.md plugin refactor
- Signal syncMessage group detection

### Summary

| Metric | Value |
|--------|-------|
| Fork commits | 27 |
| Upstream commits to absorb | 410 |
| Conflict points | 3 (in 2 files) |
| Resolution time | ~20 minutes |
| Tests after rebase | 96 pass (memory, signal, mattermost) |
| Files with semantic conflicts | 1 (`tools/memory_tool.py`) |
| Files with structural conflicts | 1 (`plugins/platforms/mattermost/adapter.py`) |
| Files that auto-merged | `agent/agent_init.py`, `gateway/run.py`, `run_agent.py`, `Dockerfile` |

### Lessons

1. **Signal adapter patches are low-risk.** Upstream had 0 changes to `gateway/platforms/signal.py` across 410 commits. Our signal patches will likely continue to rebase cleanly.

2. **Memory scoping is the highest-risk patch.** It touches `tools/memory_tool.py` which upstream actively develops (security features). However, the features compose orthogonally — scoping affects path routing, security affects content scanning. As long as both use `_path_for()` as the routing function, they won't conflict semantically.

3. **Upstream file moves create one-time pain.** The Mattermost adapter moved from `gateway/platforms/` to `plugins/platforms/mattermost/`. This required manual resolution but only happens once — future rebases will use the new path.

4. **`gateway/run.py` auto-merged despite 27 upstream changes.** Our `context_id` wiring touches a very specific, isolated part of the file (the `_context_id_for_source` static method and two `AIAgent()` call sites). The upstream changes were in completely different functions.

5. **The thin-fork model works.** 3 conflicts across 410 commits is manageable. The CI/CD system correctly detected the conflict and would have opened an issue. Weekly rebases will be far smaller (10-30 upstream commits) and likely conflict-free.

6. **Clean history matters.** Dropping the 6 debug/regen commits before rebasing reduced the commit count from 33 to 27, which meant fewer potential conflict points.

### Process for future rebases

1. `git fetch upstream`
2. Check divergence: `git log --oneline HEAD..upstream/main | wc -l`
3. Review upstream changes to our files: `git log --oneline HEAD..upstream/main -- tools/memory_tool.py`
4. `git rebase upstream/main`
5. Resolve conflicts (this journal tells you what to expect)
6. Run tests: `python3 -m pytest tests/tools/test_memory_scoping.py tests/gateway/test_signal*.py tests/gateway/test_mattermost.py -v`
7. Regenerate patches: `rm -f patches/0*.patch && git format-patch upstream/main..HEAD -o patches/`
8. Update this journal with new entry
