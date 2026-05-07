# TODO — feat/multi-provider-memory

**Rebased onto:** `upstream/main` (`50ab0a85a`)
**Branch:** `feat/multi-provider-memory` — 27 commits ahead, 0 behind
**PR:** https://github.com/NousResearch/hermes-agent/pull/17119
**Last updated:** 2026-05-05 (rebase #2 complete)

---

## ✅ Done This Session

- [x] Rebase onto `upstream/main` (`50ab0a85a`) — 3 docstring conflicts resolved
- [x] Fix `test_concurrent_interrupt.py` — upstream `_invoke_tool` signature changes
- [x] Fix `_get_current_memory_provider()` — now returns all active providers joined
- [x] Fix `_save_memory_provider()` — appends instead of nuking list
- [x] Memory tests: 148 passed
- [x] Full suite: ~17,396 passed, 34 failed (all pre-existing upstream)
- [x] Force-push to origin

## ⏳ Remaining

- [ ] Verify PR #17119 shows 27 commits after push
- [ ] Address any PR review feedback from upstream

---

## Files Changed (new commits)

| File | Change |
|------|--------|
| `hermes_cli/plugins_cmd.py` | `_get_current_memory_provider()` shows all, `_save_memory_provider()` appends |
| `tests/run_agent/test_concurrent_interrupt.py` | `_tool_guardrails`, `_append_guardrail_observation`, `**kwargs` |

## Rebase Conflicts Resolved

| File | Resolution |
|------|-----------|
| `agent/memory_manager.py` | Kept multi-provider docstring + usage example |
| `agent/memory_provider.py` | Kept multi-provider docstring with MEMORY.md / USER.md mention |
