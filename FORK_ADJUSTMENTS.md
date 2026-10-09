# Fork Adjustments Registry

Tracks divergence from NousResearch/hermes-agent upstream.
Each entry: what changed, why, upstream PR status, how to verify removal is safe.

## Active adjustments

### 1. `tools/memory_tool.py` — render-time truncation in `_render_block()`

| Field | Value |
|-------|-------|
| File | `tools/memory_tool.py` |
| Commit | `f0a00fd94` |
| Category | bug-fix |
| Upstream PR | pending (GH rate limit — file when reset) |
| Removable when | upstream merges and we rebase |

**Root cause:** `_render_block()` injected all entries without truncation.
`memory_char_limit: 2200` guarded writes only. File grew to 422KB / 4,318 entries → 118k tokens per session.

**Retained-base check:** `a071f0c11453e3db2dc3d11a20562c1955b7fd9a` has size configuration and usage counters, but no render-time entry-selection loop. This adjustment is still present in the PR diff.

**Verify safe to remove:** `grep -A5 '_render_block' tools/memory_tool.py | grep 'kept\|reversed'` — if upstream has this, our patch is redundant.

---

### 2. `plugins/platforms/slack/adapter.py` — config-driven bot loop prevention

| Field | Value |
|-------|-------|
| File | `plugins/platforms/slack/adapter.py` |
| Commits | `1edbea4de`, `9543613fc`, `38cadf794`, `current` |
| Category | feature/config |
| Upstream PR | ready to file — generic, no hardcoded IDs |
| Removable when | upstream merges; our IDs stay in plist env vars |

**Root cause:** sibling Hermes instances (prod + staging) sent each other messages in loops.
Fix: `SLACK_LOOP_BLOCK_USERS`, `SLACK_LOOP_BLOCK_BOTS`, `SLACK_LOOP_BLOCK_NAMES` env vars.
Our specific IDs live in `~/Library/LaunchAgents/ai.hermes.prod.plist`.

**Verify safe to remove:** `grep SLACK_LOOP_BLOCK plugins/platforms/slack/adapter.py` — if upstream has it, our patch is redundant but env vars still work.

---

## Superseded upstream

### `gateway/status.py` — macOS `_get_process_start_time` support

Superseded: `origin/main` already includes the psutil `Process(pid).create_time()` fallback for platforms without `/proc`.

## Previously active — now confirmed upstream-merged

| Entry | Was | Status |
|-------|-----|--------|
| `tools/memory_tool.py` size configuration | Base behavior | `_char_limit()` and usage counters are in the retained main base; newest-first render-time truncation remains Active adjustment 1 |
| `gateway/platforms/slack.py` loop prevention | Active | ✅ Upstream has `SLACK_FREE_RESPONSE_CHANNELS` + `SLACK_REQUIRE_MENTION` |
| `gateway/status.py` macOS `_get_process_start_time` | Active (PR #16) | ✅ Upstream has full psutil + `/proc` fallback |
| `tools/file_operations.py` portable `chmod =rw` | Active | ✅ Upstream has `chmod "=rw"` (confirmed 2026-07-31 via `git show upstream/main:tools/file_operations.py`) |

---

## Local-only (never upstream)

| Item | Reason |
|------|--------|
| `.github/workflows/green-gate.yml` | Our 6-gate CI harness — project-specific |
| `.github/workflows/skeptic-cron.yml` | Our auto-merge cron — project-specific |
| `.github/workflows/hermes-pr-tag-listener.yml` | Our shared-org PR tag listener dispatch — project-specific |
| `.coderabbit.yaml` | Our CR config |
| `gateway/outbound_guard.py` | Local cross-channel outbound misroute guard for our Slack/operator routing |
| `plugins/rtk/` RTK plugin | Env-specific (RTK token rewriting) |
| `.gitignore` additions | AO session files — harmless upstream but unnecessary |
| `FORK_ADJUSTMENTS.md` | Fork divergence registry |

## How to use this file

- Before rebasing on upstream: check each Active adjustment against the new upstream diff
- Before filing a PR: copy the entry's commit range into the PR body as "addresses FORK_ADJUSTMENTS entry N"
- After upstream merge: move entry from Active to "Previously active — now confirmed upstream-merged"

