# Fork Adjustments Registry

Tracks divergence from NousResearch/hermes-agent upstream.
Each entry: what changed, why, upstream PR status, how to verify removal is safe.

## Active adjustments

### 1. `tools/file_operations.py` + `tests/tools/test_file_operations.py` — portable `chmod =rw` for new files

| Field | Value |
|-------|-------|
| Files | `tools/file_operations.py`, `tests/tools/test_file_operations.py` |
| Commit | `98105f31f` |
| Category | bug-fix / portability |
| Upstream PR | jleechanorg/hermes-agent#7 (needs to be filed to NousResearch/hermes-agent) |
| Removable when | upstream merges the NousResearch PR |

**Root cause:** The `$((0666 & ~0$u))` shell arithmetic for new-file permissions breaks on zsh
(leading-zero constants parsed as decimal, not octal → silently wrong mode). Replaced with
POSIX symbolic `chmod "=rw"` which is identical across bash/dash/ash/busybox/zsh.

**Verify safe to remove:** `grep '=rw' tools/file_operations.py` — if upstream has this, patch is redundant.

---

## Previously active — now confirmed upstream-merged

| Entry | Was | Status |
|-------|-----|--------|
| `tools/memory_tool.py` render-time truncation | Active | ✅ Upstream has `_char_limit()` + `memory_char_limit` in `_render_block()` |
| `gateway/platforms/slack.py` loop prevention | Active | ✅ Upstream has `SLACK_FREE_RESPONSE_CHANNELS` + `SLACK_REQUIRE_MENTION` |
| `gateway/status.py` macOS `_get_process_start_time` | Active (PR #16) | ✅ Upstream has full psutil + `/proc` fallback |

---

## Local-only (never upstream)

| Item | Reason |
|------|--------|
| `.github/workflows/hermes-pr-tag-listener.yml` | Fork's PR tagging workflow |
| `.coderabbit.yaml` | Fork's CR config |
| `.gitignore` additions | AO session files — harmless upstream but unnecessary |
| `FORK_ADJUSTMENTS.md` | This file |

## How to use this file

- Before rebasing on upstream: check each Active adjustment against the new upstream diff
- Before filing a PR: copy the entry's commit range into the PR body as "addresses FORK_ADJUSTMENTS entry N"
- After upstream merge: move entry from Active to "Previously active — now confirmed upstream-merged"
