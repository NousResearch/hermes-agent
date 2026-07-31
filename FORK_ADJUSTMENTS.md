# Fork Adjustments Registry

Tracks divergence from NousResearch/hermes-agent upstream.
Each entry: what changed, why, upstream PR status, how to verify removal is safe.

## Active adjustments

_None._ Fork main is currently at parity with NousResearch/hermes-agent upstream,
plus the four local-only harness files below.

---

## Previously active — now confirmed upstream-merged

| Entry | Was | Status |
|-------|-----|--------|
| `tools/memory_tool.py` render-time truncation | Active | ✅ Upstream has `_char_limit()` + `memory_char_limit` in `_render_block()` |
| `gateway/platforms/slack.py` loop prevention | Active | ✅ Upstream has `SLACK_FREE_RESPONSE_CHANNELS` + `SLACK_REQUIRE_MENTION` |
| `gateway/status.py` macOS `_get_process_start_time` | Active (PR #16) | ✅ Upstream has full psutil + `/proc` fallback |
| `tools/file_operations.py` portable `chmod =rw` | Active | ✅ Upstream has `chmod "=rw"` (confirmed 2026-07-31 via `git show upstream/main:tools/file_operations.py`) |

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

## Verification command

```bash
# Should return only the 4 local-only files listed above:
git diff upstream/main --name-only
```
