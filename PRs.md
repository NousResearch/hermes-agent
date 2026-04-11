# Hermes Agent — Custom PRs to Upstream

## release/ironin

The personal release branch at `fork/release/ironin`. Contains ALL custom features + latest origin/main.

### How it's updated (NEVER rebuild from scratch)

```bash
./scripts/hermes-rebuild-release.sh --push
```

This MERGES origin/main INTO release/ironin, keeping all feature code.
Never uses `git reset` or rebuilds from scratch.
Verifies 9 key features are present before allowing push.

### Why not rebuild from scratch?

Some features (subagent-panel, stash-panel) were merged directly into release/ironin
without dedicated ironin/* branches. Rebuilding from origin/main + clean branches
would silently lose these features. The merge-into approach preserves everything.

## Open PRs to Upstream

| PR # | ironin/ Branch | PR Head | Description | Commits | Status | Link |
|------|---------------|---------|-------------|---------|--------|------|
| 7880 | `ironin/interactive-session-picker-clean` | `feat/interactive-session-picker-clean` | Interactive session picker for /resume and hermes resume | 1 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/7880 |
| 6781 | `ironin/auto-detect-chrome-cdp` | `feat/auto-detect-chrome-cdp` | Auto-detect local Chrome CDP on port 9222 | 1 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/6781 |
| 6599 | `ironin/add-qwen36-plus-paid` | `feat/add-qwen36-plus-paid` | Add qwen/qwen3.6-plus (paid) to model catalogs | 1 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/6599 |
| 5571 | `ironin/configurable-api-retries` | `fix/configurable-api-retries` | Configurable max_api_retries + max_stream_retries | 3 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/5571 |
| 5318 | `ironin/terminal-title` | `feat/terminal-title-v2` | Terminal tab/window title | 2 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/5318 |
| 5020 | `ironin/browser-auto-profile` | `feat/browser-auto-profile` | /browser connect auto-launches Chrome | 2 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/5020 |
| 4833 | `ironin/per-skill-model-routing` | `feat/per-skill-model-routing` | Per-skill model routing | 1 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/4833 |
| 4827 | `ironin/double-esc-clear` | `feat/double-esc-clear` | Double ESC clears input | 1 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/4827 |
| 4801 | `ironin/root-model-flag` | `fix/root-model-flag` | -m/--model and --provider flags | 2 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/4801 |
| 4788 | `ironin/queue-followup` | `feat/queue-followup` | Alt+Enter queues follow-up (subsumed by dual-queue) | 5 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/4788 |
| 4783 | `ironin/ctrl-d-delete-char` | `fix/ctrl-d-delete-char` | Ctrl+D deletes char under cursor | 1 | MERGEABLE | https://github.com/NousResearch/hermes-agent/pull/4783 |

## Feature Branches (ironin/*)

All custom feature branches use the `ironin/` prefix and MUST be based on `origin/main`.

### Clean branches (rebased, pushed)
- `ironin/add-qwen36-plus-paid` (1 commits)
- `ironin/async-delegation` (2 commits)
- `ironin/auto-detect-chrome-cdp` (1 commits)
- `ironin/browser-auto-profile` (2 commits)
- `ironin/configurable-api-retries` (3 commits)
- `ironin/ctrl-d-delete-char` (1 commits)
- `ironin/double-esc-clear` (1 commits)
- `ironin/history-pager` (0 commits)
- `ironin/interactive-session-picker-clean` (1 commits)
- `ironin/per-skill-model-routing` (1 commits)
- `ironin/prompt-display-fix` (1 commits)
- `ironin/queue-followup` (5 commits)
- `ironin/root-model-flag` (2 commits)
- `ironin/terminal-title` (2 commits)

### Features without dedicated branches (in release/ironin only)
- subagent-panel (Ctrl+X panel)
- stash-panel (Ctrl+S input stash)

These were merged directly into release/ironin and don't have clean ironin/* branches.
They're safe in release/ironin but can't be updated as separate PRs.

## Rules

1. **NEVER** `git reset --hard origin/main` on release/ironin
2. **NEVER** rebuild release/ironin from scratch
3. **ALWAYS** use `./scripts/hermes-rebuild-release.sh` to update
4. Feature branches: create from `origin/main`, name with `ironin/` prefix
5. Before pushing: script verifies 9 key features are present
