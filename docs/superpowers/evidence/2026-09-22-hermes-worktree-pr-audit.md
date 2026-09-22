# Hermes worktree and PR handoff audit — 2026-09-22

## Boundary and identity

- Read-only inventory source: `git -C /Users/mikedemott/Hermes-agent worktree list --porcelain`.
- GitHub repository: `mrkillbob/hermes-agent`; intended base for fork PRs: `main`.
- Upstream source: `NousResearch/hermes-agent` through the local `upstream` remote.
- Direct writer preflight: `gh api user --jq .login` returned `mrkillbob`.
- No Hermes bot write was delegated. `mrkillbobbot` credentials were not loaded or used.
- The inventory and Kanban audit ran before any task status write. This audit made no task status changes.

## Registered worktrees

The canonical Git common directory reported 28 registered worktrees. Dirty worktrees were preserved exactly as found. Commit-to-PR coverage was checked with GitHub's associated-pulls endpoint for every unique registered HEAD and with the fork's open/all PR lists.

| Worktree | Branch | HEAD | Initial dirty rows | PR/integration evidence | Action |
|---|---|---:|---:|---|---|
| `/Users/mikedemott/Hermes-agent` | `hermes/project-source-upstream-main-20260919` | `b5251aa44c` | 39 | Commit associated with merged fork PR #116; preserved source-root WIP | Preserve |
| `.codex/worktrees/0346/Hermes-agent` | detached | `c6ad576d0b` | 0 | Merged PR #123 | None |
| `.codex/worktrees/18c7/Hermes-agent` | detached | `25c9f7b2d3` | 1 | Merged PR #106 | Preserve dirty state |
| `.codex/worktrees/3407/Hermes-agent` | detached | `d76f71e36c` | 0 | Merged PR #118 | None |
| `.codex/worktrees/5395/Hermes-agent` | detached | `73672f4928` | 0 | Merged PR #119 | None |
| `.codex/worktrees/7c54/Hermes-agent` | `codex/federated-kanban-runner-pool` | `714df9e2d5` | 0 | Merged PR #124 | None |
| `.codex/worktrees/8b5d/Hermes-agent` | detached | `d76f71e36c` | 0 | Merged PR #118 | None |
| `.codex/worktrees/bbf7/Hermes-agent` | `codex/cross-engineering-memory-index-pr` | `a18a13b10a` | 0 | Superseded fork-base rebuild `65245f8dfd` has the same stable patch-id `1386962187f5` and was merged in PR #122 | None |
| `.codex/worktrees/e89e/Hermes-agent` | `codex/shared-model-provider-credentials` | `6a0700e07d` | 1 | Active seven-task recovery branch; no published head at audit time | Keep in flight; parent handoff owns publication |
| `.codex/worktrees/f589/Hermes-agent` | detached | `25c9f7b2d3` | 3 | Merged PR #106 | Preserve dirty state |
| `.codex/worktrees/Hermes-agent/ci-failures-prs-122-124-61057a` | `fix/pr-124` | `4a7ad7f151` | 0 | Merged PR #124 | None |
| `.codex/worktrees/Hermes-agent/fix-pr-123` | detached | `c1cc77a7b9` | 0 | Merged PR #123 | None |
| `.codex/worktrees/hermes-fork-main-20260922` | detached | `5d95f46b42` | 0 | Merged PR #124 | None |
| `.codex/worktrees/hermes-sync-upstream-main-20260922` | `sync/upstream-main-20260922` | `f14f86dd5c` | 0 | Open PR #125, base `main`, exact head matches; no review decision, merge state `DIRTY` | Existing PR retained |
| `worktrees/codex-pr-108` | detached | `6d8cc861c2` | 0 | Merged PR #108 | None |
| `worktrees/codex-pr-109` | detached | `10c149c38b` | 0 | Merged PR #109 | None |
| `worktrees/codex-pr-110` | detached | `be77a2cac4` | 0 | Merged PR #110 | None |
| `worktrees/codex-pr-111` | detached | `4878125878` | 0 | Merged PR #111 | None |
| `worktrees/codex-pr-114` | detached | `2a6aabfdb0` | 0 | Merged PR #114 | None |
| `worktrees/codex-ruff-encoding` | `codex/ruff-encoding-baseline` | `fc7e4f06f5` | 0 | Merged PR #120 | None |
| `worktrees/hermes-automations-debug-a25a01` | `claude/hermes-sync-pr-ci-merge-f0183e` | `30b6b4d99c` | 4 | No associated fork PR at this exact head; committed state sits beneath unrelated dirty WIP and is not safely publishable as a completed unit | Preserve dirty state |
| `worktrees/hermes-feature-forkbase-20260921` | `codex/cross-engineering-memory-index-forkbase-20260921` | `65245f8dfd` | 0 | Commit associated with merged PR #122 | None |
| `worktrees/hermes-kanban-blocked-cards-079699` | `claude/windows-worker-ssh-repair-f4bb79` | `0c24d5f0d5` | 1 | Merged PR #115 | Preserve dirty state |
| `worktrees/hermes-upstream-main-20260921` | `codex/sync-upstream-main-20260921` | `73672f4928` | 0 | Merged PR #119 | None |
| `worktrees/pr-109-conflicts` | `codex/pr-109-conflicts` | `210e4ed1dd` | 0 | Commit associated with merged PR #109 | None |
| `worktrees/pr-119-ci-workflow` | `codex/pr-119-ci-workflow` | `3f01e82f6e` | 0 | Commit associated with merged PR #119 | None |
| `worktrees/preserved-root-wip-20260919` | `codex/preserved-root-wip-20260919` | `013973d307` | 31 | Local-only preserved WIP; commit absent from fork GitHub | Preserve; no PR |
| `worktrees/security-tab-alerts-2026-09-21` | `codex/security-alerts-upstream-sync-2026-09-21` | `325668a5d9` | 0 | Merged PR #119 | None |

The fork had four open PRs. Only #125 corresponded to a registered Hermes engineering worktree. The other three were pre-existing Lunar City drafts (#8, #25, #69) and were outside this repair's publication scope.

## Active and recent Kanban audit

The current board was `tradingbot-burndown`. The audit opened its SQLite database in immutable read-only mode and selected active cards plus worktree/changed-file completions from the prior 14 days.

- `t_92e44484` was the only active Hermes Agent worktree card. It was already blocked, its recorded workspace path no longer existed, and it had no completion receipt claiming committed changes. No status was changed.
- Recent completed Hermes Agent worktree cards (`t_c4e1cfee`, `t_09f1e2d2`, `t_76bf542d`, `t_86535c7c`, `t_58e01330`, `t_0e162184`, `t_4bdf2434`, `t_47f67124`, `t_64848651`, `t_5a5a445d`, `t_af5ef9e6`) were diagnostic/no-change work. Their recorded ephemeral worktree paths had already been removed, and their receipts contained no commit or PR claim.
- `t_dd65d098` was a completed LunaBot worktree card and therefore outside the Hermes Agent repository audit.
- Scratch cards `t_47827ad7`, `t_eb47d590`, and `t_a263175a` named changed files in an ephemeral scratch workspace. They provided no Git commit, branch, or pushed head, so they were not qualifying completed repository changes and were not published.

## Publication result

No qualifying completed change lacked PR coverage. No new PR was opened, no branch was pushed, and no Kanban receipt or status was mutated. Dirty and uncommitted work remained untouched.

## Commands and evidence boundary

- `git worktree list --porcelain`, per-worktree `git status --porcelain=v1`, `git branch --show-current`, and `git rev-parse HEAD`
- `gh pr list --repo mrkillbob/hermes-agent --state open|merged|closed`
- `gh api repos/mrkillbob/hermes-agent/commits/<sha>/pulls`
- immutable SQLite queries against `/Users/mikedemott/.hermes/kanban/boards/tradingbot-burndown/kanban.db`

This is a point-in-time audit. It proves the classifications above at the recorded heads; it does not claim that historical dirty WIP is complete, reviewed, or ready to publish.
