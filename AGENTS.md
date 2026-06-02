# Hermes Agent Operating Surface

This file is the thin first-read surface for agents working in this repository.
For the detailed development guide, read `docs/hermes-agent-development-guide.md`
only when the task touches Hermes internals.

## First-Read Rules

- User-facing output should follow Kevin's current conversation language and any
  higher-priority system/developer instructions.
- Start with read-only checks such as `git status`, `rg`, `ls`, and targeted
  file reads before editing.
- Keep root docs short. Long operational or historical details belong in
  `docs/` and must be listed in `docs/DOC_INDEX.md`.
- Do not commit secrets, runtime state, auth files, logs, caches, or local-only
  compose files.
- Prefer scoped changes and narrow verification over broad refactors.

## Current Project Map

- Detailed development guide: `docs/hermes-agent-development-guide.md`
- Local Kevin Docker/dashboard runbook: `docs/kevin-local-operations.md`
- Document governance: `docs/DOC_POLICY.md`
- Session continuity: `SESSION_HANDOFF.md`
- Project Codex config: `.codex/config.toml`
- Git remote safety check: `scripts/check-git-remote-safety.sh`

## Local Runtime Boundary

Kevin's local Hermes runtime is Docker-first and project-local.

- Local compose file: `compose.hermes.local.yml` (ignored by Git)
- Runtime data: `.hermes-docker/` (ignored by Git)
- NotebookLM browser/auth state: `.notebooklm-home/` and related cache dirs
  (ignored by Git)
- Dashboard watchdog: `scripts/hermes-dashboard-watchdog.sh`
- macOS LaunchAgent template:
  `scripts/com.kevin.hermes.dashboard.watchdog.plist`

Do not run global Hermes installs, OpenClaw migration, shell profile edits, or
`~/.hermes` migrations unless Kevin explicitly asks for that operation.

## Git And Remote Safety

Before pushing or opening a PR, run:

```bash
scripts/check-git-remote-safety.sh
```

Expected remote posture:

- `upstream` fetches from `https://github.com/NousResearch/hermes-agent.git`
- `upstream` push URL is `DISABLED`
- Kevin-owned write remotes may be `origin` or `fork`
- Feature work should usually happen on `codex/*` branches, not directly on
  `main`

## Common Verification

Use the smallest relevant check:

```bash
git status --short --branch --untracked-files=all
git diff --check
scripts/check-git-remote-safety.sh
bash /Users/kevin/codex/harness/scripts/check-codex-harness-health.sh --project /Users/kevin/codex/projects/hermes
```

For Hermes internals, use the repo's test wrapper when possible:

```bash
scripts/run_tests.sh <target-tests> -q
```
