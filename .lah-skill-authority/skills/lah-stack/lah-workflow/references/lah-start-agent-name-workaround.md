# lah-start Launcher — Agent-Name Workaround

## Symptom

Repo AGENTS.md files say: "run `node tools/lah-start.mjs <agent> <MISSION>` first and
wait for `STARTUP_RECEIPT`". Running with `hermes` fails:

```
INVALID_AGENT: hermes — allowed: chatgpt, claude, codex
verdict: STARTUP_BLOCKED_INVALID_AGENT
```

Second trap: lah-core's AGENTS.md references a relative `tools/lah-start.mjs`
that does NOT exist in that repo (`MODULE_NOT_FOUND`). The launcher lives in
lah-stack-tools.

## Fix

Run the launcher from lah-stack-tools with an accepted agent name:

```bash
node /home/deploy/lah-stack-repos/lah-stack-tools/lah-start.mjs \
  claude <MISSION_ID> /home/deploy/lah-stack-repos/<repo> \
  --cartelogic /home/deploy/lah-stack-repos/cartelogic-v2
```

Use `claude` (or `chatgpt`/`codex`) as the placeholder — the receipt fields that
matter are `reasoning_allowed`, `verdict: STARTUP_PASS`, `steps_completed`,
`git_policy_observed`, `context_lock_path`. Treat STARTUP_PASS as the gate
regardless of the agent label.

## Notes

- `git_policy_observed` warnings about pre-existing dirty/untracked files do NOT
  block startup (observation mode).
- The context lock path is written under
  `/tmp/lah-stack-tools/cartelogic-context-lock/<repo>/<sha>/<agent>/<mission>/...`
  — the agent placeholder appears in the path; harmless.
