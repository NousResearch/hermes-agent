# lah-start Bootstrap Protocol (LAH Stack repos)

lah-core, lah-brain and lah-stack-tools AGENTS.md mandate: run the canonical
launcher and wait for a `STARTUP_RECEIPT` with `reasoning_allowed=true` BEFORE
reading the repo, searching it, or running any git/tests/patches/commits.

## Canonical invocation

`tools/lah-start.mjs` relative path is STALE. Use the launcher at the
lah-stack-tools repo root:

```
node /home/deploy/lah-stack-repos/lah-stack-tools/lah-start.mjs \
  <agent> <MISSION> <absolute-repo-path> \
  [--cartelogic /home/deploy/lah-stack-repos/cartelogic-v2]
```

- Agent MUST be one of: `chatgpt | claude | codex`. `hermes` →
  `STARTUP_BLOCKED_INVALID_AGENT` (first round-trip wasted, 2026-08-11).
- Repo arg: absolute path is safest. Do NOT pass cartelogic-v2 as the
  positional repo — it goes via `--cartelogic` only.
- Receipt fields to check: `verdict: STARTUP_PASS`, `reasoning_allowed`,
  `canonical_repo_path`, `git_policy_observation` (dirtiness observation mode —
  pre-existing dirty repos warn but do NOT block).

## CodeGraph bootstrap (lah-brain AGENTS.md)

Before reading lah-brain source it prescribes:
1. `node tools/codegraph/freshness-check.js --repo lahb`
2. `node tools/codegraph/refresh-pack.js --repo lahb`
3. `node tools/codegraph/mission-context-pack.js --repo lahb --mission <MISSION>`

These tools DO NOT exist in the lah-brain checkout (verified 2026-08-11:
`tools/` contains only operator-validation/). Use the documented fallback: emit
`CODEGRAPH_UNAVAILABLE_FOR_<MISSION>` and proceed with a bounded manual
inspection list. Never claim codegraph coverage that did not run.

## After STARTUP_PASS

- Use `git -C <canonical_repo_path>` for every git command — never from a
  workspace root.
- lah-brain AGENTS.md absolute doctrine: `EXOCLICK_LIVE_ENABLED=false` is
  permanent; mode ladder READ-ONLY → SHADOW → PLAN → HUMAN_APPROVAL → LIVE;
  protected campaigns 8293490/8304842 never touched; decision routes are
  read-only and never async; never import exoclick-stats.js in decision
  handlers.
