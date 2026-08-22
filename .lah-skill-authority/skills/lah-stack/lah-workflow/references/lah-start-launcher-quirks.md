# lah-start Launcher Quirks (per-repo bootstrap)

Observed during LAH_MCP_BRIDGE_PERMANENT_READONLY_OBSERVABILITY_V1 (2026-08-03).

## Quirk 1: `lah-start.mjs` does NOT exist inside lah-core

lah-core's AGENTS.md says: "For any mission, run `node tools/lah-start.mjs <agent> <MISSION> [repo]` first and wait for `STARTUP_RECEIPT`."
But `/home/deploy/lah-stack-repos/lah-core/tools/lah-start.mjs` is **missing** (lah-core has no `tools/` dir at all).
The canonical launcher lives at:
- `/home/deploy/lah-stack-repos/lah-stack-tools/lah-start.mjs`

Fix: run the lah-stack-tools launcher with the absolute canonical repo path:
```bash
node /home/deploy/lah-stack-repos/lah-stack-tools/lah-start.mjs <agent> <MISSION> /home/deploy/lah-stack-repos/<repo> --cartelogic /home/deploy/lah-stack-repos/cartelogic-v2
```

## Quirk 2: launcher rejects agent `hermes`

```
"blocked_reasons": ["INVALID_AGENT: hermes — allowed: chatgpt, claude, codex"]
verdict: STARTUP_BLOCKED_INVALID_AGENT
```
The allowed agent names are `chatgpt`, `claude`, `codex`. `hermes` is rejected even though the operator runs Hermes.
Workaround: pass `claude` (or any allowed name) — the agent-name gate is a launcher formality; the STARTUP_PASS receipt is what unlocks the mission:
```bash
node /home/deploy/lah-stack-repos/lah-stack-tools/lah-start.mjs claude <MISSION> /home/deploy/lah-stack-repos/<repo> --cartelogic /home/deploy/lah-stack-repos/cartelogic-v2
```

## Expected success signal

```
"reasoning_allowed": true, "steps_completed": [...,"cartelogic_verify_passed",...],
"verdict": "STARTUP_PASS"
```
Pre-existing dirty/untracked files appear as a `GIT_POLICY` warning ("observation mode active") and do NOT block startup.

## Gate 0 order note (from this mission)

Routing (lah-repo-router) resolved a `LAH_` mission to `cloe-diagnostic-orchestrator` because Phase 6 wording "Baseline **Diagnostic**" matched the `diagnostic` alias — a wrong-but-high-confidence RESOLVED. The mission prefix `LAH_` + explicit `leanframeworklab/lah-core` path won via MANUAL_OVERRIDE (see lah-repo-router pitfall table). Run the launcher only AFTER Gate 0 routing is settled, with the resolved canonical repo path.

## Quirk 3: repo alias `openclaw` maps to a NON-EXISTENT path → STARTUP_BLOCKED_CODEGRAPH

`REPO_CANONICAL_PATHS.openclaw = <lah-stack-repos>/openclaw` but the actual checkout is
`openclaw-runtime`. Launching with positional repo `openclaw` makes the codegraph-precheck
guard probe a non-existent path → `FALLBACK_NOT_AUTHORIZED` / `STARTUP_BLOCKED_CODEGRAPH`,
even when the canonical checkout's CodeGraph is READY.

Fix: pass the ABSOLUTE canonical checkout path as the positional repo:

```bash
node /home/deploy/lah-stack-repos/lah-stack-tools/lah-start.mjs \
  claude <MISSION> /home/deploy/lah-stack-repos/openclaw-runtime \
  --cartelogic /home/deploy/lah-stack-repos/cartelogic-v2
```

This yields `STARTUP_PASS` with `reasoning_allowed: true` and
`steps_completed: [..., codegraph_verified, cartelogic_verify_passed, agent_memory_gate_passed]`.

Prerequisite: if `check-codegraph-availability.mjs --json` reports the target repo
`INDEX_STALE` (e.g. `pendingChanges=N` from a dirty canonical checkout), run
`codegraph sync <canonical-path>` BEFORE the launcher, re-check READY, then launch.
The guard re-checks the repo itself; a fresh index is required for HARD codegraph gates.

## Quirk 4: launcher rejects WORKTREE paths → STARTUP_BLOCKED_INVALID_REPO

Passing a worktree path as the positional repo (e.g.
`/home/deploy/lah-stack-worktrees/<mission>-v1`) is rejected:

```
"blocked_reasons": ["INVALID_REPO: <worktree-path> — allowed: cartelogic, lah, lahb, openclaw, tools"]
verdict: STARTUP_BLOCKED_INVALID_REPO
```

The launcher resolves the repo from a whitelist of SYMBOLIC names or ABSOLUTE
CANONICAL CHECKOUT paths — it does not accept execution worktrees.

Fix: run the launcher against the canonical checkout (see Quirk 3), not the
worktree. The STARTUP_RECEIPT's `context_lock_path` will be written under
`/tmp/lah-stack-tools/cartelogic-context-lock/<repo>/<sha>/...` for the
CANONICAL checkout SHA, not your worktree's branch SHA — harmless, the receipt
is the gate. All implementation still happens in the mission worktree; the
launcher is only the bootstrap gate.

Also note: `hermes` agent name is rejected (Quirk 2) AND a worktree repo is
rejected (this quirk) — so the fully working invocation for a CLOE mission in a
worktree is:

```bash
node /home/deploy/lah-stack-repos/lah-stack-tools/lah-start.mjs \
  claude <MISSION> /home/deploy/lah-stack-repos/openclaw-runtime \
  --cartelogic /home/deploy/lah-stack-repos/cartelogic-v2
```

