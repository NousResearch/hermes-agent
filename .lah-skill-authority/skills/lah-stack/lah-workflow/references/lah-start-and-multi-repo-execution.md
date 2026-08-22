# LAH launcher + multi-repo execution lessons (2026-08-11, CLOE_LAHB_DUAL_FUNNEL_OBSERVABILITY_REPAIR_AND_CERTIFICATION_V1)

## lah-start.mjs gotchas (host VPS execution)

- Lives at `lah-openclaw-mvp/lah-tools-runtime/tools/lah-start.mjs` — NOT `tools/` at repo root (a root-level `node tools/lah-start.mjs` fails MODULE_NOT_FOUND).
- Validates a canonical CarteLogic repo at `../CarteLogic_v2` relative to the script — that path does not exist on this VPS. Pass `--cartelogic /home/deploy/lah-stack-repos/cartelogic-v2` (that repo has `v2/operational` and passes `isCarteLogic`).
- Allowed agents: `chatgpt|claude|codex`. `hermes` → `STARTUP_BLOCKED_INVALID_AGENT`. Use `chatgpt` as the agent arg.
- `runPython` spawns `python`; this host only has `python3`. Shim: `mkdir -p /tmp/lah-shim-bin && ln -sf /usr/bin/python3 /tmp/lah-shim-bin/python && PATH=/tmp/lah-shim-bin:$PATH node .../lah-start.mjs ...`
- `STARTUP_BLOCKED_MEMORY_GATE` may be a TOOLING gap, not a mission failure: shipped `lah-guard.mjs` in openclaw-runtime-wt-main has NO `agent-memory-gate` command — running it directly prints `[lah-guard] FAIL: unknown command: agent-memory-gate` (version skew between lah-start and lah-guard). Codegraph precheck (`codegraph-precheck <cartelogic>`) still passes. Verify the guard command directly; if unknown-command, record the launcher skew and proceed with the direct CodeGraph bootstrap sequence (`tools/codegraph/freshness-check.js --repo <repo>` → `refresh-pack.js` → `mission-context-pack.js`) — the context pack itself is already written by the bootstrap step, so the mission context is fresh.
- Receipt's `canonical_repo_path` is a virtual `lah-tools-runtime/<repo>` path. The REAL canonical worktrees are `/home/deploy/lah-stack-repos/<repo>-wt-*` (`openclaw-runtime-wt-main`, `openclaw-runtime-wt-deploy`, `lah-brain-wt-deployed`, plus `/home/deploy/lah-stack-repos/lah-core`). lah-brain and lah-core have no codegraph tooling of their own; run the openclaw worktree's tools and point `--cartelogic` explicitly.
- `lah-start` is read-only (no checkout/pull/commit) — safe to run; it only bootstraps CarteLogic context + codegraph precheck.

## Multi-repo missions: commit locally EARLY

Cross-repo missions with many LOTs routinely hit the session iteration budget mid-implementation. This mission stopped with all three repos' work UNCOMMITTED — tests green, diffs unreviewable, no PRs possible, and resume requires re-deriving state from scratch.

Durable rule: **as soon as a repo's focused tests are green, `git commit` it locally.** Branches already created per repo (`fix/<mission-scope>-v1`); stage files SELECTIVELY — never `git add -A`, because the canonical worktrees carry pre-existing dirty artifacts (test report JSONs, untracked `node_modules` not gitignored in lah-brain, lah-core MCP-bridge local mods). Local commits:
- preserve progress across iteration limits / session cuts,
- make post-resume review a diff instead of a re-derivation,
- keep unrelated worktree dirt out of the eventual PR.

Push/PR/merge only per the mission's deployment gate — but never leave green code uncommitted.

## Fixture parsing + replay certification pattern

- Diagnostic probe outputs concatenate a text header + JSON (and may append multiple JSON sections). Parse with brace-matching slice (`first '{'` → matching `}`), not whole-file `JSON.parse`.
- Replay captured provider evidence through the REAL production module (normalizer/fetch) to certify: `provider_zone_count` vs `normalized_zone_count`, `pages_fetched`, `sum_zone_spend` vs `campaign_total_spend`, `reconciliation_delta` — label the source `USER_AUTHENTICATED_PROVIDER_EVIDENCE` (operator read-only probe), never `LIVE_API`.
- T01 fixture for campaign 8529830: `/home/deploy/cloe-diagnostics/ourdream-zone-economics-v1/zone-complete.json` (541 zones / 11 pages; the 50-row `zone-raw.json` is page 1 only).
