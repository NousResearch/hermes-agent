# Operational Bootstrap Pitfalls (LAH Stack missions)

Learned 2026-08-11 (CLOE_AFFILIATE_OUTCOME_SYNTHETIC_POSTBACK_E2E_CERTIFICATION_V1
and CLOE_BOUNDED_AUTOCUT_POLICY_V1). Read BEFORE starting a gated LAH mission.

## lah-start launcher

- **Canonical launcher lives at lah-stack-tools ROOT**:
  `node /home/deploy/lah-stack-repos/lah-stack-tools/lah-start.mjs <agent> <MISSION> <repo>`
  lah-core's AGENTS.md says `node tools/lah-start.mjs` — that relative path is
  STALE (no such file in lah-core/tools). Always use the absolute lah-stack-tools path.
- **Agent whitelist**: lah-start rejects `hermes` with
  `STARTUP_BLOCKED_INVALID_AGENT` (allowed: `chatgpt`, `claude`, `codex`). Use
  `claude` — the label only goes into the receipt; the work proceeds normally.
- Pass the repo as an absolute path (`/home/deploy/lah-stack-repos/lah-core`).
- Expected success receipt: `STARTUP_PASS` with `reasoning_allowed: true` and a
  `git_policy_observation` noting pre-existing dirt (observation mode — dirt does
  NOT block startup, but you must not increase unattributed dirtiness).

## CodeGraph bootstrap fallback

- lah-brain's AGENTS.md mandates a CodeGraph bootstrap
  (`tools/codegraph/freshness-check.js --repo lahb` etc.) — **the tools/codegraph
  directory does NOT exist in the lah-brain checkout** (MODULE_NOT_FOUND).
- Per the repo contract, when CodeGraph is unavailable emit
  `CODEGRAPH_UNAVAILABLE_FOR_<MISSION>` and proceed with the BOUNDED MANUAL
  INSPECTION LIST (read only the files the mission touches). Do not block; do not
  guess-and-broaden the search.

## Repo AGENTS.md gates

- Each LAH repo (lah-core, lah-brain, lah-stack-tools) has its own AGENTS.md with
  bootstrap + command rules; they are injected as subdirectory context on first
  search. Read them and comply BEFORE touching the repo (lah-core requires
  lah-start first; lah-brain requires the CodeGraph contract; lah-stack-tools
  documents the canonical launcher).
- Git discipline: use `git -C <canonical_repo_path>` (never from workspace root);
  never `git add -A`; stage only mission-scoped paths; leave pre-existing dirt
  (e.g. `?? docs/operator/`, modified `tools/lah-mcp-bridge-adapter/server.js`)
  untouched.

## Resume discipline

When a mission resume / operator instruction scopes the fix ("Fix ONLY ...",
"Do NOT redo diagnosis", "Do NOT change semantics again unless the harness reveals
a real defect"):
- Fix exactly the named blocker — one file/one concern. Do not re-open the
  diagnosis or re-expand scope.
- After the fix, verify `git status` / `git diff` contains ONLY the intended
  changes, then commit separately per repo, create PRs, merge if local gates pass.
- Report CI honestly (if GitHub Actions was not run, say so — never claim CI passed).
- Live production trials (real postbacks, real provider calls) require EXPLICIT
  operator authorization even when marked synthetic — absent authorization, the
  certification stands on simulation + deploy proofs and no live trial is claimed.
