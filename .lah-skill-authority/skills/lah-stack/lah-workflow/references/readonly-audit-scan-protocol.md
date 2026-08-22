# Read-only Audit Scan Protocol (operator requirement — 2026-08-02)

Proven in CLOE_ARCHITECTURE_SELF_AUDIT_V1 LOT 0. The operator BLOCKED the first
scan attempt (a multi-repo inventory scan) because the command was executed
without being shown first. Re-authorization came with this protocol attached.

## The protocol (mandatory for READ_ONLY_AUDIT scans)

1. **Show the exact command BEFORE running it.** Write the scan script first
   (write_file), then in the same turn display the invocation line
   (`python3 /tmp/.../scan.py`) and state what it does. Only then execute.
2. **State the guarantees up front**, in the operator's own vocabulary:
   - `aucun write` (no writes to any repo; the only artifact written is a
     scratch file under /tmp, outside all repos)
   - `aucun index refresh` (no codegraph sync, no npm install, no test run)
   - `aucun commit`
   - `aucune modification de configuration`
3. **Read-only implementation details:**
   - Open files with `open(fp, "r", errors="ignore")` only.
   - Git metadata via `git -C <repo> rev-parse HEAD` / `git status --porcelain`
     (read-only subcommands).
   - Exclude `node_modules`, `.git`, `.codegraph`, `dist`, `build`, `vendor`,
     `data`, caches, `__pycache__`, `.venv`/`venv` from the walk.
   - Cap file size read (e.g. skip files > 2 MB).
   - One output JSON per scan stage under a scratch dir (e.g.
     `/tmp/<mission>/`), never inside a repo.
4. **Deliverable shape** (LOT 0 baseline, roadmap CLOE_ARCHITECTURE_SELF_AUDIT_V1):
   - `scan-raw.json` — per-source keyword-category hits with file paths
   - `CLOE_SELF_AUDIT_EXISTING_CAPABILITY_MATRIX_V1.json` — authority
     resolution (22 entries), duplicates, reusable bricks, deprecate
     candidates, components
   - `LOT0_CONTINUITY.json` — gate verdict + operator decisions required
   - Distinguish evidence types in the matrix: `implementation`
     (src/tools/routes — real authority), `declaration` (docs/plans — needs
     LOT 1/2 proof), `pattern` (skills — canonical method), `gap`
     (function absent, 0 hits). A keyword hit in a doc is NOT proof of an
     engine.

## Missing-repo handling (lah-stack-skills precedent)

When the roadmap names a repo that does not exist as a git checkout:
- Do NOT block the lot.
- Audit the real runtime location instead (`~/.hermes/skills/` for skills).
- Record BOTH findings explicitly:
  - `GITHUB_REPOSITORY_ABSENT` — no git repo (no HEAD/commit/PR → no
    evidence-first traceability)
  - `RUNTIME_SKILLS_DIRECTORY_PRESENT` — runtime dir exists, files scanned
- Classify it as an authority/traceability finding to resolve, not a blocker.

## Gate-0 prerequisite check list (before ANY scan)

- `dry-run-route.sh` on the mission prefix → RESOLVED + authority repo
- All roadmap repos present (list which are missing)
- Adapter/CLI prerequisites if Codex will be involved (see
  codex-bounded-executor skill)
- `free -h` ≥ 1 GiB available (8 GB VPS resource governance)

## Pitfalls proven in the field (2026-08-14, postback audit)

- **Redact before reading secret-bearing files.** Files that may embed REAL postback secrets
  (openclaw `data/execution-receipts.json`, `data/cloe-governed-action-packets.json`,
  campaign-ledger.json, any URL with `secret=`) must be read ONLY through a redaction filter
  (`re.sub(r'(secret["\'=:\s]*)[A-Za-z0-9_\-\.]{6,}', r'\1<REDACTED_POSTBACK_SECRET>', s)`),
  never grepped/dumped raw. Operator BLOCKED a raw receipts grep mid-audit.
- **Never trust a worktree by its name.** `lah-brain-wt-deployed` was STALE vs `origin/main`
  (still had `conversion→sale` + payout coercion + no `event_semantics`). Resolve deployment
  authority with `git merge-base --is-ancestor <fix_sha> origin/main` + `git rev-parse
  origin/main` before quoting any semantics from a checkout.
- **After an operator denial, STOP and present status** — do not keep probing with
  differently-phrased commands; the denial outranks mission autonomy (protocol: stop, safe-reset).
- CrakRevenue postback contract + round-trip + macro status: see lah-workflow
  `references/crakrevenue-global-postback-contract.md`.
