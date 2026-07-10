# `hermes-parity` — SPEC v2.2 (2026-07-10, Momus pass-2 APPROVE-WITH-CHANGES — RC1-RC5 folded)

One command that turns an upstream parity sync from a full-day grind into ~1 hour of real
decisions. Distilled from the 2026-07-10 sync (PR #255: 56 conflict files, 147 hunks,
~45 CI-surfaced reconciliation bugs across 4 CI rounds).

> **v2 deltas** (adversarial 4-model PRD review, verdicts folded; v2.1 reconciles the body
> sections to these — Momus pass-1 caught four header/body contradictions):
> - **JSON manifest** (`docs/sync/fork-features.json`), not YAML — stdlib has no YAML parser.
> - ★ **Fork-delta cross-reference gate** (the review's best catch): gate stage
>   `manifest+forkdelta` computes files touched only on the fork side since merge-base;
>   any such file that the merge modifies/deletes and that no manifest entry's `paths`
>   covers = **hard fail**. Closes the "feature nobody remembered to register" hole.
> - **Package layout** `scripts/hermes_parity/` (stdlib-only, `python -m hermes_parity`),
>   not a single 2000-line file.
> - **Gate ladder = 6 executing stages**: markers, imports, unbound-name lint (the other
>   3 AST linters deferred until a real escape recurs), manifest+forkdelta, full tests,
>   linux-only listing. Gitleaks/contributor/config-migration/tsc stay CI-owned; `gates`
>   prints them as a pre-push reminder checklist instead of reimplementing them.
> - **No PR-API client**: `finish` pushes and prints a ready `gh pr create` command with
>   a generated body file.
> - **Bisect hardening**: regression-classified tests re-run once (flake detection);
>   `--from-file -` piped input; merge/baseline runs in parallel (bounded, see bisect §);
>   realistic target ~10 min for a 45-failure batch, not 2.
> - **State hygiene**: atomic writes (tmp+rename); each gate result records the worktree
>   tree-SHA — edits invalidate downstream stages; `gates --resume` starts at first
>   invalid; `bisect`/`finish` refuse if fork/main moved since `start` (print recovery).
> - `finish --force` requires `--force-reason`, still runs gates and prints reds.
> - git ≥ 2.38 startup check; machine-readable `gates.jsonl` alongside human output.
>
> **Review verdict OVERRIDDEN on evidence**: reviewer wanted stage-5 tests scoped to
> changed files. Scoped/targeted gates were exactly what 2026-07-10 used — they caught
> only ~half the 45 bugs. Full suite stays the default (background it while fixing);
> `--fast` is the scoped inner loop.

## Problem

A parity sync is 95% mechanical, 5% judgment. Today the mechanical 95% is done by hand:

1. Spin an isolated worktree, freeze the upstream target, stage the merge.
2. Triage conflicts (mechanical-union vs both-sides-semantic) with ad-hoc awk.
3. After resolution, run targeted pytest gates — which catch only ~HALF the bugs.
4. Discover the other half one CI round at a time (10-slice matrix + Linux-only tests
   that skip on macOS): push → wait → red → fix → push. 4 rounds on 2026-07-10.
5. For every red, manually classify: "does it pass on fork/main baseline?" → regression
   (fix code) vs stale test (update to merged contract). Done ~15 times by hand.
6. Manually re-run each CI-gate trap locally (gitleaks pinned, contributor-check,
   config-migration dry-run, TypeScript).

## Goal

Package `scripts/hermes_parity/` (stdlib-only, no new deps), invoked as
`python -m hermes_parity` (root shim `hermes_parity.py` makes this work from repo root;
a thin `scripts/hermes-parity` console wrapper MAY exist as an alias — the hyphen form is
never the module name). Subcommands cover the full sync lifecycle. **Everything runs
locally BEFORE pushing** so CI is a confirmation, not a discovery mechanism.

## Non-goals

- Does NOT resolve conflicts (Codex/human judgment stays the resolver).
- Does NOT deploy (deploy is Ace's separate gated call, `hermes update` per host).
- Does NOT auto-merge the PR (landing is a human/Apollo decision; tool prints the
  `gh pr merge --merge` reminder — NEVER squash).
- Not a general merge tool — purpose-built for THIS fork's upstream sync.

## Command surface

### `hermes-parity status`
Cheap divergence report, runnable any time (also the payload for a monthly cron):
- fetch upstream + fork; print ahead/behind counts, merge-base, days since last sync
- `--fail-behind N` exits 2 when behind-count ≥ N (cron alert hook); exit 0 otherwise
- (cut for v1: `--predict` merge-tree conflict prediction)

### `hermes-parity start [--target SHA]`
- Preflight: refuse if live checkout dirty state would be touched — all work happens in
  `~/.hermes/worktrees/parity-YYYY-MM-DD` off `fork/main`, branch `sync/upstream-YYYY-MM-DD`
- Freeze target: `--target` or upstream HEAD at invocation; record merge-base
- Write rollback SHAs + all state to `<worktree>/.parity-state.json` — the single source
  of truth for every later subcommand and the SOLE home of rollback SHAs (no /tmp mirror;
  Momus STRIDE note — /tmp copy was an information-disclosure smell with no upside)
- Run `git merge --no-commit --no-ff <target>` (diff3 conflictStyle)
- Emit **conflict-bucketing report** (`docs/sync/review/conflict-buckets.md` in worktree):
  - per-file: hunk count, bucket = MECHANICAL (one side empty / pure add-add) vs
    SEMANTIC (both sides changed) vs ARCH-SPLIT (DU/UD: one side deleted, other modified)
  - totals + a scale comparison vs previous syncs (from a small history table in the file)
- Print next step: "resolve conflicts (spec template at docs/sync/RESOLUTION-SPEC-*.md),
  then `hermes-parity gates`"

### `hermes-parity gates [--stage NAME] [--fast] [--resume] [--strict] [TESTS...]`
The whole reason this tool exists. **6 executing stages**, ordered, fail-fast per stage,
one ✅/❌ table at the end:

| # | stage | what | catches |
|---|-------|------|---------|
| 1 | `markers` | grep conflict markers + `git diff --diff-filter=U` | unresolved hunks |
| 2 | `imports` | `py_compile` all changed .py + import-smoke of top-level packages | syntax/import breaks |
| 3 | `traps` | unbound-name AST lint on merge-changed files (see below) | silent merge corruption |
| 4 | `manifest+forkdelta` | fork-feature manifest tests + fork-delta cross-reference (see below) | silent fork-feature drops |
| 5 | `tests` | **FULL suite** via `scripts/run_tests.sh` (hermetic, parallel) — not targeted subsystems | the ~45-bug class |
| 6 | `linuxonly` | list tests skipped on darwin (`skipif.*darwin\|platform`) that CHANGED files touch; print as "CI-only risk — review by hand" | systemd exit-code class |

After the table, `gates` prints the **CI-owned pre-push reminder checklist** (NOT
executed locally — these gates already run in CI and reimplementing them drifts):
gitleaks (pinned in CI workflow) · contributor-check · config-migration dry-run ·
`tsc` on apps/desktop (reminder includes the zsh-`zle` trap: run via bash; exit 194
with 0 errors = tsc never ran).

- `--stage NAME` runs one stage; `--fast` = stages 1-4 only (inner-loop while fixing)
- `--resume` starts at the first stage not green at the current tree-SHA (RC3)
- `--strict` promotes stage-3 lint warnings to a hard fail (RC3)
- Every stage prints its exact repro command so a failure is immediately actionable
- Results appended to `.parity-state.json` (which stages green, when, at what tree-SHA)
  AND to a machine-readable `gates.jsonl` in the worktree root (one JSON object per
  gate run: ts, gate, passed, detail, seconds, metadata) (RC3)

### `hermes-parity bisect <pytest-node-or-file> [...]`
Auto-classifies failing tests against the fork baseline — the by-hand protocol from
2026-07-10, mechanized:
- Maintains a cached baseline worktree at `~/.hermes/worktrees/parity-baseline`
  pinned to the recorded pre-merge `fork/main` SHA (from state file). **Cross-sync
  invalidation (RC4): if the cached worktree's HEAD ≠ the state file's
  `fork_main_at_start`, it is removed and re-created at the recorded SHA** — a stale
  baseline from a previous sync silently mis-classifies (regression↔inherited).
- Runs each given test on BOTH the merge worktree and the baseline, then prints class:
  - **pass-on-baseline / fail-on-merge → MERGE REGRESSION** — fix CODE, preserve fork behavior
  - **fail-on-both → INHERITED/FLAKY** — not merge-caused; check order-pollution
  - **absent-on-baseline → UPSTREAM TEST** — new contract; reconcile test vs fork intent
  - **pass-on-both → ORDER-POLLUTION** — fails only in full-suite context
- Accepts multiple tests (`--from-file -` reads node ids from stdin); merge/baseline runs
  in parallel with **bounded concurrency (default 2 workers, `--jobs N` capped at
  os.cpu_count()//2)** — an unbounded 45-test fan-out thrashes worktrees/CPU
- REGRESSION-classified tests re-run once before reporting (flake detection)
- Prints a classification table

### `hermes-parity finish`
- Re-verify stage table all-green at current tree-SHA (refuse otherwise; `--force`
  requires `--force-reason`, still runs gates and prints reds)
- Commit as ONE 2-parent merge commit (template body: target SHA, base, bucket stats,
  behavior-changes section the operator fills in)
- Push branch; print a ready `gh pr create` command with a generated body file
  (no PR-API client in-tool)
- Print the landing rules: land with `--merge` (NEVER `--squash`); if PR goes BEHIND,
  `git merge fork/main` in (NEVER rebase)

### `hermes-parity clean`
Remove merge + baseline worktrees, temp files, state. Prints rollback SHAs one last time.

## Merge-trap AST lint (stage 3)

**All FOUR linters ship active** (2026-07-10: the three originally deferred were
promoted at closeout on Ace's call — `lint_merge_traps.py`):
- **unbound-name** (`lint_unbound.py`): a `Name` load in a function with no matching
  binding in scope (the stray `known_tool_ids.discard()` after the set→Counter migration)
- **dead-code-after-return**: statements after an unconditional return/raise in the same
  block (the unreachable cron-gate in `check_dangerous_command`); empty-generator
  `return`+`yield` idiom and trailing docstrings excluded
- **getattr-lambda-fallback**: `getattr(obj, "name", <callable fallback>)` where `name`
  is defined NOWHERE in the scanned tree (the removed `should_defer_preflight_to_real_usage`
  silently returning False); repo-wide name collection so cross-file methods don't false-fire
- **duplicate-function-bodies**: two structurally-identical sibling function bodies
  (positions/docstrings normalized, <4-stmt bodies skipped) — the scheduler's two
  dispatch bodies; ~27 pre-existing findings in the current tree are review-list
  warnings, not failures

False positives allowed — output is a REVIEW list, not a hard fail (stage passes with
warnings unless `--strict`). Each finding prints file:line + the incident class.

## Fork-feature manifest + fork-delta gate (stage 4)

Checked-in file **`docs/sync/fork-features.json`** (JSON — stdlib-parseable): the
"must survive any merge" list. Each entry:

```json
{
  "feature": "cron-subagent approval gate reads ContextVar not raw env",
  "tests": ["tests/tools/test_cron_subagent_session.py::TestCronSubagentApprovalGating"],
  "paths": ["hermes_cli/approval.py", "hermes_cli/gateway/session_context.py"],
  "why": "cron children run in bare workers; raw-env check auto-approves dangerous cmds"
}
```

- `tests` — pytest node(s) proving the behavior; stage 4 runs exactly these.
- `paths` — the files/globs this feature lives in; **this is what makes the fork-delta
  gate computable** (Momus B3).
- Seeded with every silent-drop found 2026-07-10; grows each sync's postmortem.

**Fork-delta cross-reference (same stage, RC1-sharpened):** the non-vacuous trigger is
a **fork-side file the merge changed or deleted relative to `fork/main`** — for a file
touched only on the fork side, a clean three-way merge takes the fork version unchanged,
so the merge touching it at all means upstream deleted/renamed it (the DU/UD arch-split
case). Any such file with no manifest entry's `paths` covering it = **hard fail** with
the uncovered file list. Coverage split, stated honestly: this path check closes the
*delete/rename* subset; upstream **modifications** of fork-touched files are caught by
the manifest `tests` nodes, not by this check — so registering good `tests` per feature
remains load-bearing.

**Acknowledge path (RC2):** `hermes-parity ack --reason <why> <path>...` records a
reviewed-and-intentional exception (path + reason + timestamp) in the state file's
`forkdelta_acks`. Acknowledged paths clear stage 4 as first-class "reviewed" — distinct
from `finish --force`, which stays reserved for overriding *red gates*, not for routine
intentional drops. This keeps the hard-fail honest without making `--force` the habit.

**Manifest `tests` quality note (Momus Tests lens):** stage 4's "dropped feature cannot
reach CI" guarantee is only as strong as each registered test's mutation-sensitivity —
a test that passes trivially gives false assurance. When registering a feature, verify
its test actually fails if the feature is removed (RED-proof it once).

## State file schema (`.parity-state.json`)

```json
{
  "created": "2026-07-10T...",
  "target_sha": "...", "merge_base": "...",
  "fork_main_at_start": "...", "branch": "sync/upstream-YYYY-MM-DD",
  "rollback": {"fork_main": "...", "local_main": "..."},
  "buckets": {"files": 56, "hunks": 147, "semantic_hunks": 140, "arch_split_files": 2},
  "forkdelta_acks": [{"path": "...", "reason": "...", "at": "..."}],
  "gates": {"markers": {"ok": true, "at": "...", "tree_sha": "..."},
            "tests": {"ok": false, "failed": ["..."]}}
}
```

Bucket counts are explicit about their axis (Momus Impl lens): `files` and
`arch_split_files` count files; `semantic_hunks` counts hunks — 140 hunks across 56
files is the real 2026-07-10 shape.

Writes are atomic (tmp+rename). Each gate result records the worktree tree-SHA; a later
edit invalidates downstream stages and `gates --resume` restarts at the first invalid one.

## Implementation constraints

- **Name pinned (RC5):** the module is `hermes_parity` (matching the repo's dominant
  `hermes_*` module naming — `hermes_cli`, `hermes_constants`); if the fleet-wide
  Hermes→Hermes rename ever reaches module paths, this package renames in that same
  sweep, not before. The hyphenated `hermes-parity` is prose/console-alias only, never
  the module name.
- Package `scripts/hermes_parity/` (stdlib only, `python -m hermes_parity` via root shim).
- Never touches the live checkout's working tree; all mutations in worktrees.
- Every destructive step prints its rollback command first.
- Testable: core logic (bucketing, classification, linter, fork-delta coverage) as pure
  functions; `tests/scripts/test_hermes_parity.py` unit-tests them against fixture repos
  built in tmpdir (git init + synthetic commits).
- Runtime deps discovered at run time (test-runner path probed) — no hardcoded versions
  to drift. The CI-owned reminder checklist reads the pinned gitleaks version from the CI
  workflow file for display; **if that parse fails it prints an explicit ⚠️ "could not
  determine pinned gitleaks version — check .github/workflows" line, never a silent skip**.
- git ≥ 2.38 checked at startup (worktree + merge-tree behavior).

## Success criteria

- Next sync: zero CI rounds spent discovering **locally-discoverable** reconciliation
  bugs (all found by stages 1-5 before push). **Honest residual: Linux-only test classes
  (systemd exit codes etc.) structurally CANNOT be caught on a macOS operator box — stage
  6 surfaces them as a named CI-only risk list, and they may still cost a CI round.**
- `bisect` classifies a 45-failure batch in ~10 min without a human building baseline
  worktrees (single red: ~2 min).
- A dropped fork feature named in the manifest CANNOT reach CI (stage 4 blocks).
- A fork-side file the merge touches with NO manifest coverage CANNOT reach CI silently
  (fork-delta hard-fails with the file list).
- Total operator time for a comparable (56-file) sync: ~1 hour of judgment calls.
