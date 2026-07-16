# SPEC: hermes_parity tooling improvements (faster, safer next sync)

**Author:** Apollo · **Date:** 2026-07-16 · **Status:** v0.4 (post Momus pass-3 BLOCK on C3; C3 + carried lens notes folded)
**Motivation:** The 2026-07-15/16 syncs surfaced concrete tool gaps that cost real time or
nearly shipped a regression. Each item below is a small, testable change to
`scripts/hermes_parity/` (the parity CLI) or a new lint. Ordered by risk-reduction value.

---

## 1. Fix `bisect` mis-classification (HIGHEST — nearly shipped a regression)

**Symptom (07-15):** `hermes_parity bisect` classified 2 telegram queue-preservation reds as
`INHERITED/FLAKY`. A direct baseline run proved they PASS on `fork/main` → they were real MERGE
REGRESSIONS. Trusting the tool would have shipped a broken telegram recovery path.

**Root cause:** the bisect baseline run doesn't reproduce the merge-side run's environment
faithfully (env/cwd/venv drift between the cached baseline worktree and the merge worktree), so
a test that's deterministic in one runs differently in the other → both sides "fail" → labeled
INHERITED.

**Fix:**
- Run BOTH sides (baseline + merge) through the IDENTICAL invocation: same `env -i` allowlist,
  same `-p no:randomly -o addopts=""`, same cwd-relative import root — but **each side under its
  OWN faithful venv (C3):** the baseline runs old code under `fork/main`'s own resolved
  environment (that is fork/main's REAL state — the state the 07-15 direct baseline run proved
  PASS on), the merge side under the merge's resolved environment. When the merge's
  requirements/lock diff is non-empty, ALSO run the baseline under the merge venv: a test that
  passes under fork-venv but fails under merge-venv is classified **MERGE REGRESSION
  (dep-driven)** — a dependency bump that came in WITH the merge is the merge's regression,
  never INHERITED. (Pinning both sides to the merge venv — v0.3's wording — silently
  substitutes "old code + new deps" for fork/main's real state and re-manufactures the exact
  false-INHERITED §1 exists to kill.)
- Re-run each side N=3 and require a STABLE verdict; a side that flips across runs is flagged
  `NONDETERMINISTIC` (a distinct class), never silently folded into INHERITED.
  One-side-stable/other-side-flipping is classified by the STABLE side: stable-pass baseline +
  flipping merge = `MERGE REGRESSION (flaky)`, stable-fail baseline + flipping merge =
  `INHERITED (flaky)` — the flake annotation rides along, the verdict stays actionable.
- **Emit the actual pytest output of both sides** in the report so the operator can eyeball a
  disagreement instead of trusting the label — and persist that report into `gates.jsonl` so it
  survives into the `finish` PR body for audit (not stdout-only). Persisted output is BOUNDED +
  SCRUBBED (STRIDE-I): last-N-lines tail per side, absolute home/venv paths rewritten to
  `$HOME`/`$VENV`, routed through `agent.redact.redact_sensitive_text(force=True)` before write —
  raw tracebacks can echo env/test-printed secrets into a public PR body.
- Self-test corpus covers ALL FOUR verdict cells (R1) — pass/pass → CLEAN, pass/fail →
  MERGE REGRESSION (the 07-15 shape, the load-bearing cell), fail/pass → FIXED-BY-MERGE,
  fail/fail → INHERITED — plus a NONDETERMINISTIC fixture whose alternation is GUARANTEED
  per-run (a persisted on-disk counter flipping parity each invocation, never a probabilistic
  tmpfile race that could land stable across 3 CI runs), plus a **dep-bump fixture (C3):** a
  test that passes on fork/main under its own venv and fails post-merge solely because the
  merge bumps a dependency — MUST classify MERGE REGRESSION (dep-driven). One-directional
  over-correction (everything becomes "regression") is exactly the trust-eroding failure in
  the other direction; the six-fixture corpus prevents both directions.

**Acceptance:** replay the 07-15 telegram tests through the fixed bisect → both classify as
MERGE REGRESSION, not INHERITED.

## 2. `fork-features.json` path linter (prevents the forkdelta blind-spot)

**Symptom (07-15):** the manifest's `paths` pointed at pre-refactor locations
(`hermes_cli/approval.py`, `hermes_cli/gateway/run.py`) that no longer exist → forkdelta
coverage computed 0 covered paths → the gate was vacuous on the path axis.

**Fix:** a gate stage (or a standalone `hermes_parity lint-manifest`) that:
- Fails if any `paths` entry matches ZERO files on disk (glob-aware).
- Fails if any `tests` nodeid doesn't collect (`pytest --collect-only <nodeid>` returns empty).
- **Vacuous-coverage floor (R2):** independent of path rot, a forkdelta run that computes
  **0 covered paths** while the manifest is non-empty and the merge touched ≥1 fork-delta file
  goes RED (absent an explicit ack) — so ANY future cause of vacuousness is caught, not just
  the one that bit on 07-15.
- Runs in CI on every PR touching `fork-features.json` OR the guarded paths, so rot is caught
  at authoring time, not next sync. (Trigger on manifest change AND guarded-path change — a PR
  deleting a guarded path doesn't "touch" the manifest, so both triggers are load-bearing.)

**Acceptance:** intentionally break a path → lint RED; fix it → GREEN.

## 3. `catchup` subcommand (the 07-16 mini-merge, mechanized)

**Symptom (07-16):** the 39-commit post-sync drift was merged by hand — the full
`start→gates→finish` ceremony is overkill for <~50-commit drift, but the by-hand path skips the
manifest/fork-feature guards.

**Fix:** `hermes_parity catchup [--max-commits N]`:
- Refuses (points at `start`) if behind > N (default **50** — matched to the observed safe band;
  the 07-16 catch-up was 39. Raising it is an explicit operator call, and C1's scope risk scales
  with it).
- Stages the merge, runs the fast gates (markers → imports → traps → manifest+forkdelta), PLUS
  **test selection by dependency closure over the merge's full CHANGED-file set (C1), not just
  git-conflicted files.** Merge regressions are routinely NOT textually conflicted (an upstream
  signature change with a fork caller elsewhere breaks with zero markers — the exact class §1
  exists to catch). Selection seed: `git diff --name-only <PRE-MERGE fork/main tip>..HEAD`
  (pinned per the Design lens — the pre-merge fork tip, NOT the merge-base, so the seed is
  exactly "everything this merge brought in"). Mapping: (a) a REAL import graph via
  `importlab`/AST module resolution (resolves `__init__.py` re-exports and package-relative
  imports — a literal `^from|^import` grep misses the M→`__init__`→T chain, C2), plus (b) the
  `tests/<mirror-path>` convention; run the union. The fork-feature manifest tests run
  unconditionally on top.
- **Per-changed-file coverage map (C2 — partial-miss made visible):** the report lists EVERY
  changed file with the tests the closure selected for it. A file with zero selected tests is
  flagged UNTESTED-BY-CATCHUP; a file with suspiciously few is eyeball-able on the same line.
  Partial-miss (file has SOME consumers but the one load-bearing test was missed) is thereby a
  visible per-line fact, not silence.
- `finish`-compatible (2-parent merge, PR body auto-gen).

**Acceptance:** re-run against a synthetic 20-commit drift → completes in <20 min with all
guards enforced (budget measured WITH the AST import-graph resolver in the loop, not the grep
it replaced); refuses a 500-commit drift; a synthetic non-conflicted upstream signature
change with a fork caller (the C1 shape) MUST go RED through the closure-selected tests — and
the fixture's fork caller must NOT be covered by the fork-feature manifest (RC1: the RED must
be attributable solely to the closure, else a dead closure still passes via the manifest).
A second fixture reaches the fork caller ONLY through an `__init__.py` re-export and MUST also
go RED (C2's dismissal condition (c)).

## 4. Mass-ack ergonomics (already worked around 07-15, formalize)

**Symptom:** 200+ reviewed uncovered paths needed acking; 200 CLI calls is absurd, so I looped
`state.record_ack` from python inline.

**Fix (R3 — snapshot, not live set):** `hermes_parity ack --from-file <paths.txt> --reason
"<why>"` (bulk, from an explicit reviewed list), and a two-step snapshot flow replacing the
footgun `--all-uncovered`: `hermes_parity forkdelta --emit-uncovered > reviewed.txt` (operator
reviews/edits THIS file against the decisions doc) → `ack --from-file reviewed.txt`. Acking
re-evaluates NOTHING at execution time — a merge landing between review and ack cannot smuggle
unreviewed paths in (TOCTOU closed). Per-path reasons supported via `path<TAB>reason` lines in
the file for attribution; a bare path inherits the `--reason` flag — and `--reason` is REQUIRED
whenever any bare-path line is present (STRIDE-R: no empty-reason acks). Keep single-path `ack`.
The §2 vacuous-coverage RED has its own DISTINCT ack kind (`ack --vacuous-ok --reason ...`),
not satisfiable by a §4 bulk path-ack sharing the store.

## 5. `gates --stage tests` default-corpus honesty (avoid the 1.8s fake-green)

**Symptom:** `gates --stage tests` with no explicit tests ran only the tool's OWN test file
(1.8s "PASS") — looks like a full-suite pass, isn't.

**Fix:** when no `tests` arg is given, `--stage tests` must EITHER run the real
`scripts/run_tests.sh` full suite OR print a loud `SKIPPED (no corpus; run scripts/run_tests.sh
for the full suite)` — never a green that implies the suite ran. Default to the loud skip;
require `--full` to actually launch the hours-long suite. **SKIP exit semantics (R4): the skip
exits NONZERO (exit 2, distinct from failure's 1)** and writes `{"gate":"tests","status":"SKIP"}`
to gates.jsonl; `finish`'s all-green check treats SKIP as NOT-satisfied (it demands a real PASS
recorded for the tests stage). A machine consumer can never read the skip as green.

## 6. (documented, no code) The enum-adoption sweep pattern

When upstream adds an enum member (this sync: reasoning effort `max`/`ultra`), fork "rejects X"
tests go red across MANY surfaces at once. Add a `hermes_parity note enum-adoption <member>`
helper that greps the enum literal repo-wide and lists every test asserting rejection, so the
operator fixes them in ONE pass instead of one CI wave at a time. Low priority; the skill
already documents the pattern.

---

## Rollout
- Items 1, 2 are the safety-critical pair (one nearly shipped a bug, one blinded a gate) — land
  those first, each with a RED-proof test.
- 3, 4, 5 are ergonomics — land as a batch.
- All changes are to `scripts/hermes_parity/` (a fork-owned tool, not upstream surface) so they
  go straight to `fork/main` via normal PR; no upstream coordination.
- Each item ships with a unit test in `tests/scripts/test_hermes_parity.py`; item 1's test is
  the load-bearing one (it encodes the exact 07-15 failure so the classifier can never regress
  to calling that shape INHERITED again).
