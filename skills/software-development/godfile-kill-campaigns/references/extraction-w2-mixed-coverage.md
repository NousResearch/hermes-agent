# Extraction w2 — Mixed-Coverage Adjudication (cli.py s2, witness w2b)

Recipe for the wave-2 witness case where the two w1 patches overlap on ONE cluster and each
covers a disjoint extra cluster. Verified end-to-end on cli.py shard s2 (18,485 lines):
w1a = c18(19) + c7(9); w1b = c18(19) + c22(11); overlap = c18 only.

## 0. Read order + scope
1. `shard-plan.json` → `shards.<sN>` cluster list + slice_order (extract the covered clusters'
   method names+lines).
2. Both sandboxes: `orig/`, `new/`, patch, MANIFEST.json (note the covered-cluster sets FIRST —
   disjoint vs overlapping determines the whole adjudication).
3. Live godfile READ-ONLY; brief; then the skill's w2 section.

## 1. Snapshot + blob gate (before anything else)
- `md5sum live/<godfile> <sandbox>/orig/<godfile>`; w1a orig was byte-identical to live HEAD,
  w1b orig was 4 lines SHORTER (stale snapshot).
- `git hash-object <witness orig>` vs `git rev-parse HEAD:<godfile>` — w1b orig blob
  `aed3992...` ≠ HEAD `9583383f...`, but its patch `index aed3992..a96b08c` == its own orig
  blob → internally consistent, generated from an older checkout.
- Stale-snapshot tolerance: diff ONLY the shard's line region (s2 = 3697-7392) — byte-identical
  → the patch context lies in unchanged territory → `git apply --check` against live passes.
  Record as minor finding, not rejection.

## 2. Freshness + reconstruction (format-agnostic, decisive)
Regenerate-then-compare is format-confounded (raw `diff -ruN` vs git-format headers differ).
Instead:
```
cp -r <sandbox>/orig t/ && cp <sandbox>/orig/cli.py t/cli.py   # repo-relative layout
cd t && git init -q && git add -A
git apply <sandbox>/<patch>          # w1a: raw orig//new/-prefixed patch — PLAIN apply works
git status --porcelain               # expect: AM cli.py, ?? new dirs only
diff -rq --exclude=__pycache__ t <sandbox>/new
```
Clean `diff -rq` (only `.git` differs) proves BOTH freshness and that the patch reproduces
`new/` exactly. w1a (raw format) and w1b (git format) both passed this way.

## 3. Byte-fidelity (fast AST line-slicing)
- Parse orig + new cli.py once; map `(kind, name) -> "\n".join(lines[lineno-1:end_lineno])`.
  NEVER `ast.get_source_segment` per def on 10K+ line files — O(n²), times out (>300s).
- removed-set = defs in orig not in new; added-set = defs in new not in orig (expect: 0 added).
- removed names must equal the covered clusters' method names exactly (w1b did; w1a had a
  documented benign extra: `_handle_battery_command` = plan cluster c19 embedded inside the
  c18 span 5078-5525 — c19 is itself a move cluster proposing the SAME module).
- Every remaining def in new/cli.py must be byte-identical (only HermesCLI class line — and
  for w1a, `_reload_skills` — may differ; verify those diffs are exactly the documented edits).
- Moved bodies: locate each removed def in the new module files; allow ONLY the documented
  lazy-import shims, stripping shim lines ANYWHERE in the body (mid-body insertions and
  shim+blank-line pairs both occur) — top-only stripping false-flags verbatim methods.
- Verify class-level attrs referenced via `self.` stay on the main class (MRO), e.g.
  `_PET_FRAME_INTERVAL` referenced 2× in pet_mixin but defined nowhere there.
- Orphan gate: grep remaining references in new/cli.py — re-imported names + call sites must
  resolve (w1a re-imports all 9 skill helpers; `_reload_skills` writes the moved registry via
  `import hermes_cli.skill_command_helpers as _skill_helpers`).

## 4. Conventions + seams
- Mixin modules: stdlib-only top-level imports OK; `from __future__ import annotations` OK;
  a top-level `from hermes_cli.banner import _format_context_length` is cycle-free ONLY if
  banner doesn't import cli (`grep -n '^import cli\|^from cli' hermes_cli/banner.py`).
- Orphaned section comments and blank-line separators: see head-to-head reference.
- `git apply --stat` on raw patches shows the resolved repo-relative targets — use it to
  confirm the apply mode before believing `--check` results.

## 5. Tests — the conftest trap
- Build the validation replica from the LIVE repo: `cli.py` + full package dir + the touched
  test dirs AND **`tests/conftest.py` + every touched dir's `conftest.py`** (autouse fixtures
  reset caches / sandbox env). Without conftest, tests fail spuriously: stale skill-scan
  cache → `resolve_skill_command_key("claude-code")` returns None; leaked gateway session
  vars → secret-capture callback fires. Both were replica artifacts, not regressions —
  confirmed by copying conftest in and re-running: failure set matched live baseline exactly.
- Run: `PYTHONDONTWRITEBYTECODE=1 <repo-venv>/python -m pytest -p no:cacheprovider -q
  --basetemp=C:/tmp/pytest-<unique>` (fresh basetemp per run — stale `pytest-of-<user>\pytest-current`
  throws WinError 5 PermissionError under parallel lanes).
- Baseline-verify: run the SAME existing suites on the unmodified live repo; failure sets
  must match (s2: patched 96 passed/2 failed == baseline 77 passed/2 failed for the same 9
  files; the 2 = pre-existing env-dependent).
- The witnesses' MANIFEST claims should reproduce exactly (w1b: 52 passed = 25 new + 27 existing).

## 6. Canonical adjudication (shared cluster tie-breaks)
Both VERIFIED and interchangeable (15/19 bodies byte-identical between witnesses; the 4 shim
methods differ only in shim placement). Tie-breaks that decided c18→w1a:
1. orig snapshot freshness (byte-identical to live HEAD beats 4-line-stale),
2. plan-consistent superset (c19 method rides along into the plan's own proposed module),
3. conventions (top-level cycle-free import instead of a 5th lazy shim).
Record that the loser's c18 module remains a valid drop-in (merger freedom).
Disjoint clusters: canonical = the only covering witness (c7→w1a, c22→w1b).

## 7. Verdict JSON + composition notes (write for the merger)
- Schema: `patch_verdicts` per patch (VERIFIED + byte/plan/conventions/tests/freshness/
  format/orig-snapshot findings) + `canonical_per_cluster` + `composition_notes_for_merger`.
- Composition notes MUST spell out: disjoint cli.py regions (c18 spans 5078-5525 & 5885-6163;
  c22 span 5625-5830 sits between → patches compose), final class line
  (`class HermesCLI(CLIAgentSetupMixin, CLICommandsMixin, CLIBillingMixin, StatusBarMixin, PetMixin):`),
  final import block, per-patch apply mode (w1a: PLAIN `git apply`, never `-p2`), REQUIRED seam
  edits (`_reload_skills` registry re-point), idempotency check (`git status --porcelain` first).
- Interlock: verify claims read-only with `gh pr view 79365 --json number,title,state,isDraft`
  (w1b's `interlocked_with [79365]` confirmed OPEN, non-draft, same pet cluster — Extract-ALWAYS
  satisfied).
- Self-verify the JSON afterwards (parse, canonical map 1:1 with plan clusters, counts,
  module paths, interlock list) — see s1/s4 recipes.

## Evidence artifacts (this run)
- `C:/tmp/tg-campaign/godfile/cli/extraction/w2/analyze_s2.py` — the line-slicing fidelity
  analyzer (reusable pattern for other shards; parametrize the cluster-name sets).
- Verdict: `.../w2/s2-w2b.json` (CANONICAL_SET; 29 + 30 moved methods; 0 regressions).
