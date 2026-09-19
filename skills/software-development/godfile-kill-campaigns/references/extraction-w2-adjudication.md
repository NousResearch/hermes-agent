# Extraction w2 adjudication — full recipe (cli.py shard s1, witness w2b)

Blind cross-check of two wave-1 extraction patches; verdict JSON per shard.
Session-tested 2026-08-05 on cli.py s1 (w1a: c2+c3; w1b: c10+c11+c18, disjoint).

## 1. Read order
1. `shard-plan.json` → `shards.<sN>` (clusters, classifications, proposed modules, slice order).
2. Both w1 sandboxes: `MANIFEST.json`, `<sN>-impl-w1{a,b}.patch`, minimal `orig/` + `new/` trees.
3. Live godfile (READ ONLY) + `brief-godfile-extraction.md`.
4. The `godfile-kill-campaigns` skill (this file).

## 2. Freshness + apply checks (before anything else)
```bash
cd <live-repo> && git rev-parse HEAD
git hash-object cli.py                          # live blob
git hash-object 'C:/.../s1-w1b/orig/cli.py'     # native C:/ path — git rejects MSYS /c/ paths
# git-format patch: compare live blob against the patch's "index <old>..<new>" line
grep -m1 '^index ' s1-impl-w1b.patch
# diff -ruN patch: byte-compare orig vs live
git diff --no-index --stat 'C:/.../orig/cli.py' cli.py
git apply --check s1-impl-w1a.patch   # NO pipe — capture git's own exit code
```
Findings that shaped the verdict:
- w1b orig blob == live blob == patch index line → FRESH, applies.
- w1a orig was 18481 lines vs live 18485 (4-line drift at line ~10249, OUTSIDE its moved regions) → whole-file hunk `@@ -1,18481 +1,18377 @@` cannot match → `git apply --check` FAILS. Content unaffected (drift outside moved spans) but the artifact must be regenerated.

## 3. CRLF corruption detection (the 1.7MB patch)
```bash
file orig/cli.py new/cli.py        # 'with CRLF line terminators' on one side = the tell
git diff --no-index --ignore-cr-at-eol --stat orig/cli.py new/cli.py   # true content delta
diff -ruN orig new > /tmp/regen.patch && cmp -s /tmp/regen.patch s1-impl-w1a.patch
```
- Whole-file hunk + 10-100x oversized patch = EOL mismatch, not a real rewrite.
- `git apply --check` may still PASS whole-file hunks; an applied worktree then shows the whole godfile modified (`git diff --stat` = N insertions/N deletions).
- Committed patch can be post-hoc LF-normalized: regen differs from committed by ≈ #added-lines bytes; worktree won't reverse-apply the committed patch. Verdict: content OK, artifact = regenerate minimal LF diff.

## 4. Byte-fidelity (per witness)
Script: parse live godfile top-level blocks (regex `^(async )?(def|class) (\w+)` at col 0), for each moved method take [def line, next top-level def) and assert `.rstrip()` is contained in the target mixin text AND absent from new cli.py.
Pitfalls hit:
- Nested defs (MANIFEST `nested_in`) are not top-level blocks — check the enclosing parent's block, or grep the indented `def name` in the mixin.
- Block boundary can include the NEXT section's header comment/constants → containment "MISSING" is a false positive; diff the live block vs the mixin block to confirm only the trailing section header differs.
- CR-normalize both sides before containment (`s.replace('\r\n','\n')`) when a witness's tree is CRLF.

## 5. Re-export seam + module-state rebind check
- Verify new cli.py has `from hermes_cli.<module> import (...)` blocks naming every moved symbol (grep the block; note it may sit at the ORIGINAL region location mid-file, which is fine).
- Moved module constants (`_REASONING_TAGS`, `_IMAGE_EXTENSIONS`, `_WORKTREE_MERGE_CACHE_MAX`) must be re-exported if anything outside the moved region references them — grep live for other consumers (cli.py AND tui_gateway/ etc.).
- Module-state rebind divergence (`_active_worktree` moved; staying `main()` does `global _active_worktree; _active_worktree = wt_info`): grep ALL callers of the moved reader — if every call passes the state explicitly (`_cleanup_worktree(wt_info)`), the divergence is unobservable and the move is sound.

## 6. Tests
- w1a: run in its own validation worktree (stale HEAD ok for content) — 41 passed (25 new + 16 regression).
- w1b: fresh worktree at CURRENT HEAD, `git apply`, pytest the 2 new + 2 updated test files (scope to changed classes when files are large): 43 passed; `py_compile` all 7 touched files.
- Stale-patch content against CURRENT live: PYTHONPATH shadow — copy the new modules + tests into `scratch/hermes_cli/` + `scratch/tests/hermes_cli/`, `touch hermes_cli/__init__.py tests/__init__.py tests/hermes_cli/__init__.py`, then
  `PYTHONPATH="scratch;<repo>" <repo>/.venv/Scripts/python.exe -m pytest scratch/tests/hermes_cli/ --basetemp=...`
  Without the `__init__.py` files the repo's real `hermes_cli` package wins and the injected submodule is unreachable (ModuleNotFoundError) → false FAIL.
- Updated tests must retarget monkeypatches to the NEW module namespace (patch `hermes_cli.worktree_mixin._worktree_merge_cache_path`, not `cli._...`) — patching the re-export has no effect on the mixin's internal globals.

## 7. Adjudication + verdict JSON
- Canonical per cluster = the covering witness's extraction; VERIFIED only if content + tests pass; a fresh applying patch is VERIFIED as-is; a stale/CRLF artifact is CONTENT-VERIFIED with explicit `patch_artifact: REGENERATE` merge instructions.
- Skips: c13 (module-global invariant `_LIGHT_MODE_CACHE` + entanglement) accepted from w1b; c15 skipped by BOTH witnesses independently → SKIPPED_DOUBLE_BLIND_AGREEMENT.
- Not covered by either witness (c1/c4/c6/c7/c9/c12/c14/c17/c19): `NOT_COVERED_WAVE1`, stay in godfile.
- JSON must cover ALL plan clusters (19/19), canonical map 1:1 with the cluster map (no grouped keys), classification + proposed_module matching the plan, every canonical method present in the covering patch/manifest.
- Self-verify: throwaway script (tempfile under %TEMP%, prefix `hermes-verify-`, auto-clean in `finally`) asserting: schema fields, 19/19 coverage, key parity, class/module match, method presence, verdict↔canonical_source consistency. It caught: a missing cluster (c1) and grouped canonical keys — both fixed before ship.

## 8. Cleanup
`git worktree remove --force <w2-wt> && git worktree prune`; `rm -rf` scratch/pytmp dirs (retry once if 'Device or resource busy'); NEVER delete other witnesses' temp files (their evidence).
