# Wave-3 witness validation procedure (extraction phase 2)

Empirical validation of the canonical extraction patch per shard. Verified in the
main.py s5 w3b run (2026-08-05) — every step below was executed and the numbers
reproduced the wave-2 claims exactly.

## Procedure (per shard, canonical patch)

1. **Locate the pristine target**: `git rev-parse main` + `md5sum <(git show main:<godfile>)`.
   Compare against the wave-2 verdict's recorded baseline md5 — they MUST match.
   The live repo may sit on a PR branch (e.g. `pr-79129`) with unrelated local
   mods and a DIFFERENT godfile md5. Validate against `main`, never the live
   working tree.
2. **Fresh worktree**: `git worktree add C:/tmp/.../w3/wt-<shard>-w3b -b gfg/cx-<shard>-w3b-val main`
   (native C:/ paths). Before adding, check `git branch --list 'gfg/cx-*'` AND
   `git worktree list` — died lanes leave orphaned branch refs with NO registered
   worktree; reuse only if a registered worktree actually exists, otherwise use a
   fresh unique suffix (`-val`).
3. **Apply**: `git apply --check -p1` then `git apply -p1` (patch paths carry
   `orig/`/`new/` prefixes from `diff -ruN orig new`, so `-p0` fails by design).
   Record whitespace warnings (cosmetic EOF-blank-line warnings are known-good).
   `git status --porcelain` must show exactly the godfile modified + new modules
   + new tests.
4. **Baseline md5**: md5 the worktree's godfile AFTER add, BEFORE apply — must
   equal the wave-2 baseline.
5. **py_compile**: venv python on godfile + new modules + test files.
6. **Pristine fidelity source**: after apply the worktree godfile is modified —
   compare against `git show HEAD:<godfile>` (git object, read-only), NOT the
   worktree file. AST line spans (incl. decorators) + char-for-char compare of
   every moved name vs the new modules; assert 0 diffs, 0 moved names still
   DEFINED in the new godfile, all names present in the re-export import block.
   Also byte-check relocated comment blocks (comment text above moved constants).
7. **Import + identity probe** (cwd = worktree, `sys.path.insert(0,'.')`):
   godfile and new modules must resolve from the worktree path, and
   `main.<name> is module.<name>` for every re-exported name.
8. **Tests**: shipped tests + the pre-existing regression tests that string-patch
   or from-import the moved names (they pin the re-export seam). Reproduced
   counts must equal the wave-2 record (e.g. 31 shipped + 5 regression = 36).
9. **Reverse-apply**: `git apply --check -R -p1` exit 0 on the patched tree.
10. **Leftover-refs audit**: moved names may appear only in the re-export block,
    legitimate call sites, and comments — never as defs.
11. **Cleanup**: `git worktree remove --force <wt> && git worktree prune` +
    `git branch -D <branch>`. CONFIRM removal with the full unique path —
    same-named worktrees exist per campaign (`wt-s5-w3b` under godfile/main,
    godfile/kanban_db, godfile/cli, ...), and a bare `grep wt-s5-w3b` matches
    other campaigns' trees. Never remove/prune a tree you didn't create.
12. **JSON + receipt**: wave-3 schema (`patch_results[]` with
    apply/py_compile/import/tests/leftover_refs/verdict/evidence, recommendation,
    findings, receipt), then a 3-line receipt. Live repo untouched throughout.

## Re-verifying after the worktree is gone

The verification-hygiene pass asks for fresh evidence on the JSON deliverable
after the worktree was removed. Re-run the fidelity core WITHOUT a worktree:
pristine `git show main:<godfile>` vs the wave-1 sandbox `new/` tree (the exact
files the patch was generated from) — AST span extraction + char-for-char.
Write the temp verify script via `tempfile.mkstemp(prefix="hermes-verify-")`
under %TEMP%, run with the venv python, delete after. This re-exercises the same
logic and all artifacts still exist.

## Pitfalls that cost real time

- Validating the live working tree instead of `main` → false failures (live
  branch's godfile md5 differed from the baseline).
- Grepping `git worktree list` by short name → matched another campaign's
  identical-named worktree; nearly misreported cleanup status.
- Trusting an orphaned branch ref as "already validated" — branches without
  registered worktrees are leftovers, not evidence.
- Patching over dirty state: always `git status --porcelain` first; skip
  already-applied worktrees; never re-apply (corrupts — staged `D` deletions).
