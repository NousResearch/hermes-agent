# Web_server kill closeout — merger rebuild at validated HEAD (2026-08-05)

The final godfile's merger exposed the wrong-ref trap at scale. Session-specific
receipts, complementary to `extraction-merger-pitfalls.md`.

## Symptom chain

1. Merger printed `s1: winner=w1b -> ok`, `s2/s4/s5: winner=w1a -> ok`, `s3: BLOCKED`
   — all against worktrees created from LOCAL `main` (f40fbcf).
2. Manual ship-patch applies failed at `hermes_cli/web_server.py:8417/8418`
   (`patch failed: ... patch does not apply`) for BOTH raw and fixed s3 patches.
3. Diagnosis: the w2/w3 waves validated at `0577116f83` (the live checkout's
   HEAD = `pr-79129`); the merger's `git worktree add ... main` used stale local
   main. `git rev-parse 0577116f83:hermes_cli/web_server.py` blob ≠
   `f40fbcf:...` blob. The whole-file hunks die on the drift.
4. Also caught: the s5 union builder (`build_union_s5_w3a.py`) writes to its OWN
   hardcoded OUT (`w3/s5-w3a-union/`), leaving the worktree at 0 changes.

## Fix sequence (worked)

1. Remove ALL stale merger worktrees + branches:
   `git worktree remove --force <wt>` per region, `git branch -D <branch>` per
   region, `git worktree prune`.
2. Re-add each at the VALIDATED sha (NOT `main`):
   `git worktree add <wt> -b <branch> 0577116f83`.
3. Confirm the godfile blob matches the wave records:
   `git rev-parse 0577116f83:hermes_cli/web_server.py`.
4. Apply the SHIP artifact per region (from the w3 verdict's
   recommendation/ship field — never the pick() winner):
   - s1: `w1/s1-w1b/s1-impl-w1b.patch` (VERIFIED w1b, 20/20, 38/38)
   - s2: `w3/s2-impl-w1b-fixed.patch` (binary-stripped + 2 logger `__name__`→
     `"hermes_cli.web_server"` fixes; byte-deterministic rebuild from raw)
   - s3: `w3/s3-impl-w1b-fixed.patch` (LateState fix)
   - s4: `w2/s4-canonical-merged-w2a.patch` (per-cluster w1a/w1b/w1b/w1b merged)
   - s5: run `w3/build_union_s5_w3a.py` (writes to its own OUT), then SYNC the
     delta into the worktree via file-walk copy (byte-compare every file under
     OUT vs worktree; copy differing; verify `git status --porcelain` count).
5. Verify ALL regions in the worktrees: py_compile + shipped tests
   (s1 11, s2 42, s3 12, s4 21, s5 79 = 165 green).
6. Commit EVERY worktree (`git add -A && git commit`) — the merger never
   commits — then push + create PRs. (422 "No commits between" = forgot this.)

## Ship-artifact notes

- `s2-impl-w1b-fixed.patch`: rebuild from raw = strip 4 binary lines + 2 logger
  fixes, byte-identical — deterministic transform, quoted in verdict.
- `s3-impl-w1b-fixed.patch`: raw canonical + exactly 2 LateState lines.
- The w2/w3 dirs carry these; grep `*-fixed.patch` / `-canonical-merged-*.patch`
  / `build_union_*` before rebuilding.
- Result: web_server PRs #79774-#79778 (6/6/6/9/9 files, real diffs, epic
  interlock in all five).
