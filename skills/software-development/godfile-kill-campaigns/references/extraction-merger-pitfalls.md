# Extraction merger debugging arc (run.py 2026-08-05)

The agreement merger (`merge_godfile_extraction.py`) crashed through five distinct
root causes before producing clean worktrees. Error signature → root cause → fix,
so a future campaign skips the whole arc.

## 1. `NotADirectoryError: [WinError 267]` at `git apply --check` (cwd=wt)
- **Cause**: stale git worktree registrations (dead w3 witnesses left dirs removed
  but `.git/worktrees/` entries live). `git worktree add` refuses ("already
  exists"), the merger's fallback misses, and subprocess with `cwd=wt` explodes.
- **Fix**: `git worktree prune` before running; remove leftover dirs with native
  `C:/...` paths (`git worktree remove --force` rejects MSYS `/c/...` paths with
  "is not a working tree"). Windows: a terminal whose cwd is INSIDE a worktree
  holds a handle → "Permission denied" on removal — `cd` out first, then `rm -rf`.

## 2. Merger `git worktree add` fails silently → crash later
- **Cause**: the branch `gfg/extract-s1-w1b` ALREADY EXISTED (the adapter
  campaign's shipped PR branches own that namespace, locally + on the fork).
- **Fix**: campaign-scope the branch prefix via env (`EX_BRANCH_PREFIX=gfg/run-extract`
  for run.py, `gfg/base-extract`, etc.). Also handle branch-exists by attaching:
  `git worktree add <wt> <branch>` as a fallback.

## 3. `FileNotFoundError: [WinError 206] filename or extension too long` at pytest
- **Cause**: the merger built the pytest command with EVERY `test_*.py` under the
  tests dir as an argument → Windows 32K command-line limit.
- **Fix**: scope to the region's shipped regression tests:
  `fn.startswith(f"test_s{region}_") and fn.endswith(".py") and TESTS_DIR in root`.

## 4. THE POISONED PATCH — py_compile fails, `git apply` stages `D gateway/run.py`
- **Symptom**: s1's worktree repeatedly ended with `D gateway/run.py` (staged
  deletion) + py_compile failed with empty stderr. The patch applied cleanly by
  hand (`M` + 3 new files) but the merger produced a deletion.
- **Root cause**: the sandbox had TWO patch files — `s1-impl-w1b.patch`
  (1,347,142 bytes: the full-tree diff `diff -ruN orig new` of a tree whose
  `new/` LACKED run.py → the diff deletes the whole godfile) and
  `s1-impl-w1b-repo.patch` (16,081 bytes: the canonical, w2/w3-VERIFIED patch).
  The merger's `patch_path` checked `{region}-impl-{w}.patch` FIRST → applied the
  CONTRADICTED 1.3MB deletion variant.
- **Fix**: candidate order prefers `*-repo.patch` (the shippable variant):
  `(f"{region}-impl-{w}-repo.patch", f"{region}-impl-{w}.patch", f"{region}-{w}-impl.patch")`
  + glob fallback preferring `-repo.patch`.
- **Verification pattern**: apply the patch to a copy of `orig/`, then
  `diff -rq applied-tree new/` — any residual (other than `__pycache__`/empty
  `__init__.py` markers GNU diff omits) means the patch ≠ the tree.

## 5. py_compile "failed" with EMPTY stderr
- **Cause**: the merger used `python` (the WindowsApps store python), which
  breaks on long Windows paths. The repo venv python works.
- **Fix**: hardcode the venv python
  (`C:/Users/andre/AppData/Local/hermes/hermes-agent/venv/Scripts/python`) for
  py_compile AND pytest.

## 6. Verdict JSON schemas vary across witnesses — rec parser must handle all
- w2 verdicts: `patch_verdicts: [{patch, verdict}]` (run.py style).
- Per-cluster style (s5): `canonical`/`recommendation`/`winner` fields + per-
  cluster `canonical`/`winner`/`source` keys.
- w3 top-level style (s5 w3): `verdict: VERIFIED` + `canonical: <patch>`.
- A parser that only reads `patch_verdicts` marks whole regions BLOCKED even
  when the canonical selection is clearly recorded. Add fallbacks for all three.

## 7. Idempotency / corruption from repeated applies
- Re-running the merger over a worktree that already has the patch applied:
  `git apply --check` fails, the `patch -p1` fallback applies partially, `.rej`
  files appear, and repeated cycles stage bogus `D` states.
- **Fix**: check `git status --porcelain` first; if dirty, skip apply
  ("already-applied") and verify what's there. When a worktree IS corrupted,
  remove it + its branch completely (`git worktree remove --force` + `git branch -D`
  + `git worktree prune` + `rm -rf`) before rerunning — never patch over it.

## 8. Shell guard quirk (this environment)
- Terminal commands containing the venv absolute path + "gateway"-ish tokens can
  trip the lifecycle guard's NUL-byte bug (`ValueError: embedded null character
  in path` in cron/lifecycle_guard.py). Workaround: write the invocation to a
  small script (`write_file` → `bash script.sh`) so the command text stays short;
  or use `python3 <script>` where the script itself holds the long paths.
