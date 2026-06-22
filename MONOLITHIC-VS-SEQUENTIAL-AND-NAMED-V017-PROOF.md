# Why "monolithic apply FAILED" but "per-PR apply PASSES" — and both are honest

The Council rightly flagged: don't wave away the monolithic v0.17.0 failure as
"methodology error" without proving success against the **named tag commit**
`2bd1977d8fad185c9b4be47884f7e87f1add0ce3`. This note explains the two results and
shows the proof.

## The two tests are measuring different things

**Monolithic apply** = `git diff v0.16.0..src-HEAD` applied as ONE patch onto v0.17.0.
This FAILS (18 conflicts) — and *correctly so*. The patch's hunk context lines are
v0.16.0's text. For files v0.17.0 itself rewrote by thousands of lines (cli.py:
1999+/3213-, gateway/run.py: 2713+/5093-, hermes_cli/main.py: 2685+/6094-), the
v0.16.0-based context no longer matches the v0.17.0 file, so 3-way apply conflicts.
This is NOT how the PRs pull down — no one applies the entire 144-file overlay delta
as a single patch.

**Sequential per-PR apply** = each open PR's OWN net change (`base..head`, where base
= merge-base(PR, origin/main)) applied onto v0.17.0, one PR at a time, with the
#50111 resolution patches for the 6 drifted files. This is EXACTLY the "pull them
down onto a later release" operation. It PASSES.

The distinction is not hand-waving: a PR is a small, targeted change (#49916 = 1 file,
16 lines). Pulling it onto v0.17.0 means applying *that* change, not re-deriving the
whole overlay. `base..head` isolates the PR's own additions with their own (recent)
context, which matches v0.17.0 far better than v0.16.0-rooted context.

## Proof against the NAMED v0.17.0 commit (not origin/main)

`verify_against_named_v017_commit.sh` checks out the worktree at EXACTLY
`2bd1977d8fad185c9b4be47884f7e87f1add0ce3` and ASSERTS `HEAD == named commit` before
applying. origin/main is used ONLY to compute each PR's `base..head` net diff (what
the PR adds) — never as the apply target.

```
NAMED v0.17.0 apply target = 2bd1977d8fad185c9b4be47884f7e87f1add0ce3
  (= 2bd1977d8 chore: release v0.17.0 (2026.6.19))
worktree HEAD = 2bd1977d8fad185c9b4be47884f7e87f1add0ce3
✓ ASSERT: worktree is checked out at the NAMED v0.17.0 commit (not origin/main)

apply onto NAMED v0.17.0: clean=30 resolved=9 failed=0  (total=39)
changed .py compiled=111 compile-fail=0
representative pytest: 609 passed, 28 skipped, 5 xfailed, 0 failed
RESULT: PASS — 39/39 PRs apply onto NAMED v0.17.0 commit (30 clean + 9 resolved), tree compiles
```

## Reproducible
- Canonical tree: 39/39, 117 .py compile, 460 pytest pass (integration_v017_sequential.sh)
- NAMED-commit assertion run: 39/39, 111 .py compile, 609 pytest pass (this script)
- Clean independent clone (after `git fetch origin main` to correct the stale ref):
  identical 30 clean + 9 resolved, 0 failed.

Both scripts are committed here and re-runnable: `SRC=<checkout> bash <script>`.
