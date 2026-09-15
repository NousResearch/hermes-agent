# Fork Workflow: NousResearch/hermes-agent → personal fork

Solo-maintainer workflow for keeping `main` in sync with `upstream/main`, preserving a
small set of personal customizations without recurring merge pain, and landing
feature/PR branches (e.g. `fix/state-db-registry-cli-double-writer`) that are meant to be
upstreamed, not just kept local.

## TL;DR recommendation

**Model (a), with a twist: `local-patches` is not one flat branch but a short stack of
independently-identifiable commits (or small per-customization branches merged into it),
rebased onto upstream after every sync.**

- `main` = pure mirror of `upstream/main`. Fast-forward only. It never diverges, so
  syncing is a no-op merge conflict-wise.
- `local-patches` = long-lived branch = `main` + your personal commits, rebased onto the
  new `main` after every sync.
- Feature/PR branches (upstream-bound work) branch from `main` (or `upstream/main`
  directly), **never** from `local-patches`. They stay clean and mergeable upstream, and
  don't inherit your personal customizations' conflict surface.
- Once a PR branch merges upstream, you delete it locally and it "returns" to you for
  free on the next `main` sync — no manual reconciliation needed.

This is (a) because (b) guarantees repeated identical merge-conflict resolution forever
(git has no memory of merge conflict resolutions the way `rebase --rerere` or rebase-onto
does across a *merge* history as cleanly), and (c) is over-engineered for "a small set of
local patches" — a scripted per-topic-branch rebase farm is the right answer at 10+ active
customizations, not 2-4.

## Why not (b): merge commits directly on `main`

- Every `git fetch upstream && git merge upstream/main` replays your same personal diffs
  against upstream's evolving files. If upstream ever touches the same lines (very likely
  for `hermes_state.py`/`cli.py`-adjacent files you're already patching), you re-resolve
  variations of the *same* conflict indefinitely, and resolutions live buried in merge
  commits instead of as clean, auditable commits.
- `main` becomes a private mutant of upstream: you can no longer fast-forward, can't
  cleanly diff "what did I change" vs upstream, and can't easily open a clean PR *from*
  `main` because it's polluted with unrelated personal commits.
- History is not simplifiable: `git log --oneline main ^upstream/main` shows merge noise,
  not a clean patch list.

## Why not (c) as the default: per-customization topic branches + rebase-bot

- Right idea at scale (many independent customizations with different lifetimes,
  possibly a team). For "a small set" solo-maintained patches, running N topic branches
  through a rebase-and-report script is more moving parts (script maintenance, per-branch
  drift, remembering which branch is "installed" where) than benefit.
- You *can* fold this in later without changing the model: each customization commit in
  `local-patches` is still individually identifiable (one commit = one customization, or
  one small merge from a short-lived topic branch), so if the set grows past ~6-8 patches
  or one starts needing independent testing/toggling, promote it out of `local-patches`
  into its own topic branch and rebase it independently. The recommendation below is
  written so that migration is mechanical, not a redesign.

## The model in one picture

```
upstream/main ──●──●──●──●──●──●──●──●──▶  (NousResearch, moves on its own)
                 \  \  \  \  \  \  \  \
main (mirror)     ●──●──●──●──●──●──●──●   fast-forward only, == upstream/main always
                                        \
local-patches                           ●──p1──p2──p3   (rebased onto main after each sync)

fix/state-db-registry-cli-double-writer
   branches off main (== upstream/main) directly, rebased forward the same way,
   PR'd to NousResearch, then deleted once merged upstream.
```

## Step-by-step setup (one-time)

```bash
cd /home/ian-koratsky/Development/claude-code
git remote -v                      # confirm origin=fork, upstream=NousResearch (already done)

# main becomes a pure mirror — protect it from accidental local commits
git checkout main
git reset --hard upstream/main
git push origin main --force-with-lease   # only needed once, to align fork's main with upstream

# create the long-lived patch branch on top of current main
git checkout -b local-patches main
# (re-)apply your existing personal customizations here as clean, one-customization-per-commit
git push -u origin local-patches
```

Optional guardrail so you never `git commit` on `main` by accident:
```bash
# .git/hooks/pre-commit (or a repo-wide hook) — reject commits while on main
#!/bin/sh
branch=$(git symbolic-ref --short HEAD)
if [ "$branch" = "main" ]; then
  echo "Refusing commit on 'main' — it must stay a pure upstream mirror. Use local-patches or a feature branch." >&2
  exit 1
fi
```

## Weekly (or on-demand) sync cadence

```bash
cd /home/ian-koratsky/Development/claude-code

# 1. Fast-forward main to upstream — this can never conflict by construction
git fetch upstream
git checkout main
git merge --ff-only upstream/main      # fails loudly if main ever drifted; that's the tripwire
git push origin main                    # keep your fork's main mirrored too

# 2. Rebase your personal patch stack onto the new main
git checkout local-patches
git rebase main
# resolve any conflicts commit-by-commit (see policy below), then:
git push origin local-patches --force-with-lease

# 3. Roll every open feature/PR branch forward the same way
git checkout fix/state-db-registry-cli-double-writer
git rebase main
git push origin fix/state-db-registry-cli-double-writer --force-with-lease
```

Turn steps 1–3 into a single script once it's routine (`scripts/sync-fork.sh`):
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git fetch upstream
git checkout main && git merge --ff-only upstream/main && git push origin main
for b in local-patches fix/state-db-registry-cli-double-writer; do
  git checkout "$b"
  if git rebase main; then
    git push origin "$b" --force-with-lease
  else
    echo "CONFLICT rebasing $b onto main — resolve manually, then: git rebase --continue"
    exit 1
  fi
done
git checkout main
```
Enabling `git config rerere.enabled true` repo-wide means any conflict you resolve once
(e.g. a recurring hunk in `hermes_state.py`) is auto-replayed by git on the next rebase,
which is the practical antidote to "endless reconflicting."

## Worktrees (matches your existing convention)

Keep using worktrees for isolated feature work; nothing above changes:
```bash
git worktree add ../claude-code-wal-fix fix/state-db-registry-cli-double-writer main
```
`local-patches` can also live in its own permanent worktree
(`../claude-code-local`) if you want your daily-driver checkout to always run *with*
customizations while `main`'s worktree stays a clean upstream mirror for comparison/bisect.

## Conflict-resolution / drop-if-superseded policy for personal patches

Apply this checklist every time `local-patches` rebase hits a conflict on one of your
commits:

1. **Read what upstream did to that hunk first** (`git log -p -- <file>` on
   `upstream/main` since your patch's base) before touching your side. Determine intent,
   per AGENTS.md's "verify the claim and the intent" rule — don't fight upstream's design
   blind.
2. **Superseded** (upstream now does what your patch did, equivalently or better): drop
   your commit. `git rebase --skip` (rebase) or `git rm`/revert the change, and record it
   in a short `local-patches/CHANGELOG.md` entry: *"dropped patch X, superseded by
   upstream commit `<sha>` on `<date>`."* This log is your audit trail for "why is my
   patch list shrinking."
3. **Partially superseded** (upstream solved the core problem differently, but your patch
   also carried an unrelated tweak): split the commit — drop the superseded part, keep the
   rest as a smaller commit. Never keep a whole patch alive just because part of it is
   still needed; that's how patches silently regain conflict surface.
4. **Still needed, just conflicts syntactically** (upstream refactored nearby code but the
   underlying gap your patch fills still exists): resolve normally, `git add`, `git rebase
   --continue`. Let `rerere` remember it.
5. **Should actually be upstreamed** (you realize during a conflict that this "personal"
   patch is a real bug fix or generally useful): promote it — cherry-pick it out of
   `local-patches` onto a fresh branch off `main`, open a PR to NousResearch (see next
   section), and once merged, drop it from `local-patches` per rule #2. Don't let good
   fixes rot as private patches.

Re-review the full `local-patches` stack (not just the commit that happened to conflict)
every few months — upstream may quietly obsolete a patch without ever touching the same
lines (e.g. it adds a config flag that makes your hardcoded override moot).

## How upstream-bound PR branches fit in (e.g. `fix/state-db-registry-cli-double-writer`)

These branches are **not** customizations and must never live inside or be based on
`local-patches` — mixing them would (a) force your personal patches onto NousResearch's
PR diff, and (b) make the eventual "delete after merge" step re-litigate whether any
personal commit accidentally rode along.

- **Branch point:** always `main` (== `upstream/main`) or `upstream/main` directly, never
  `local-patches`.
  ```bash
  git fetch upstream
  git checkout -b fix/state-db-registry-cli-double-writer upstream/main
  # or via worktree, as already set up:
  git worktree add ../claude-code-wal-fix fix/state-db-registry-cli-double-writer upstream/main
  ```
- **While open:** rebase it onto `main` on the same weekly cadence as `local-patches`
  (step 3 of the sync script above), so it never goes stale relative to upstream and the
  eventual PR diff stays minimal and current.
- **Do you also want the fix locally before it's merged upstream?** If yes (e.g. you need
  the double-writer fix in your daily driver now, not after upstream review lands), that's
  fine and doesn't violate the model: merge or rebase `fix/state-db-registry-cli-double-writer`
  into `local-patches` as a normal "temporary customization" commit, but **tag it clearly**
  (commit message prefix `[upstream-pending]` or a note in `CHANGELOG.md`) so future-you
  knows to drop it from `local-patches` the moment it lands upstream — it's rule #2
  (superseded) applied proactively, not a special case.
- **After NousResearch merges the PR:** delete the local and remote feature branch; the
  fix reappears automatically in `main` on your next `git merge --ff-only upstream/main`.
  If you'd temporarily folded it into `local-patches`, drop that now-redundant commit in
  the same sync pass.
  ```bash
  git branch -d fix/state-db-registry-cli-double-writer
  git push origin --delete fix/state-db-registry-cli-double-writer
  ```
- **If NousResearch merges a different implementation of the same fix:** treat it exactly
  like the "superseded" case above — diff your version against theirs, drop yours, and if
  your approach caught something theirs didn't, that's a fast, small, focused follow-up PR
  (in the spirit of AGENTS.md's "fix the whole bug class" guidance), not a fork-only patch.

## When to graduate to model (c)

Signals it's time to split `local-patches` into per-customization topic branches with a
rebase-and-report script:
- More than ~6–8 independent personal patches, or
- Any single patch needs to be toggled on/off independently (e.g. testing upstream's
  fix vs. yours side by side), or
- You start wanting CI to verify each customization rebases cleanly *before* you're forced
  to discover it interactively.

Migration path (mechanical, not a rewrite): for each commit in `local-patches`,
`git checkout -b patch/<name> <that commit's parent>` then cherry-pick just that commit;
`local-patches` itself becomes `git merge --no-ff patch/a patch/b patch/c` onto `main` each
sync, or you retire it entirely and check out whichever combination of `patch/*` branches
you want merged for your daily driver.
