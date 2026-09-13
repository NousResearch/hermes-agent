# Fork diff: what this fork changes, and keeping that page honest

This repository is a fork of [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent).
Everything it changes relative to upstream is published as a browsable page:

**https://ethenotethan.github.io/harness/**

The page is rendered by [`protolambda/forkdiff`](https://github.com/protolambda/forkdiff)
from [`fork.yaml`](../fork.yaml) at the repo root, in the style of
[op-geth's go-ethereum fork diff](https://op-geth.optimism.io/). `fork.yaml`
groups the changed files into sections with a paragraph each, names the exact
upstream commit the fork is rebased onto (`base.hash`), and lists files that
are not code (`ignore`). Every push to `main` re-renders and redeploys it
(`.github/workflows/forkdiff-pages.yml`).

## The gate

A fork-diff page is only useful while it is true, and two things make it go
stale silently. `scripts/check_forkdiff.py` runs on every PR
(`forkdiff-check.yml`, part of **All required checks pass**) and fails on both:

| Drift | Check | Why it matters |
|-------|-------|----------------|
| **Rebase onto newer upstream** | `base.hash` must equal `git merge-base HEAD upstream/main` and be an ancestor of upstream `main` | After a rebase the old base makes upstream's own commits look like fork changes: thousands of files, none of them ours. |
| **New fork change nobody described** | every path in `git diff --name-only base.hash HEAD` must match a section glob or a global `ignore` | An undescribed file is a change the page can't explain. |
| **Section describing code we no longer carry** | every glob must match at least one changed path | Stale sections are as misleading as missing ones. |

The check also renders the page, so a `fork.yaml` that forkdiff itself rejects
cannot merge. The deploy workflow runs the same check before publishing, so a
stale analysis is never served.

## Day to day

**Adding or changing fork files in a PR.** If the check lists uncovered files,
add each to the section that explains it in `fork.yaml` (or start a new
section with a short description). Files that are not code — lockfiles, CI,
contributor metadata — go under the top-level `ignore`. Keep globs specific:
a `tests/**` catch-all would swallow upstream's test changes after a bad rebase
and defeat the gate.

**Rebasing onto newer upstream.** The gate will fail with the new merge-base
in its message:

```
git fetch https://github.com/NousResearch/hermes-agent.git main:refs/remotes/upstream/main
git merge-base HEAD refs/remotes/upstream/main      # → new base.hash
```

Set `base.hash` to that value, then run the check locally and fix what it
reports — usually files upstream absorbed (stale globs to delete) and files
that moved (globs to rename):

```
python3 scripts/check_forkdiff.py --upstream-ref refs/remotes/upstream/main
```

**Previewing the page locally** (Go 1.21+):

```
go run github.com/protolambda/forkdiff@v0.1.1 -repo . -fork fork.yaml -out tmp/index.html
open tmp/index.html
```

## Design notes

- `base.hash` is a full 40-hex commit id, never a branch name: a symbolic base
  would move underneath the page and the gate alike.
- Section `ignore` lists count as coverage (forkdiff still lists those files,
  grayed out); the top-level `ignore` is for things that aren't code at all.
- The glob semantics are forkdiff's: `*` and `?` stop at `/`, `**` spans
  directories (and may match none), `[!x]` negates a class.
- The check is pure git + PyYAML so the same command runs locally and in CI;
  tests live in `tests/scripts/test_check_forkdiff.py` and exercise it against
  throwaway repositories, including a real rebase.
