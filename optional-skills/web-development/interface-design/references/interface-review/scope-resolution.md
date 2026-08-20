# Scope Resolution

Exact commands for turning a review target into a file list. Every command here is read-only against the working tree: `git fetch` writes to `.git`, everything else only reads. Never run `gh pr checkout`, `git checkout`, `git switch`, or `git stash`; resolving a scope never requires moving the author's files. An isolated `git worktree add` at a throwaway path is the one exception, and only for opt-in rendered verification as described in the skill.

## Default branch

Try in order, and stop at the first that answers:

```bash
git symbolic-ref --quiet --short refs/remotes/origin/HEAD   # -> origin/main
gh repo view --json defaultBranchRef -q .defaultBranchRef.name
git config --get init.defaultBranch
```

If `refs/remotes/origin/HEAD` is missing, ask the remote rather than guessing with `git remote set-head origin --auto`. It needs the network and writes a ref under `.git`, leaving the working tree untouched, so it is permitted; note it in Verification. With no remote at all, fall back to a local `main` or `master` and state which base you assumed.

## Merge base

```bash
BASE=$(git merge-base origin/main HEAD)
git rev-list --count "$BASE"..HEAD          # commits in the change
git diff --name-status "$BASE"...HEAD       # files, with rename detection
```

Use three dots (`"$BASE"...HEAD`) so the diff is against the merge base and not against whatever has landed on the base branch since. Two dots reports every upstream commit as part of the change.

If the base branch is stale, refresh the remote ref before computing the merge base:

```bash
git fetch origin main --no-tags
```

## Targets

These are the accepted targets. Anything else in the invocation after the mode is treated as a `<ref>`.

| Target | Commands |
| --- | --- |
| `working` | `git diff --name-status HEAD` plus `git ls-files --others --exclude-standard` for untracked files |
| `staged` | `git diff --name-status --cached` |
| `branch` | `git diff --name-status "$BASE"...HEAD`, where `BASE` is the merge base above |
| `branch` with uncommitted work | The `branch` diff, plus `git diff --name-status HEAD` **and** `git ls-files --others --exclude-standard`; report the two counts separately |
| `pr <n>` | Fetch, then diff (see below) |
| `<ref>` | `git diff --name-status "$(git merge-base <ref> HEAD)"...HEAD` |
| `<a>..<b>` | `git diff --name-status <a>..<b>` (two dots as entered) |
| `<a>...<b>` | `git diff --name-status <a>...<b>` (three dots as entered) |

`git diff --name-status HEAD` reports tracked changes only, so any target including uncommitted work must pair it with `git ls-files --others --exclude-standard`. Otherwise a newly added component or stylesheet is silently dropped from a scope the report claims to cover in full.

Use the dots the user wrote. `<a>..<b>` compares the endpoints; `<a>...<b>` compares `merge-base(<a>, <b>)` with `<b>`. Rewriting `release..feature` to three dots drops everything between `release` and the merge base, which is often exactly what was asked for. State the resolved range in the scope block.

## Pull requests

Fetch the head into a remote-tracking ref and review it in place. This works for forks, which `origin/<branch>` does not:

```bash
gh pr view <n> --json title,body,headRefName,headRefOid,baseRefName
git fetch origin "pull/<n>/head:refs/remotes/pr/<n>" --no-tags
BASE=$(git merge-base origin/<baseRefName> "refs/remotes/pr/<n>")
git diff --name-status "$BASE"..."refs/remotes/pr/<n>"
```

Read files at that ref with `git show refs/remotes/pr/<n>:path/to/file`. Do not open the working-tree copy; on a fork PR it is a different file.

`gh pr diff <n>` is a fine shortcut for the patch text, but it gives no way to read unchanged context or expand to consumers, so fetch the ref as well.

**Citations.** `better-interface` requires `path/to/file:line`. Line numbers from a fetched ref do not necessarily match the working tree. Cite against the head ref, and declare that ref and its SHA in the scope block so the numbers are resolvable.

**Intent.** The `title` and `body` from `gh pr view` are the stated intent for principle 6. Add the commit subjects when the body is empty:

```bash
git log --format='%s%n%b' "$BASE".."refs/remotes/pr/<n>"
```

## Awkward repository states

Three worth handling explicitly. Everything else (no remote, unrelated histories, a repo with no commits, a moved submodule pointer) fails loudly at `merge-base`: say the base is unresolvable and stop rather than reviewing a range you cannot name.

**Detached HEAD.** `git symbolic-ref --quiet HEAD` fails. Use the merge base against the default branch and name the SHA, not a branch, in the scope block.

**Shallow clone**, the CI default. `git rev-parse --is-shallow-repository` is `true`, or `merge-base` returns nothing. Run `git fetch --deepen=50 origin` and retry, once more at `--deepen=200`, then report the scope as unresolvable. Deepening writes to `.git` and not to the working tree, so it is permitted; note it in Verification.

**Mid-rebase or mid-merge**, the one that does not fail loudly. `git diff` succeeds and returns something that is not the change, so the review looks fine and is wrong. Detect it through git rather than testing `.git/` paths, which are not directories inside the linked worktree this skill recommends:

```bash
for state in rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD; do
  test -e "$(git rev-parse --git-path $state)" && echo "in progress: $state"
done
```

Stop and say the tree is mid-operation.

## Nothing to review

The tree is clean and `HEAD` is not ahead of the merge base. Gather the facts before asking, so the offer is accurate rather than a guess about why the scope came back empty:

```bash
git rev-parse --abbrev-ref HEAD                 # current branch, or HEAD when detached
git status --porcelain                          # empty: nothing uncommitted, tracked or untracked
git rev-list --count "$BASE"..HEAD              # 0: nothing ahead of the base
git log -1 --format='%h %s'                     # the last commit, to name in the offer
gh pr status --json number,title,headRefName    # `currentBranch`: this branch's open PR, if any
```

`gh pr status` succeeds with no pull request open — it simply omits `currentBranch` — so an empty result is an answer, not an error. It fails outright without `gh`, without authentication, and on a repository with no GitHub remote. Treat any failure as "no pull request found", say so, and offer the remaining routes rather than stopping.

Report the current branch, whether it is the default branch, that the tree is clean, and whether a pull request is open, then offer the three routes in principle 2. State the last commit's SHA and subject inside the offer: the user recognises "a1b2c3d Merge pull request #482" as not what they wanted, and cannot recognise "the last commit".

A whole-repository audit is a different review, not this one with a wider net. Hand the repository to `better-interface` directly, without a scope block, statuses, or a pre-existing section.

## Renames

Rename detection is on by default for `--name-status`, which reports `R100 old/path new/path`. Raise the similarity window when a file was moved and edited in the same change:

```bash
git diff --find-renames=40% --find-copies-harder --name-status "$BASE"...HEAD
```

Review a rename as a move, not as a delete plus an add. Everything that survived the move is unchanged code, and only the genuine edits are in scope.

## Excluded paths

Exclude these from the change scope and name what you excluded in the scope block. They are machine-authored and carry no interface rules.

| Category | Patterns |
| --- | --- |
| Lockfiles | `package-lock.json`, `pnpm-lock.yaml`, `yarn.lock`, `bun.lock`, `bun.lockb`, `Cargo.lock`, `composer.lock`, `Gemfile.lock`, `poetry.lock`, `uv.lock` |
| Snapshots and fixtures | `__snapshots__/`, `*.snap`, `*.approved.*`, `test-results/`, `playwright-report/` |
| Generated output | `dist/`, `build/`, `out/`, `.next/`, `.turbo/`, `.svelte-kit/`, `coverage/`, `storybook-static/`, `*.min.js`, `*.min.css`, `*.map` |
| Generated sources | `*.gen.ts`, `*.generated.*`, `*.d.ts` emitted by a build, GraphQL and Prisma client output |
| Vendored code | `vendor/`, `third_party/`, `node_modules/` |
| Binaries and media | `*.png`, `*.jpg`, `*.webp`, `*.avif`, `*.woff2`, `*.mp4`, `*.pdf` |

Two exceptions worth keeping in scope: a **font file** added or swapped is a `better-typography` change, and an **image** added to a component is a `better-ui` and `better-accessibility` change through its `alt` text and outline. Review the code that references them, not the bytes.

Apply the exclusions as pathspecs so the file count in the scope block is the reviewed count:

```bash
git diff --name-only "$BASE"...HEAD -- . \
  ':(exclude,glob)**/*.lock' ':(exclude,glob)**/*-lock.json' \
  ':(exclude,glob)**/*lock.yaml' ':(exclude,glob)**/*.lockb' \
  ':(exclude,glob)**/dist/**' ':(exclude,glob)**/build/**' \
  ':(exclude,glob)**/__snapshots__/**'
```

Two traps silently under-exclude and leave the scope block claiming a count it did not deliver. `*.lock` catches `yarn.lock` and `Cargo.lock` but not `package-lock.json` or `pnpm-lock.yaml`, so cover every suffix the table lists. And `**` needs `glob` magic: without it `*` crosses `/`, so `**/dist/**` excludes `packages/a/dist/` but misses a root-level `dist/`.

Run the diff with and without the pathspecs and confirm the count dropped by exactly the files you named.

## Expanding to consumers

Principle 3 expands one hop, two for tokens and primitives. Use the project's own resolver where one exists, otherwise import paths.

`git grep` searches the working tree by default. Pass the reviewed ref after the pattern instead, or on a pull request you search a different revision and miss importers the change itself added:

```bash
REV=refs/remotes/pr/482          # or HEAD for a local branch
git grep -l "from ['\"].*<module-name>" "$REV" -- '*.ts' '*.tsx' '*.js' '*.jsx' '*.vue' '*.svelte'
git grep -ln "<ComponentName>" "$REV" -- '*.tsx' '*.vue' '*.svelte'
```

Results come back as `<rev>:path/to/file`. Read them with `git show "$REV":path/to/file`, never the working-tree copy.

For a changed token or theme value, search the token name rather than the file, since consumers reference the name and never import it. Use `-e` when the pattern starts with a dash, or git parses it as an option:

```bash
git grep -n -e '--color-accent' "$REV" -- '*.css' '*.tsx' 'tailwind.config.*'
```

Order the consumers by a rule you can evaluate, so the cutoff is reproducible instead of a guess:

1. **Route and layout entry points first**, whatever the framework treats as a rendered surface: `app/**/page.*`, `app/**/layout.*`, `pages/**`, `routes/**`, `src/views/**`, `*.astro` pages. Everything else only appears inside one.
2. **Then by importer count**, since a component pulled in by twenty files carries more of the change than one pulled in by two:

   ```bash
   git grep -l "<ComponentName>" "$REV" -- '*.tsx' '*.vue' '*.svelte' | wc -l
   ```

3. **Break ties by proximity**: same package or feature directory first.

Review the first five, state how many you did not expand, and say so plainly if the ordering was arbitrary past a point.
