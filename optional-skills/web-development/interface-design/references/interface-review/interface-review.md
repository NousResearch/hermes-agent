# Review the change, not just the code it left behind

A diff is not a surface. The lines a change deletes matter as much as the lines it adds, and the file it touches is rarely the whole of what it affects.

This skill owns change scope only: resolving the target, expanding changed files to affected surfaces, reading both sides of the diff, and classifying each finding. Domain rules belong to the `better-*` skills. Mode, severity, consolidation, coverage, the cap, the output format, and the verdict belong to `better-interface`, which this skill hands the review to. Never duplicate or override their rules here.

Correctness, tests, security, and performance belong to the project's general code review. Name the concern once and move on.

## Quick Reference

| Category | When to Use |
| --- | --- |
| [Scope Resolution](scope-resolution.md) | Targets and commands, default branch, merge-base, PR and fork refs, repository states, nothing to review, renames, exclusions, consumer expansion |
| [Removed Signals](removed-signals.md) | What to look for on the `-` side of a hunk and which skill owns each removal |

## Core Principles

### 1. Resolve the Change Scope First

`better-interface` owns mode parsing; everything after the mode is the target, so `/interface-review quick pr 482` is a `quick` review of pull request 482. [Scope Resolution](scope-resolution.md) holds the accepted targets and the command for each.

With no target supplied, resolve in this order and stop at the first match:

1. `HEAD` is ahead of `git merge-base origin/<default-branch> HEAD`: that range **plus** any uncommitted changes, with the commit count and uncommitted file count stated separately.
2. The working tree is dirty: the uncommitted changes.
3. Neither: there is no change to review. Stop and ask, per principle 2.

Order matters: checking the working tree first lets one stray formatting edit shadow a twelve-commit branch while the report still claims full coverage.

Exclude lockfiles, snapshots, generated output, vendored code, and binaries, and name what you excluded. If the scope is empty after exclusions, that is principle 2 reached by a different route: name the excluded files and ask.

### 2. With No Change, Ask Rather Than Invent One

A clean tree with nothing ahead of the merge base means the user asked to review a change that does not exist. Never fall back to `HEAD~1..HEAD` on your own. The last commit is whatever happened to land — often a merge, often someone else's work — and a report on it is indistinguishable from a report on what the user meant.

State the repository facts you found, then offer the routes and wait. [Nothing to Review](scope-resolution.md#nothing-to-review) holds the commands:

- **The last commit**, `HEAD~1..HEAD`, named by short SHA and subject so the user sees what they would get before choosing it.
- **A target they name**: `pr <n>`, a branch, a ref, or a range, resolved per principle 1.
- **A whole-repository interface audit**, which is not a change review. Hand it to `better-interface` as a repository-scope review and drop this skill's scope block, statuses, and pre-existing section: with no change, every finding is pre-existing and the classification carries no information.

Check for an open pull request on the current branch before asking, and offer it first when one exists. A branch whose commits already landed on the base resolves to no change while its pull request is still exactly what the user meant.

An empty scope after exclusions is the same situation reached a different way. Say which files were excluded and ask the same way, rather than reporting a review of nothing as `Approve`.

### 3. A Diff Is Not a Surface

A changed file is evidence, not the review subject. Its **blast radius** is the set of surfaces it renders in; review those.

Expand the blast radius one hop by default: the direct importers and callers. Expand a second hop only for design tokens, theme values, and shared primitives, where one line reaches the whole product.

Review at most five consumers, ordered by [the rule in Scope Resolution](scope-resolution.md#expanding-to-consumers), and state how many you did not expand. An unbounded sweep produces coverage claims you cannot support; an unstated cutoff produces a report that looks complete and is not.

### 4. Read the Removed Lines

Regressions are invisible in the post-change state. Read the `-` side of every hunk against [Removed Signals](removed-signals.md).

A signal is a lead, not a finding. A removal is only a regression when nothing in the change replaces it, and the domain skill owns that judgement. Route each unmatched removal to its owner and report only what that skill confirms. Then status it `Regression`, which tells the author they broke something that worked rather than made a new mistake.

### 5. Classify Every Finding

Give every finding one status:

- `Introduced`: the change created it.
- `Regression`: the change weakened something previously correct.
- `Pre-existing`: present in the touched code but not caused by this change.

Status by what the diff touched, not by which file it sits in: a line the change never touched is `Pre-existing` even three lines from a hunk. Confirm against the base ref when it matters:

```bash
git blame -L <line>,<line> "$BASE" -- path/to/file
```

Hand every finding up with its status attached and let `better-interface` apply its cap and verdict rules.

### 6. Hold the Change to Its Stated Intent

Read the pull request title and body, the linked issue, and the commit messages, then review whether the interface delivers what they claim.

This is what surfaces the **incomplete** change, which a surface review cannot see because it inspects states when present and here the point is that they are absent:

- A new variant, size, or theme applied to some states but not all: hover, focus, active, disabled, loading, selected.
- A new user-facing string with no entry in the translation catalogue the project maintains.
- A new component with no empty, loading, error, disabled, or narrow-width state.
- A control added to one surface but not to the siblings that already carry its peers.

Do not report scope creep. Whether a change does too much is a process question, not an interface one.

### 7. Hand the Review to `better-interface`

With the scope, the affected surfaces, and both sides of the diff in hand, hand the review to `better-interface` with the scope block and a status on every finding. It routes to the domain skills, applies severity, consolidates, enforces the cap, and issues the verdict, including the change-scoped rules under its **Change-Scoped Reviews** section.

If `better-interface` is unavailable, report the resolved scope and the file inventory, name it as the missing skill, and stop. Do not invent a severity scale, a cap, or a verdict.

### 8. Never Mutate the Working Tree

A change review is read-only, including the checkout. Fetch pull request refs; never check them out. `git fetch` writes only to `.git` and is permitted. `gh pr checkout`, `git checkout`, `git switch`, and `git stash` rewrite the files the author has open, failing against local edits or discarding them, and are never permitted in any mode.

Rendered verification is opt-in: mark visual and runtime claims **Not verified** unless the project exposes a cheap preview or the user asks for a rendered review. When they do, use an isolated worktree (`git worktree add /tmp/review-<n> refs/remotes/pr/<n>`) and remove it when done. That leaves the author's tree untouched, which a checkout does not, so a checkout is not an alternative here.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| One stray edit reviewed instead of the branch | Check `merge-base` before the working tree, and report both counts |
| The last commit reviewed because there was no change | State the repository facts and offer the last commit, a named target, or a repository audit |
| Hunks reviewed without their consumers | Expand one hop, two for tokens and primitives, and name what you skipped |
| Only the `+` side of the diff read | Search the `-` side for removed accessibility, focus, motion, and text signals |
| An equivalent replacement reported as a regression | Route the removal to the owning skill and report only what it confirms |
| A removal reported as a new mistake | Status it `Regression` so the author knows it used to work |
| A line near a hunk statused `Introduced` | Status by what the diff touched, confirmed with `git blame` against the base ref |
| A pull request checked out to review it | Fetch the ref and review it in place |
| Line numbers cited that do not exist on the reviewed ref | Cite against the head ref named in the scope block |
| Mode, severity, caps, the output format, or the verdict restated here | Defer to `better-interface` |
| Correctness, test, or security findings in the report | Name the concern once, point at the project's code review, and drop it |

## Review Output Format

`better-interface` owns the format, including the four change-scoped additions under its **Change-Scoped Reviews** section. Follow it as written and add nothing here.

This skill supplies the scope block:

| Field | Value |
| --- | --- |
| Target | `branch`, `working`, `staged`, `pr 482`, or the range as entered |
| Base ref | `origin/main` at `a1b2c3d` |
| Head ref | `refs/remotes/pr/482` at `e4f5g6h` |
| Commits | 7 committed, 2 files uncommitted |
| Files in scope | 12 after exclusions |
| Excluded | `pnpm-lock.yaml`, `src/__snapshots__/`: lockfile and snapshots |
| Surfaces expanded | `CheckoutPage`, `SettingsPanel`; 3 further `Button` consumers not expanded |

Plus a status on every finding, per principle 5.

Under `better-interface`'s **Verification**, list the exact `git` and `gh` commands and their results, including every write to `.git` (a fetch, a deepen, a `set-head`, a worktree), so the read-only claim in principle 8 is auditable.
