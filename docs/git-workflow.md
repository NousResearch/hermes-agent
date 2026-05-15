# Git Workflow

This repository moves too fast for a single long-lived dirty checkout.
The default workflow here should optimize for:
- small diffs
- fast review
- cheap rollback
- isolation between concurrent tasks
- minimal accidental repo-wide churn

## Goals
- Keep `main` usable as a clean sync/review branch.
- Do feature work in short-lived branches.
- Use separate `git worktree` directories for parallel tasks.
- Run narrow, relevant validation instead of expensive whole-repo sweeps.
- Keep local operator state and generated artefacts out of tracked diffs.

## Repository reality
Current repo characteristics that drive this workflow:
- active fork with both `origin` and `upstream`
- mixed Python + Node codebase
- large test surface
- tooling that can create noisy repo-wide diffs if run broadly
- local agent/runtime artefacts that should stay untracked

## Branch roles

### `main`
Use `main` for:
- syncing with the canonical base branch
- reading/reviewing
- creating fresh task branches

Avoid using `main` as a long-running work branch.

### Task branches
Use short-lived branches by intent:
- `feat/<name>` — new functionality
- `fix/<name>` — bug fixes
- `chore/<name>` — tooling, docs, hygiene, CI, config
- `refactor/<name>` — internal cleanup without behavior change
- `spike/<name>` — experiments and exploratory work
- `sync/<name>` — upstream sync or vendor-wide updates

Examples:
- `fix/telegram-send-locks`
- `chore/secret-scan-baseline`
- `feat/xai-oauth-refresh`
- `spike/tui-command-debug`

## Worktree-first development
Prefer a separate worktree per task.

## Directory layout
Suggested sibling directory layout:
```bash
../wt/feat-...
../wt/fix-...
../wt/chore-...
../wt/spike-...
../wt/review-...
../wt/sync-...
```

## Create a new task worktree
Start from fresh remote state:
```bash
git fetch origin
git worktree add ../wt/fix-telegram-send-locks -b fix/telegram-send-locks origin/main
```

Then work inside that worktree:
```bash
cd ../wt/fix-telegram-send-locks
```

Benefits:
- no task mixing in one checkout
- less accidental staging of unrelated files
- easier local review
- easier cleanup of abandoned work
- parallel feature and review streams without stash gymnastics

## Fork and upstream model
This repo already has:
- `origin` — working fork
- `upstream` — upstream source repository

Recommended model:
- fetch both remotes regularly
- branch from `origin/main` for normal local work
- use `upstream/main` for upstream sync awareness
- publish task branches to `origin`

Typical sync check:
```bash
git fetch --all --prune
git log --oneline --decorate --graph origin/main..upstream/main
```

If maintaining a local sync flow for the fork:
```bash
git checkout main
git fetch upstream origin
git rebase origin/main
# inspect upstream drift separately before integrating it
```

## Commit architecture
Commits should be organized by intent, not by time.

Preferred order inside a task branch:
1. tests or fixtures proving the change
2. minimal implementation
3. docs/config updates
4. optional cleanup/refactor follow-up

Commit types:
- `test:` test-only changes
- `fix:` bug fixes
- `feat:` features
- `chore:` tooling/config/docs hygiene
- `docs:` docs-only
- `refactor:` non-behavioral cleanup

Rules:
- do not mix broad formatting with logic changes
- do not batch unrelated fixes into one commit
- keep hygiene/config changes separate from feature logic when possible

## Validation model
This repo should use targeted validation.

### Safe default checklist before push
```bash
git status --short
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
```

Then run only what matches the change:
- targeted `pytest` for touched area
- targeted `pre-commit run --files ...`
- `gitleaks detect --no-banner --source . --config .gitleaks.toml` for security-sensitive or hygiene changes

### Avoid by default
Do not use this casually:
```bash
pre-commit run --all-files
```

Reason:
- it can trigger broad repo-wide formatting/churn in this repository
- it is appropriate only for an explicit format-only or repo-wide cleanup branch

## Review model
Review each branch in 3 layers:

### 1. Scope review
Check that only intended files changed:
```bash
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
```

### 2. Behavior review
Inspect commits and diff:
```bash
git log --oneline origin/main..HEAD
git diff origin/main...HEAD
```

### 3. Hygiene/security review
Run relevant checks only:
- tests for changed area
- targeted pre-commit hooks
- secret scan when relevant

## Local artefact containment
Keep a strong boundary between:
- source files meant for version control
- local operator/runtime state
- generated temporary outputs

Recommendations:
- keep runtime state under ignored paths such as `.hermes/`, caches, or profile-home paths
- avoid ad-hoc helper files at repo root
- if a helper is local-only, either ignore it explicitly or place it under a clearly local path like `scripts/local/`

## Standard flows

### Small bugfix
```bash
git fetch origin
git worktree add ../wt/fix-issue -b fix/issue origin/main
cd ../wt/fix-issue
# edit files
pytest path/to/relevant_test.py -q
pre-commit run --files file1 file2
git add file1 file2
git commit -m "fix: describe change"
git push -u origin fix/issue
```

### Hygiene or config change
```bash
git fetch origin
git worktree add ../wt/chore-hygiene -b chore/hygiene origin/main
cd ../wt/chore-hygiene
pre-commit run --files .pre-commit-config.yaml .gitleaks.toml
gitleaks detect --no-banner --source . --config .gitleaks.toml
git add .pre-commit-config.yaml .gitleaks.toml
git commit -m "chore: describe hygiene change"
```

### Risky exploration
```bash
git fetch origin
git worktree add ../wt/spike-foo -b spike/foo origin/main
cd ../wt/spike-foo
# experiment
# if result is useful: create a fresh feat/fix branch and cherry-pick only the good commits
# if not: remove the worktree and drop the branch
```

## Worktree cleanup
List worktrees:
```bash
git worktree list
```

Remove an abandoned worktree:
```bash
git worktree remove ../wt/spike-foo
```

Delete its branch if no longer needed:
```bash
git branch -D spike/foo
```

## Suggested aliases
Use shell aliases or git aliases for common flows.

### Shell aliases
Add to your shell rc if desired:
```bash
alias gs='git status --short --branch'
alias gup='git fetch --all --prune'
alias gwl='git worktree list'

mkwt() {
  local branch="$1"
  local base="${2:-origin/main}"
  local dir="../wt/${branch//\//-}"
  git fetch origin && git worktree add "$dir" -b "$branch" "$base"
}

reviewdiff() {
  local base="${1:-origin/main}"
  git diff --stat "$base"...HEAD
  echo '---'
  git diff --name-only "$base"...HEAD
  echo '---'
  git log --oneline "$base"..HEAD
}
```

### Git aliases
```bash
git config alias.wtl 'worktree list'
git config alias.rstat 'diff --stat origin/main...HEAD'
git config alias.rfiles 'diff --name-only origin/main...HEAD'
git config alias.rlog 'log --oneline origin/main..HEAD'
```

## Minimal adoption sequence
1. Stop starting new work directly on `main`.
2. Start every new task in a dedicated worktree.
3. Keep task branches short-lived and scoped.
4. Use targeted validation only.
5. Treat repo-wide formatting as a dedicated branch/PR class, never as incidental cleanup.
