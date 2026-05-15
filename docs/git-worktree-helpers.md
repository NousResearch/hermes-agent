# Git / Worktree Helpers

Optional shell helpers for this repository workflow.

## Bash/Zsh snippet
Add to `~/.bashrc`, `~/.zshrc`, or a local shell include:

```bash
alias gs='git status --short --branch'
alias gup='git fetch --all --prune'
alias gwl='git worktree list'
alias grs='git diff --stat origin/main...HEAD'
alias grf='git diff --name-only origin/main...HEAD'
alias grl='git log --oneline origin/main..HEAD'

mkwt() {
  if [ -z "$1" ]; then
    echo "usage: mkwt <branch-name> [base-ref]"
    return 1
  fi
  local branch="$1"
  local base="${2:-origin/main}"
  local dir="../wt/${branch//\//-}"
  git fetch origin && git worktree add "$dir" -b "$branch" "$base"
}

rmwt() {
  if [ -z "$1" ]; then
    echo "usage: rmwt <worktree-path> [branch-name]"
    return 1
  fi
  local dir="$1"
  local branch="$2"
  git worktree remove "$dir" || return 1
  if [ -n "$branch" ]; then
    git branch -D "$branch"
  fi
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

## Example
```bash
mkwt fix/telegram-send-locks
cd ../wt/fix-telegram-send-locks
gs
reviewdiff
```
