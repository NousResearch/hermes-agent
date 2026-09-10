---
name: create-pull-request
description: 'Create a GitHub pull request following project conventions. Use when asked to create a PR, submit changes for review, or open a pull request.'
version: 1.0.0
author: Hermes Agent (adapted from oz-skills)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, PR, pull-request, git, collaboration]
    related_skills: [github-pr-workflow, github-code-review, gstack-review, ci-fix]
---

# Create Pull Request

Create a GitHub pull request following project conventions.

## Prerequisites

```bash
gh auth status
```

## Workflow

### 1. Assess Current State

```bash
git status --short --branch
git log --oneline origin/main..HEAD
git diff --stat origin/main..HEAD
```

### 2. Determine PR Details

- **Title**: conventional commit format (`feat:`, `fix:`, `docs:`, `chore:`) followed by short description
- **Body**: what changed, why, testing notes, screenshots if UI change
- **Base branch**: `main` (or project default)
- **Draft**: set draft if WIP, tests failing, or awaiting feedback

### 3. Push and Create

```bash
git push -u origin HEAD
gh pr create \
  --title "feat: add user authentication" \
  --body "## What changed
- Added login/signup endpoints
- Added JWT token handling
- Added password hashing with bcrypt

## Testing
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Manual testing complete

## Screenshots
[if applicable]" \
  --base main
```

### 4. Post-Creation

- Add labels if applicable
- Request reviewers
- Link related issues with `Closes #123`
- Monitor CI status

## PR Template

If the repo has a `.github/PULL_REQUEST_TEMPLATE.md`, follow it. Otherwise:

```markdown
## What changed

[Brief description]

## Why

[Context and motivation]

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing complete

## Screenshots (if UI change)

[before/after]

## Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Breaking changes documented
```

## Hermes Integration

- Use `terminal` for all `gh` and `git` commands
- Use `read_file` to check PR templates and conventions
- Combine with `gstack-review` for pre-PR review
- Combine with `verification-loop` for pre-push quality gates
- Combine with `ci-fix` if CI fails after PR creation
