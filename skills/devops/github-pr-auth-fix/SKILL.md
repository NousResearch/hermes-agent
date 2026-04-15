---
name: github-pr-auth-fix
description: >
  Resolves "fatal: could not read Username" errors when pushing to GitHub from a headless
  environment using the gh CLI. Provides the correct pattern for authenticating git push
  via the gh auth token.
metadata:
  author: Indigo Karasu
  version: \"1.0.0\"
  hermes:
    tags: [github, git, auth, devops]
    category: devops
---

# GitHub PR Auth Fix

When using the `gh` CLI in a headless environment, `gh auth status` may show you are logged in, but standard `git push` commands often fail with `fatal: could not read Username for 'https://github.com': No such device or address`. This happen because the `gh` CLI manages its own token, but the underlying `git` process does not automatically inherit it for HTTPS operations.

## The Solution

The most reliable way to push changes in this environment without interactive prompts is to interpolate the `gh auth token` directly into the remote URL.

### Implementation Pattern

Run the following command to update the remote URL and push in one sequence:

```bash
git remote set-url origin https://$(gh auth token)@github.com/owner/repo.git && git push origin branch-name
```

## Why this works
The `$(gh auth token)` subshell command retrieves the active session token, which Git accepts as the password for the HTTPS protocol. This bypasses the need for a credential helper or SSH keys in temporary environments (like cloud sandboxes).

## Common Pitfalls

### 1. SSH Failures
If you attempt to switch to SSH (`git@github.com:...`), you will likely encounter `Host key verification failed` unless the host's public key has been manually added to `~/.ssh/known_hosts`. Stick to the HTTPS token method for simplicity in ephemeral sessions.

### 2. Credential Helper Conflicts
Avoid using `git config --global credential.helper store` if you are in a shared or restricted environment, as it writes tokens in plain text to the filesystem. The URL-interpolation method is cleaner as it doesn't persist the token in `.git/config`.

## Verification
After pushing, verify the remote state:
```bash
gh pr list
# or
gh pr create --title "Title" --body "Description"
```
