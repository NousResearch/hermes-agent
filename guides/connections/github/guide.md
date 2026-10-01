# GitHub

Read `../guide.md` first. Use Git and the official `gh` CLI for repository and
GitHub access. Use native authentication and remote operations.

## Check existing access

```bash
gh --version
gh auth status --hostname github.com
gh api --hostname github.com user --jq .login
gh repo view OWNER/REPO --json nameWithOwner,viewerPermission
```

Match the authenticated account and repository to the requested work. Use
explicit `--repo OWNER/REPO` on commands that support it; do not infer the target
from an unrelated current checkout. For Enterprise, use its hostname and
`HOST/OWNER/REPO` where supported. Never print tokens with `gh auth token` or
`gh auth status --show-token`.

## Connect

Install `gh` using its official platform instructions if absent:
https://cli.github.com/manual/installation

For a user login, run the native flow on the Hermes host:

```bash
gh auth login --hostname github.com --git-protocol https --web
```

Let the user finish the displayed browser/device authorization. On a headless
host they can open the URL on their own device; the CLI handles the exchange.
Use an existing appropriately scoped `GH_TOKEN` supplied through native secrets
or the deployment environment when that is the intended access method. Do not
copy the token into chat, manuals, scripts or Git remote URLs.

For HTTPS Git access through the selected gh account:

```bash
gh auth setup-git --hostname github.com
```

Keep an already-working SSH setup if that is the intended method. Do not
overwrite SSH keys or change remotes merely to standardize authentication.
If several gh accounts exist, use `gh auth switch --hostname github.com --user LOGIN`
when changing the active account is intended, then verify again. Environment
tokens take precedence over stored logins; resolve that before switching.

`gh` uses its own credential/configuration storage, not `$HERMES_HOME` by
default. Record the actual account, host and credential source used.

## Verify and keep operating patterns

Repeat the account and repository checks above. For an existing checkout:

```bash
git remote -v
git ls-remote origin HEAD
```

A successful read does not prove push access. Record the verified scope without
pushing a test commit. If a remote embeds credentials, remove them from the URL
and use native credential storage; never copy that URL to the manual.

Useful patterns to adapt into `$HERMES_HOME/connections/github/manual.md`:

```bash
gh issue list --repo OWNER/REPO
gh pr list --repo OWNER/REPO
gh pr view NUMBER --repo OWNER/REPO
gh run list --repo OWNER/REPO
```

Record repository/account selection, Git authentication method, any required
host/config environment-variable names, and helper paths. Link this guide for
reconnecting instead of copying login instructions. Respect user or
responsibility authority for writes, reviews and pushes.

## Native references

- https://cli.github.com/manual/gh_auth_login
- https://cli.github.com/manual/gh_auth_setup-git
- https://cli.github.com/manual/gh_auth_switch
- https://cli.github.com/manual/gh_auth_status
