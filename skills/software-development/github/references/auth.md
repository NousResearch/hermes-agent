# GitHub Authentication Setup

This skill sets up authentication so the agent can work with GitHub repositories, PRs, issues, and CI. It covers two paths:

- **`git` (always available)** — uses HTTPS personal access tokens or SSH keys
- **`gh` CLI (if installed)** — richer GitHub API access with a simpler auth flow

## Detection Flow

When a user asks you to work with GitHub, run this check first:

```bash
# Check what's available
git --version
gh --version 2>/dev/null || echo "gh not installed"

# Check if already authenticated
gh auth status 2>/dev/null || echo "gh not authenticated"
git config --global credential.helper 2>/dev/null || echo "no git credential helper"
```

**Decision tree:**
1. If `gh auth status` shows authenticated → you're good, use `gh` for everything
2. If `gh` is installed but not authenticated → use "gh auth" method below
3. If `gh` is not installed → use "git-only" method below (no sudo needed)

---

## Method 1: Git-Only Authentication (No gh, No sudo)

This works on any machine with `git` installed. No root access needed.

### Option A: HTTPS with Personal Access Token (Recommended)

This is the most portable method — works everywhere, no SSH config needed.

**Step 1: Create a personal access token**

Tell the user to go to: **https://github.com/settings/tokens**

- Click "Generate new token (classic)"
- Give it a name like "hermes-agent"
- Select scopes:
  - `repo` (full repository access — read, write, push, PRs)
  - `workflow` (trigger and manage GitHub Actions)
  - `read:org` (if working with organization repos)
- Set expiration (90 days is a good default)
- Copy the token — it won't be shown again

**Step 2: Configure a credential manager or in-memory cache**

Prefer the OS credential manager, or git's in-memory `cache` helper. Do **not** embed the token in a remote URL, and do **not** enable plaintext on-disk credential storage as routine setup.

```bash
# macOS — Keychain
git config --global credential.helper osxkeychain

# Windows — Git Credential Manager
git config --global credential.helper manager

# Linux — libsecret when available
git config --global credential.helper libsecret

# Portable fallback: cache in memory for 8 hours (not written to disk)
git config --global credential.helper 'cache --timeout=28800'
```

Pick one helper for the machine, then trigger auth once:

```bash
# Username: <their-github-username>
# Password: <paste the personal access token, NOT their GitHub password>
git ls-remote https://github.com/<their-username>/<any-repo>.git
```

After entering credentials once, the helper reuses them. The `cache` helper forgets them when the timeout expires.

**Step 3: Configure git identity**

```bash
# Required for commits — set name and email
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

**Step 4: Verify**

```bash
# Test push access (this should work without any prompts now)
git ls-remote https://github.com/<their-username>/<any-repo>.git

# Verify identity
git config --global user.name
git config --global user.email
```

### Option B: SSH Key Authentication

Good for users who prefer SSH or already have keys set up.

**Step 1: Check for existing SSH keys**

```bash
ls -la ~/.ssh/id_*.pub 2>/dev/null || echo "No SSH keys found"
```

**Step 2: Generate a key if needed**

```bash
# Generate an ed25519 key (modern, secure, fast).
# ssh-keygen will prompt for a passphrase — set one; do not leave it empty.
ssh-keygen -t ed25519 -C "their-email@example.com" -f ~/.ssh/id_ed25519

# Display the public key for them to add to GitHub
cat ~/.ssh/id_ed25519.pub
```

Tell the user to add the public key at: **https://github.com/settings/keys**
- Click "New SSH key"
- Paste the public key content
- Give it a title like "hermes-agent-<machine-name>"

**Step 3: Test the connection**

```bash
ssh -T git@github.com
# Expected: "Hi <username>! You've successfully authenticated..."
```

**Step 4: Configure git to use SSH for GitHub**

```bash
# Rewrite HTTPS GitHub URLs to SSH automatically
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

**Step 5: Configure git identity**

```bash
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

---

## Method 2: gh CLI Authentication

If `gh` is installed, it handles both API access and git credentials in one step.

### Interactive Browser Login (Desktop)

> **PITFALL (agent-driven sessions on Windows):** when driving `gh auth login` through a pty background process, answer prompts with `process(submit)` — never `process(write)` with a bare `\n`. Enter on a Windows PTY (ConPTY/pywinpty) is a carriage return; a lone `\n` is not delivered as a line terminator, so gh's "Press Enter to open the browser" prompt (a blocking line read) silently never returns and the login hangs. Also note the browser may not open on the user's desktop from a background session — if they report that, use gh's built-in device-code option or the `--with-token` flow below.

```bash
gh auth login
# Select: GitHub.com
# Select: HTTPS
# Authenticate via browser, or choose the device-code option when no browser is available
```

Note: on Windows winget installs, gh lands at `/c/Program Files/GitHub CLI` — add it to PATH in the same shell: `export PATH="$PATH:/c/Program Files/GitHub CLI"`.

### Token-Based Login (Headless / SSH Servers)

Prefer a user-provided PAT piped into gh. Do **not** poll GitHub's OAuth device endpoint yourself, and do **not** hand-build gh's host config as a routine fallback.

```bash
echo "<THEIR_TOKEN>" | gh auth login --with-token

# Set up git credentials through gh
gh auth setup-git
```

> **PITFALL (headless Linux): `gh auth login --with-token` can hang forever.**
> On keyring-less/headless boxes (VPS, containers, no dbus session), gh's
> credential storage may block indefinitely waiting on a secret-service
> keyring. Wrap it in `timeout 20 …` to detect this. If it hangs, fall back
> to git-only HTTPS with an OS credential manager or the in-memory `cache`
> helper (Method 1) — not an agent-written gh host-config file.

### Verify

```bash
gh auth status
```

---

## Using the GitHub API Without gh

When `gh` is not available, you can still access the full GitHub API using `curl` with a personal access token. This is how the other GitHub skills implement their fallbacks.

### Setting the Token for API Calls

```bash
# Option 1: Export as env var (preferred — keeps it out of commands)
export GITHUB_TOKEN="<token>"

# Then use in curl calls:
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user
```

### Extracting the Token from Git Credentials

If git credentials are already configured (OS credential manager or cache helper), extract the token with the shipped helper — do not parse credential files with ad-hoc tools:

```bash
uv run python "${HERMES_HOME:-$HOME/.hermes}/skills/software-development/github/scripts/git-credential-token.py"
```

### Helper: Detect Auth Method

Use this pattern at the start of any GitHub workflow:

```bash
# Try gh first, fall back to git + curl
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  echo "AUTH_METHOD=gh"
elif [ -n "$GITHUB_TOKEN" ]; then
  echo "AUTH_METHOD=curl"
elif _hermes_env="${HERMES_HOME:-$HOME/.hermes}/.env"; [ -f "$_hermes_env" ] && grep -q "^GITHUB_TOKEN=" "$_hermes_env"; then
  export GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" "$_hermes_env" | head -1 | cut -d= -f2 | tr -d '\n\r')
  echo "AUTH_METHOD=curl"
elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
  export GITHUB_TOKEN=$(uv run python "${HERMES_HOME:-$HOME/.hermes}/skills/software-development/github/scripts/git-credential-token.py")
  echo "AUTH_METHOD=curl"
else
  echo "AUTH_METHOD=none"
  echo "Need to set up authentication first"
fi
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `git push` asks for password | GitHub disabled password auth. Use a personal access token as the password, or switch to SSH |
| `remote: Permission to X denied` | Token may lack `repo` scope — regenerate with correct scopes |
| `fatal: Authentication failed` | Cached credentials may be stale — run `git credential reject` then re-authenticate |
| `ssh: connect to host github.com port 22: Connection refused` | Try SSH over HTTPS port: add `Host github.com` with `Port 443` and `Hostname ssh.github.com` to `~/.ssh/config` |
| Credentials not persisting | Check `git config --global credential.helper` — use an OS credential manager (`osxkeychain`, `manager`, `libsecret`) or git's in-memory `cache` helper |
| Multiple GitHub accounts | Use SSH with different keys per host alias in `~/.ssh/config`, or a per-repo credential helper |
| `gh: command not found` + no sudo | Use git-only Method 1 above — no installation needed |
