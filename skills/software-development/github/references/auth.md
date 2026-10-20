# GitHub Authentication Setup

Set up authentication for GitHub repositories, PRs, issues, and CI. Keep the
git-only HTTPS/SSH and `gh` paths distinct; authentication setup is not permission
to handle credentials in an agent-visible command or conversation.

## Detection Flow

Use `terminal` for non-secret preflight:

```bash
git --version
gh --version 2>/dev/null || echo "gh not installed"
gh auth status 2>/dev/null || echo "gh not authenticated"
git config --global credential.helper 2>/dev/null || echo "no git credential helper"
```

1. If `gh auth status` succeeds, use `gh`.
2. If `gh` exists but is not authenticated, the user completes gh login below.
3. If `gh` is unavailable, use git-only setup (no sudo required).

## Method 1: Git-Only Authentication (No gh, No sudo)

### Option A: HTTPS with Personal Access Token

**Step 1: The user creates a token** at https://github.com/settings/tokens.
Choose the least permissions and repository access needed for the task, with an
expiration. For a classic token, `repo` enables private-repo operations;
`workflow` is needed for workflow-file changes and `read:org` for relevant org
operations. The user keeps the token private; never paste it into chat.

**Step 2: Configure Git authentication**

Credential entry is a user-only interactive step in a trusted shell. The agent
must not receive, type, print, or put a PAT in command text or a remote URL.
Prefer the platform credential manager; the user can choose a memory cache:

```bash
# User-run setup; no credential value appears in this command.
git config --global credential.helper cache
# The user enters credentials privately at Git's prompt.
git ls-remote https://github.com/<their-username>/<any-repo>.git
```

Persistent plaintext storage is not the default: `credential.helper store`
writes to `~/.git-credentials` and requires the user's explicit informed choice.
Never embed a token in a remote URL; URLs leak through config, process listings,
logs, and shell history.

**Step 3: Configure git identity**

```bash
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

**Step 4: Verify**

```bash
# Read-access check; this does not prove write/push permission.
git ls-remote https://github.com/<their-username>/<any-repo>.git
git config --global user.name
git config --global user.email
```

### Option B: SSH Key Authentication

Use `search_files` to check for existing public keys. Key generation/passphrase
entry is user-only in a trusted shell; never read a private key or display it.
The user can generate an ed25519 key with `ssh-keygen -t ed25519` and add only the
public key at https://github.com/settings/keys. Do not overwrite an existing key
or default to an empty passphrase. Use `terminal` for verification:

```bash
ssh -T git@github.com
# Expected: "Hi <username>! You've successfully authenticated..."
```

SSH git remotes use `git@github.com:OWNER/REPO.git`. If the user explicitly wants
a global HTTPS-to-SSH rewrite:

```bash
git config --global url."git@github.com:".insteadOf "https://github.com/"
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

## Method 2: gh CLI Authentication

### Interactive Browser Login (Desktop)

The user runs `gh auth login` in their trusted terminal, chooses GitHub.com and
HTTPS, and completes the browser/device approval themselves. Authentication and
verification codes must not enter chat or agent-controlled input.

**Windows PTY pitfall:** non-secret prompt navigation requires a submitted Enter
(carriage return on ConPTY), not a bare newline write. A background shell may not
open the browser on the user's desktop; the user should use gh's displayed
device-flow instructions in their own terminal instead.

### OAuth Device Flow (Headless)

Use the device flow provided by `gh auth login`; let the user open
https://github.com/login/device and complete it privately. Do not hand-roll an
agent-controlled device-code/token polling loop, parse OAuth token responses, or
write gh's credential store.

**Headless keyring pitfall:** `gh auth login --with-token` may stall when no
working keyring/secret-service session exists. Stop and ask the user to repair
the keyring or configure authentication outside the agent. Never fall back to
writing raw tokens into `~/.config/gh/hosts.yml`, or silently opt into insecure
storage. Storage mode is the user's informed decision.

On Windows winget installs, gh may be at `/c/Program Files/GitHub CLI`; verify
its installation path before adjusting PATH in that shell.

### Token-Based Login (Headless / SSH Servers)

Token entry is user-only: the user runs `gh auth login --with-token` in a trusted
shell and supplies the token privately through standard input. The agent must
not receive, type, print, or put a token in command text. After the user confirms
success, use `terminal` to verify without revealing credentials:

```bash
gh auth setup-git
gh auth status
```

## Using the GitHub API Without gh

Use `curl` only with `GITHUB_TOKEN` already configured in the process environment
or profile `.env`. Do not ask for a token in chat/command text or extract a PAT
from Git's credential store. Load the canonical installed detector:

```bash
_helper="${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/gh-env.sh"
if [ ! -f "$_helper" ]; then
  _helper="${HERMES_HOME:-$HOME/.hermes}/skills/software-development/github/scripts/gh-env.sh"
fi
source "$_helper"
unset _helper
```

When that standalone skill is absent, the bundled safe detector is
[`../scripts/gh-env.sh`](../scripts/gh-env.sh), invoked from the installed profile
as `skills/software-development/github/scripts/gh-env.sh`.
`GH_AUTH_METHOD` is `gh`, `curl`, `git`, or `none`. `git` means Git credentials
were detected by a presence-only check: clone/fetch/push may work, but API calls
require `gh` or a separately preconfigured `GITHUB_TOKEN`. `none` means setup is
required. A detector result is not proof of permission for a target repository.
Never print the token, enable shell tracing, or dump the environment.

`git-credential-token.py` is a **legacy, operator-only** extraction utility, not
an agent authentication route. Agents must not invoke it, including via command
substitution, or read credential files to obtain tokens. Retain it for explicit
human maintenance; presence checks use `gh-env.sh`, not that utility.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Git asks for a password | The user enters a PAT privately at Git's prompt, or chooses SSH; GitHub account passwords are not supported |
| Permission denied | Have the token owner check permissions and target repository access; do not request the token itself |
| Authentication failed | Have the user reject stale cached credentials and re-authenticate privately |
| SSH port 22 blocked | Try SSH over port 443 with `Hostname ssh.github.com` in the user's SSH config |
| Credentials not persisting | Prefer platform credential manager or `cache`; `store` is plaintext and requires explicit informed user choice |
| Multiple accounts | Use SSH host aliases or `gh auth switch`; never token-bearing per-repo URLs |
| gh unavailable and no sudo | Use git-only Method 1; API calls still require a preconfigured environment token |

## Verification

- Verify authentication with `gh auth status` or a read-access Git check.
- Preserve git-only operation without treating stored Git credentials as API tokens.
- Verify the intended target repo's permissions separately; credentials stay out
  of chat, command text, URLs, logs, and agent-managed secret files.
