---
name: 1password
description: Set up op CLI, sign in, and read or inject secrets.
version: 1.1.0
author: arceus77-7, enhanced by Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, secrets, 1password, op, cli]
    category: security
---

# 1Password CLI

Use this skill when the user wants secrets managed through 1Password instead of plaintext env vars or files.

## Requirements

- 1Password account
- 1Password CLI (`op`) installed
- One of: desktop app integration or a Connect server for `op` in the Hermes terminal
- `tmux` available for stable authenticated sessions during Hermes terminal calls (desktop app flow only)

Hermes never passes `OP_SERVICE_ACCOUNT_TOKEN` to the terminal or `execute_code`, even when it is in
`.env` or `terminal.env_passthrough`: it unlocks every secret the service account can read. A service
account is still the right way to feed provider keys to Hermes itself (`hermes secrets onepassword setup`).

## When to Use

- Install or configure 1Password CLI
- Sign in with `op signin`
- Read secret references like `op://Vault/Item/field`
- Inject secrets into config/templates using `op inject`
- Run commands with secret env vars via `op run`

## Authentication Methods

### Service Account (Hermes's own secret resolution, or your own shell)

A service-account token in `${HERMES_HOME:-~/.hermes}/.env` lets Hermes resolve `op://` references for
its own provider keys at startup (`hermes secrets onepassword setup`). It is stripped from the processes the
agent spawns, so `op` run through the Hermes terminal does not see it. To use the token with `op`
directly, run those commands in your own shell, outside Hermes:

```bash
export OP_SERVICE_ACCOUNT_TOKEN="your-token-here"
op whoami  # verify — should show Type: SERVICE_ACCOUNT
```

### Desktop App Integration (interactive)

1. Enable in 1Password desktop app: Settings → Developer → Integrate with 1Password CLI
2. Ensure app is unlocked
3. Run `op signin` and approve the biometric prompt

### Connect Server (self-hosted)

```bash
export OP_CONNECT_HOST="http://localhost:8080"
export OP_CONNECT_TOKEN="your-connect-token"
```

## Setup

1. Install CLI:

```bash
# macOS
brew install 1password-cli

# Linux (official package/install docs)
# See references/get-started.md for distro-specific links.

# Windows (winget)
winget install AgileBits.1Password.CLI
```

2. Verify:

```bash
op --version
```

3. Choose an auth method above and configure it.

## Hermes Execution Pattern (desktop app flow)

Hermes terminal commands are non-interactive by default and can lose auth context between calls.
For reliable `op` use with desktop app integration, run sign-in and secret operations inside a dedicated tmux session.

Note: a service-account token is never available here (see Requirements), so headless `op` use from the
Hermes terminal needs a Connect server instead.

```bash
SOCKET_DIR="${TMPDIR:-${HERMES_HOME:-$HOME/.hermes}/cache/scratch}/hermes-tmux-sockets"
mkdir -p "$SOCKET_DIR"
SOCKET="$SOCKET_DIR/hermes-op.sock"
SESSION="op-auth-$(date +%Y%m%d-%H%M%S)"

tmux -S "$SOCKET" new -d -s "$SESSION" -n shell

# Sign in (approve in desktop app when prompted)
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "eval \"\$(op signin --account my.1password.com)\"" Enter

# Verify auth
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op whoami" Enter

# Example read
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op read 'op://Private/Npmjs/one-time password?attribute=otp'" Enter

# Capture output when needed
tmux -S "$SOCKET" capture-pane -p -J -t "$SESSION":0.0 -S -200

# Cleanup
tmux -S "$SOCKET" kill-session -t "$SESSION"
```

## Common Operations

### Read a secret

```bash
op read "op://app-prod/db/password"
```

### Get OTP

```bash
op read "op://app-prod/npm/one-time password?attribute=otp"
```

### Inject into template

```bash
echo "db_password: {{ op://app-prod/db/password }}" | op inject
```

### Run a command with secret env var

```bash
export DB_PASSWORD="op://app-prod/db/password"  # example op:// reference, resolved by `op run`
op run -- sh -c '[ -n "$DB_PASSWORD" ] && echo "DB_PASSWORD is set" || echo "DB_PASSWORD missing"'
```

## Guardrails

- Never print raw secrets back to user unless they explicitly request the value.
- Prefer `op run` / `op inject` instead of writing secrets into files.
- If command fails with "account is not signed in", run `op signin` again in the same tmux session.
- If desktop app integration is unavailable (headless/CI), use a Connect server from the Hermes terminal; a service-account token only works in the user's own shell or for Hermes's own `op://` resolution.

## CI / Headless note

For non-interactive use outside Hermes, authenticate with `OP_SERVICE_ACCOUNT_TOKEN` and avoid interactive
`op signin`. Service accounts require CLI v2.18.0+. Inside the Hermes terminal, use a Connect server.

## References

- `references/get-started.md`
- `references/cli-examples.md`
- https://developer.1password.com/docs/cli/
- https://developer.1password.com/docs/service-accounts/
