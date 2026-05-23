---
name: 1password
description: Use 1Password CLI with the fleet service-account token for non-interactive secret reads, writes, injection, and validation. Also covers fallback interactive sign-in only for non-fleet machines without a service token.
version: 1.0.0
author: arceus77-7, enhanced by Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, secrets, 1password, op, cli]
    category: security
setup:
  help: "Create a service account at https://my.1password.com → Settings → Service Accounts"
  collect_secrets:
    - env_var: OP_SERVICE_ACCOUNT_TOKEN
      prompt: "1Password Service Account Token"
      provider_url: "https://developer.1password.com/docs/service-accounts/"
      secret: true
---

# 1Password CLI

Use this skill when the user wants secrets managed through 1Password instead of plaintext env vars or files.

Fleet default: use the service-account token. Do not use desktop auth, `op signin`, `op account add`, or tmux auth flows on Hermes/Forge/Bastion/Skynet unless the user explicitly says this is a non-fleet/manual 1Password setup.

## Requirements

- 1Password account
- 1Password CLI (`op`) installed
- Service account token (`OP_SERVICE_ACCOUNT_TOKEN`) for fleet/headless use
- Desktop app integration only for explicit non-fleet/manual setup
- `tmux` only for explicit desktop app flow, never for normal fleet service-account use

## When to Use

- Install or configure 1Password CLI
- Verify service-account auth with `op whoami`
- Read secret references like `op://Vault/Item/field`
- Inject secrets into config/templates using `op inject`
- Run commands with secret env vars via `op run`

## Authentication Methods

### Service Account (fleet default)

Use the fleet service-account token. Default path is `~/.openclaw/.op-service-token`; on Mac Studio with sandboxed agent homes, use `/Users/alexgierczyk/.openclaw/.op-service-token`. No desktop app needed. Supports `op read`, `op item get|create|edit`, `op inject`, and `op run`.

```bash
TOKEN_FILE="${OP_SERVICE_ACCOUNT_TOKEN_FILE:-$HOME/.openclaw/.op-service-token}"
[[ -f "$TOKEN_FILE" ]] || TOKEN_FILE="/Users/alexgierczyk/.openclaw/.op-service-token"
OP_SERVICE_ACCOUNT_TOKEN="$(cat "$TOKEN_FILE")" gtimeout 15 op whoami
```

### Desktop App Integration (non-fleet/manual only)

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

## Hermes Execution Pattern

Hermes terminal commands are non-interactive by default. For fleet use, export `OP_SERVICE_ACCOUNT_TOKEN` inline and wrap networked `op` calls with `gtimeout 15` on macOS or `timeout 15` on Linux.

The tmux desktop-auth flow below is only for explicit non-fleet/manual setups. It is not needed when using `OP_SERVICE_ACCOUNT_TOKEN`.

```bash
SOCKET_DIR="${TMPDIR:-/tmp}/hermes-tmux-sockets"
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
TOKEN_FILE="${OP_SERVICE_ACCOUNT_TOKEN_FILE:-$HOME/.openclaw/.op-service-token}"
[[ -f "$TOKEN_FILE" ]] || TOKEN_FILE="/Users/alexgierczyk/.openclaw/.op-service-token"
OP_SERVICE_ACCOUNT_TOKEN="$(cat "$TOKEN_FILE")" gtimeout 15 op read "op://app-prod/db/password"
```

### Get OTP

```bash
op read "op://app-prod/npm/one-time password?attribute=otp"
```

### Inject into template

```bash
TOKEN_FILE="${OP_SERVICE_ACCOUNT_TOKEN_FILE:-$HOME/.openclaw/.op-service-token}"
[[ -f "$TOKEN_FILE" ]] || TOKEN_FILE="/Users/alexgierczyk/.openclaw/.op-service-token"
OP_SERVICE_ACCOUNT_TOKEN="$(cat "$TOKEN_FILE")" \
  gtimeout 15 sh -c 'echo "db_password: {{ op://app-prod/db/password }}" | op inject'
```

### Run a command with secret env var

```bash
export DB_PASSWORD="op://app-prod/db/password"
op run -- sh -c '[ -n "$DB_PASSWORD" ] && echo "DB_PASSWORD is set" || echo "DB_PASSWORD missing"'
```

## Guardrails

- Never print raw secrets back to user unless they explicitly request the value.
- Never print `OP_SERVICE_ACCOUNT_TOKEN`; redact command output if it may contain it.
- Prefer `op run` / `op inject` instead of writing secrets into files.
- If `op` hangs, inspect and clean stale CLI state before declaring the token bad: `ps -ef | grep '[ /]op '`, kill stale `op`/`op daemon` processes if needed, and remove `~/.config/op/op-daemon.sock`.
- Validate with `op whoami` using `OP_SERVICE_ACCOUNT_TOKEN`. Do not infer token validity from random 1Password HTTP endpoints; many return 401/403/HTML for valid service-account tokens.
- Interactive `op signin`, desktop app integration, `op account add`, and tmux auth flows are for non-fleet/manual setups only.

## CI / Headless note

For non-interactive use, authenticate with `OP_SERVICE_ACCOUNT_TOKEN` and avoid interactive `op signin`.
Service accounts require CLI v2.18.0+.

## References

- `references/get-started.md`
- `references/cli-examples.md`
- https://developer.1password.com/docs/cli/
- https://developer.1password.com/docs/service-accounts/
