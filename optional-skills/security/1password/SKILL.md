---
name: 1password
description: Set up op CLI, sign in, and read or inject secrets.
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

## Requirements

- 1Password account
- 1Password CLI (`op`) installed
- One of: desktop app integration, service account token (`OP_SERVICE_ACCOUNT_TOKEN`), or Connect server
- `tmux` available for stable authenticated sessions during Hermes terminal calls (desktop app flow only)

## When to Use

- Install or configure 1Password CLI
- Sign in with `op signin`
- Read secret references like `op://Vault/Item/field`
- Inject secrets into config/templates using `op inject`
- Run commands with secret env vars via `op run`

## Authentication Methods

### Service Account (recommended for Hermes)

Set `OP_SERVICE_ACCOUNT_TOKEN` in `${HERMES_HOME:-~/.hermes}/.env` (the skill will prompt for this on first load).
No desktop app needed. Supports `op read`, `op inject`, `op run`.

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

Note: This is NOT needed when using `OP_SERVICE_ACCOUNT_TOKEN` — the token persists across terminal calls automatically.

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
- If desktop app integration is unavailable (headless/CI), use service account token flow.

## CI / Headless note

For non-interactive use, authenticate with `OP_SERVICE_ACCOUNT_TOKEN` and avoid interactive `op signin`.
Service accounts require CLI v2.18.0+.

## ⚠️ Silent-hang failure mode (macOS headless / stripped env)

`op` can **hang indefinitely with no stdout/stderr and no exit** instead of failing fast. Three independent triggers, same symptom:

1. **Missing auth context.** If `OP_SERVICE_ACCOUNT_TOKEN` is absent from the *child process* env (e.g. Hermes' `execute_code`/`terminal` sandboxes deliberately scrub secret-looking vars — `TOKEN` matches the filter), `op` falls back to desktop-app auth and blocks on a socket/prompt that can never answer.
2. **`op` daemon + TCC (upstream bug).** Even with a valid service-account token, `op` spawns `op daemon --background`; on macOS 26 (Tahoe) headless/LaunchAgent contexts that daemon triggers a TCC permission dialog that never resolves → hang. `OP_CACHE=false` / `--cache=false` do NOT suppress daemon spawning. Upstream: https://github.com/1Password/op-js/issues/216
3. **Wedged desktop-app container** — the desktop-integration probe itself can block with no timeout.

**Rule: never call bare `op` in automation. Always wrap with a hard timeout**, so failure is loud and fast instead of a silent stall:

```bash
# macOS has no `timeout(1)` by default — use a subprocess-level timeout:
python3 -c 'import subprocess,sys; r=subprocess.run(["op"]+sys.argv[1:],capture_output=True,text=True,timeout=30); print(r.stdout,end=""); sys.exit(r.returncode)' read "op://Vault/Item/field"

# or in Python directly:
subprocess.run(["op","read","op://Vault/Item/field"], timeout=30, capture_output=True, text=True)
```

**Hermes-specific:** to make `op` work *inside* `execute_code`/`terminal` sandboxes (instead of avoiding them), opt the token through the secret scrubber explicitly:

```yaml
# ~/.hermes/config.yaml
terminal:
  env_passthrough:
    - OP_SERVICE_ACCOUNT_TOKEN        # SA auth itself (scrubbed as a secret by default)
    - OP_LOAD_DESKTOP_APP_SETTINGS    # set to "false": skip the desktop-app probe that wedges headless
    - OP_CACHE                        # set to "false": belt-and-braces vs the op-daemon TCC spawn
```

with matching values in `~/.hermes/.env`:

```bash
OP_LOAD_DESKTOP_APP_SETTINGS=false
OP_CACHE=false
```

(Skills that declare `required_environment_variables: [OP_SERVICE_ACCOUNT_TOKEN]` in frontmatter get the same passthrough effect automatically on `skill_view` — but the two `OP_` behavior flags are non-secret so they're stripped unless listed in `env_passthrough`.)

**Verified recipe (macOS 26.6.1, op 2.39.0, gateway on Hermes 2026.8.13):** with all three vars passed through, `op whoami` inside `execute_code` returns in ~0.1s; missing any one of the token / `OP_LOAD_DESKTOP_APP_SETTINGS=false`, it hangs until externally killed.

**For long-lived unattended paths** (cron, LaunchAgents, HA `shell_command`), the most robust pattern is **resolve-and-stash**: resolve the secret once in an interactive shell, store it in a `chmod 600` file, read the file at runtime, keep `op` as a manual-only fallback. This removes `op` from the hot path entirely.

## References

- `references/get-started.md`
- `references/cli-examples.md`
- https://developer.1password.com/docs/cli/
- https://developer.1password.com/docs/service-accounts/
