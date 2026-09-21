---
name: hermes-secret-vault
description: Use Sahil's local encrypted Hermes secret vault for project/API credentials without exposing values in chat logs.
version: 1.0.0
adoption_status: provisional
---

# Hermes Secret Vault

Use this skill when Sahil asks to store, retrieve, inject, or manage project secrets/API credentials outside Hermes provider auth.

## Location

- CLI on PATH: `~/.local/bin/hermes-secret`
- Explicit Hermes CLI symlink: `~/.hermes/bin/hermes-secret`
- Wrapper: `~/.hermes/scripts/secret-vault.sh`
- Vault dir: `~/.hermes/secret-vault/`
- User guide: `~/.hermes/secret-vault/SECRET_VAULT.md`

## Rules

1. Never ask Sahil to paste a real secret into chat.
2. Never run commands that print plaintext secrets into tool output unless Sahil explicitly asks and accepts the risk.
3. Prefer interactive `hermes-secret set KEY` or stdin-based writes.
4. Prefer `hermes-secret run KEY:ENV_VAR -- command` for injection because it does not print the secret.
5. Use `hermes-secret get KEY` and `hermes-secret list` only for masked verification.
6. Treat `hermes-secret get-plain KEY` and `hermes-secret env KEY:ENV_VAR` as script-only. Their output is plaintext.

## Common flows

Initialise:

```bash
hermes-secret init
```

If the old prototype left partial files and Sahil confirms nothing inside matters:

```bash
hermes-secret init --force
```

Store interactively with hidden input:

```bash
hermes-secret set TWITTER_API_KEY
```

Store interactively with visible input when Sahil needs to check what he typed:

```bash
hermes-secret set TWITTER_API_KEY --visible
```

Warn that visible input should only be used when nobody can see the terminal.

Store from an existing local environment variable without echoing:

```bash
printf '%s' "$TWITTER_API_KEY" | hermes-secret set TWITTER_API_KEY --stdin
```

Verify masked:

```bash
hermes-secret get TWITTER_API_KEY
hermes-secret list
```

Both commands prompt for your vault password when needed.

Run a command with injected environment:

```bash
HERMES_SECRET_VAULT_PASSWORD='...' hermes-secret run TWITTER_API_KEY:TWITTER_API_KEY -- python3 script.py
```

Check status and permissions:

```bash
hermes-secret status
stat -c '%a %n' ~/.hermes/secret-vault ~/.hermes/secret-vault/*
```

## Pitfalls

### Pitfall 1: PTY/getpass fails when running via Hermes agent

`hermes-secret set KEY` and `hermes-secret get KEY` prompt for the vault master password via `getpass()`, which requires a real TTY. When called from a Hermes agent session via PTY terminal, `getpass` fails with `termios.error: (25, 'Inappropriate ioctl for device')` followed by `EOFError`.

This happens because the PTY transport doesn't expose a real terminal device that Python's `getpass` module can control echo on.

### Workaround: Two-step temp-file flow

When the interactive `hermes-secret set` command fails, use this user-facing pipe pattern.

**Step 1** — Ask the user to write the secret to a temp file on the VPS via their own terminal (SSH session):

**Simple version (recommended):** Shows what they type so they can verify. Includes a character count so they know it landed:

```bash
printf 'Paste key and press Enter: '; read k && printf '%s' "$k" > /tmp/sec && chmod 600 /tmp/sec && echo "done - $(wc -c < /tmp/sec) chars written"
```

What this command does in plain English:
1. Prompts "Paste key and press Enter"
2. Reads what they type into memory
3. Writes it to /tmp/sec (a temp file, no echo in terminal)
4. Locks the file so only they can read it
5. Reports back how many characters were stored

**Silent variant** (use when they're comfortable with invisible input — nothing shows as they type):

```bash
read -s -p "Paste key: " k && printf '%s' "$k" > /tmp/sec && chmod 600 /tmp/sec && echo "done"
```

**When to pick which:**
- Default to the simple visible variant above. The user can see what they're doing.
- Use the silent (-s) variant only if they specifically ask for it, or if there's a genuine shoulder-surfing risk and they know what they're doing.
- Always include a character-count confirmation (`$(wc -c < /tmp/sec)`) so they know it worked.

**Step 2** — The agent reads from the temp file and propagates to all target files. The agent never sees the plaintext value in its chat context — only the path. After propagation, wipe the temp file:

```bash
rm -f /tmp/sec
```

**Step 3 (optional)** — After propagation, verify the key is present in the target files without exposing the value:

```bash
grep "KEY_NAME" /home/kensei/.hermes/.env
# Expect: "KEY_NAME=tvly-..."  (masked by the agent before output)
```

### Alternative: Direct .env injection (skip vault)

For API keys that Hermes reads from `.env` files (Tavily, Firecrawl, etc.), the encrypted vault is an unnecessary extra hop. The simpler pattern is:

1. User writes key to /tmp/sec via the command in Step 1 above
2. Agent reads key, propagates to all 11 `.env` files (root + 10 profiles), wipes temp file
3. No vault master password needed, no PTY issues

This is the preferred path for `.env`-based API keys on this user's setup. Reserve the vault for secrets that don't have an established `.env` home.

### Pitfall 2: vault master password prompt cannot be bypassed via `HERMES_SECRET_VAULT_PASSWORD` on `set`

The `HERMES_SECRET_VAULT_PASSWORD` env var works for `hermes-secret run` (inject into a child process) and `hermes-secret env` (export). However, `hermes-secret set` and `hermes-secret get` do NOT read this env var — they always prompt interactively.

Combined with Pitfall 1, this means:
- `set` and `get` are only usable from a real SSH/terminal session, never from an agent PTY
- `run` and `env` work from any context as long as `HERMES_SECRET_VAULT_PASSWORD` is set in the environment
- For agent-driven key swaps that need to propagate to `.env` files, the two-step temp-file flow above is the only reliable path

### Pitfall 3: auth config changes need explicit approval

Trigger: any time you are about to edit provider fallbacks, credentials, auth config, or API key routing.

These affect all crons, dispatches, and profile operations. Always:
1. State what you would change and why
2. Ask: "Shall I proceed?"
3. Wait for explicit user approval

Exception: Adding a known-good secondary key to an existing fill_first pool is low-risk if the user explicitly asks for it.

## Notes

- Inference provider keys should still use native `hermes auth add` where possible.
- This vault is for project secrets and interim local automation.
- It is encrypted at rest with AES-256-GCM and PBKDF2-HMAC-SHA256.
- It remains external to Hermes core to avoid fork/merge debt and to migrate cleanly when upstream native secret support lands.
