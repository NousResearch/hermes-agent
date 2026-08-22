---
name: secrets-protocol
description: "All Hermes secret sources: encrypted-only cache, no leaks."
version: 1.0.0
author: Axl Ibiza
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, secrets, bitwarden, bws, onepassword, secrets-manager, credential-hygiene, exfiltration-prevention]
    related_skills: [hermes-agent]
---

# Secrets Protocol — Hermes Secrets-Handling for All Sources

The authoritative protocol for operating every Hermes secret source — Bitwarden Secrets Manager (`bws`), 1Password (`op`), and the user-configured command helper (`secrets.command`) — under one secrets-handling posture: **secrets are never persisted as plaintext, never emitted into status lines, logs, or provider-bound output, and never inherited by child processes — under any name**. If you are configuring, rotating, debugging, or auditing any secret source, follow this skill.

## When to Use

- Configuring or re-configuring any secret source: `secrets.bitwarden` (setup, token rotation, region/project changes), `secrets.onepassword` (setup, token rotation, `op://` reference mapping), `secrets.command` (helper command changes).
- Debugging "Bitwarden rejected the machine-account access token", `bws exited 1`, `op read failed`, helper-command failures, timeouts, or checksum mismatches.
- Auditing whether any secret path can leak: disk cache, stderr, logs, subprocess environments, provider-bound messages.
- Answering "is my secret safe in Hermes?" — the answer must be grounded in this protocol's invariants, not vibes.

## Protocol invariants (non-negotiable)

1. **No plaintext at rest, ever.** Every source's disk cache is AES-GCM encrypted — `~/.hermes/cache/bws_cache.enc.json` for Bitwarden, `~/.hermes/cache/op_cache.enc.json` for 1Password — keyed off the source's auth material. The command source keeps no disk cache at all: its values live only in the in-memory applied-secrets snapshot. There is no plaintext write branch. `encrypted_cache.enabled: false` means **memory-only** — it disables disk persistence entirely; it never re-enables plaintext.
2. **Legacy plaintext is destroyed on first read.** A legacy plaintext cache (e.g. `~/.hermes/cache/bws_cache.json`) is re-encrypted and removed on first read — including in memory-only mode — for every source that ever had one. Verify no plaintext cache file exists after any run.
3. **No secret values in output.** Secret-source error, remediation-hint, warning, and conflict lines are masked before reaching stderr — for every source. `RedactingFormatter` masks opaque credential values in all log output (shape-based regex + exact-value pass). If you see a raw secret value in any status line or log, that is a regression — stop and fix, do not route around.
4. **No credentials in child environments by name.** `BWS_ACCESS_TOKEN` (under its exact configured name), `OP_SERVICE_ACCOUNT_TOKEN`, `OP_CONNECT_TOKEN`, `OP_SESSION_*`, and every `*_PASSWORD` are stripped from spawned children on every surface: terminal (`build_subprocess_env`) and non-terminal (browser worker, ACP executor, computer-use driver, TUI/Node host). The only children that receive auth material are the source CLIs themselves — `bws` and `op` — explicitly, never by inheritance. The command helper is the deliberate exception: it is the user's own configured command and runs with the user's shell environment by design, but its stderr is discarded, the requested key travels only as `HERMES_SECRET_KEY` data (never interpolated into the command string), and everything it emits joins the applied-secrets snapshot below.
5. **Applied secrets are stripped from children by provenance, not just by name** (#77193). Every value present in the per-home applied-secrets snapshot (`get_secret_source_values`) is removed from spawned children regardless of variable-name shape — `DATABASE_URL`, `FOO`, or any arbitrary 1Password item key resolve to nothing in a child's environment. Only `env_passthrough`-registered variables survive, deliberately.
6. **Secret values are redacted before they reach the model provider** (#77198). Applied secret values are masked in tool-result content (`make_tool_result_message`), in the pre-send sanitizers (`redact_sensitive_text` → context engine + chat-completion helpers), and in terminal output (`redact_terminal_output` → terminal tool + process registry) before anything is sent to the provider.
7. **The 1Password integration ships the same posture** (#77168). The 1Password cache is encrypted-only (`~/.hermes/cache/op_cache.enc.json`), and `OP_SERVICE_ACCOUNT_TOKEN`, `OP_CONNECT_TOKEN`, and `OP_SESSION_*` are stripped from spawned children the same way. The `op` child receives auth material through an explicit allowlist — never a copy of the full environment.
8. **Fail-closed read scope.** Credential-shaped environment reads route through `agent.secret_scope.get_secret()`. The credential-read audit (#77031) documents every read path; any new env read of credential-shaped names must go through the gate.
9. **The gate is tested.** The end-to-end no-exfiltration test (`tests/test_secrets_exfiltration.py`) pins invariants 3–7: a loaded secret's name and value never surface in stdout, stderr, or formatted output, and the provider-bound message carries the mask, never the value. Run it as part of any change to any secret source.

## Quick Reference

| Task | Command |
|---|---|
| Bitwarden interactive setup | `hermes secrets bitwarden setup` |
| Bitwarden non-interactive setup | `hermes secrets bitwarden setup --access-token "$BWS_ACCESS_TOKEN" --server-url https://vault.bitwarden.eu --project-id <uuid>` |
| Bitwarden status / validation | `hermes secrets bitwarden status` |
| Bitwarden rotate token (masked prompt) | `hermes secrets bitwarden token` |
| Bitwarden rotate token (non-interactive) | `hermes secrets bitwarden token --access-token 0.…` |
| Bitwarden dry-run sync | `hermes secrets bitwarden sync` |
| Bitwarden apply to current shell | `hermes secrets bitwarden sync --apply` |
| Bitwarden install binary only | `hermes secrets bitwarden install` |
| Bitwarden disable (keeps config) | `hermes secrets bitwarden disable` |
| 1Password interactive setup | `hermes secrets onepassword setup` |
| 1Password non-interactive setup | `hermes secrets onepassword setup --account my.1password.com --token-env OP_SERVICE_ACCOUNT_TOKEN --token "$OP_SERVICE_ACCOUNT_TOKEN"` |
| 1Password status / validation | `hermes secrets onepassword status` |
| 1Password map var → reference | `hermes secrets onepassword set ENV_VAR "op://Vault/Item/field"` |
| 1Password remove mapping | `hermes secrets onepassword remove ENV_VAR` |
| 1Password rotate service-account token | `hermes secrets onepassword token` |
| 1Password dry-run sync | `hermes secrets onepassword sync` |
| 1Password apply to current shell | `hermes secrets onepassword sync --apply` |
| 1Password disable (keeps config) | `hermes secrets onepassword disable` |
| Command source | `secrets.command.enabled: true` + `secrets.command.command: "<helper>"` in `~/.hermes/config.yaml` |
| Config keys | `secrets.bitwarden.*`, `secrets.onepassword.*`, `secrets.command.*` in `~/.hermes/config.yaml` |
| Bootstrap tokens | `BWS_ACCESS_TOKEN` / `OP_SERVICE_ACCOUNT_TOKEN` in `~/.hermes/.env` |

(`op` and `1password` are accepted aliases for `onepassword`.)

## Procedure

### 1. Bitwarden setup

1. In the Bitwarden web app: create a machine account, grant it Read access to a project, create a never-expiring (or dated) access token (starts with `0.`). Bitwarden shows the token once — copy it immediately.
2. Run `hermes secrets bitwarden setup`. It installs the pinned `bws` binary (SHA-256 verified) into `~/.hermes/bin/`, prompts for the token (hidden input), records the region (`BWS_SERVER_URL`), lists projects, test-fetches, and flips `enabled: true`.
3. Confirm with `hermes secrets bitwarden status`.

### 2. 1Password setup

1. Install the official 1Password CLI (`op`) and authenticate: a **service-account token** for headless boxes (exported as `OP_SERVICE_ACCOUNT_TOKEN` in `~/.hermes/.env`), or a desktop/interactive session (`OP_SESSION_*`). Verify with `op whoami`.
2. Run `hermes secrets onepassword setup`. It verifies `op` is on `PATH`, records account/token settings, checks for an active session, and flips `secrets.onepassword.enabled: true`.
3. Map each env var to an `op://vault/item/field` reference: `hermes secrets onepassword set OPENAI_API_KEY "op://Private/OpenAI/api key"`.
4. Confirm with `hermes secrets onepassword sync` (dry-run) and `hermes secrets onepassword status`.

### 3. Command source setup

1. Configure the helper in `config.yaml` — never in `.env` (the command is configuration; `.env` holds values):

   ```yaml
   secrets:
     command:
       enabled: true
       command: "cat /run/user/1000/hermes-secrets.env"
   ```

2. The helper runs once at startup via `/bin/sh -c`, prints `KEY=VALUE` lines on stdout, and must be fast and non-interactive (hard 3s timeout; 1 MiB output cap). POSIX-only — on Windows the source reports itself unconfigured and startup continues.

### 4. Rotation — user action only

**Rotating any secret source's credential is exclusively a user action.** The agent never performs it and never asks for the token value. Rotation happens in the source's own UI/CLI and in the user's terminal (hidden input) — the agent has no legitimate role in either step.

The agent's role is to **instruct and verify**. Rotation is routine posture maintenance: the bootstrap credential unlocks every applied secret, so rotating it on a schedule — and immediately if it may have leaked — keeps the guarantee airtight.

#### Bitwarden

- Instruct the user to revoke the old machine-account access token in the web app, create a new one, and run `hermes secrets bitwarden token` in their terminal, pasting the new value when prompted. Do not proceed with anything that depends on fresh secrets until they confirm this is done.
- **Clipboard discipline during rotation.** The new token should go straight from creation to the terminal: create it in the Bitwarden web app, copy it to the clipboard, paste it into the `hermes secrets bitwarden token` prompt, and **do not save it anywhere else in between** — no notes app, no file, no chat, no screenshot. The clipboard is a temporary staging area, not storage. If the paste fails, re-copy from the web app rather than retyping the token.
- After the user reports rotating, verify with `hermes secrets bitwarden status` (token present and valid) — but the token VALUE never enters the conversation. If the user pastes a token value into chat, do not repeat it, log it, or store it; tell them to run `hermes secrets bitwarden token` themselves and delete the message.

`hermes secrets bitwarden token` probes the new token against Bitwarden **before** writing anything — a rejected token leaves the current `.env` untouched. On success it stores the token, clears fetch caches, and warns if the configured project is invisible to the new machine account.

#### 1Password

- Instruct the user to revoke the old service-account token in the 1Password web app (or `op`), create a new one, and run `hermes secrets onepassword token` in their terminal, pasting the new value when prompted. The same clipboard discipline applies: token straight from creation to the prompt, **do not save it anywhere else in between**.
- `hermes secrets onepassword token` validates the new token with `op whoami` **before** writing anything — a rejected token leaves the current `.env` untouched. Desktop/interactive sessions rotate by signing in again (`op signin`); the new `OP_SESSION_*` values flow to the `op` child through the same allowlist.
- The token VALUE never enters the conversation — same rule as Bitwarden.

#### Command source

- There is nothing to rotate inside Hermes: the helper is the user's own command, and any credential it reads lives in the user's store (KeePassXC, GNOME Keyring, `pass`, a tmpfs env file, …). Rotation happens at that store, user action only.
- If the helper itself changes, the user updates `secrets.command.command` in `config.yaml` and restarts — the agent never edits the helper or handles values it resolves.

### 5. Verification after any change

1. Per-source status: `hermes secrets bitwarden status` / `hermes secrets onepassword status` — token presence, binary version, project/account, references all correct.
2. Plaintext check: `ls ~/.hermes/cache/bws_cache* ~/.hermes/cache/op_cache*` — must show only `*.enc.json` (or nothing in memory-only mode). Any plaintext cache file is a regression.
3. Child-env check: spawn a probe child and confirm `BWS_ACCESS_TOKEN`, `OP_SERVICE_ACCOUNT_TOKEN`, `OP_CONNECT_TOKEN`, `OP_SESSION_*`, every `*_PASSWORD`, and every applied secret value (under any name) are absent from its environment, except `env_passthrough`-registered variables (see Pitfalls for the exact check).
4. If touching the integration code: run the no-exfiltration gate and the per-source secret-source test modules.

## Pitfalls

- **Never echo a token.** Do not paste `BWS_ACCESS_TOKEN` or `OP_SERVICE_ACCOUNT_TOKEN` values into chat, logs, or issue reports. Use the masked rotation path.
- **Region mismatch masquerades as rejection (Bitwarden).** An EU token hitting the US identity endpoint fails as `invalid_client`. Check `secrets.bitwarden.server_url` before assuming revocation.
- **`op` failures usually mean auth, not config.** `op read failed` after a token change means the new token wasn't stored or the session is stale — run `hermes secrets onepassword status`, then `hermes secrets onepassword token` (service account) or `op signin` (desktop session). References are validated `op://`-shaped and passed after `--`, so a crafted value can't become an `op` flag.
- **`bws` must be pinned, not "latest".** Hermes downloads a pinned version (v2.0.0 at time of writing) with checksum verification. If you need a newer version, that is a repo PR, not a runtime auto-upgrade.
- **Do not store the bootstrap tokens as secrets inside the project.** Hermes refuses to overwrite its own `BWS_ACCESS_TOKEN` / `OP_SERVICE_ACCOUNT_TOKEN` even with `override_existing: true` — the secret is silently skipped.
- **A failed cache write must never block a fetch.** The encrypted-cache write is best-effort by design. A cache failure surfaces as a warning, never as a startup failure.
- **Memory-only mode still destroys legacy plaintext.** `encrypted_cache.enabled: false` never reads or writes plaintext; it still removes leftover plaintext caches. Do not "temporarily" re-enable plaintext — there is no such option.
- **The child-env scrub is provenance-based, not name-based.** Any value from the applied-secrets snapshot is stripped from children even under an unrelated name (`DATABASE_URL`, `FOO`, arbitrary item keys). Only `env_passthrough`-registered variables survive. When auditing child environments, check values, not just names.
- **The command helper's stderr is discarded on purpose** — vault-CLI diagnostics can carry secret material. To debug a failing helper, run it manually in a shell; Hermes logs only structured fields (exit code / signal / errno), never the command string or helper output.
- **Secret-source state is per-home.** Caches, the applied-secrets snapshot, and profile resolution are scoped to `HERMES_HOME`: a multiplexed profile resolves only its own `.env` plus global-safe values — never a sibling profile's secrets.
- **A raw secret value in tool results, sanitized context, or terminal output is a provider-egress regression.** If a secret value appears anywhere the provider could see it, quarantine the output, reproduce, and file a fix against the redaction path (`redact_sensitive_text` / `redact_terminal_output` / `make_tool_result_message`).
- **If a raw secret value appears in any status line, warning, or log: that is a security regression in the masking layer, not a cosmetic issue.** Quarantine the output, reproduce, and file a fix against `agent/redact.py` / `hermes_cli/env_loader.py`.

## Verification

- [ ] `hermes secrets bitwarden status` reports valid token + project + region; `hermes secrets onepassword status` reports valid auth + references.
- [ ] Only `*.enc.json` caches (or nothing) exist under `~/.hermes/cache/`; no plaintext cache file from any source.
- [ ] A spawned child process (terminal and non-terminal surfaces) sees neither `BWS_ACCESS_TOKEN`, `OP_SERVICE_ACCOUNT_TOKEN`, `OP_CONNECT_TOKEN`, `OP_SESSION_*`, any `*_PASSWORD`, nor any applied secret value under an arbitrary name, unless explicitly `env_passthrough`-registered.
- [ ] A simulated fetch error/warning emits masked output (`***`) — no raw secret values on stderr.
- [ ] `tests/test_secrets_exfiltration.py` (no-exfiltration gate, wire-path assertions included) and the per-source secret-source test modules pass.
