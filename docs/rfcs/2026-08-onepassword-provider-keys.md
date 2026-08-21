# LLM provider API keys via 1Password — no secrets on disk in Hermes home

Date: 2026-08-15
Status: proposed (revised — see "Alternatives considered" for the original
macOS Keychain draft this supersedes)

## Problem

`~/.hermes/.env` stores every credential Hermes knows about — including LLM
provider API keys — as plaintext on disk. Goal: **no secret value is ever
written to disk anywhere under Hermes home**, for LLM provider keys, using
the user's existing 1Password subscription as the source of truth.

## Why 1Password over macOS Keychain

Both can reach "zero secret bytes in Hermes home." The deciding factors were
sync (1Password syncs across machines; Keychain is single-Mac) and reuse
(1Password's resolver already exists in this codebase — see
`agent/secret_sources/onepassword.py`). See "Alternatives considered" below
for the full comparison.

## What already exists (and works as-is)

`agent/secret_sources/onepassword.py` + `hermes_cli/onepassword_secrets_cli.py`
already implement: map an env var name to an `op://vault/item/field`
reference in `config.yaml` (`secrets.onepassword.env`), and
`load_hermes_dotenv()` resolves those references into `os.environ` at every
process start by shelling out to `op read`. This needs **no code changes**
to function. Two things about its *defaults* work against the "zero disk"
goal and need explicit configuration (not code changes) to avoid:

1. **Disk cache.** Resolved values are cached in plaintext (0600, not
   encrypted) at `<hermes_home>/cache/op_cache.json` with a default 5-minute
   TTL (`secrets.onepassword.cache_ttl_seconds: 300`,
   `agent/secret_sources/onepassword.py:404,586-609`,
   `agent/secret_sources/_cache.py:94-216`). Setting
   `cache_ttl_seconds: 0` disables both the in-process and on-disk cache
   layers (`_cache.py:136,174`: both `read`/`write` no-op when
   `ttl_seconds <= 0`) — every process start does a live `op read` instead.
2. **Service-account token.** `hermes secrets onepassword setup --token …`
   / `hermes secrets onepassword token` persist
   `OP_SERVICE_ACCOUNT_TOKEN` into `~/.hermes/.env`
   (`hermes_cli/onepassword_secrets_cli.py:162,347`) — a secret, in Hermes
   home. Service accounts are a Teams/CI feature anyway; the personal/
   premium-subscription equivalent is the 1Password 8 desktop app's
   **"Integrate with 1Password CLI"** setting (Settings → Developer),
   which lets `op` authenticate via biometric-gated IPC to the already
   signed-in app. `cmd_setup` already handles this without a token: if no
   `--token` is passed and none is in the environment, it checks for an
   active `op` session (`_op_whoami`, `onepassword_secrets_cli.py:169-177`)
   and completes setup with **nothing written to `.env`**.

So the "zero disk" setup is:

```
# 1Password app → Settings → Developer → enable "Integrate with 1Password CLI"
hermes secrets onepassword setup          # no --token
# then hand-edit ~/.hermes/config.yaml: secrets.onepassword.cache_ttl_seconds: 0
hermes secrets onepassword set ANTHROPIC_API_KEY "op://Private/Anthropic/credential"
```

(the key itself is created as a vault item in the 1Password app, same as any
other 1Password item — not through Hermes.)

## What actually needs code changes

Two gaps, both in `hermes_cli/web_server.py`, found by tracing the desktop
Settings "API keys" tab end-to-end:

1. **`GET /api/env` shows a 1Password-resolved key as "not set."**
   `_get_env_vars_sync()` (`web_server.py:7207-7264`) computes `is_set` /
   `redacted_value` purely from `env_on_disk = load_env()` — i.e. `.env`
   contents. A key resolved into `os.environ` by the onepassword source but
   never written to `.env` reads as unset in Settings, which would confuse
   the user (or worse, invite them to re-paste it, landing it in plaintext
   `.env`). Fix: fall back to `os.environ.get(var_name)` when the var isn't
   in `.env` — the web server process itself already went through
   `load_hermes_dotenv()` at its own startup, so the resolved value is
   already sitting in its `os.environ`.
2. **`PUT /api/env` would silently shadow a 1Password-mapped key with a
   plaintext copy.** `save_env_value()` already has a precedent for "this
   key is managed elsewhere, refuse the plain write" — the existing
   `managed_scope.is_env_managed(key)` guard (`hermes_cli/config.py:
   3999-4008`). Add the same shape of guard for secret-source-mapped keys:
   if `key` appears in an *enabled* `secrets.onepassword.env` mapping,
   `save_env_value` refuses with a clear message pointing at
   `hermes secrets onepassword set`, instead of writing `.env`. Functionally
   the 1Password source would re-win at next startup anyway
   (`override_existing: true` by default), but the plaintext value would
   still have touched disk in the meantime — this guard is what actually
   makes the "never" in "no secret ever touches disk" true.

Desktop UI treatment for a mapped key: show it as "Managed via 1Password"
(from the same `os.environ`-sourced `is_set`, redacted value for display),
with editing disabled — direct the user to the 1Password app + `hermes
secrets onepassword set/remove`. No new UI for creating/editing vault items
or mappings from the desktop app; that's out of scope (see below).

## Scope

Same eligible-key set as the original draft: `provider_catalog()` entries
with `tab == "keys"`, plus custom endpoint keys
(`HERMES_CUSTOM_<slug>_API_KEY`). Keys the user hasn't mapped in 1Password
keep working exactly as today (`.env`), so this can be adopted per-key,
incrementally — consistent with wanting to map keys "one by one... not all
at one go."

## Explicitly out of scope

- Building a "write to 1Password" flow (creating/editing vault items, or
  managing the `env` mapping) from the desktop Settings UI. Keys are
  created in the 1Password app and mapped via `hermes secrets onepassword
  set`, same as `sync`/`status` are CLI-only today.
- Any change to the Bitwarden or `command` secret sources.
- Non-macOS OS keystores.
- A `--cache-ttl-seconds` flag on `hermes secrets onepassword setup` (the
  one-time hand-edit of `config.yaml` is a single line and this is a
  one-time setup step, not a repeated operation).

## Known limitation to verify during implementation

`op` CLI desktop-app integration may prompt for biometric approval the
first time a given calling process is seen, and 1Password's own vault
access may be briefly unavailable if the desktop app isn't running/unlocked
— an unattended gateway/cron run in that state gets a warning and proceeds
without that credential (per the existing "never block startup" contract in
`agent/secret_sources/base.py`), not a crash, but the model call using that
key would fail for that run. Worth confirming in practice whether repeated
approval prompts occur across Hermes's various spawn paths (CLI, gateway,
desktop-spawned backend) before relying on this for anything unattended.

## Testing

- `web_server.py` unit tests: `GET /api/env` reports `is_set`/redacted value
  for a var present in `os.environ` but absent from `.env`.
- `PUT /api/env` / `save_env_value` unit tests: a save attempt for a key
  with an active `secrets.onepassword.env` mapping is refused with a clear
  error (mirroring existing `managed_scope` guard tests in
  `tests/hermes_cli/test_credential_lifecycle.py`); a save for an unmapped
  key is unaffected.
- No change needed to the onepassword source's own tests — it's used as-is.

## Alternatives considered: macOS Keychain (original draft)

The first draft of this RFC proposed a new `hermes_cli/keychain_store.py`
wrapping the `security` CLI, with `save_env_value`/`get_env_value`/
`remove_env_value` routing eligible keys to the macOS Keychain instead of
`.env`. That reaches "zero secrets in Hermes home" too, with no caching
workaround needed and no dependency on the 1Password desktop app being
unlocked — better suited to a single Mac and to unattended/background
gateway runs. It was set aside in favor of 1Password for multi-machine sync
and because the 1Password resolver already exists, making this revision
far smaller. If cross-machine sync turns out not to matter in practice, or
the desktop-integration prompt/reliability issue above is a real problem
for unattended runs, this is the fallback.
