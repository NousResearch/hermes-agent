---
sidebar_position: 15
sidebar_label: "Encryption at Rest"
title: "Encryption at Rest"
description: "Encrypt credentials and conversation history on disk with hermes encrypt"
---

# Encryption at Rest

Hermes Agent can keep the sensitive files under `~/.hermes/` **encrypted on
disk** so a stolen laptop, a leaked backup, or a decommissioned VPS volume does
not expose your API keys, OAuth tokens, or conversation history.

Encryption is **opt-in and off by default** — nothing changes until you run
`hermes encrypt enable`.

## What it protects (and what it doesn't)

Encryption at rest protects data on a **cold disk** — when the machine is off
or the files are copied elsewhere (a backup, a synced folder, a disposed
drive).

It does **not** protect a running agent. The agent needs cleartext to work, so
while Hermes is running the keys live in process memory; a malicious plugin or
skill, or anyone with access to your logged-in account, can still read them.
This is defense-in-depth on top of file permissions and OS-level isolation —
not a replacement for either. See `SECURITY.md` for the full threat model.

## Runtime isolation — pair with OpenShell

Encryption at rest and runtime isolation cover the two different halves of the
problem:

| Threat | Covered by |
|---|---|
| Cold disk — stolen laptop, leaked backup, decommissioned VPS volume, synced folder | **Encryption at rest** (this feature) |
| Running process — a malicious skill/plugin, an escaped shell command, exfiltration | **A whole-process sandbox** |

For the second half, run Hermes inside a sandbox. Hermes's `SECURITY.md` (§2.2)
recommends **[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)** — it
confines the agent with filesystem, network-egress, and syscall policy, and
keeps credentials off the sandbox filesystem entirely. The two features compose:
encryption keeps your `~/.hermes` safe when the machine is off, OpenShell keeps
the agent contained while it runs.

`hermes encrypt status` and `hermes doctor` report whether Hermes is currently
running inside an OpenShell sandbox.

## What gets encrypted

| Data | File(s) | Covered by |
|---|---|---|
| API keys | `~/.hermes/.env` | `encrypt_credentials` |
| OAuth tokens, credential pool | `auth.json`, `.anthropic_oauth.json`, `auth/google_oauth.json`, `mcp-tokens/*` | `encrypt_credentials` |
| Conversation history | `state.db` | `encrypt_databases` |
| Kanban tasks | `kanban.db` (+ per-board DBs) | `encrypt_databases` |

Database encryption uses [SQLCipher](https://www.zetetic.net/sqlcipher/);
full-text session search keeps working on the encrypted database.

## Quick start

```bash
# Install the encryption dependencies and encrypt credential files.
hermes encrypt enable

# Also encrypt the conversation-history databases.
hermes encrypt enable --databases

# See what is encrypted and how the key is held.
hermes encrypt status
```

`enable` makes a plaintext backup of every file under
`~/.hermes/.encryption/backup/`, verifies each file round-trips, and only then
flips the config flag. If anything fails it restores from backup and aborts.
Remove those backups with `hermes encrypt clean-backups` once you have verified
encryption — `hermes encrypt status` reports how many remain.

## Choosing a key source

`hermes encrypt enable --key-source <source>`:

- **`keyring`** (default) — a random key is stored in the OS keyring (Windows
  Credential Manager / macOS Keychain / Linux Secret Service). Unlocks
  automatically with your OS login, no prompt. Best for a personal desktop.
- **`passphrase`** — the key is derived from a passphrase with Argon2id. Works
  on headless servers. On a VPS / gateway, set `HERMES_ENCRYPTION_PASSPHRASE`
  so the agent can unlock at startup (see below).
- **`keyfile`** — a random key in `~/.hermes/.encryption/keyfile`. Zero-friction
  for automation, but the weakest option: the key sits next to the data.

> **Lose the key, lose the data.** There is no backdoor. In passphrase mode,
> `enable` offers a one-time **recovery code** — write it down and keep it
> offline. Add or replace one later with `hermes encrypt add-recovery`.

### Passphrase environment variable

On headless hosts, export `HERMES_ENCRYPTION_PASSPHRASE` before starting
Hermes. Two independent protections apply — both on purpose:

1. **Parent process** — at startup the variable is moved out of
   `os.environ` into process memory before anything can spawn a subprocess,
   so no child process — terminal commands, LSP servers, media probes,
   platform scripts — can inherit it. The keystore keeps unlocking from the
   in-memory copy.
2. **Subprocesses** — every spawned shell and code-execution environment
   additionally strips `HERMES_ENCRYPTION_PASSPHRASE` unconditionally (even
   when other credential scrubbing is off), in case the variable was re-set
   later in the process lifetime.

Neither layer alone covers every path; keeping both is deliberate.

## Managing encryption

| Command | Purpose |
|---|---|
| `hermes encrypt status` | Show state, key source, and per-file status |
| `hermes encrypt enable` | Encrypt files and create the keystore |
| `hermes encrypt disable` | Decrypt everything back to plaintext |
| `hermes encrypt rotate-key` | Re-wrap the key (e.g. change passphrase) — does not rewrite data |
| `hermes encrypt rotate-key --full` | Generate a new data key and re-encrypt every sealed artifact (after suspected DEK compromise) |
| `hermes encrypt add-recovery` | Generate a new recovery code |
| `hermes encrypt unlock` | Verify the passphrase / key works |
| `hermes encrypt clean-backups` | Remove plaintext migration backups (use `--older-than N` to keep recent ones) |
| `hermes encrypt read-log` | Decrypt and print an encrypted rotated log segment or session transcript |
| `hermes encrypt sweep-sessions` | Seal plaintext session transcripts left behind by a crashed writer |

### Database re-keying and WAL checkpoints

`rotate-key --full` re-encrypts the SQLCipher main `.db` file along with its
`-wal` and `-shm` sidecars, snapshotting them best-effort under
`.encryption/backup/` so a rollback can restore the consistent triple. With a
live writer (the case when you pass `--force` to bypass the concurrent-Hermes
detector), the OS may hold the sidecars exclusively and the snapshot is not
guaranteed; uncheckpointed WAL transactions can be lost if a verification
failure later triggers a rollback. The recommended operator practice is to
stop the gateway before running `rotate-key --full` on an encrypted database.
The concurrent-Hermes detector is the default guard against this; `--force`
opts out and accepts the risk.

## Configuration

`hermes encrypt` manages these `config.yaml` keys for you — do not edit
`enabled` by hand, the on-disk files must match:

```yaml
security:
  encryption:
    enabled: false
    key_source: keyring        # keyring | passphrase | keyfile
    encrypt_credentials: true
    encrypt_databases: false   # needs the SQLCipher dependency
    encrypt_logs: false        # rotated logs + closed/session transcripts
    argon2:                    # passphrase-mode key-derivation cost
      time_cost: 3
      memory_cost_kib: 131072
      parallelism: 4
```

## Notes

- Once `.env` is encrypted, edit it with `hermes setup` / `hermes secrets`
  rather than a text editor — the file is no longer plaintext.
- Run `hermes encrypt enable --databases` with the gateway stopped: an open
  database cannot be migrated safely.
- Checkpoint snapshots (`~/.hermes/checkpoints/`) are a live git repository and
  are **not** encrypted by this feature; use full-disk encryption if those are
  a concern.
- Plugin-managed databases (for example the memory plugins'
  `memory_store.db` / `retaindb_queue.db`) open their own SQLite connections
  and are **not** covered by `encrypt_databases` in this iteration — only
  `state.db` and `kanban.db` are.
