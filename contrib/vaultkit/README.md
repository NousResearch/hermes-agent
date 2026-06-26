# vaultkit — portable KeePassXC secret vault for agents

A self-contained toolkit for storing secrets in an offline KeePassXC vault and
injecting them into your agent/services at runtime — **without** writing them to
plaintext `.env` files. Built for people who want offline, no-cloud secret
management but **don't necessarily have a NAS or a TPM**.

It is the generic, machine-agnostic version of a setup originally hardcoded to
one box. Nothing here assumes a username, home directory, UID, service list, or
backup target — every path is resolved from config with sane defaults.

## What you get

| Command            | What it does                                                        |
|--------------------|---------------------------------------------------------------------|
| `vault-setup`      | Bootstrap a fresh vault from nothing (wizard). TPM/age optional.     |
| `vault-add KEY`    | Add/update a secret via hidden input (never on the command line).   |
| `vault-import .env`| Bulk-onboard an existing `.env` file into the vault (idempotent).    |
| `vault-unlock`     | Read the vault group into a tmpfs env file **+ a name-only manifest**. |
| `vault-lock`       | Stop services and securely wipe the tmpfs files.                    |
| `vault-inject -- CMD` | Run one command with the secrets in its env (no services needed). |
| `vault-seal-tpm`   | Seal the key file to this machine's TPM (Linux Keychain-equivalent). |
| `vault-breakglass` | **NAS-free** offline recovery: 24-word phrase → age-encrypted bundle. |

## Quick start

```bash
./install.sh                 # checks prereqs, makes scripts executable
vault-setup                  # creates vault + key file + config (wizard)
vault-add OPENAI_API_KEY     # paste the value at the hidden prompt
vault-unlock                 # writes secrets to $XDG_RUNTIME_DIR (tmpfs)
```

Point your services at the env file:

```ini
# in a systemd --user unit
EnvironmentFile=%t/vaultkit-secrets.env
```

(`%t` is `$XDG_RUNTIME_DIR`, i.e. `/run/user/<uid>` — a tmpfs wiped on logout.)

## Prerequisites

- **Required:** `keepassxc-cli` (or snap `keepassxc.cli`), Python 3.8+.
- **Optional:** `age` + `age-keygen` (breakglass recovery), `tpm2-tools` + a TPM
  (sealed key file).
- **Not required:** the Python `cryptography` package. The breakglass key
  derivation uses it when present (C-backed, constant-time) but **falls back to
  a pure-Python X25519 implementation** that produces the *identical* age key,
  so recovery works on a bare machine with nothing but Python + `age` installed.

Everything optional **degrades gracefully** — the kit tells you what's missing
and keeps working without it.

## Snap / Flatpak compatibility (important)

If your `keepassxc-cli` came from **snap** or **flatpak**, it runs sandboxed and
can only reach a limited set of paths. This trips people up because the CLI's
error is cryptic (`Creating KeyFile ... failed: No such file or directory` or
`Permission denied`) even when the file is right there.

The empirically-verified rules for the **snap** `keepassxc.cli`:

| Path                                   | Reachable by snap CLI? |
|----------------------------------------|------------------------|
| `~/vaultkit/...` (non-hidden in $HOME) | ✅ yes                 |
| `~/snap/...`                           | ✅ yes                 |
| `/media/...`, `/mnt/...` (removable)   | ✅ yes (if interface connected) |
| `~/.local/...`, `~/.config/...`, `~/.cache/...` (**hidden**) | ❌ **no** |
| `/tmp/...`, `/run/user/...`, `/etc/...` | ❌ no                  |

**What the kit does about it automatically:**

- It **detects** snap/flatpak packaging (`cli_confinement()`).
- When confined, the **default vault directory** switches from the XDG-hidden
  `~/.local/share/vaultkit` to a **non-hidden `~/vaultkit`** the CLI can reach.
- A **preflight check** (`preflight_paths()`) runs before `db-create` and before
  unlock; if your vault or key file is on an unreachable path it **stops with an
  actionable message** instead of the raw CLI error.

The tmpfs env file and manifest are written by the kit's own Python (not the
keepassxc CLI), so they can safely live on `/run/user/<uid>` even for snap users
— the confinement only constrains the vault + key file.

**Recommendation:** for the least friction, install the **native** package
(`sudo apt install keepassxc`, `sudo dnf install keepassxc`, `brew install
keepassxc`) — it has no sandbox and the XDG-standard paths just work. The kit
supports both; this is only about avoiding surprises.

## The subprocess-leak protection (why the manifest exists)

When secrets are injected into a process's environment, every **child process**
that process spawns normally inherits them too. A blocklist of "known secret
names" can't catch a vault key with an arbitrary name. So `vault-unlock` writes
a sidecar **manifest** (`vaultkit-secrets.keys`) listing the **names** (never the
values) of everything it injected.

A consumer can read that manifest and **default-deny those names** from the
environment it hands to child processes — protection by *provenance* (it came
from the vault) rather than by a hardcoded list. Adding a new secret is then
covered automatically. The Hermes agent integration does exactly this via
`HERMES_VAULT_KEYS_MANIFEST` (point it at this kit's manifest path).

## Configuration

Resolved in order: environment var `VAULTKIT_<KEY>` → config file → default.
Config file: `$XDG_CONFIG_HOME/vaultkit/config` (written by `vault-setup`).

| Key             | Default                                   | Meaning                              |
|-----------------|-------------------------------------------|--------------------------------------|
| `DIR`           | `~/.local/share/vaultkit`                 | vault directory                      |
| `VAULT`         | `<DIR>/vault.kdbx`                         | the KeePassXC database               |
| `KEYFILE`       | `<DIR>/vault.key`                         | on-disk key file (Model B)           |
| `TPM_KEYFILE`   | `$XDG_RUNTIME_DIR/vaultkit-vault.key`     | TPM-unsealed key in tmpfs (optional) |
| `ENVFILE`       | `$XDG_RUNTIME_DIR/vaultkit-secrets.env`   | tmpfs env file services read         |
| `KEYS_MANIFEST` | `$XDG_RUNTIME_DIR/vaultkit-secrets.keys`  | tmpfs name-only manifest             |
| `GROUP`         | `hermes`                                  | vault group to read                  |
| `SERVICES`      | *(empty)*                                 | services to restart after unlock     |
| `CLI`           | `keepassxc-cli`                           | the KeePassXC CLI binary             |

## Breakglass recovery (no NAS required)

```bash
vault-breakglass export                 # prints 24 words + writes breakglass.age
# -> write the 24 words on PAPER. Copy breakglass.age anywhere (it's ciphertext).
```

On a brand-new machine:

```bash
vault-breakglass restore breakglass.age ~/vaultkit-restored
# -> type the 24 words back in; the vault + key file are decrypted and unpacked.
```

The 24-word phrase is your **only** recovery factor; it is never stored with the
bundle. The encrypted `.age` bundle is useless without it — so the two can live
in different places (paper in a safe, ciphertext on a USB stick / cloud).
This is the "breakglass: build for others, recovery factor not co-located"
principle, made NAS-free.

## Onboarding existing `.env` secrets

If you already have a plaintext `.env`, migrate it in one shot:

```bash
vault-import .env                  # add every KEY=value into the vault group
vault-import .env --dry-run        # preview (add / update / skip) without writing
vault-import .env --update         # overwrite entries that already exist
```

- **Idempotent by default:** an entry that already exists is *skipped*, not
  clobbered. Re-run it as often as you like. Use `--update` only when you want
  the `.env` values to win.
- **Values never hit the command line** — each is streamed to `keepassxc-cli`
  on stdin, so nothing leaks into `ps` or shell history.
- Parsing handles `export KEY=val`, single/double quotes, comments, and blank
  lines; empty values and invalid env-var names are reported and skipped; a
  duplicate key uses its **last** occurrence.

Once imported, delete the plaintext `.env` and switch your services to the
tmpfs env file written by `vault-unlock`.

## TPM sealing (optional, Linux) — `vault-seal-tpm`

This is the Linux counterpart to macOS "seal the key file into the login
Keychain". It uses `systemd-creds encrypt --with-key=tpm2`, so the key file can
only be recovered on **this machine** (bound to its TPM) and never sits on disk
in plaintext after sealing.

Recommended first-run order (matches `vault-setup`'s "key file" auth model):

```bash
vault-setup                         # 1. create vault + key file
vault-breakglass export recovery.age   # 2. SAVE the 24-word phrase on paper FIRST
vault-seal-tpm seal --print-sudoers    # 3. seal the key file to the TPM
# (optionally) vault-seal-tpm seal --shred-source  # remove the plaintext key
```

> **Save your recovery phrase before you shred.** Once the plaintext key file is
> gone, the key exists only as the TPM blob (this machine) and, at runtime, in
> tmpfs. The 24-word breakglass phrase is your only way back on a new machine.

At boot, a tiny root oneshot unseals the key into tmpfs:

```bash
vault-seal-tpm unseal               # decrypt the blob -> tmpfs key file
vault-seal-tpm teardown             # wipe the tmpfs key (on lock/shutdown)
vault-seal-tpm status               # report TPM availability + seal state
```

**Gotcha — do NOT prefix these with `sudo` yourself.** `vault-seal-tpm`
*self-elevates* (it re-execs under sudo for the TPM ops) and forwards the
resolved, absolute paths to the root side. If you run `sudo vault-seal-tpm …`
directly, you start already-root, the path-forwarding is skipped, and `HOME`
becomes `/root` — so it looks for the vault under `/root/.local/share/vaultkit`
instead of yours. Run the commands as your normal user; you'll get one sudo
password prompt. (The kit *also* re-anchors paths to the owning user via
`$SUDO_USER` when it does land as root, so the boot oneshot — which runs as root
with no user env — resolves correctly too.)

`--print-sudoers` emits a narrow NOPASSWD rule scoped to exactly the `unseal`
and `teardown` invocations, so your boot oneshot needs no password and can do
nothing else. Write it to `/etc/sudoers.d/vaultkit-tpm-unseal` (chmod 440).

A CI round-trip test lives at `tests/test_tpm_roundtrip.sh`; it skips cleanly on
machines without a TPM or passwordless root.

## Security notes

- Secrets live only in the vault (encrypted at rest) and in the tmpfs runtime
  files (0600, RAM-only). Nothing is written to a plaintext `.env`.
- The manifest contains **names only**, never values.
- `vault-add` and password prompts use hidden input — values never hit the
  command line or shell history.
- If `$XDG_RUNTIME_DIR` is not tmpfs, the kit **warns** you (secrets could touch
  disk). On a normal logind session it's `/run/user/<uid>`, which is tmpfs.

## Layout

```
portable-kit/
  install.sh
  lib/
    vaultkit_lib.py        # machine-agnostic shared core
    bip39_english.txt      # canonical BIP39 wordlist (sha256-verified)
  bin/
    vault-setup vault-add vault-import vault-unlock vault-lock
    vault-inject vault-seal-tpm vault-breakglass
  tests/
    test_import_roundtrip.sh   # .env onboarding (offline, no root)
    test_tpm_roundtrip.sh      # TPM seal/unseal (skips without a TPM)
    test_breakglass_fallback.py test_breakglass_tar_safety.py
    test_confinement.py test_kit_e2e.sh
```
