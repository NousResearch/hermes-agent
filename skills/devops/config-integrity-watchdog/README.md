# config-integrity-watchdog

A tamper-evident integrity system for `~/.hermes/config.yaml` using git-backed fingerprinting.

## After Install

Hermes automatically syncs this skill to `~/.hermes/skills/devops/config-integrity-watchdog/` on next startup.

**Initial seal** (run once after install to establish baseline):
```bash
python3 ~/.hermes/skills/devops/config-integrity-watchdog/scripts/seal.py
```

## Why git-backed?

The existing `.sha256` sidecar file is mutable — any process that can write `config.yaml` can also overwrite the sidecar, masking the tampering. By committing the integrity log to the dotfiles git repository, the fingerprint gains the tamper-evidence of git history: a malicious process without git commit credentials cannot silently forge an entry.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `HERMES_CONFIG` | `~/.hermes/config.yaml` | Path to the config file to protect |
| `HERMES_DOTFILES_DIR` | `~/Dev/dotfiles` | Path to the dotfiles git repository |

## Cron job setup

Replace or augment the existing Config Integrity Watchdog cron job:

**verify** (runs every hour):
```bash
python3 ~/.hermes/skills/devops/config-integrity-watchdog/scripts/verify.py
```

**restore** (runs on verify failure):
```bash
python3 ~/.hermes/skills/devops/config-integrity-watchdog/scripts/restore.py
```

## Graceful fallback

If the dotfiles directory is not a git repository, `seal.py` still writes the log file but skips the git commit and prints a warning. Verification still works (hash comparison), but the log itself is not tamper-evident in that mode.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | Tampered or error |
| 2 | Log file has uncommitted changes (log tampering) |
| 3 | No baseline sealed yet |
