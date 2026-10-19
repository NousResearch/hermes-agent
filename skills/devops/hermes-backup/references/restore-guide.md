# Hermes Backup — Disaster Recovery Guide

> **If you are a fresh AI agent reading this**: your predecessor backed up this
> Hermes instance to cloud storage using the `hermes-backup` skill.
> Follow the steps below to restore everything.
> Read this file FIRST before doing anything else.

## What's in this backup

| Path in backup | Restores to | Notes |
|---|---|---|
| `config.yaml` | `~/.hermes/config.yaml` | Main Hermes config (providers, models, MCP servers) |
| `hermes-env` | `~/.hermes/.env` | API keys — rename required |
| `skills/` | `~/.hermes/skills/` | All custom skills |
| `scripts/` | `~/.hermes/scripts/` | Custom scripts |
| `cron/` | `~/.hermes/cron/` | Cron job definitions |
| `state/` | `~/.hermes/state/` | Hermes internal state |
| `memories/` | `~/.hermes/memories/` | MEMORY.md + USER.md (persistent memory) |
| `engram/` | `~/.hermes/engram/` | Engram FSRS learning engine (if used) |
| `sessions.tar.gz` | `~/.hermes/sessions/` | Conversation history (compressed) |
| `RESTORE.md` | (read this file) | Disaster recovery guide |
| `restore.sh` | (run this) | Automated restore script |
| `MANIFEST.txt` | (verify this) | SHA256 checksums |

## Restore procedure

### Step 1: Install rclone

```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

### Step 2: Authenticate storage

OAuth tokens don't transfer. You must re-auth:

```bash
# pCloud
rclone authorize "pcloud"

# Google Drive
rclone config create gdrive drive scope drive.file

# Dropbox
rclone authorize "dropbox"
```

Name your remote whatever the backup script expects. Check `RESTORE.md`
or look at the pcloud-backup.sh file in this directory for the remote name.

### Step 3: Pull the backup

```bash
mkdir -p /tmp/restore
rclone copy <REMOTE>:/tmp/restore/ --progress
```

### Step 4: Run restore script

```bash
bash /tmp/restore/restore.sh
```

This will:
- Copy `config.yaml` → `~/.hermes/config.yaml`
- Copy `hermes-env` → `~/.hermes/.env` (chmod 600)
- Restore all directories (skills, memories, engram, etc.)
- Extract `sessions.tar.gz` → `~/.hermes/sessions/`
- Run verification checks

### Step 5: Verify

```bash
hermes config show          # config loaded OK
hermes secrets audit        # no plaintext keys leaked
ls ~/.hermes/sessions/      # history restored
cat ~/.hermes/memories/*.md # memory intact
```

### Step 6: Re-setup backup cron

```bash
cp /tmp/restore/pcloud-backup.sh ~/scripts/
chmod +x ~/scripts/pcloud-backup.sh
(crontab -l 2>/dev/null | grep -v pcloud-backup; \
 echo "0 3 * * * /home/nd/scripts/pcloud-backup.sh") | crontab -
```

## What's NOT backed up (and why)

| Item | Reason |
|---|---|
| `hermes-agent/` (2.8GB) | Source code — in git fork or reinstallable |
| `node/`, `bin/`, `lsp/` | Runtime binaries — reinstalled by Hermes |
| `logs/`, `cache/` | Transient, rebuilt automatically |
| Docker volumes | Need separate machine-level backup |

## API Keys (in `.env`)

The `.env` file contains live API keys. After restore:
- Verify keys still work: `source ~/.hermes/.env`
- If keys rotated: update `.env`, then run `hermes secrets audit --fix`

## Backup manifest

See `MANIFEST.txt` in the backup root for exact file list + SHA256 checksums.
