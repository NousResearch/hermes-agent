---
name: hermes-backup
description: Use when setting up backup/disaster recovery for Hermes, or when recovering from a crash. Supports pCloud, Google Drive, Dropbox, S3-compatible, local, and any rclone-compatible storage.
version: 1.1.0
author: AndyN (@andynguyendk)
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [backup, disaster-recovery, restore, rclone, pcloud]
    related_skills: [hermes-agent]
---

# Hermes Backup

Disaster recovery system for Hermes AI agents. Backs up config, skills, memories, sessions, and scripts to any rclone-compatible storage — then lets a fresh agent self-restore by reading `RESTORE.md` from the backup itself.

## Problem

Hermes stores valuable agent state in `~/.hermes/` — skills learned over months, persistent memory, conversation history. There's no built-in backup. If the machine dies or the install corrupts, everything is lost.

This skill solves that with a backup that **is its own restore manual**.

## How It Works

1. **Backup script** packages Hermes state + restore assets into a staging folder
2. **rclone** syncs to your storage (pCloud, Google Drive, Dropbox, S3, local, 40+ backends)
3. **`RESTORE.md` and `restore.sh`** are stored INSIDE the backup on cloud — a fresh agent can download the backup and read one file to know what to do
4. **Manifest** with SHA256 checksums verifies integrity

## Supported Storage Backends

Any [rclone-compatible storage](https://rclone.org/overview/). The backup script only needs the `REMOTE` variable:

| Storage | `REMOTE` value |
|---|---|
| pCloud | `pcloud:hermes-backup` |
| Google Drive | `gdrive:hermes-backup` |
| Dropbox | `dropbox:hermes-backup` |
| S3-compatible | `s3:my-bucket/hermes-backup` |
| Local USB/dir | `/mnt/usb/hermes-backup` |

Setup rclone first (one-time):

```bash
# Install
curl https://rclone.org/install.sh | sudo bash

# Google Drive
rclone config create gdrive drive scope drive.file

# pCloud
rclone authorize "pcloud"

# Dropbox
rclone authorize "dropbox"

# S3-compatible
rclone config create s3 aws access_key_id XXX secret_access_key YYY
```

## Setup

### 1. Deploy backup assets

```bash
mkdir -p ~/.hermes/backup-assets

# Copy restore files from skill
cp -r $(hermes skills inspect hermes-backup 2>/dev/null || echo "$HOME/.hermes/skills/devops/hermes-backup")/* ~/.hermes/backup-assets/
# OR find the skill manually:
find ~/.hermes -path "*/hermes-backup/scripts" -type d 2>/dev/null | head -1
```

### 2. Create backup script from template

```bash
cp ~/.hermes/backup-assets/templates/pcloud-backup.sh.template ~/scripts/pcloud-backup.sh
# Edit REMOTE to point to your storage
```

### 3. Add cron (daily 3AM)

```bash
(crontab -l 2>/dev/null | grep -v pcloud-backup; \
 echo "0 3 * * * /home/nd/scripts/pcloud-backup.sh") | crontab -
```

### 4. Run once to verify

```bash
~/scripts/pcloud-backup.sh
tail -5 ~/.hermes/logs/pcloud-backup.log
```

## Disaster Recovery

If Hermes is destroyed, a new agent can self-restore:

```bash
# 1. Install rclone + auth your storage
curl https://rclone.org/install.sh | sudo bash
rclone authorize "pcloud"  # or gdrive, dropbox, etc.

# 2. Pull the backup
mkdir -p /tmp/restore
rclone copy pcloud:hermes-backup/ /tmp/restore/

# 3. Read the restore guide
cat /tmp/restore/RESTORE.md

# 4. Run automated restore
bash /tmp/restore/restore.sh
```

No prior knowledge needed — the backup contains everything.

## Backup Contents

| Item | Why it matters | Size (typical) |
|---|---|---|
| `config.yaml` | Provider config, MCP servers, settings | ~17 KB |
| `hermes-env` | API keys (.env, renamed for safety) | ~2 KB |
| `skills/` | Cumulated knowledge — most valuable asset | ~35 MB |
| `memories/` | MEMORY.md + USER.md — persistent memory | ~12 KB |
| `sessions.tar.gz` | Full conversation history | ~40 MB |
| `engram/` | FSRS learning engine state | ~52 KB |
| `restore.sh` | Automated restore script | ~5 KB |
| `MANIFEST.txt` | SHA256 checksums for all files | ~200 KB |

## Common Pitfalls

- **pCloud DNS failures**: `dial tcp: lookup api.pcloud.com:53: server misbehaving`. Transient — retry with `--retries 5`.
- **OAuth expired**: Re-run `rclone authorize "pcloud"` — tokens don't survive crashes.
- **Local storage**: If using a local path, skip rclone — just use `rsync` or `cp -r`.
- **Sessions backup timeout**: For large history, rclone may time out during hash comparison. The script retries up to 3 times.

## Template Configuration

The backup script template (`templates/pcloud-backup.sh.template`) is fully configurable:

```bash
# === CONFIGURATION — customize these ===
REMOTE="pcloud:hermes-backup"           # rclone remote:path
BACKUP_DIRS=(skills scripts cron state career memories engram)  # dirs to back up
CONFIG_FILES=("$HOME/.hermes/config.yaml")
ENV_FILE="$HOME/.hermes/.env"
```

## Verification

```bash
# Check backup integrity
rclone check pcloud:hermes-backup/ /tmp/verify/ --size-only

# Show size and file count
rclone size pcloud:hermes-backup/
rclone lsf pcloud:hermes-backup/ -R --files-only | wc -l
```

## Files

- `references/restore-guide.md` — Full disaster recovery guide (also stored inside backup)
- `scripts/restore.sh` — Automated restore script (also stored in backup)
- `scripts/generate-manifest.sh` — SHA256 manifest generator
- `templates/pcloud-backup.sh.template` — Configurable backup script template
