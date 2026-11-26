# Termux ↔ Shared Storage Bridge Sync

Termux's `~/.hermes/` lives in Android private storage (`/data/data/com.termux/files/home/.hermes/`),
invisible to Syncthing. We bridge it to `/storage/emulated/0/HermesSync/` (shared storage).

## Prerequisites

```bash
pkg install termux-api cronie termux-services rsync
# Also install the Termux:API companion APK on Android (from F-Droid)
```

**`rsync` is not pre-installed in Termux** — it must be explicitly installed.

Without `termux-api` + APK, Termux **cannot read** files Syncthing writes to shared storage.

## One-Time Setup

Create sync directories via **Android file manager** (not Termux — see FUSE caveat below):

```
/storage/emulated/0/HermesSync/
  ├── memories/
  ├── skills/
  └── sessions/
```

> ⚠️ **FUSE caveat**: `~/storage/shared/` in Termux is a separate FUSE view. Files written there by Termux MAY be visible to Syncthing (observed working on Android 13+), but files Syncthing writes are NEVER visible to Termux via `~/storage/shared/`. Always use `/storage/emulated/0/` as the canonical path — it works for both read and write with `termux-api`.

## Bidirectional Sync Script

Create `~/.hermes/scripts/sync_hermes.sh` (see `../scripts/sync_hermes.sh` in this skill).

```bash
chmod +x ~/.hermes/scripts/sync_hermes.sh
```

> 🚫 **DO NOT use `rsync --delete` in bidirectional sync.** With `-u` (update-only), both sides can hold different supersets of files. `--delete` will remove files on the destination that don't exist on the source — in a bidirectional sync, this means whichever side syncs first deletes files the other side has. This caused verified data loss (DIALOGUE_LOG.md deleted from both sides). Use `rsync -rtu` only — it preserves newer files and adds missing ones without ever deleting.

## Cron (every 5 minutes)

```bash
(crontab -l 2>/dev/null; echo "*/5 * * * * bash /data/data/com.termux/files/home/.hermes/scripts/sync_hermes.sh") | crontab -
crond
```

Verify: `crontab -l` and `ps aux | grep crond`

> To survive Termux being killed: `termux-wake-lock` and `Termux:Boot` plugin for auto-start on reboot.
