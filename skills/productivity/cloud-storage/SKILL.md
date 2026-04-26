---
name: cloud-storage
version: 1.0.0
description: |
  Universal cloud storage integration powered by rclone. Works with OneDrive,
  Google Drive, Dropbox, pCloud, Box, Backblaze B2, Wasabi, S3-compatible
  providers, WebDAV, SFTP, and 70+ other backends. No per-service app
  registrations required for major providers — rclone handles its own OAuth
  clients.
author: subimpact
license: MIT
source: https://github.com/subimpact/hermes-agent/tree/main/skills/productivity/cloud-storage
metadata:
  hermes:
    tags: [Cloud, Storage, rclone, OneDrive, Google-Drive, Dropbox, Backblaze, S3, Sync]
    related_skills: [google-workspace]
---

# Cloud Storage

A universal cloud storage skill for Hermes Agent. Upload, download, list, sync,
and manage files across **70+ cloud providers** — all through a single
interface powered by `rclone`.

## Supported Providers

Just a few highlights — rclone supports many more via config:

| Provider | Auth Type | Notes |
|----------|-----------|-------|
| **OneDrive** / OneDrive for Business | OAuth (rclone built-in) | Personal, Work/School accounts |
| **Google Drive** | OAuth (rclone built-in) | Shared drives supported |
| **Dropbox** | OAuth (rclone built-in) | Team folders supported |
| **pCloud** | OAuth (rclone built-in) | EU and US regions |
| **Box** | OAuth (rclone built-in) | Enterprise friendly |
| **Backblaze B2** | Key + App Key | Dead easy |
| **Amazon S3** / Wasabi / DigitalOcean Spaces | Access Key + Secret | S3-compatible |
| **WebDAV** | Username + Password | Nextcloud, ownCloud, etc. |
| **SFTP** | SSH key / password | Any SSH server |
| **FTP** | Username + Password | Legacy support |
| **HTTP** | None | Read-only public mirrors |

Full list: https://rclone.org/#providers

## Prerequisites

- `rclone` installed (`pkg install rclone` in Termux, or `apt install rclone`
  on Linux)
- At least one remote configured via `rclone config`

## First-Time Setup

Run the setup helper to check if rclone is installed and see your remotes:

```bash
CS_SETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/cloud-storage/scripts/setup.py"
$CS_SETUP --check
```

### One-Time Interactive Setup

Cloud provider OAuth (OneDrive, Google Drive, Dropbox, etc.) requires one
**interactive** authentication per remote. This only happens once.

**In Termux / on your phone:**

```bash
rclone config
```

1. Type `n` for **New remote**
2. Name it (e.g. `onedrive`, `gdrive`, `dropbox`)
3. Pick your provider number from the list
4. For OAuth providers, it will say: `Use auto config?` → type `y`
5. It prints a URL. Open it in your phone browser, sign in, approve
6. Return to Termux — it detects the auth automatically
7. Type `q` to quit config

**If auto-config fails (headless server):**

Select `n` for auto-config. It'll print a URL for you to open on a device with
a browser, then paste the code back.

### Verifying Your Remotes

```bash
rclone listremotes
rclone about onedrive:       # shows used / free space
rclone lsf onedrive:         # lists root folder
```

## Usage

All commands go through the CLI script. Set a shorthand:

```bash
CS="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/cloud-storage/scripts/cloud_storage.py"
```

### Upload

```bash
# Upload a file
$CS upload /path/to/file.md onedrive:Documents/

# Upload to root
$CS upload /path/to/file.md onedrive:

# Upload a folder recursively
$CS upload /local/folder/ gdrive:Backups/ --recursive

# Upload with progress
$CS upload /sdcard/DCIM/photo.jpg onedrive:Photos/ --progress
```

### Download

```bash
# Download a file
$CS download onedrive:Documents/file.md /local/path/file.md

# Download a folder
$CS download gdrive:Projects/ ./Projects/ --recursive
```

### List

```bash
# List root
$CS list onedrive:

# List a folder
$CS list onedrive:Documents/

# JSON output for scripting
$CS list gdrive: --format json

# Fast directory-only listing
$CS list onedrive: --dirs-only
```

### Sync

```bash
# Make remote match local (uploads new, deletes removed)
$CS sync /local/folder/ onedrive:backup/folder/

# Make local match remote (downloads new, deletes removed)
$CS sync onedrive:backup/folder/ /local/folder/

# Dry run to preview changes
$CS sync /local/folder/ onedrive:backup/folder/ --dry-run
```

> ⚠️ **Sync deletes files** that don't exist on the source. Always use
> `--dry-run` first.

### Delete

```bash
# Delete a file
$CS delete onedrive:Documents/old.txt

# Delete a folder recursively
$CS delete onedrive:OldFolder/ --recursive

# Empty trash (provider-dependent)
$CS empty-trash onedrive:
```

### Create Folders

```bash
$CS mkdir onedrive:Projects/NewProject
$CS mkdir gdrive:2024/Reports/Q3
```

### Search

```bash
# Search filename across a remote
$CS search onedrive: "report"
$CS search gdrive: --include "*.pdf" --format json
```

### Get Info

```bash
# Remote storage usage
$CS about onedrive:

# File size, modtime, hash
$CS info onedrive:Documents/file.md
```

## Provider-Specific Shortcuts

The helper auto-detects common remotes and offers shortcuts:

```bash
$CS upload file.md o:Documents/      # if "o" is configured = onedrive
$CS upload file.md g:Projects/         # if "g" is configured = gdrive
$CS upload file.md d:                   # if "d" is configured = dropbox
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `rclone: not found` | Install rclone: `pkg install rclone` (Termux) or `apt install rclone` |
| `Failed to create file system` | Remote not configured. Run `rclone config` first. |
| `Unauthorized` / `token expired` | Re-auth: `rclone config reconnect onedrive:` or `rclone reconnect onedrive:` |
| Upload fails with large files | Add `--transfers 1 --checkers 1` to reduce concurrency |
| Slow speeds | `rclone` is single-threaded. Add `--transfers 4` or `--multi-thread-cutoff 50M` |
| OneDrive "name already exists" | Delete the conflicting file first, or add `--checksum --ignore-existing` |

## Security Notes

- rclone stores its config (including tokens) in `~/.config/rclone/rclone.conf`
- If you're on a shared device, consider `rclone config password` to encrypt
  the config file with a password
- Our scripts never touch token storage — they just call `rclone`

## Contributing

This skill is part of the [hermes-agent](https://github.com/NousResearch/hermes-agent)
community collection. Found a bug or want to add another provider shortcut?
PR welcome at `github.com/NousResearch/hermes-agent`.
