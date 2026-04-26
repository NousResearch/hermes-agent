---
name: onedrive
version: 1.0.0
description: |
  Upload, download, list, delete and manage files on Microsoft OneDrive via the
  Microsoft Graph API. Built for headless / Termux environments using the OAuth
  device-code flow — no browser on the device required.
author: subimpact
license: MIT
source: https://github.com/subimpact/hermes-agent/tree/main/skills/productivity/onedrive
metadata:
  hermes:
    tags: [OneDrive, Microsoft, Graph, Storage, Productivity, Cloud]
    related_skills: [google-workspace]
---

# OneDrive Integration

A portable OneDrive file manager for Hermes Agent. Works on phones, servers, and
any headless environment.

## What It Does

- **Upload files** to OneDrive (auto-creates folders, supports overwrite/rename)
- **Download files** by path or file ID
- **List files and folders** (with size, modified date, webUrl)
- **Delete files and folders**
- **Create folders**

## Prerequisites

- A Microsoft / Azure personal account (Outlook.com, Hotmail, Live, Xbox, etc.)
- Python 3.8+ with `requests` (or `urllib` for minimal installs)

## First-Time Setup

### 1. Register an Azure AD App (one-time, ~3 minutes)

Go to **Azure Portal → App registrations** (or visit this link):
https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade

1. Click **New registration**
2. Name it anything (e.g. `hermes-agent`)
3. Supported account types: **"Accounts in any organizational directory and
   personal Microsoft accounts"**
4. Redirect URI: **leave empty** (not needed for device code flow)
5. Click **Register**
6. On the Overview page, copy **Application (client) ID**
7. Left sidebar → **API permissions** → **Add a permission** → **Microsoft Graph**
   → **Delegated permissions**, then add:
   - `Files.ReadWrite`
   - `User.Read`
8. Click **Grant admin consent** (not required for personal Microsoft accounts)

> **Important:** Save the Client ID. You'll need it below.

### 2. Run Setup

```bash
OD_SETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/onedrive/scripts/setup.py"
```

**Step A: Start device login**

```bash
$OD_SETUP --client-id YOUR_CLIENT_ID --auth --format json
```

This prints a JSON object containing:
- `user_code` — a short alphanumeric code
- `verification_uri` — the URL to visit (usually `https://microsoft.com/devicelogin`)

**Step B: Authorize**

1. On any device with a browser, go to the `verification_uri` (usually https://microsoft.com/devicelogin)
2. Enter the `user_code`
3. Sign in with your Microsoft account and approve the permissions
4. Return to Hermes agent

**Step C: Complete**

The script will poll in the background for up to 5 minutes. Once you approve in
the browser, the token is saved to `~/.hermes/onedrive_token.json`.

**Step D: Verify**

```bash
$OD_SETUP --check
```

Should print `AUTHENTICATED`. Token auto-refreshes from now on.

### Notes

- Token stored at: `~/.hermes/onedrive_token.json` (auto-refreshes)
- To revoke / re-auth: delete `~/.hermes/onedrive_token.json` and run `--auth` again
- Device codes expire in 15 minutes. If setup times out, just run `--auth` again

## Usage

All commands go through the API script. Set a shell shorthand:

```bash
OD="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/onedrive/scripts/onedrive_api.py"
```

### List Files

```bash
# List root folder
$OD list /

# List a specific folder
$OD list Documents
$OD list Documents/Projects

# JSON output for scripts
$OD list Documents --format json
```

### Upload Files

```bash
# Upload to root (file keeps its name)
$OD upload /path/to/local/file.pdf

# Upload with a different remote name
$OD upload /path/to/local/file.pdf Documents/report.pdf

# Upload to a folder (creates intermediate folders automatically)
$OD upload /path/to/photo.jpg Pictures/Vacations/photo.jpg

# Overwrite existing file
$OD upload /path/to/photo.jpg Pictures/photo.jpg --conflict replace

# Keep both (rename if duplicate)
$OD upload /path/to/photo.jpg Pictures/photo.jpg --conflict rename
```

### Download Files

```bash
# Download by remote path
$OD download Documents/report.pdf /local/path/report.pdf

# Download by item ID (useful for shared files or when path has special chars)
$OD download --id ITEM_ID /local/path/report.pdf
```

### Create Folders

```bash
$OD mkdir Documents/NewProject
$OD mkdir Documents/NewProject/Notes
```

### Delete Files or Folders

```bash
# Delete by path
$OD delete Documents/old_report.pdf

# Delete by item ID
$OD delete --id ITEM_ID
```

### Search

```bash
# Search across OneDrive
$OD search "budget"
$OD search "budget 2024" --format json
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `invalid_client` | Double-check the client ID. No client secret needed for device code. |
| `access_denied` / `authorization_pending` | You haven't completed the browser step yet. Open the link, enter the code, and approve. |
| `InvalidAuthenticationToken` | Token expired and auto-refresh failed. Delete `~/.hermes/onedrive_token.json` and re-run `--auth`. |
| Upload fails with large files | Currently limited to ~60 MB. Large file chunked upload will be added in a future version. |
| `AADSTS70002: The provided client is not supported` | The Azure app is registered as a **Web** app. Device code requires a **Public client**. Fix: Azure Portal → your app → Authentication → Advanced settings → **Allow public client flows: Yes** → Save. |
| `Insufficient privileges` | Remove and re-add the API permissions in Azure Portal, then grant admin consent. |

## Contributing

This skill is part of the [hermes-agent](https://github.com/NousResearch/hermes-agent)
community skills collection. Found a bug or want to add large-file chunked upload?
Open a PR at `github.com/NousResearch/hermes-agent`.
