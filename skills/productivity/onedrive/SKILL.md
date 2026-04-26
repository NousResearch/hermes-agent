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

## Authentication Options

Pick the path that works for your setup.

### Option 1: rclone (Easiest — recommended for personal Microsoft accounts)

Microsoft makes it hard to use OAuth with custom apps for personal accounts.
**rclone already has a pre-registered public OAuth client**, so you don't need
an Azure app at all.

Install rclone (Termux: `pkg install rclone`, Ubuntu: `apt-get install rclone`),
then run once:

```bash
rclone config
# n) New remote
# name: onedrive
# Storage: 26 (OneDrive)
# Follow the interactive prompts — it will open a browser on your phone
# Choose "Personal" account when asked
```

From then on Hermes uses the rclone backend automatically. No Azure portal
needed, no app registration.

### Option 2: Direct Microsoft Graph API (No external dependencies)

This gives you full control but **requires an Azure app configured as a
Public client**.

**Register an Azure AD App (one-time, ~3 minutes):**

1. Go to https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade
2. Click **New registration** → name it `hermes-agent`
3. Supported account types: **"Accounts in any organizational directory and
   personal Microsoft accounts"**
4. Redirect URI: **leave empty**
5. Click **Register**
6. Copy the **Application (client) ID**
7. Left sidebar → **Authentication**
8. Under **Advanced settings**
   → Toggle **"Allow public client flows"** → **Yes** ← *Required for device code*
9. Left sidebar → **API permissions** → **Add a permission** → **Microsoft Graph**
   → **Delegated permissions**, add:
   - `Files.ReadWrite`
   - `User.Read`
10. Click **Grant admin consent**

> **CRITICAL:** If you see `AADSTS70002: The provided client is not supported
> for this feature` during auth, you skipped step 8. The app **must** be a
> Public client for device-code flow.
   - `User.Read`
10. Click **Grant admin consent** (not required for personal Microsoft accounts)

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
| `invalid_client` / `AADSTS70002: The provided client is not supported for this feature` (device-code) | Your Azure app is a **Web** application. Go to Azure Portal → Authentication → **Allow public client flows** → **Yes** → Save. Device-code flow only works with **Public** clients. |
| `invalid_client` (PKCE) | Your Azure app doesn't have the `nativeclient` redirect URI. Either add `https://login.microsoftonline.com/common/oauth2/nativeclient` as a **Mobile and desktop** platform, or switch to the device-code flow. |
| `InvalidAuthenticationToken` | Token expired and auto-refresh failed. Delete `~/.hermes/onedrive_token.json` and re-run setup. |
| Upload fails with large files | Currently limited to ~60 MB. Use rclone for files above that. |
| `Insufficient privileges` | Remove and re-add the API permissions in Azure Portal, then grant admin consent. |
| `Insufficient privileges` | Remove and re-add the API permissions in Azure Portal, then grant admin consent. |

## Contributing

This skill is part of the [hermes-agent](https://github.com/NousResearch/hermes-agent)
community skills collection. Found a bug or want to add large-file chunked upload?
Open a PR at `github.com/NousResearch/hermes-agent`.
