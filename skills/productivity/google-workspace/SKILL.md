---
name: google-workspace
description: "Gmail, Calendar, Drive, Docs, Sheets via gws CLI or Python."
version: 1.1.0
author: Nous Research
license: MIT
platforms: [linux, macos, windows]
required_credential_files:
  - path: google_token_joncoenen_gmail_com.json
    description: Google OAuth2 token for joncoenen@gmail.com (created by setup script)
  - path: google_token_salofren_gmail_com.json
    description: Google OAuth2 token for salofren@gmail.com (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials (downloaded from Google Cloud Console)
metadata:
  hermes:
    tags: [Google, Gmail, Calendar, Drive, Sheets, Docs, Contacts, Email, OAuth]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [himalaya]
---

# Google Workspace

Gmail, Calendar, Drive, Contacts, Sheets, and Docs — through Hermes-managed OAuth and a thin CLI wrapper. Gmail, Calendar, and Drive commands require an explicit Google account: `joncoenen@gmail.com` or `salofren@gmail.com`. When `gws` is installed, the skill uses it as the execution backend for broader Google Workspace coverage; otherwise it falls back to the bundled Python client implementation.

## References

- `references/gmail-search-syntax.md` — Gmail search operators (is:unread, from:, newer_than:, etc.)

## Scripts

- `scripts/setup.py` — OAuth2 setup (run once to authorize)
- `scripts/google_api.py` — compatibility wrapper CLI. It prefers `gws` for operations when available, while preserving Hermes' existing JSON output contract.

## First-Time Setup

The setup is fully non-interactive — you drive it step by step so it works
on CLI, Telegram, Discord, or any platform.

Define a shorthand first:

```bash
GSETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/setup.py"
```

### Step 0: Check if already set up

For Jonas's multi-account setup, check each account explicitly:

```bash
$GSETUP --check --account joncoenen@gmail.com
$GSETUP --check --account salofren@gmail.com
```

If both print `AUTHENTICATED`, skip to Usage — setup is already done for both accounts.

### Step 1: Triage — ask the user what they need

Before starting OAuth setup, ask the user:

**"Does your Google account use Advanced Protection (hardware
security keys required to sign in)? If you're not sure, you probably don't
— it's something you would have explicitly enrolled in."**

- **No / Not sure** → Normal setup. Continue below.
- **Yes** → Their Workspace admin must add the OAuth client ID to the org's
  allowed apps list before Step 4 will work. Let them know upfront.

### Step 2: Create OAuth credentials (one-time, ~5 minutes)

Tell the user:

> You need a Google Cloud OAuth client. This is a one-time setup:
>
> 1. Create or select a project:
>    https://console.cloud.google.com/projectselector2/home/dashboard
> 2. Enable the required APIs from the API Library:
>    https://console.cloud.google.com/apis/library
>    Enable: Gmail API, Google Calendar API, Google Drive API,
>    Google Sheets API, Google Docs API, People API
> 3. Create the OAuth client here:
>    https://console.cloud.google.com/apis/credentials
>    Credentials → Create Credentials → OAuth 2.0 Client ID
> 4. Application type: "Desktop app" → Create
> 5. If the app is still in Testing, add the user's Google account as a test user here:
>    https://console.cloud.google.com/auth/audience
>    Audience → Test users → Add users
> 6. Download the JSON file and tell me the file path
>
> Important Hermes CLI note: if the file path starts with `/`, do NOT send only the bare path as its own message in the CLI, because it can be mistaken for a slash command. Send it in a sentence instead, like:
> `The JSON file path is: /home/user/Downloads/client_secret_....json`

Once they provide the path:

```bash
$GSETUP --client-secret /path/to/client_secret.json
```

If they paste the raw client ID / client secret values instead of a file path,
write a valid Desktop OAuth JSON file for them yourself, save it somewhere
explicit (for example `~/Downloads/hermes-google-client-secret.json`), then run
`--client-secret` against that file.

### Step 3: Get authorization URL

Generate a URL for the account being authorized. Examples:

```bash
$GSETUP --auth-url --account joncoenen@gmail.com
$GSETUP --auth-url --account salofren@gmail.com
```

The script prints the OAuth URL directly and stores account-specific pending OAuth state under the Hermes profile directory.

Agent rules for this step:
- Send the printed URL to the user as a single line.
- Tell the user that the browser will likely fail on `http://localhost:1` after approval, and that this is expected.
- Tell them to copy the ENTIRE redirected URL from the browser address bar.
- If the user gets `Error 403: access_denied`, send them directly to `https://console.cloud.google.com/auth/audience` to add themselves as a test user.

### Step 4: Exchange the code

The user will paste back either a URL like `http://localhost:1/?code=4/0A...&scope=...`
or just the code string. Either works. The `--auth-url` step stores a temporary
pending OAuth session locally so `--auth-code` can complete the PKCE exchange
later, even on headless systems:

```bash
$GSETUP --auth-code "THE_URL_OR_CODE_THE_USER_PASTED" --account joncoenen@gmail.com
$GSETUP --auth-code "THE_URL_OR_CODE_THE_USER_PASTED" --account salofren@gmail.com
```

If `--auth-code` fails because the code expired, was already used, or came from
an older browser tab, it now returns a fresh `fresh_auth_url`. In that case,
immediately send the new URL to the user and have them retry with the newest
browser redirect only.

### Step 5: Verify

```bash
$GSETUP --check --account joncoenen@gmail.com
$GSETUP --check --account salofren@gmail.com
```

Should print `AUTHENTICATED`. Setup is complete — token refreshes automatically from now on.

### Notes

- Tokens are stored per account, e.g. `~/.hermes/google_token_joncoenen_gmail_com.json` and `~/.hermes/google_token_salofren_gmail_com.json`, and auto-refresh.
- Pending OAuth session state/verifier are stored per account until exchange completes.
- If `gws` is installed, `google_api.py` points it at the selected account's token credentials file. Users do not need to run a separate `gws auth login` flow.
- To revoke one account: `$GSETUP --revoke --account joncoenen@gmail.com`

## Usage

All commands go through the API script. Set `GAPI` as a shorthand:

```bash
GAPI="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/google_api.py"
```

### Gmail

Gmail commands require `--account joncoenen@gmail.com` or `--account salofren@gmail.com`.

```bash
# Search (returns JSON array with id, from, subject, date, snippet)
$GAPI gmail --account joncoenen@gmail.com search "is:unread" --max 10
$GAPI gmail --account salofren@gmail.com search "from:boss@company.com newer_than:1d"
$GAPI gmail --account joncoenen@gmail.com search "has:attachment filename:pdf newer_than:7d"

# Read full message (returns JSON with body text)
$GAPI gmail --account joncoenen@gmail.com get MESSAGE_ID

# Send
$GAPI gmail --account joncoenen@gmail.com send --to user@example.com --subject "Hello" --body "Message text"
$GAPI gmail --account joncoenen@gmail.com send --to user@example.com --subject "Report" --body "<h1>Q4</h1><p>Details...</p>" --html
$GAPI gmail --account joncoenen@gmail.com send --to user@example.com --subject "Hello" --from '"Research Agent" <user@example.com>' --body "Message text"

# Reply (automatically threads and sets In-Reply-To)
$GAPI gmail --account joncoenen@gmail.com reply MESSAGE_ID --body "Thanks, that works for me."
$GAPI gmail --account joncoenen@gmail.com reply MESSAGE_ID --from '"Support Bot" <user@example.com>' --body "Thanks"

# Labels
$GAPI gmail --account joncoenen@gmail.com labels
$GAPI gmail --account joncoenen@gmail.com modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail --account joncoenen@gmail.com modify MESSAGE_ID --remove-labels UNREAD
```

### Calendar

Calendar commands require `--account joncoenen@gmail.com` or `--account salofren@gmail.com`.

```bash
# List events (defaults to next 7 days)
$GAPI calendar --account joncoenen@gmail.com list
$GAPI calendar --account joncoenen@gmail.com list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# Create event (ISO 8601 with timezone required)
$GAPI calendar --account joncoenen@gmail.com create --summary "Team Standup" --start 2026-03-01T10:00:00-06:00 --end 2026-03-01T10:30:00-06:00
$GAPI calendar --account joncoenen@gmail.com create --summary "Lunch" --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z --location "Cafe"
$GAPI calendar --account joncoenen@gmail.com create --summary "Review" --start 2026-03-01T14:00:00Z --end 2026-03-01T15:00:00Z --attendees "alice@co.com,bob@co.com"

# Delete event
$GAPI calendar --account joncoenen@gmail.com delete EVENT_ID
```

### Drive

Drive commands require `--account joncoenen@gmail.com` or `--account salofren@gmail.com`.

```bash
# Search existing files
$GAPI drive --account joncoenen@gmail.com search "quarterly report" --max 10
$GAPI drive --account joncoenen@gmail.com search "mimeType='application/pdf'" --raw-query --max 5

# Get metadata for a single file
$GAPI drive --account joncoenen@gmail.com get FILE_ID

# Upload a local file (auto-detects MIME type)
$GAPI drive --account joncoenen@gmail.com upload /path/to/report.pdf
$GAPI drive --account joncoenen@gmail.com upload /path/to/image.png --name "Logo.png" --parent FOLDER_ID

# Download (binary files download as-is; Google-native files export to a
# sensible default — Docs→pdf, Sheets→csv, Slides→pdf, Drawings→png)
$GAPI drive --account joncoenen@gmail.com download FILE_ID
$GAPI drive --account joncoenen@gmail.com download DOC_ID --output ~/doc.pdf
$GAPI drive --account joncoenen@gmail.com download DOC_ID --export-mime text/plain --output ~/doc.txt

# Create a folder
$GAPI drive --account joncoenen@gmail.com create-folder "Reports"
$GAPI drive --account joncoenen@gmail.com create-folder "Q4" --parent FOLDER_ID

# Share
$GAPI drive --account joncoenen@gmail.com share FILE_ID --email alice@example.com --role reader
$GAPI drive --account joncoenen@gmail.com share FILE_ID --email alice@example.com --role writer --notify
$GAPI drive --account joncoenen@gmail.com share FILE_ID --type anyone --role reader        # anyone with link
$GAPI drive --account joncoenen@gmail.com share FILE_ID --type domain --domain example.com --role reader

# Delete — defaults to trash (reversible). Use --permanent to skip the trash.
$GAPI drive --account joncoenen@gmail.com delete FILE_ID
$GAPI drive --account joncoenen@gmail.com delete FILE_ID --permanent
```

### Contacts

```bash
$GAPI contacts list --max 20
```

### Sheets

```bash
# Create a new spreadsheet
$GAPI sheets create --title "Q4 Budget"
$GAPI sheets create --title "Inventory" --sheet-name "Stock"

# Read
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# Write
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# Append rows
$GAPI sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

### Docs

```bash
# Read
$GAPI docs get DOC_ID

# Create a new Doc (optionally seeded with body text)
$GAPI docs create --title "Meeting Notes"
$GAPI docs create --title "Draft" --body "First paragraph..."

# Append text to the end of an existing Doc
$GAPI docs append DOC_ID --text "Additional content to append"
```

## Output Format

All commands return JSON. Parse with `jq` or read directly. Key fields:

- **Gmail search**: `[{id, threadId, from, to, subject, date, snippet, labels}]`
- **Gmail get**: `{id, threadId, from, to, subject, date, labels, body}`
- **Gmail send/reply**: `{status: "sent", id, threadId}`
- **Calendar list**: `[{id, summary, start, end, location, description, htmlLink}]`
- **Calendar create**: `{status: "created", id, summary, htmlLink}`
- **Drive search**: `[{id, name, mimeType, modifiedTime, webViewLink}]`
- **Drive get**: `{id, name, mimeType, modifiedTime, size, webViewLink, parents, owners}`
- **Drive upload**: `{status: "uploaded", id, name, mimeType, webViewLink}`
- **Drive download**: `{status: "downloaded", id, name, path, mimeType}`
- **Drive create-folder**: `{status: "created", id, name, webViewLink}`
- **Drive share**: `{status: "shared", permissionId, fileId, role, type}`
- **Drive delete**: `{status: "trashed" | "deleted", fileId, permanent}`
- **Contacts list**: `[{name, emails: [...], phones: [...]}]`
- **Sheets get**: `[[cell, cell, ...], ...]`
- **Sheets create**: `{status: "created", spreadsheetId, title, spreadsheetUrl}`
- **Docs create**: `{status: "created", documentId, title, url}`
- **Docs append**: `{status: "appended", documentId, inserted_at, characters}`

## Rules

1. **Explicit Google account required for Gmail/Calendar/Drive** — pass `--account joncoenen@gmail.com` or `--account salofren@gmail.com`; invalid accounts are rejected before any API call.
2. **Never send email, create/delete calendar events, delete Drive files, share files, or modify Docs/Sheets without confirming with the user first.** Show what will be done (recipients, file IDs, content, share role) and ask for approval. For `drive delete`, prefer the default trash (reversible) over `--permanent`.
3. **Check auth before first use** — run `setup.py --check --account joncoenen@gmail.com` or `setup.py --check --account salofren@gmail.com`. If it fails, guide the user through setup.
4. **Use the Gmail search syntax reference** for complex queries — load it with `skill_view("google-workspace", file_path="references/gmail-search-syntax.md")`.
5. **Calendar times must include timezone** — always use ISO 8601 with offset (e.g., `2026-03-01T10:00:00-06:00`) or UTC (`Z`).
6. **Respect rate limits** — avoid rapid-fire sequential API calls. Batch reads when possible.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NOT_AUTHENTICATED` | Run setup Steps 2-5 above |
| `REFRESH_FAILED` | Token revoked or expired — redo Steps 3-5 |
| `HttpError 403: Insufficient Permission` | Missing API scope — `$GSETUP --revoke --account ACCOUNT` then redo Steps 3-5 |
| `AUTHENTICATED (partial)` or "Token missing scopes" | New write capabilities (Drive write/delete, Docs create/edit) require re-authorization. `$GSETUP --revoke --account ACCOUNT` then redo Steps 3-5 to grant the upgraded scopes. |
| `HttpError 403: Access Not Configured` | API not enabled — user needs to enable it in Google Cloud Console |
| `ModuleNotFoundError` | Run `$GSETUP --install-deps` |
| Advanced Protection blocks auth | Workspace admin must allowlist the OAuth client ID |

## Revoking Access

```bash
$GSETUP --revoke --account joncoenen@gmail.com
$GSETUP --revoke --account salofren@gmail.com
```
