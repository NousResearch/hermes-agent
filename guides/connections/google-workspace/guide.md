# Google Workspace

Gmail, Calendar, Drive, Contacts, Sheets, and Docs — through Hermes-managed OAuth and a thin CLI wrapper. When `gws` is installed, the helper uses it as the execution backend for broader Google Workspace coverage; otherwise it falls back to the bundled Python client implementation.

Read `../guide.md` first for connection verification and documentation rules.

## References

- `references/gmail-search-syntax.md` — Gmail search operators (is:unread, from:, newer_than:, etc.)

## Scripts

- `scripts/setup.py` — OAuth2 setup (run once to authorize)
- `scripts/google_api.py` — compatibility wrapper CLI. It prefers `gws` for operations when available, while preserving Hermes' existing JSON output contract.

## First-Time Setup

The setup is fully non-interactive — you drive it step by step so it works
on CLI, Telegram, Discord, or any platform.

Run the setup script with Python from the Hermes environment, not an unrelated
system Python. `--install-deps` syncs Hermes' declared Google extra through PM;
after syncing, restart Hermes and rerun the OAuth command. If Hermes is not
importable, use `hermes setup` first rather than installing packages with pip.

Define a shorthand first:

```bash
google_setup_script="$(python -c 'from pathlib import Path; import guides; print(Path(guides.__file__).parent / "connections/google-workspace/scripts/setup.py")')"
```

### Step 0: Check if already set up

```bash
python "$google_setup_script" --check
```

If it prints `AUTHENTICATED`, skip to Usage — setup is already done.

### Step 1: Choose the required access

Use the services already requested; ask only when the intended work is unclear.
For email-only access, `../email/guide.md` also covers Himalaya IMAP/SMTP where
App Passwords are supported.

The native Google helper requests its fixed Workspace scope set: Gmail read,
send and modify; Calendar; Drive; Contacts read; Sheets; and Docs. It has no
service-selection flag. Explain that consent scope before authorization; the
user can deselect unneeded permissions. The helper preserves granted scopes,
and subsequent service calls require the corresponding permissions. Do not
claim a narrow consent request or expand access without the user's authority.

If account or organization policy blocks the OAuth client, follow Google's
reported policy requirements before retrying.

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
> `The JSON file path is: ~/Downloads/client_secret_....json`

Once they provide the path:

```bash
python "$google_setup_script" --client-secret /path/to/client_secret.json
```

Have the user provision the downloaded credential file on the Hermes host;
do not ask them to paste reusable client secrets into chat.

### Step 3: Get authorization URL

```bash
python "$google_setup_script" --auth-url
```

The helper prints the authorization URL as plain text and saves the pending
PKCE session locally. It does not return JSON or save a separate URL file.

Agent rules for this step:
- Send the exact printed authorization URL to the user as a single line.
- Tell the user that the browser will likely fail on `http://localhost:1` after approval, and that this is expected.
- Tell them to copy the ENTIRE redirected URL from the browser address bar.
- If the user gets `Error 403: access_denied`, send them directly to `https://console.cloud.google.com/auth/audience` to add themselves as a test user.

### Step 4: Exchange the code

The user will paste back either a URL like `http://localhost:1/?code=4/0A...&scope=...`
or just the code string. Either works. The `--auth-url` step stores a temporary
pending OAuth session locally so `--auth-code` can complete the PKCE exchange
later, even on headless systems:

```bash
python "$google_setup_script" --auth-code "THE_URL_OR_CODE_THE_USER_PASTED"
```

If exchange fails because the code expired, was used, or belongs to an older
session, run `python "$google_setup_script" --auth-url` again. Send that new URL
and use only the newest browser redirect. No replacement URL is returned
automatically by a failed exchange.

### Step 5: Verify

```bash
python "$google_setup_script" --check
```

Should print `AUTHENTICATED`. Then perform a harmless read for the requested
service and verify the account/resource. Record working usage in the connection
manual following `../guide.md`. Tokens refresh automatically. This helper stores
one Google account per Hermes home; do not overwrite an existing account to add
another without resolving that choice with the user.

### Notes

- Token is stored at `$HERMES_HOME/google_token.json` and auto-refreshes.
- Pending OAuth session state/verifier are stored temporarily at `$HERMES_HOME/google_oauth_pending.json` until exchange completes.
- If `gws` is installed, `google_api.py` points it at the same `$HERMES_HOME/google_token.json` credentials file. Users do not need to run a separate `gws auth login` flow.
- To revoke: `python "$google_setup_script" --revoke`

## Usage

All commands go through the API script. Locate the shipped helper:

```bash
google_api_script="$(python -c 'from pathlib import Path; import guides; print(Path(guides.__file__).parent / "connections/google-workspace/scripts/google_api.py")')"
```

### Gmail

```bash
# Search (returns JSON array with id, from, subject, date, snippet)
python "$google_api_script" gmail search "is:unread" --max 10
python "$google_api_script" gmail search "from:boss@company.com newer_than:1d"
python "$google_api_script" gmail search "has:attachment filename:pdf newer_than:7d"

# Read full message (returns JSON with body text)
python "$google_api_script" gmail get MESSAGE_ID

# Send
python "$google_api_script" gmail send --to user@example.com --subject "Hello" --body "Message text"
python "$google_api_script" gmail send --to user@example.com --subject "Report" --body "<h1>Q4</h1><p>Details...</p>" --html
python "$google_api_script" gmail send --to user@example.com --subject "Hello" --from '"Research Agent" <user@example.com>' --body "Message text"

# Reply (automatically threads and sets In-Reply-To)
python "$google_api_script" gmail reply MESSAGE_ID --body "Thanks, that works for me."
python "$google_api_script" gmail reply MESSAGE_ID --from '"Support Bot" <user@example.com>' --body "Thanks"

# Labels
python "$google_api_script" gmail labels
python "$google_api_script" gmail modify MESSAGE_ID --add-labels LABEL_ID
python "$google_api_script" gmail modify MESSAGE_ID --remove-labels UNREAD
```

### Calendar

```bash
# List events (defaults to next 7 days)
python "$google_api_script" calendar list
python "$google_api_script" calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# Create event (ISO 8601 with timezone required)
python "$google_api_script" calendar create --summary "Team Standup" --start 2026-03-01T10:00:00-06:00 --end 2026-03-01T10:30:00-06:00
python "$google_api_script" calendar create --summary "Lunch" --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z --location "Cafe"
python "$google_api_script" calendar create --summary "Review" --start 2026-03-01T14:00:00Z --end 2026-03-01T15:00:00Z --attendees "alice@co.com,bob@co.com"

# Delete event
python "$google_api_script" calendar delete EVENT_ID
```

### Drive

```bash
# Search existing files
python "$google_api_script" drive search "quarterly report" --max 10
python "$google_api_script" drive search "mimeType='application/pdf'" --raw-query --max 5

# Get metadata for a single file
python "$google_api_script" drive get FILE_ID

# Upload a local file (auto-detects MIME type)
python "$google_api_script" drive upload /path/to/report.pdf
python "$google_api_script" drive upload /path/to/image.png --name "Logo.png" --parent FOLDER_ID

# Download (binary files download as-is; Google-native files export to a
# sensible default — Docs→pdf, Sheets→csv, Slides→pdf, Drawings→png)
python "$google_api_script" drive download FILE_ID
python "$google_api_script" drive download DOC_ID --output ~/doc.pdf
python "$google_api_script" drive download DOC_ID --export-mime text/plain --output ~/doc.txt

# Create a folder
python "$google_api_script" drive create-folder "Reports"
python "$google_api_script" drive create-folder "Q4" --parent FOLDER_ID

# Share
python "$google_api_script" drive share FILE_ID --email alice@example.com --role reader
python "$google_api_script" drive share FILE_ID --email alice@example.com --role writer --notify
python "$google_api_script" drive share FILE_ID --type anyone --role reader        # anyone with link
python "$google_api_script" drive share FILE_ID --type domain --domain example.com --role reader

# Delete — defaults to trash (reversible). Use --permanent to skip the trash.
python "$google_api_script" drive delete FILE_ID
python "$google_api_script" drive delete FILE_ID --permanent
```

### Contacts

```bash
python "$google_api_script" contacts list --max 20
```

### Sheets

```bash
# Create a new spreadsheet
python "$google_api_script" sheets create --title "Q4 Budget"
python "$google_api_script" sheets create --title "Inventory" --sheet-name "Stock"

# Read
python "$google_api_script" sheets get SHEET_ID "Sheet1!A1:D10"

# Write
python "$google_api_script" sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# Append rows
python "$google_api_script" sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

### Docs

```bash
# Read (a tabbed Doc returns a "tabs" array; single-tab and legacy Docs also return "body")
python "$google_api_script" docs get DOC_ID
python "$google_api_script" docs get DOC_ID --tab TAB_ID     # read one tab of a tabbed Doc

# Create a new Doc (optionally seeded with body text)
python "$google_api_script" docs create --title "Meeting Notes"
python "$google_api_script" docs create --title "Draft" --body "First paragraph..."

# Append text to the end of an existing Doc
python "$google_api_script" docs append DOC_ID --text "Additional content to append"
python "$google_api_script" docs append DOC_ID --tab TAB_ID --text "..."   # --tab required when the Doc has multiple tabs
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

1. **Use the authority already granted by the user or responsibility.** Before writes outside that authority, show what will be done (recipients, file IDs, content, share role) and ask for approval. For `drive delete`, prefer the default trash (reversible) over `--permanent`.
2. **Check auth before first use** — run `setup.py --check`. If it fails, guide the user through setup.
3. **Use the Gmail search syntax reference** for complex queries — read `references/gmail-search-syntax.md`.
4. **Calendar times must include timezone** — always use ISO 8601 with offset (e.g., `2026-03-01T10:00:00-06:00`) or UTC (`Z`).
5. **Respect rate limits** — avoid rapid-fire sequential API calls. Batch reads when possible.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NOT_AUTHENTICATED` | Run setup Steps 2-5 above |
| `REFRESH_FAILED` | Token revoked or expired — redo Steps 3-5 |
| `HttpError 403: Insufficient Permission` | Missing API scope — `python "$google_setup_script" --revoke` then redo Steps 3-5 |
| `AUTHENTICATED (partial)` or "Token missing scopes" | New write capabilities (Drive write/delete, Docs create/edit) require re-authorization. `python "$google_setup_script" --revoke` then redo Steps 3-5 to grant the upgraded scopes. |
| `HttpError 403: Access Not Configured` | API not enabled — user needs to enable it in Google Cloud Console |
| `ModuleNotFoundError` | Run `python "$google_setup_script" --install-deps` |
| Advanced Protection blocks auth | Workspace admin must allowlist the OAuth client ID |

## Revoking Access

```bash
python "$google_setup_script" --revoke
```
