---
name: google-workspace
description: "Gmail, Calendar, Drive, Docs, Sheets — identity-scoped."
version: 2.0.0
author: Nous Research
license: MIT
platforms: [linux, macos, windows]
required_credential_files:
  - path: google_token.json
    description: JID's own Google OAuth2 token (identity "jid" — other identities live under family_credentials/<identity>/, not covered by this declaration)
  - path: google_client_secret.json
    description: JID's own Google OAuth2 client credentials (identity "jid")
metadata:
  hermes:
    tags: [Google, Gmail, Calendar, Drive, Sheets, Docs, Contacts, Email, OAuth, identity, family]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [himalaya]
---

# Google Workspace

Gmail, Calendar, Drive, Contacts, Sheets, and Docs — through Hermes-managed OAuth and a thin CLI wrapper. When `gws` is installed, the skill uses it as the execution backend for broader Google Workspace coverage; otherwise it falls back to the bundled Python client implementation.

## READ THIS FIRST — identity is mandatory, every single call, no exceptions

**As of 2026-08-12, this system supports multiple independent Google accounts, one per resolved identity — this is not a single-account skill anymore.** `--identity <name>` is a **required** argument on every invocation of `setup.py`, `google_api.py`, and `gws_bridge.py` — the underlying scripts fail closed (argparse error, then a fail-closed `UnknownGoogleIdentityError`) if it's missing or unrecognized. But a required flag only prevents a *crash* — it does nothing to stop you from passing the *wrong* value, and passing the wrong value is a genuine privacy breach, not a bug: it means one person's private Gmail/Calendar/Drive/Contacts content gets read out loud to a different person. **This already happened once (2026-08-12) — Zee asked about "his" calendar and was shown JID's, because nothing told the agent to resolve identity before calling this skill.** Do not repeat it.

**Before calling any command in this skill, resolve who this turn is actually for:**

1. Use the exact same identity-resolution procedure `obsidian-vault-governance` already uses for Family Rules/Actions routing — platform ID against `Profile/JID Profile.md` or the matching `Profile/Family/<Name>/<Name> Profile.md`, never self-identification in chat text.
2. Map the resolved person to the identity key registered in `_google_identities.py`:
   - JID → `--identity jid`
   - Zarkash / Zee → `--identity zarkash`
   - Any other family member → **they have no Google identity registered yet.** Do not silently fall back to `jid` or to any other identity. Tell them Google Workspace access isn't set up for them and stop — this needs a deliberate JID-approved OAuth setup for that person first (same process as Zee's), not an improvised substitution.
3. If identity resolution itself is ambiguous or fails (unmatched platform ID, CLI/desktop context with no clear family member specified), **do not guess and do not default to `jid`.** Stop and ask, the same fail-closed posture as everywhere else in this system.
4. **Every single command below in this file is missing its `--identity` flag in the printed examples for brevity — that is a documentation shorthand, not permission to omit it in practice.** Insert the resolved identity into every real invocation.

**A JID-resolved turn and a Zarkash-resolved turn must never see each other's Gmail, Calendar, Drive, or Contacts data, under any framing, the same hard-line firewall that already applies to vault Family/ folders.**

## References

- `references/gmail-search-syntax.md` — Gmail search operators (is:unread, from:, newer_than:, etc.)
- `references/daily-brief.md` — daily/morning brief procedure: schedule + conflicts + meeting prep + urgent mail from Gmail and Calendar. Load it when the user asks for a morning brief, meeting preparation, or "what's on my calendar and what email needs attention." Same identity-resolution rule applies before using it.

## Scripts

- `scripts/setup.py` — OAuth2 setup (run once per identity to authorize that identity's account)
- `scripts/google_api.py` — compatibility wrapper CLI. It prefers `gws` for operations when available, while preserving Hermes' existing JSON output contract.
- `scripts/_google_identities.py` — the identity → (credential directory, OAuth scopes) registry. Read this file directly if you need to confirm which identities currently exist or what scope a given identity actually has — do not assume every identity has the same access (Zee's Gmail scope, for example, is read+draft-only, never send).

## First-Time Setup (per identity — repeat for each new person)

The setup is fully non-interactive — you drive it step by step so it works
on CLI, Telegram, Discord, or any platform. **Every step below needs
`--identity <name>` for the specific person being set up.**

Define a shorthand first (fill in the resolved identity — do not leave this as a literal placeholder):

```bash
IDENTITY="jid"   # or "zarkash", etc. — the resolved identity for THIS setup, never guessed
GSETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/setup.py --identity $IDENTITY"
```

### Step 0: Check if already set up

```bash
$GSETUP --check
```

If it prints `AUTHENTICATED`, skip to Usage — setup is already done for this identity. A different identity being authenticated does NOT mean this one is — always check the specific identity you're about to act as.

### Step 1: Triage — ask the user what they need

Before starting OAuth setup, ask the user TWO questions:

**Question 1: "What Google services do you need? Just email, or also
Calendar/Drive/Sheets/Docs?"**

- **Email only** → They don't need this skill at all. Use the `himalaya` skill
  instead — it works with a Gmail App Password (Settings → Security → App
  Passwords) and takes 2 minutes to set up. No Google Cloud project needed.
  Load the himalaya skill and follow its setup instructions.

- **Email + Calendar** → Continue with this skill.

- **Calendar/Drive/Sheets/Docs only** → Continue with this skill.

- **Full Workspace access** → Continue with this skill.

Current note: this setup script requests whatever scope set is registered for
that identity in `_google_identities.py` — scopes are NOT necessarily
identical across identities (JID's router account has full Gmail
send/modify/Contacts write; Zee's is deliberately narrower — read+draft-only
Gmail, no send). Check the registry rather than assuming a new identity
should get the same scopes as an existing one; that's a decision point for
JID, not a default to copy.

**Question 2: "Does your Google account use Advanced Protection (hardware
security keys required to sign in)? If you're not sure, you probably don't
— it's something you would have explicitly enrolled in."**

- **No / Not sure** → Normal setup. Continue below.
- **Yes** → Their Workspace admin must add the OAuth client ID to the org's
  allowed apps list before Step 4 will work. Let them know upfront.

### Step 2: Create OAuth credentials (one-time per Google Cloud project, not per identity)

A single Google Cloud OAuth client (client_secret.json) can authorize
multiple different Google accounts — it represents the *app*, not a specific
user. If a client_secret.json already exists for another identity and the
new person's account has been added as a test user on that same project's
consent screen, you can reuse the same file rather than creating a new GCP
project. Confirm with JID which applies before assuming either way — don't
silently reuse another identity's file without checking, and don't tell the
user to create a new GCP project if reuse was actually intended.

If a new one is genuinely needed, tell the user:

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
> `The JSON file path is: /path/to/Downloads/client_secret_....json`

Once they provide the path:

```bash
$GSETUP --client-secret /path/to/client_secret.json
```

This writes to the resolved identity's own credential directory — `HERMES_HOME` root for `jid`, `HERMES_HOME/family_credentials/<identity>/` for everyone else. It never touches another identity's files.

If they paste the raw client ID / client secret values instead of a file path,
write a valid Desktop OAuth JSON file for them yourself, save it somewhere
explicit (for example `~/Downloads/hermes-google-client-secret.json`), then run
`--client-secret` against that file.

### Step 3: Get authorization URL

```bash
$GSETUP --auth-url
```

This prints the authorization URL directly, scoped to the resolved identity's registered scope set.

Agent rules for this step:
- Extract the printed URL and send that exact URL to the user as a single line.
- Tell the user that the browser will likely fail on `http://localhost:1` after approval, and that this is expected.
- Tell them to copy the ENTIRE redirected URL from the browser address bar.
- If the user gets `Error 403: access_denied`, send them directly to `https://console.cloud.google.com/auth/audience` to add themselves as a test user.
- **If this identity is sharing a device/browser with someone already signed into a different Google account (e.g. a family member's phone that also has JID's account logged in), remind them to sign out of any other active Google session first** — otherwise the consent screen can silently authorize the wrong account.

### Step 4: Exchange the code

The user will paste back either a URL like `http://localhost:1/?code=4/0A...&scope=...`
or just the code string. Either works. The `--auth-url` step stores a temporary
pending OAuth session locally (inside the resolved identity's own credential
directory) so `--auth-code` can complete the PKCE exchange later, even on
headless systems:

```bash
$GSETUP --auth-code "THE_URL_OR_CODE_THE_USER_PASTED"
```

If `--auth-code` fails because the code expired, was already used, or came from
an older browser tab, generate a fresh URL with `--auth-url` again and have the
user retry with the newest browser redirect only.

### Step 5: Verify

```bash
$GSETUP --check
```

Should print `AUTHENTICATED`. Setup is complete for this identity — token refreshes automatically from now on, for this identity only.

### Notes

- Token is stored at the resolved identity's own path: `~/.hermes/google_token.json` for `jid`, `~/.hermes/family_credentials/<identity>/google_token.json` for everyone else. Auto-refreshes independently per identity.
- **Google enforces a ~7-day refresh-token expiry while the OAuth consent screen is in "Testing" publishing status** — this applies to every identity, not just one. If `--check` returns `TOKEN_REVOKED`, this is very likely why; re-run Steps 3-5 for that identity. See `Governance/Operations.md`'s "Known Recurring Maintenance Items" for the full note.
- Pending OAuth session state/verifier are stored temporarily inside the resolved identity's own credential directory until exchange completes.
- If `gws` is installed, `google_api.py` points it at the resolved identity's own token file. Users do not need to run a separate `gws auth login` flow.
- To revoke a specific identity's access: `$GSETUP --revoke` (with that identity's `--identity` already set in `$GSETUP`).

## Usage

All commands go through the API script. Set `GAPI` as a shorthand — **`$IDENTITY` must already be resolved and correct before this line runs, every time, for every turn:**

```bash
GAPI="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/google_api.py --identity $IDENTITY"
```

### Gmail

```bash
# Search (returns JSON array with id, from, subject, date, snippet)
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"

# Read full message (returns JSON with body text)
$GAPI gmail get MESSAGE_ID

# Send — NOTE: only identities with the gmail.send scope can do this. Check
# the identity's registered scopes first (_google_identities.py) — if the
# scope isn't there, the API call itself will fail; don't attempt it and
# then explain the failure, check first and tell the user plainly that
# identity is read/draft-only.
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"
$GAPI gmail send --to user@example.com --subject "Report" --body "<h1>Q4</h1><p>Details...</p>" --html
$GAPI gmail send --to user@example.com --subject "Hello" --from '"Research Agent" <user@example.com>' --body "Message text"

# Draft only (no send) — this is what identities with gmail.compose-only scope use
$GAPI gmail draft-create --to user@example.com --subject "Hello" --body "Message text"

# Reply (automatically threads and sets In-Reply-To) — requires gmail.send, same restriction as send above
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."
$GAPI gmail reply MESSAGE_ID --from '"Support Bot" <user@example.com>' --body "Thanks"

# Labels — requires gmail.modify
$GAPI gmail labels
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

### Calendar

```bash
# List events (defaults to next 7 days)
$GAPI calendar list
$GAPI calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# Create event (ISO 8601 with timezone required)
$GAPI calendar create --summary "Team Standup" --start 2026-03-01T10:00:00-06:00 --end 2026-03-01T10:30:00-06:00
$GAPI calendar create --summary "Lunch" --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z --location "Cafe"
$GAPI calendar create --summary "Review" --start 2026-03-01T14:00:00Z --end 2026-03-01T15:00:00Z --attendees "alice@co.com,bob@co.com"

# Delete event
$GAPI calendar delete EVENT_ID
```

### Drive

```bash
# Search existing files
$GAPI drive search "quarterly report" --max 10
$GAPI drive search "mimeType='application/pdf'" --raw-query --max 5

# Get metadata for a single file
$GAPI drive get FILE_ID

# Upload a local file (auto-detects MIME type)
$GAPI drive upload /path/to/report.pdf
$GAPI drive upload /path/to/image.png --name "Logo.png" --parent FOLDER_ID

# Download (binary files download as-is; Google-native files export to a
# sensible default — Docs→pdf, Sheets→csv, Slides→pdf, Drawings→png)
$GAPI drive download FILE_ID
$GAPI drive download DOC_ID --output ~/doc.pdf
$GAPI drive download DOC_ID --export-mime text/plain --output ~/doc.txt

# Create a folder
$GAPI drive create-folder "Reports"
$GAPI drive create-folder "Q4" --parent FOLDER_ID

# Share
$GAPI drive share FILE_ID --email alice@example.com --role reader
$GAPI drive share FILE_ID --email alice@example.com --role writer --notify
$GAPI drive share FILE_ID --type anyone --role reader        # anyone with link
$GAPI drive share FILE_ID --type domain --domain example.com --role reader

# Delete — defaults to trash (reversible). Use --permanent to skip the trash.
$GAPI drive delete FILE_ID
$GAPI drive delete FILE_ID --permanent
```

### Contacts

```bash
$GAPI contacts list --max 20
$GAPI contacts create --name "Alice Example" --email alice@example.com --phone "+1-555-0100"
$GAPI contacts update RESOURCE_NAME --email newalice@example.com
$GAPI contacts delete RESOURCE_NAME
```

Create/update/delete require the full `contacts` scope (not `contacts.readonly`) — check the identity's registered scope before attempting.

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
# Read (a tabbed Doc returns a "tabs" array; single-tab and legacy Docs also return "body")
$GAPI docs get DOC_ID
$GAPI docs get DOC_ID --tab TAB_ID     # read one tab of a tabbed Doc

# Create a new Doc (optionally seeded with body text)
$GAPI docs create --title "Meeting Notes"
$GAPI docs create --title "Draft" --body "First paragraph..."

# Append text to the end of an existing Doc
$GAPI docs append DOC_ID --text "Additional content to append"
$GAPI docs append DOC_ID --tab TAB_ID --text "..."   # --tab required when the Doc has multiple tabs
```

## Output Format

All commands return JSON. Parse with `jq` or read directly. Key fields:

- **Gmail search**: `[{id, threadId, from, to, subject, date, snippet, labels}]`
- **Gmail get**: `{id, threadId, from, to, subject, date, labels, body}`
- **Gmail send/reply**: `{status: "sent", id, threadId}`
- **Gmail draft-create**: `{status: "draft_created", id, messageId}`
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
- **Contacts create/update**: `{status: "created" | "updated", resourceName}`
- **Contacts delete**: `{status: "deleted", resourceName}`
- **Sheets get**: `[[cell, cell, ...], ...]`
- **Sheets create**: `{status: "created", spreadsheetId, title, spreadsheetUrl}`
- **Docs create**: `{status: "created", documentId, title, url}`
- **Docs append**: `{status: "appended", documentId, inserted_at, characters}`

## Rules

1. **Resolve identity first, every turn, before anything else in this skill runs.** See "READ THIS FIRST" above — this is rule zero, everything else is secondary to it.
2. **Never send email, create/delete calendar events, delete Drive files, share files, or modify Docs/Sheets without confirming with the user first.** Show what will be done (recipients, file IDs, content, share role) and ask for approval. For `drive delete`, prefer the default trash (reversible) over `--permanent`.
3. **Check auth before first use, for the correct identity** — run `setup.py --identity <name> --check`. If it fails, guide that specific person through setup — do not substitute a different identity's working credentials just to give an answer.
4. **Use the Gmail search syntax reference** for complex queries — load it with `skill_view("google-workspace", file_path="references/gmail-search-syntax.md")`.
5. **Calendar times must include timezone** — always use ISO 8601 with offset (e.g., `2026-03-01T10:00:00-06:00`) or UTC (`Z`).
6. **Respect rate limits** — avoid rapid-fire sequential API calls. Batch reads when possible.
7. **Check scopes before attempting a write-shaped call.** An identity's registered scope set (`_google_identities.py`) tells you up front what's possible — don't discover a permission boundary by trying and failing when you could have known in advance (e.g. Zee's Gmail is read+draft-only; don't attempt `gmail send` for him and then explain the error).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NOT_AUTHENTICATED` | Run setup Steps 2-5 above, for the correct `--identity` |
| `TOKEN_REVOKED` / `invalid_grant` | Very likely the ~7-day Testing-mode refresh-token expiry (see Notes above) — redo Steps 3-5 for that identity |
| `REFRESH_FAILED` | Token revoked or expired — redo Steps 3-5 |
| `HttpError 403: Insufficient Permission` | Missing API scope for this identity — check `_google_identities.py`, don't assume it's the same as another identity's scope set |
| `AUTHENTICATED (partial)` or "Token missing scopes" | New write capabilities require re-authorization. `$GSETUP --revoke` then redo Steps 3-5 to grant the upgraded scopes. |
| `HttpError 403: Access Not Configured` | API not enabled — user needs to enable it in Google Cloud Console |
| `ModuleNotFoundError` | Run `$GSETUP --install-deps` |
| Advanced Protection blocks auth | Workspace admin must allowlist the OAuth client ID |
| Agent shows one person's Google data in response to a different person's question | **Stop immediately, this is the exact failure mode from the 2026-08-12 incident.** Identity was resolved wrong or not at all before the call. Do not attempt to patch the specific answer — flag it to JID as a repeat of that incident. |

## Revoking Access

```bash
$GSETUP --revoke
```
(with `--identity <name>` already set in `$GSETUP` for whichever identity you mean to revoke)
