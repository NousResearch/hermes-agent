---
name: google-workspace
description: Gmail, Calendar, Drive, Contacts, Sheets, and Docs integration via Python. Uses OAuth2 with automatic token refresh. No external binaries needed — runs entirely with Google's Python client libraries in the Hermes venv.
version: 1.0.0
author: Nous Research
license: MIT
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials (downloaded from Google Cloud Console)
metadata:
  hermes:
    tags: [Google, Gmail, Calendar, Drive, Sheets, Docs, Contacts, Email, OAuth]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [himalaya]
---

# Google Workspace

Gmail, Calendar, Drive, Contacts, Sheets, and Docs — all through Python scripts in this skill. No external binaries to install.

## References

- `references/gmail-search-syntax.md` — Gmail search operators (is:unread, from:, newer_than:, etc.)

## Scripts

- `scripts/setup.py` — OAuth2 setup (run once to authorize)
- `scripts/google_api.py` — API wrapper CLI (agent uses this for all operations)

## First-Time Setup

The setup is fully non-interactive — you drive it step by step so it works
on CLI, Telegram, Discord, or any platform.

Define a shorthand first:

```bash
GSETUP="python ~/.hermes/skills/productivity/google-workspace/scripts/setup.py"
```

### Step 0: Check if already set up

```bash
$GSETUP --check
```

If it prints `AUTHENTICATED`, skip to Usage — setup is already done.

### Step 1: Triage — ask the user what they need

Before starting OAuth setup, ask the user TWO questions:

**Question 1: "What Google services do you need? Just email, or also
Calendar/Drive/Sheets/Docs?"**

- **Email only** → They don't need this skill at all. Use the `himalaya` skill
  instead — it works with a Gmail App Password (Settings → Security → App
  Passwords) and takes 2 minutes to set up. No Google Cloud project needed.
  Load the himalaya skill and follow its setup instructions.

- **Calendar, Drive, Sheets, Docs (or email + these)** → Continue with this
  skill's OAuth setup below.

**Question 2: "Does your Google account use Advanced Protection (hardware
security keys required to sign in)? If you're not sure, you probably don't
— it's something you would have explicitly enrolled in."**

- **No / Not sure** → Normal setup. Continue below.
- **Yes** → Their Workspace admin must add the OAuth client ID to the org's
  allowed apps list before Step 4 will work. Let them know upfront.

### Step 2: Create OAuth credentials (one-time, ~5 minutes)

Tell the user:

> You need a Google Cloud OAuth client. This is a one-time setup:
>
> 1. Go to https://console.cloud.google.com/apis/credentials
> 2. Create a project (or use an existing one)
> 3. Click "Enable APIs" and enable: Gmail API, Google Calendar API,
>    Google Drive API, Google Sheets API, Google Docs API, People API
> 4. Go to Credentials → Create Credentials → OAuth 2.0 Client ID
> 5. Application type: "Desktop app" → Create
> 6. Click "Download JSON" and tell me the file path

Once they provide the path:

```bash
$GSETUP --client-secret /path/to/client_secret.json
```

### Step 3: Get authorization URL

```bash
$GSETUP --auth-url
```

This prints a URL. **Send the URL to the user** and tell them:

> Open this link in your browser, sign in with your Google account, and
> authorize access. After authorizing, you'll be redirected to a page that
> may show an error — that's expected. Copy the ENTIRE URL from your
> browser's address bar and paste it back to me.

### Step 4: Exchange the code

The user will paste back either a URL like `http://localhost:1/?code=4/0A...&scope=...`
or just the code string. Either works. The `--auth-url` step stores a temporary
pending OAuth session locally so `--auth-code` can complete the PKCE exchange
later, even on headless systems:

```bash
$GSETUP --auth-code "THE_URL_OR_CODE_THE_USER_PASTED"
```

### Step 5: Verify

```bash
$GSETUP --check
```

Should print `AUTHENTICATED`. Setup is complete — token refreshes automatically from now on.

### Notes

- Token is stored at `google_token.json` under the active profile's `HERMES_HOME` and auto-refreshes.
- Pending OAuth session state/verifier are stored temporarily at `google_oauth_pending.json` under the active profile's `HERMES_HOME` until exchange completes.
- Hermes now refuses to overwrite a full Google Workspace token with a narrower re-auth token missing Gmail scopes, so one profile's partial consent cannot silently break email actions later.
- To revoke: `$GSETUP --revoke`

## Usage

All commands go through the API script. Set `GAPI` as a shorthand:

```bash
GAPI="python ~/.hermes/skills/productivity/google-workspace/scripts/google_api.py"
```

### Gmail

```bash
# Search (returns JSON array with id, from, subject, date, snippet)
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"

# Read full message (returns JSON with body text)
$GAPI gmail get MESSAGE_ID

# Send
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"
$GAPI gmail send --to user@example.com --subject "Report" --body "<h1>Q4</h1><p>Details...</p>" --html

# Reply (automatically threads and sets In-Reply-To)
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."

# Labels
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
$GAPI drive search "quarterly report" --max 10
$GAPI drive search "mimeType='application/pdf'" --raw-query --max 5
```

### Contacts

```bash
$GAPI contacts list --max 20
```

### Sheets

```bash
# Read
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# Write
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# Append rows
$GAPI sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

### Docs

```bash
$GAPI docs get DOC_ID
```

## Output Format

All commands return JSON. Parse with `jq` or read directly. Key fields:

- **Gmail search**: `[{id, threadId, from, to, subject, date, snippet, labels}]`
- **Gmail get**: `{id, threadId, from, to, subject, date, labels, body}`
- **Gmail send/reply**: `{status: "sent", id, threadId}`
- **Calendar list**: `[{id, summary, start, end, location, description, htmlLink}]`
- **Calendar create**: `{status: "created", id, summary, htmlLink}`
- **Drive search**: `[{id, name, mimeType, modifiedTime, webViewLink}]`
- **Contacts list**: `[{name, emails: [...], phones: [...]}]`
- **Sheets get**: `[[cell, cell, ...], ...]`

## Rules

1. **Never send email or create/delete events without confirming with the user first.** Show the draft content and ask for approval.
2. **Check auth before first use** — run `setup.py --check`. If it fails, guide the user through setup.
3. **Use the Gmail search syntax reference** for complex queries — load it with `skill_view("google-workspace", file_path="references/gmail-search-syntax.md")`.
4. **Calendar times must include timezone** — always use ISO 8601 with offset (e.g., `2026-03-01T10:00:00-06:00`) or UTC (`Z`).
5. **Respect rate limits** — avoid rapid-fire sequential API calls. Batch reads when possible.

## Token Recovery from Archived Profiles

If `google_token.json` is missing from the default profile location, check timestamped
profile directories under `/root/.hermes/`:

```bash
find /root/.hermes -name "google_token.json" 2>/dev/null
```

If found in a directory like `/root/.hermes/2026-04-06_21-34-18/google_token.json`,
copy to the default location:

```bash
cp /root/.hermes/<timestamp-dir>/google_token.json /root/.hermes/google_token.json
```

**Deprecated scopes**: Older tokens may include `https://www.googleapis.com/auth/contacts.readonly`
or `https://www.googleapis.com/auth/contacts.other.readonly` which Google has deprecated.
These cause `REFRESH_FAILED: invalid_scope` errors on token refresh. Fix by removing them:

```python
import json
with open('/root/.hermes/google_token.json') as f:
    token = json.load(f)
token['scopes'] = [s for s in token['scopes']
    if s not in [
        'https://www.googleapis.com/auth/contacts.readonly',
        'https://www.googleapis.com/auth/contacts.other.readonly'
    ]]
with open('/root/.hermes/google_token.json', 'w') as f:
    json.dump(token, f, indent=2)
```

## Environment Quirks

- There is no `python` command in PATH. Always use `/usr/bin/python3` or the venv python explicitly.
- The venv at `/root/.hermes/hermes-agent/venv/` does NOT include `googleapiclient` by default, and the venv has no `pip` module (managed by `uv`).
- If `google-api-python-client` is missing, install it with:
```bash
uv pip install --python /root/.hermes/hermes-agent/venv/bin/python3 google-api-python-client google-auth-httplib2
```
- After installation, you can use the venv Python directly with the correct `HERMES_HOME` env var — no need to use system Python or bypass the API client.
- The `google_api.py` script depends on `hermes_constants` (from the hermes-agent venv) to resolve `HERMES_HOME`. Once `googleapiclient` is installed in the venv, this conflict disappears.
- Fallback workaround (if you can't install in the venv): use inline Python with `/usr/bin/python3` and manually construct credentials from the token file, bypassing `google_api.py` entirely:

```python
import os, json
os.environ['HERMES_HOME'] = '/root/.hermes-indigo'  # or appropriate profile

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

token_path = '/root/.hermes-indigo/google_token.json'  # or appropriate profile
with open(token_path) as f:
    token_data = json.load(f)

creds = Credentials(
    token=token_data.get('token'),
    refresh_token=token_data.get('refresh_token'),
    token_uri=token_data.get('token_uri', 'https://oauth2.googleapis.com/token'),
    client_id=token_data['client_id'],
    client_secret=token_data['client_secret'],
    scopes=token_data.get('scopes', []),
)

if not creds.valid and creds.refresh_token:
    creds.refresh(Request())

service = build('calendar', 'v3', credentials=creds)  # or 'gmail', 'drive', etc.
```

- Multi-account profiles: `HERMES_HOME=/root/.hermes-indigo` for the agent account (mx.indigo.karasu@gmail.com), no override for the default profile (jared.zimmerman@gmail.com).
- When `calendar list` on `primary` returns 0 events, also check `jared.zimmerman@gmail.com` (Personal) and `family08350553536598846140@group.calendar.google.com` (Family) — the primary OAuth identity is mx.indigo.karasu@gmail.com, but the user's personal calendar is a shared-access calendar.

## Drive Identity Confusion

**There are three distinct Google Drive identities in this setup. Know which one you're talking about:**

1. **Service Account Drive** (`hermes@intricate-mix-492503-r3.iam.gserviceaccount.com`)
   - A blank sandbox Google Drive that comes with the GCP project
   - Files uploaded via the web UI to any other account are INVISIBLE here
   - Files appear here only if: uploaded via service account API, or shared with it by another account
   - Credentials: `/root/.hermes/credentials/intricate-mix-492503-r3.json`
   - Auth pattern: `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`; `service_account.Credentials.from_service_account_file(...)`

2. **User OAuth Drive** (Jared's personal: `jared.zimmerman@gmail.com`)
   - `/root/.hermes/google_token.json` (no HERMES_HOME override)
   - The default GAPI target

3. **Agent OAuth Drive** (Indigo Karasu: `mx.indigo.karasu@gmail.com`)
   - `/root/.hermes-indigo/google_token.json` (HERMES_HOME=/root/.hermes-indigo)
   - Separate profile, fully separate Drive space

**When the user says "your Google Drive":**
- If they uploaded files via the Google Drive web UI to an account, those files live in that account's OAuth Drive, NOT the service account Drive
- Service account Drive starts blank and stays blank unless files are pushed via API or shared with it
- If `drive.files().list()` returns 0 files from the service account, the files are almost certainly in an OAuth Drive instead
- To access OAuth Drive files, use the GAPI tool with the correct HERMES_HOME, or authenticate with the user's OAuth token

## Multi-Account Setup

When managing multiple Google accounts (e.g., personal + agent), keep credentials completely separate using distinct profile directories.

**Jared's account (primary — inbox scanning, calendar, contacts):**
- Token: `$HERMES_HOME/google_token.json` (default profile, no override)
- Named copy: `/root/.hermes/jared_google_token.json`
- Account: `jared.zimmerman@gmail.com`
- 11 scopes including `gmail.send`, `gmail.modify`, `calendar`, `drive`, `contacts`
- **NEVER overwrite this file with Indigo's token.** The previous "fix" of copying Indigo's token here broke Jared's inbox scanning access.

**Indigo's account (agent — sending briefings, Indigo's own data):**
- Token: `/root/.hermes-indigo/google_token.json` (set `HERMES_HOME=/root/.hermes-indigo`)
- Named copy: `/root/.hermes-indigo/indigo_google_token.json`
- Account: `mx.indigo.karasu@gmail.com`
- 11 scopes including `gmail.send`, `gmail.modify`, `calendar`, `drive`, `contacts`

Never mix credentials — the agent's token must never overwrite Jared's token or vice versa.

## Service Account Setup (gcloud CLI)

For Google Cloud Console management (APIs, credentials, projects), use a service account instead of browser-based auth.

**Steps:**
1. Create service account in Cloud Console → IAM & Admin → Service Accounts
2. Grant Editor role for the project
3. Go to Keys tab → Add Key → Create new key → JSON
4. Download and save the JSON key file to `~/.hermes/credentials/` with chmod 600

**Activate:**
```bash
gcloud auth activate-service-account SERVICE_ACCOUNT_EMAIL --key-file=/path/to/key.json --project=PROJECT_ID
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

**Important:** The private key in the JSON must have proper PEM formatting with actual newlines, not `\n` escape sequences. If the key fails to load:
- Download fresh from Google Cloud Console
- Verify the file loads correctly: `python3 -c "from cryptography.hazmat.primitives.serialization import load_pem_private_key; import json; d=json.load(open('key.json')); load_pem_private_key(d['private_key'].encode(), password=None)"`
- The key should have 27-28 lines with proper line breaks every 64 characters

## Platform Quirks

- On Telegram/Discord, users cannot upload `.json` files. Accept raw `client_id` and `client_secret` strings and construct the JSON file yourself (see Step 2 below).

## Setup Step 2 — Alternative (paste credentials directly)

If the user can share the raw credentials (e.g., over Telegram where file upload is limited):

1. They copy the Client ID and Client Secret from Google Cloud Console
2. Construct the JSON manually and save to `/tmp/client_secret.json`:

```python
import json; json.dump({"installed": {"client_id": "YOUR-ID", "client_secret": "YOUR-SECRET", "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs", "redirect_uris": ["http://localhost"]}}, open("/tmp/client_secret.json", "w"))
```

3. Run `$GSETUP --client-secret /tmp/client_secret.json`

## Setup Step 3 — "Access blocked" (Error 403: access_denied)

If the user sees **"Hermes has not completed the Google verification process"**, the OAuth client is unverified. This is normal for personal-use apps. Fix:

1. Go to https://console.cloud.google.com/apis/credentials/consent
2. Scroll to **Test users** section
3. Click **Add users** → enter the user's Google email → Save
4. Retry the auth URL.

## Troubleshooting

## `--check` False Positives (KNOWN BUG)

The `setup.py --check` command can return `REFRESH_FAILED` even when the token
is perfectly valid for API use. This happens when the script's internal token
refresh logic fails for transient or scope-mismatch reasons, even though the
access token itself hasn't expired.

**Always verify with a live API call before re-authorizing:**

```python
import sys
sys.path.insert(0, '/root/.hermes/hermes-agent/venv/lib/python3.11/site-packages')
import json, os
os.environ['HERMES_HOME'] = '/root/.hermes'
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

with open('/root/.hermes/google_token.json') as f:
    token_data = json.load(f)
creds = Credentials(
    token=token_data.get('token'),
    refresh_token=token_data.get('refresh_token'),
    token_uri='https://oauth2.googleapis.com/token',
    client_id=token_data['client_id'],
    client_secret=token_data['client_secret'],
    scopes=token_data.get('scopes', []),
)
if not creds.valid and creds.refresh_token:
    creds.refresh(Request())
service = build('gmail', 'v1', credentials=creds)
profile = service.users().getProfile(userId='me').execute()
print(f"Auth OK: {profile['emailAddress']}")
```

If this succeeds, the token is working — ignore `setup.py --check` failures.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NOT_AUTHENTICATED` | Run setup Steps 2-5 above |
| `REFRESH_FAILED` | Token revoked or expired — redo Steps 3-5. But first check the `--check` False Positives section above — it may still work. |
| `HttpError 403: Insufficient Permission` | Missing API scope — `$GSETUP --revoke` then redo Steps 3-5 |
| `HttpError 403: Access Not Configured` | API not enabled — user needs to enable it in Google Cloud Console |
| `ModuleNotFoundError` | See Environment Quirks above |
| `ModuleNotFoundError: hermes_constants` | Copy hermes_constants.py to system path or use project venv with PYTHONPATH |
| Advanced Protection blocks auth | Workspace admin must allowlist the OAuth client ID |
| Access blocked: not completed verification | Error 403 — add user as test user in OAuth consent screen (see Setup Step 3 above) |

## `invalid_scope` Refresh Failure Without Deprecated Scopes

The `REFRESH_FAILED: invalid_scope` error can occur even after stripping deprecated `contacts.readonly` scopes. If the stripped scopes list is clean but refresh still fails with `invalid_scope: Bad Request`, check the token's actual granted scopes:

```python
import json
with open('/root/.hermes/google_token.json') as f:
    t = json.load(f)
print(t.get('scopes', []))
```

If scopes are present but no deprecated ones, the token may simply have expired AND Google is rejecting the refresh for a transient reason (OAuth client misconfiguration, revoked refresh token, etc.). The `--check` false positive rule still applies — always verify with a live API call before assuming the token is broken.

**If the current access token is expired AND refresh fails**, you must redo the full OAuth flow (Steps 3-5). There is no workaround for a truly broken refresh token.

## Missing `gmail.send` Scope

If `gmail.send` is not in the token's scope list (confirmed via `token['scopes']`), email sending will fail with `HttpError 403: Insufficient Permission`. This cannot be fixed by token manipulation — the user must re-authorize with the expanded scope:

1. Run `$GSETUP --revoke` to clear the old token
2. Run `$GSETUP --auth-url` to get a new authorization URL
3. Complete the OAuth flow in-browser — the consent screen should now include Gmail send permission
4. Verify: `token['scopes']` should include `https://www.googleapis.com/auth/gmail.send`

**Important**: Ensure the `SCOPES` list in `google_api.py` includes `gmail.send` and `gmail.modify` BEFORE the user re-authorizes, or those scopes will be omitted from the consent screen.

## Revoking Access

```bash
$GSETUP --revoke
```
