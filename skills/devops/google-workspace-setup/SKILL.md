---
name: google-workspace-setup
description: Set up Google Workspace OAuth and service account access for Hermes Agent. Covers multi-account isolation, OAuth flow with localhost redirects, service account activation, and enabling APIs via Google Cloud SDK.
category: devops
metadata: {"hermes":{"emoji":"🔑"}}
---

# Google Workspace Setup

Configures Google Workspace access for Hermes Agent with proper account isolation.

## When to Use

- Setting up Gmail, Calendar, Drive, Contacts, Sheets, Docs access
- Adding a new Google account (personal and agent accounts must use separate profiles)
- Enabling Google Cloud APIs via service account
- Fixing authentication issues with existing Google integrations

## Account Isolation Rules

- The user's personal account (e.g. jared.zimmerman@gmail.com) uses /root/.hermes/
- The agent's account (e.g. mx.indigo.karasu@gmail.com) must use a SEPARATE profile directory
- Never mix tokens, clients, or credentials between accounts
- Each account needs its own OAuth Client ID + Client Secret from Google Cloud Console

## OAuth Setup (User Account)

1. Create OAuth Client in Google Cloud Console
2. Add Authorized JavaScript origins: http://localhost
3. Add Authorized redirect URIs: http://localhost:1
4. Configure scopes needed:
   - Gmail: gmail.readonly, gmail.send, gmail.modify
   - Calendar: calendar
   - Drive: drive.readonly
   - Contacts: contacts.readonly
   - Sheets: spreadsheets
   - Docs: documents.readonly
   - People: contacts or directory.readonly
5. Generate authorization URL with PKCE flow
6. User clicks URL in browser, authenticates with Google account
7. User pastes back the callback URL from browser address bar
8. Exchange code for tokens and save to /root/.hermes/google_token.json

For agent account (mx.indigo.karasu@gmail.com):
```bash
# Create separate profile directory
mkdir -p /root/.hermes-indigo/
# Use separate client ID/secret for agent
# Save token to /root/.hermes-indigo/google_token.json
```

## Service Account Setup

**Step 1: Create Service Account**
1. Go to Google Cloud Console → IAM & Admin → Service Accounts
2. Create service account (e.g., hermes@project.iam.gserviceaccount.com)
3. Grant appropriate roles (Editor for full access)
4. Create key → JSON format → download

**Step 2: Save Key**
```bash
mkdir -p ~/.hermes/credentials && chmod 700 ~/.hermes/credentials
# Save JSON key to: ~/.hermes/credentials/{project-name}.json
```

**Step 3: Activate with gcloud**
```bash
gcloud auth activate-service-account {sa-email} --key-file={path-to-key} --project={project}
```

**Step 4: Enable APIs**
```bash
# Enable via gcloud
gcloud services enable gmail.googleapis.com --project={project}
gcloud services enable drive.googleapis.com --project={project}
gcloud services enable people.googleapis.com --project={project}

# Or enable via Python API (works when gcloud has OpenSSL issues)
python3 -c "
from google.oauth2 import service_account
from googleapiclient.discovery import build
creds = service_account.Credentials.from_service_account_file('{key-path}')
su = build('serviceusage', 'v1', credentials=creds)
for api in ['gmail.googleapis.com', 'drive.googleapis.com', 'people.googleapis.com']:
    su.services().enable(name=f'projects/{project}/services/{api}', body={}).execute()
    print(f'Enabled {api}')
"
```

## Common API Endpoints to Enable

| API | When Needed |
|-----|-------------|
| gmail.googleapis.com | Email access |
| calendar.googleapis.com | Calendar management |
| drive.googleapis.com | File organization |
| people.googleapis.com | Google Contacts |
| sheets.googleapis.com | Spreadsheet access |
| docs.googleapis.com | Document editing |
| admin.googleapis.com | Workspace admin operations |
| cloudresourcemanager.googleapis.com | Project management |
| iam.googleapis.com | Service account management |
| serviceusage.googleapis.com | API enablement |

## Pitfalls

### Token Refresh Fails with `invalid_scope` (Corrupted Token)

Even when `google_token.json` shows correct scopes, Google may reject token refresh with:
```
google.auth.exceptions.RefreshError: ('invalid_scope: Bad Request', {'error': 'invalid_scope'})
```

This means the token is corrupted and must be re-issued. Do not try to repair it.

**Re-authentication flow:**

```bash
# 1. Check current auth status (requires PYTHONPATH)
PYTHONPATH=/root/.hermes/hermes-agent python3 /root/.hermes/skills/productivity/google-workspace/scripts/setup.py --check

# 2. Generate fresh auth URL
PYTHONPATH=/root/.hermes/hermes-agent python3 /root/.hermes/skills/productivity/google-workspace/scripts/setup.py --auth-url

# 3. User visits URL, authorizes, copies the `code=` from redirect URL

# 4. Exchange code for new token
PYTHONPATH=/root/.hermes/hermes-agent python3 /root/.hermes/skills/productivity/google-workspace/scripts/setup.py --auth-code {CODE}

# 5. Verify
PYTHONPATH=/root/.hermes/hermes-agent python3 /root/.hermes/skills/productivity/google-workspace/scripts/setup.py --check
```

The `--check` command returns exit code 0 on success, 1 on failure.

### Private Key Format Issues
- If gcloud says "Could not deserialize key data", check:
  - The key file must have actual newlines, not \n strings
  - The key must be exactly 64 chars per line
  - Check if the key is corrupted during paste/transfer

### Service Account Can't Access Personal Drive Files
- Service accounts have their own isolated Drive
- To access user's files: share files WITH the service account email
- For shared Drive access: add service account as member

### gcloud OpenSSL Compatibility
- gcloud sometimes has OpenSSL 3.0 compatibility issues
- Python google-api-python-client works reliably as alternative
- Always have both installed for fallback

### OAuth Token Expiry
- Access tokens expire in ~1 hour
- Refresh tokens should persist indefinitely
- Always check token validity before API calls:
```python
from google.auth.transport.requests import Request
if not creds.valid and creds.expired and creds.refresh_token:
    creds.refresh(Request())
```

### Google Calendar API Script Limitations

The `google_api.py` script has specific behaviors not obvious from usage:

**No Update Command:**
- `calendar update` does NOT exist
- To modify an event: delete then recreate
- Delete uses positional event_id: `calendar delete {event_id}` (not `--id` or `--event-id`)

**Timezone Required:**
- Start/end times MUST include timezone offset
- Valid: `2026-04-16T11:30:00-07:00`
- Invalid: `2026-04-16T11:30:00` (returns "Missing time zone definition")

**Create Command Arguments:**
```bash
python3 google_api.py calendar create \
  --summary "Event Title" \
  --start "2026-04-16T11:30:00-07:00" \
  --end "2026-04-16T12:30:00-07:00" \
  --location "Full Address" \
  --description "Details"
```

### Cron Job Fallback When Token is Expired

When a **cron job** encounters `invalid_grant` (no user present for re-auth):

1. **Do NOT suppress the error.** Surface it prominently in the briefing/output.
2. **Fall back to cached data** from `events.jsonl` or previous journal runs. The events.jsonl in `{agent_root}/commons/data/ocas-sands/` contains queried events with timestamps.
3. **Label all output as "cached/stale"** — make it clear the data may not reflect current calendar state.
4. **Generate the re-auth URL** so the user can fix it when they're next available:
   ```bash
   PYTHONPATH=/root/.hermes/hermes-agent python3 /root/.hermes/skills/productivity/google-workspace/scripts/setup.py --auth-url
   ```
5. **Log the auth failure** to decisions.jsonl and the skill journal.
6. **Write the briefing anyway** from cached data — a stale briefing is better than no briefing.

**Pattern for reading cached events:**
```python
# Read events.jsonl for previously-queried events on target date
import json
events = []
with open(f"{data_dir}/events.jsonl") as f:
    for line in f:
        ev = json.loads(line.strip())
        if target_date in ev.get("start", ""):
            events.append(ev)
```

**Key insight:** The Sands evening briefing queries events for the *next* day and writes them to events.jsonl. So the morning briefing can fall back to the previous evening's data even when the Google API is unreachable.

### Skill Invocation Pattern

OCAS skills (sands, weave, scout, etc.) are NOT CLI executables. Do NOT attempt:
- `hermes sands.event.create` — fails (not a hermes CLI command)
- `openclaw weave.upsert.person` — fails (openclaw CLI doesn't exist)
- `hermes chat --skill sands` — fails (no such flag)

**Correct approaches:**
1. Use the Google Workspace API scripts directly for Calendar/Gmail/Drive
2. Use `delegate_task` with a subagent that has the skill loaded
3. Access LadybugDB (Weave) directly via Python if the module is installed
4. For Scout: use web_search/web_extract directly for OSINT

If a skill's CLI entry point is missing, fall back to direct API calls or database access.

## Verification

Test each service after setup:
```python
# Gmail
from googleapiclient.discovery import build
service = build('gmail', 'v1', credentials=creds)
profile = service.users().getProfile(userId='me').execute()
print(f"Gmail connected: {profile['emailAddress']}")

# Drive
drive = build('drive', 'v3', credentials=creds)
results = drive.files().list(pageSize=5, fields="files(id, name)").execute()
files = results.get('files', [])
print(f"Drive: {len(files)} files visible")

# Calendar
calendar = build('calendar', 'v3', credentials=creds)
calendars = calendar.calendarList().list().execute()
print(f"Calendar: {len(calendars.get('items', []))} calendars")

# Contacts
people = build('people', 'v1', credentials=creds)
count = 0
page_token = None
while True:
    result = people.people().connections().list(
        resourceName='people/me',
        pageSize=1000,
        personFields='names,emailAddresses',
        pageToken=page_token
    ).execute()
    count += len(result.get('connections', []))
    page_token = result.get('nextPageToken')
    if not page_token:
        break
print(f"Contacts: {count} contacts")
```

## Environment Variables

```bash
# User's personal account
export GOOGLE_CLIENT_ID=your-oauth-client-id
export GOOGLE_CLIENT_SECRET=your-oauth-client-secret

# Agent's account (if separate)
export AGENT_GOOGLE_CLIENT_ID=agent-oauth-client-id
export AGENT_GOOGLE_CLIENT_SECRET=agent-oauth-client-secret

# Service account (alternative auth method)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

## Storage Locations

```
~/.hermes/
├── google_token.json              # User's OAuth token
├── google_client_secret.json      # User's OAuth client secrets
├── credentials/
│   ├── project-name.json          # Service account key
│   └── ...
└── hermes-indigo/                 # Agent's separate profile
    └── google_token.json
```

## Cron Job Pattern for Background Tasks

When registering background tasks from skills:
```bash
hermes cron list  # Check existing first

# Create new job
hermes cron add --name {skill:task} \
  --schedule "{cron_expression}" \
  --command "{skill.command}" \
  --sessionTarget isolated \
  --lightContext true \
  --timezone America/Los_Angeles
```

Common schedules:
- Updates: `0 0 * * *` (midnight daily)
- Morning briefs: `0 6 * * *`
- Evening briefs: `0 20 * * *`
- Weekly deep scans: `0 1 * * 0` (Sunday 1am)
- Weekday morning tasks: `0 6 * * 1-5`

All cron jobs:
- deliver to `local` to avoid chat spam
- Use isolated sessions
- Light context enabled
- America/Los_Angeles timezone
