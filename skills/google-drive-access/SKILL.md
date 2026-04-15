---
name: google-drive-access
description: Access Google Drive files via service account or user OAuth — visibility rules, credential setup, and file sharing patterns.
---

# Google Drive Access

Access patterns for Google Drive from the Hermes environment.

## Service Account vs User Drive

**Critical:** A Google Cloud service account has its OWN isolated Drive. It cannot see files in the user's personal Drive unless those files are explicitly shared with the service account email.

The service account email for this environment: `hermes@intricate-mix-492503-r3.iam.gserviceaccount.com`
Credentials file: `~/.hermes/credentials/intricate-mix-492503-r3.json`

## Two Access Modes

### Mode 1: Direct Service Account Access (fastest)
Use when files are shared with the service account email.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.hermes/credentials/intricate-mix-492503-r3.json
python3 -c "
from googleapiclient.discovery import build
from google.oauth2 import service_account
creds = service_account.Credentials.from_service_account_file(
    '~/.hermes/credentials/intricate-mix-492503-r3.json',
    scopes=['https://www.googleapis.com/auth/drive.readonly']
)
drive = build('drive', 'v3', credentials=creds)
results = drive.files().list(q=\"name contains 'Indigo'\", fields='files(name,id,mimeType)').execute()
print(results.get('files', []))
"
```

**Setup required:** User shares the folder with `hermes@intricate-mix-492503-r3.iam.gserviceaccount.com` as Viewer.

### Mode 2: User OAuth Access
Use when files are in personal Drive and not shared with service account.

Uses the OAuth token at `~/.hermes/google_token.json`. Requires `drive.readonly` scope.

```bash
# Check if Drive scope is present in existing token
python3 -c "
import json
with open('~/.hermes/google_token.json') as f:
    token = json.load(f)
print(token.get('scopes', []))
"

# If drive scope is missing, re-auth with --include-scopes including drive scope
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Service account returns 0 files | User hasn't shared the folder with the service account email |
| `DefaultCredentialsError` | Set `GOOGLE_APPLICATION_CREDENTIALS` env var or use `service_account.Credentials.from_service_account_file()` directly |
| `Insufficient Permission` | Token doesn't include drive scope — re-auth with broader scope |
| `access_tokens.db` errors | Service account auth is stored in gcloud, but Python needs `GOOGLE_APPLICATION_CREDENTIALS` env var |

## Quick Check

To verify Drive access works:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.hermes/credentials/intricate-mix-492503-r3.json
python3 -c "
from googleapiclient.discovery import build
from google.oauth2 import service_account
creds = service_account.Credentials.from_service_account_file(
    '$GOOGLE_APPLICATION_CREDENTIALS',
    scopes=['https://www.googleapis.com/auth/drive.readonly']
)
drive = build('drive', 'v3', credentials=creds)
me = drive.files().list(fields='files(name)', pageSize=5).execute()
print('Drive access OK' if 'files' in me else 'No files visible')
"
```

## Pitfalls

- `gcloud auth application-default login` requires interactive browser input — not usable in headless sessions
- The gcloud active account (`hermes@intricate-mix-492503-r3`) and the OAuth user account (`jared.zimmerman@gmail.com`) are completely separate
- Service account credentials in `~/.hermes/credentials/` work with Python's `google.oauth2.service_account` module, not with `gcloud adc`

## Large Drive Operations

When Drive contains 100k+ files, delegated subagent tasks will timeout due to API pagination exceeding iteration limits. Each Drive API page (~100 files) counts as one tool call. For 100k files you need ~1000 tool calls.

**Solution**: Write a standalone Python script that uses the Google Drive API directly in a foreground terminal process. Use `googleapiclient` with `list_next()` for pagination. Checkpoint progress every 1000 files. Run as a background terminal process with a timeout.

## Creating Google Docs

The service account has a Drive storage quota. If the quota is exceeded, `files().create()` returns HTTP 403 with `storageQuotaExceeded`. This blocks Google Doc creation entirely — the service account cannot create new files until existing files are deleted or the quota is increased.

**Fallback**: When Google Doc creation fails, save the report as a local Markdown file and record the local path in the link file. The report can be shared manually or migrated to a shared Drive with available quota.

## Using the Google Workspace CLI Script

A ready-made CLI wrapper exists at `~/.hermes/skills/productivity/google-workspace/scripts/google_api.py`. It handles auth, token refresh, and scope checking automatically.

```bash
cd ~/.hermes/skills/productivity/google-workspace/scripts

# Drive search
python3 google_api.py drive search "budget" --max 10

# The script uses build_service(api, version) and get_credentials() internally
python3 -c "
import sys; sys.path.insert(0, '.')
from google_api import build_service, get_credentials
svc = build_service('drive', 'v3')
results = svc.files().list(q='trashed=false', pageSize=10, fields='files(id,name,mimeType)').execute()
print(results)
"
```

**Important:** This script uses **OAuth** credentials (user's personal Drive), not the service account. When Bower or other OCAS skills need Drive access, they must decide which credential source to use — see the **Bower credential mismatch** pitfall below.

## Pitfalls

- **Quota exceeded**: If `files().create()` returns 403 `storageQuotaExceeded`, the service account Drive is full. Delete unused files from the service account's Drive or fall back to local storage.
- **Service account Drive isolation**: The service account's Drive is separate from the user's Drive. Files created by the service account are not visible in the user's Drive unless explicitly shared.
- **Bower credential mismatch**: The `folder_index.json` in `~/.hermes/commons/data/ocas-bower/` may have been built with service account credentials (which see a different, isolated Drive with 73,900+ folders). The user's personal Drive (via OAuth) has far fewer files. If Bower's scan results look wrong (0 files in scanned folders, or far fewer folders than expected), check which credential source was used. The `google_api.py` script in the google-workspace skill uses OAuth; a founding scan must use the same credential source to match.
- **API `list()` in inline Python**: Complex `svc.files().list()` calls in `-c "..."` one-liners fail with SyntaxError due to quote escaping. Always write multi-statement Drive API scripts to a file first, then execute with `python3 script.py`.