---
name: google-cloud-api-setup
category: devops
description: Setup Google Cloud service accounts, enable APIs, manage Google Workspace access programmatically via Python google-auth.
---

# Google Cloud API & Service Account Setup

Trigger: Setting up Google Cloud access, enabling APIs, managing service accounts, or interacting with Google Workspace via API.

**Critical: Credential Handling**
- Save service account keys to `~/.hermes/credentials/` with chmod 600.
- NEVER paste keys directly into `terminal` commands — terminal security scanning blocks PEM private key blocks. Always use `write_file` to save JSON keys.
- When the user pastes a service account JSON, the `private_key` field contains `\n` as JSON escape sequences (literal backslash+n). `write_file` writes them as-is. Python's `json.load()` (reading from file) handles these correctly. `json.loads()` on an inline string may corrupt them — prefer `json.load(open(path))`.
- Keep Indigo Karasu credentials completely separate from Jared Zimmerman's — different profiles, different files.

**Authentication (Python Only)**
- `gcloud auth activate-service-account` FAILS on modern Linux with OpenSSL 3.0 — error "Could not deserialize key data, unsupported PEM format".
- Use Python google-auth instead:
```python
from google.oauth2 import service_account
from googleapiclient.discovery import build
cred_path = '/root/.hermes/credentials/YOUR-KEY.json'
creds = service_account.Credentials.from_service_account_file(cred_path)
```

**Verify Authentication**
- Test by accessing an API:
```python
service = build('cloudresourcemanager', 'v1', credentials=creds)
response = service.projects().get(projectId='YOUR-PROJECT-ID').execute()
```

**Enabling APIs Programmatically**
- Use the `serviceusage` API to enable APIs:
```python
su = build('serviceusage', 'v1', credentials=creds)
su.services().enable(name='projects/YOUR-PROJECT/services/gmail.googleapis.com').execute()
```

**Core APIs to Enable for Google Workspace**
- gmail.googleapis.com
- calendar.googleapis.com (403 if no Workspace subscription)
- drive.googleapis.com
- sheets.googleapis.com
- docs.googleapis.com
- cloudresourcemanager.googleapis.com
- iam.googleapis.com
- iamcredentials.googleapis.com
- servicemanagement.googleapis.com
- serviceusage.googleapis.com
- people.googleapis.com
- admin.googleapis.com

**GitHub PAT Authentication**
- Authenticate GH CLI: `cat PAT_FILE | gh auth login --with-token`
- Use `GH_CONFIG_DIR` env var for multi-account separation.

**Verification Checklist**
- [ ] Service account key loaded successfully by Python `google-auth`
- [ ] Service account email matches expected
- [ ] At least one GCP API test call succeeds
- [ ] Core APIs (gmail, drive, sheets, docs, cloudresourcemanager) enabled
- [ ] No credential mixing between Indigo and Jared accounts