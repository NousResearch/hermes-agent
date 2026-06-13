---
name: google-workspace
description: "Google Workspace helpers for Gmail, Calendar, Drive, Docs, Sheets, and Contacts."
version: 2.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
required_credential_files:
  - path: google_token.json
  - path: google_client_secret.json
metadata:
  hermes:
    tags: [Google Workspace, Gmail, Calendar, Drive, Docs, Sheets]
    homepage: https://developers.google.com/workspace
---

# Google Workspace

Use the scripts in `scripts/` to bootstrap OAuth and call Google Workspace APIs
through a short-lived access token. The token and client secret live under
`${HERMES_HOME:-~/.hermes}` and are declared in `required_credential_files` so
remote sandboxes can mount only the files this skill needs.

## Setup

1. Save an OAuth desktop client secret as:
   `${HERMES_HOME:-~/.hermes}/google_client_secret.json`
2. Generate a consent URL:
   `python scripts/setup.py --auth-url`
3. Open the URL, approve the scopes, and copy the redirected URL or auth code.
4. Exchange it:
   `python scripts/setup.py --exchange-code "<code-or-redirect-url>"`

The resulting `${HERMES_HOME:-~/.hermes}/google_token.json` is an authorized
user credential and may be refreshed by `scripts/gws_bridge.py`.
