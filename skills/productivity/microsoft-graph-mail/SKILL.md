---
name: microsoft-graph-mail
description: "Search and read the authenticated user's Microsoft 365 or Outlook mailbox via Microsoft Graph."
version: 1.0.0
author: Nous Research
license: MIT
required_credential_files:
  - path: microsoft_graph_client.json
    description: Microsoft Entra public-client app metadata for delegated Graph auth
  - path: microsoft_graph_token.json
    description: Microsoft Graph delegated OAuth token for this Hermes profile
  - path: microsoft_graph_oauth_pending.json
    description: Temporary OAuth state used between auth-url and auth-code setup steps
metadata:
  hermes:
    tags: [Microsoft, Graph, Outlook, Email, Mail, OAuth]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [google-workspace, himalaya]
---

# Microsoft Graph Mail

Search and read mail from the Microsoft mailbox that the current Hermes profile authorized. This skill uses delegated Microsoft Graph OAuth with PKCE and only calls `/me` mailbox endpoints, so it is scoped to the signed-in user rather than an application-wide mailbox grant.

## Scripts

- `scripts/setup.py` - non-interactive Microsoft OAuth setup
- `scripts/graph_mail.py` - read-only Microsoft Graph mail commands

## Entra App Requirements

Use a Microsoft Entra app registration configured as a public/native client. Do not paste or store a client secret for this skill.

Required app settings:

- Supported account types: choose the account population that owns the target mailboxes. For a company Microsoft 365 tenant, prefer single-tenant accounts in that tenant.
- Redirect URI: add `http://localhost:1` as a mobile/desktop or public-client redirect URI.
- Public client/native flows: enabled.
- Implicit grant: disabled.
- Microsoft Graph API permissions: delegated `Mail.Read` and `User.Read`.
- Admin consent: grant it if your tenant requires consent for `Mail.Read`.

Avoid application `Mail.Read` for this skill. Application mail access is app-wide and does not match per-user mailbox access unless your administrator also constrains it outside Hermes.

## First-Time Setup

Define shorthands:

```bash
MSETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/microsoft-graph-mail/scripts/setup.py"
MMAIL="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/microsoft-graph-mail/scripts/graph_mail.py"
```

### Step 0: Check if already set up

```bash
$MSETUP --check
```

If it prints `AUTHENTICATED`, setup is already complete.

### Step 1: Save the Entra app metadata

Use the app/client ID from the Entra app registration. For a company tenant, pass the tenant ID or verified tenant domain. Use `common`, `organizations`, or `consumers` only when the app intentionally supports those account populations.

```bash
$MSETUP --configure --client-id YOUR_CLIENT_ID --tenant YOUR_TENANT_ID
```

If your app uses a redirect URI other than `http://localhost:1`:

```bash
$MSETUP --configure --client-id YOUR_CLIENT_ID --tenant YOUR_TENANT_ID --redirect-uri http://localhost:1
```

### Step 2: Get the authorization URL

```bash
$MSETUP --auth-url
```

Open the printed URL. The browser will likely fail after approval when it redirects to `http://localhost:1`; that is expected. Copy the entire redirected URL from the address bar.

### Step 3: Exchange the authorization code

Paste the full redirected URL, or just the `code` value:

```bash
$MSETUP --auth-code "http://localhost:1/?code=...&state=..."
```

### Step 4: Verify

```bash
$MSETUP --check
```

The token is stored under the current Hermes profile and refreshes automatically.

To revoke local Hermes access:

```bash
$MSETUP --revoke
```

This deletes the local token. To fully revoke consent, remove the app from the user's Microsoft account or revoke it from Entra.

## Usage

Search messages:

```bash
$MMAIL search "from:alice@example.com subject:invoice" --max 10
$MMAIL search "quarterly planning" --max 5
```

Read a message returned by search:

```bash
$MMAIL get MESSAGE_ID
```

List the latest messages when you do not have a query yet:

```bash
$MMAIL list --max 10
```

All commands return JSON. Search/list results include message IDs, sender/recipient summaries, subject, date, and body preview. `get` returns the normalized body text, capped for model context safety.

## Scope

This skill is read-only. It does not send, reply, delete, archive, categorize, or download attachments.