---
name: instagram-mcp-setup
description: >
  Workflow for deploying and configuring Instagram MCP servers (e.g., mcpware/instagram-mcp). 
  Details the gap between account creation and API access, and how to navigate Meta's strict bot-detection.
metadata:
  author: Indigo Karasu
  version: \"1.0.0\"
  hermes:
    tags: [instagram, mcp, meta, api]
---

# Instagram MCP Setup

Integrating an AI agent with Instagram requires navigating the strict divide between \"App Login\" (username/password) and \"Graph API access\" (tokens). Most MCP servers rely on the official Graph API, which cannot be used for account creation.

## The Meta Integration Gap

1. **Account Creation**: Cannot be done via API. Must be done manually via the mobile app to avoid instant bans.
2. **Trust Requirements**: Meta requires a \"Trusted\" identity to grant API access. The most reliable path is linking the Instagram account to a trusted, existing Facebook Page.
3. **Professional Status**: The Instagram account MUST be switched to a **Business or Creator** account in settings to be visible to the Graph API.

## Implementation Steps

### 1. Manual Account Setup
- Create the Instagram account via the **mobile app** (highest trust).
- **Settings** $\rightarrow$ **Account Type** $\rightarrow$ **Switch to Professional Account**.
- Link it to a Facebook Page (via an existing trusted FB account if a new one is blocked).

### 2. API Credentialing (Meta Developer Portal)
- Create an app at `developers.facebook.com`.
- Add the **Instagram Graph API** product.
- Grant necessary permissions: `instagram_basic`, `instagram_content_publish`, `instagram_manage_insights`, `pages_show_list`, `pages_read_engagement`.
- Generate a **Long-Lived Access Token** (60 days) using the Graph API Explorer.

### 3. MCP Server Deployment
- Clone the preferred MCP server (e.g., `mcpware/instagram-mcp`).
- Install dependencies using the environment's specific pip (e.g., `/root/.hermes/hermes-agent/venv/bin/pip`).
- Configure `.env` with:
  - `INSTAGRAM_ACCESS_TOKEN`
  - `INSTAGRAM_BUSINESS_ACCOUNT_ID`
  - `FACEBOOK_APP_ID`
  - `FACEBOOK_APP_SECRET`

## Pitfalls & Lessons Learned

- **The \"Username/Password\" Trap**: Do not attempt to use standard login credentials for MCP servers; they require OAuth tokens.
- **Bot Detection**: Avoid using Headless browsers or automated scripts for signup; Meta will trigger a block on the email or phone number.
- **Dependency Conflicts**: When installing MCP servers in a shared environment, use `--break-system-packages` or `--ignore-installed` if encountering `externally-managed-environment` errors in the global site-packages.
- **Database Pathing**: Ensure you are targeting the correct system database path (e.g., `/root/.hermes/commons/db/` vs `~/openclaw/`) to avoid creating duplicate/empty records.

## Verification
- Run the server's `setup.py` or a manual `GET` request to `/me` to verify token validity.
