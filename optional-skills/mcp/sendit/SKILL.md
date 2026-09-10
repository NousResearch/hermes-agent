---
name: sendit
description: Use SendIt from Hermes Agent for social publishing, scheduling, platform connection, media upload, content validation, previews, and analytics through remote MCP OAuth.
version: 0.2.1
author: SendIt / Infinite Apps AI
license: MIT
licenseUrl: https://github.com/Shree-git/sendit-hermes-skills/blob/main/LICENSE
platforms: [linux, macos]
category: social-media
tags: [sendit, social-media, mcp, oauth, publishing, scheduling, analytics]
repository: https://github.com/Shree-git/sendit-hermes-skills
sourceUrl: https://github.com/Shree-git/sendit-hermes-skills/tree/main/skills/sendit
summary: SendIt gives Hermes a remote OAuth MCP workflow for connecting social accounts, validating content, uploading media, publishing or scheduling posts, and reviewing analytics.
icon: https://sendit.infiniteappsai.com/favicon.ico
metadata:
  hermes:
    category: social-media
    tags: [sendit, social-media, mcp, oauth, publishing, scheduling, analytics]
    homepage: https://sendit.infiniteappsai.com
    repository: https://github.com/Shree-git/sendit-hermes-skills
    config:
      - key: sendit.mcp_server_name
        description: Hermes MCP server name for SendIt.
        default: sendit
        prompt: SendIt MCP server name
      - key: sendit.mcp_url
        description: SendIt remote MCP endpoint.
        default: https://sendit.infiniteappsai.com/api/mcp
        prompt: SendIt MCP endpoint
---

# SendIt

## When To Use

- Publishing or scheduling social posts to connected platforms.
- Listing or connecting LinkedIn, Instagram, TikTok, Threads, X, and other
  SendIt-supported accounts.
- Validating text, media, captions, hashtags, and platform-specific limits.
- Creating upload sessions for local images, videos, or chat attachments.
- Checking previews and analytics before or after publishing.
- Working in a SendIt team context after discovering teams.

Do not use SendIt for local-only drafting.

## Quick Reference

Use `https://sendit.infiniteappsai.com/api/mcp` as the remote MCP URL. Do not
use `https://sendit.infiniteappsai.com/mcp`; that endpoint is reserved for the
ChatGPT app profile and exposes a reduced catalog.

```yaml
mcp_servers:
  sendit:
    url: "https://sendit.infiniteappsai.com/api/mcp"
    auth: oauth
```

Hermes exposes tools as `mcp_<server>_<tool>`. Default SendIt tools include:

- `mcp_sendit_list_connected_accounts`
- `mcp_sendit_list_teams`
- `mcp_sendit_connect_platform`
- `mcp_sendit_get_platform_requirements`
- `mcp_sendit_validate_content`
- `mcp_sendit_create_upload_session`
- `mcp_sendit_preview_content`
- `mcp_sendit_publish_content`
- `mcp_sendit_schedule_content`
- `mcp_sendit_get_analytics`

- Remote VPS, Telegram, and OAuth callback replay: `references/remote-oauth.md`
- Publishing and scheduling examples: `references/publishing-workflows.md`
- File, network, and sensitive-data scope: `references/security.md`

## Permissions

Helper scripts may copy this skill into `~/.hermes/skills/social-media/sendit`,
read or update `~/.hermes/config.yaml` or `$HERMES_HOME/config.yaml`, run
`hermes mcp login sendit`, and write `/tmp/sendit-hermes` PID/log files.
Callback replay only accepts `localhost` or `127.0.0.1` `/callback` URLs and
redacts sensitive query parameters.

## Core Workflows

### Configure Hermes

Run the installer to set up SendIt or repair an older `/mcp` URL:

```bash
node ${HERMES_SKILL_DIR}/scripts/install-sendit-hermes.mjs
```

### Complete Remote OAuth

For a VPS, Telegram-only handoff, or browser-on-another-device setup, follow
`references/remote-oauth.md`.

Minimal sequence:

```bash
node ${HERMES_SKILL_DIR}/scripts/start-oauth-login.mjs
node ${HERMES_SKILL_DIR}/scripts/complete-oauth-callback.mjs '<PASTED_CALLBACK_URL>'
```

Treat pasted callback URLs as sensitive because their `code` and `state` query
parameters can complete OAuth.

### Publish Or Schedule

List accounts, choose the team when relevant, validate content, and preview
when possible. Publish immediately only on clear user request. Confirm date,
time, and timezone before scheduling. See
`references/publishing-workflows.md`.

### Connect A Platform

1. Call `mcp_sendit_list_connected_accounts`.
2. For missing accounts, call `mcp_sendit_connect_platform`.
3. Send the returned platform OAuth URL to the user.
4. Re-run `mcp_sendit_list_connected_accounts` after authorization.

## Safety Rules

- Do not publish, schedule, delete, reply, or trigger scheduled posts unless the
  user clearly asks for that action.
- Validate before publish or schedule.
- Preview before publish when the preview tool is available.
- Confirm destructive actions such as deleting published or scheduled posts.
- Do not ask the user for a SendIt API key. This setup uses MCP OAuth.
- Never echo full OAuth `code` or `state` values in chat, logs, or summaries.
- Only replay callback URLs whose host is `127.0.0.1` or `localhost` and whose
  path is `/callback`.

## Verification

After OAuth completes and MCP reloads, ask Hermes:

```text
Use the SendIt skill and list my connected social accounts.
```

Hermes should call `mcp_sendit_list_connected_accounts` or the equivalent
SendIt MCP tool for the configured server name.
