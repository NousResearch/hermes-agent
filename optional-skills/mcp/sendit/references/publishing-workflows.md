# Publishing Workflows

Use this reference for tool order, media handling, scheduling, and analytics
requests. Tool names assume the MCP server is named `sendit`; if the server name
differs, replace the `mcp_sendit_` prefix.

## Account And Team Discovery

Call before publishing, scheduling, connecting, or reporting:

```text
mcp_sendit_list_connected_accounts
```

If the user works across teams, discover teams and carry the intended `team_id`
into later tool calls when the tool schema supports it:

```text
mcp_sendit_list_teams
```

## Platform Requirements

Use requirements when the user asks about limits, when content includes media,
or when the target platform has platform-specific constraints:

```text
mcp_sendit_get_platform_requirements
```

Then validate the exact content before any publish or schedule call:

```text
mcp_sendit_validate_content
```

## Media Uploads

For local images, videos, or chat attachments, create a SendIt upload session
and use the returned HTTPS asset URL in later publish or schedule calls:

```text
mcp_sendit_create_upload_session
mcp_sendit_get_upload_session
```

Do not pass arbitrary local file paths to publishing tools unless the tool
description explicitly supports local paths.

## Preview

Preview whenever the tool is available, especially for multi-platform posts:

```text
mcp_sendit_preview_content
```

Ask the user to confirm material changes when previews reveal truncation,
platform warnings, or media problems.

## Publish Now

Use immediate publishing only after the user clearly asks to publish now:

```text
mcp_sendit_publish_content
```

Good confirmations mention target platforms and account/team context, for
example: "Publish this to the company LinkedIn page and Threads now."

## Schedule

Use scheduling for delayed posts:

```text
mcp_sendit_schedule_content
```

Confirm the exact date, time, and timezone before scheduling. If the user gives
relative time such as "tomorrow morning", resolve it to a concrete timestamp
before calling the tool.

## Platform Connection

For missing platforms:

```text
mcp_sendit_connect_platform
```

Send the returned OAuth URL to the user, ask them to complete authorization,
then call `mcp_sendit_list_connected_accounts` again to verify.

## Analytics

Use analytics for performance summaries, post status, or platform comparison:

```text
mcp_sendit_get_analytics
```

Clarify date range, platform, account/team, and whether the user wants aggregate
or post-level reporting when the request is ambiguous.
