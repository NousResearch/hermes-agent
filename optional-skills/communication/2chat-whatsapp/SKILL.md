---
name: 2chat-whatsapp
description: WhatsApp, SMS, and contacts via the 2Chat MCP server.
version: 1.0.0
author: Carlos Ruiz-Díaz (@2ChatCo), 2Chat
license: MIT
homepage: https://2chat.co
platforms: [macos, linux, windows]
metadata:
  hermes:
    category: communication
    tags: [whatsapp, sms, 2chat, waba, messaging, mcp, contacts]
---

# 2Chat — WhatsApp, SMS & Voice

Connects Hermes to **2Chat's official remote MCP server** so the agent can send and read WhatsApp
messages (WhatsApp Web *and* the WhatsApp Business API), send SMS, manage WABA templates, work with
contacts and groups, publish WhatsApp statuses, browse catalogs, and pull call records — through the
user's existing 2Chat account.

- **Server:** `https://mcp.2chat.io/mcp`
- **Transport:** Streamable HTTP (Hermes default for remote servers)
- **Auth:** OAuth 2.1 (browser sign-in, PKCE). No API key. Hermes stores tokens at `~/.hermes/mcp-tokens/2chat.json` and refreshes them automatically.

## When to Use

Use this skill whenever the user wants to:

- Send a WhatsApp or SMS message, or check whether a number is on WhatsApp.
- Send WhatsApp **Business API** messages with approved templates, or list/sync/price WABA templates.
- Read conversations, group messages, or group participant lists.
- Publish a WhatsApp text/image/video **status** (story).
- Create, search, update, or delete **contacts**.
- Inspect connected channels (WhatsApp Web, WABA, virtual numbers) or pull **call records**.

## Prerequisites

- A [2Chat](https://2chat.co) account with at least one connected channel.
- The `2chat` MCP server configured in Hermes (see Quick Reference). No API key is required.

## Quick Reference — one-time setup

Add the 2Chat server to your Hermes config (`~/.hermes/config.yaml`) under `mcp_servers` — or use the
dashboard **Profile Builder → MCP Servers** with the same URL and OAuth:

```yaml
mcp_servers:
  2chat:
    url: "https://mcp.2chat.io/mcp"
    auth: oauth
```

A copy is in `references/mcp-server.yaml`. On first connection Hermes opens the 2Chat sign-in page in
your browser and persists the tokens. Confirm with: *"List my connected 2Chat WhatsApp channels."*

## Procedure

1. Ensure the `2chat` MCP server is connected. If its tools aren't available, it isn't configured
   yet — add it per Quick Reference and complete the browser sign-in.
2. Identify the sending channel. If multiple numbers exist, call `get_whatsapp_numbers` /
   `get_waba_numbers` and confirm which one to use.
3. For a first-time WhatsApp recipient, verify reachability with `check_if_number_is_on_whatsapp`.
4. Compose the message, **echo recipient + exact body back to the user for confirmation**, then send
   with `send_whatsapp_message`, `send_waba_message`, or `send_sms`.
5. Report the resulting message id / status to the user.

## Tools

Full descriptions are in `references/tools.md`. Summary by area:

- **Account:** `get_who_am_i`, `get_billing_info`
- **WhatsApp Web:** `send_whatsapp_message`, `check_if_number_is_on_whatsapp`, `get_whatsapp_messages`
- **WABA:** `send_waba_message`, `get_waba_templates`, `sync_waba_templates`, `calculate_waba_template_cost`
- **SMS:** `send_sms`
- **Channels:** `get_whatsapp_numbers`, `get_whatsapp_number`, `execute_whatsapp_channel_command`, `get_waba_numbers`, `get_waba_number`
- **Conversations & groups:** `list_whatsapp_conversations`, `list_whatsapp_groups`, `list_whatsapp_group_participants`, `get_whatsapp_group_messages`
- **Status:** `set_whatsapp_text_status`, `set_whatsapp_image_status`, `set_whatsapp_video_status`
- **Catalog:** `list_whatsapp_catalog_products`
- **Contacts:** `create_contact`, `get_contact`, `list_contacts`, `search_contacts`, `update_contact`, `delete_contact`
- **Calls:** `list_virtual_numbers`, `get_call_history`, `get_call_details`, `get_call_price`

## Pitfalls

- **Sending costs money and reaches real people.** Always confirm recipient and exact body before any
  `send_*` call. Never batch-send without explicit confirmation.
- **WABA needs an approved template** outside the 24-hour service window; use `get_waba_templates`
  (and `calculate_waba_template_cost` if cost matters) first.
- **`delete_contact` is permanent** — confirm the contact UUID first.
- **No channels connected?** Read tools return empty; connect a number in the 2Chat dashboard first.
- **Auth expired?** Re-trigger the OAuth sign-in; Hermes refreshes tokens automatically, but a revoked
  grant needs a fresh browser login.

## Verification

- `get_who_am_i` returns the authenticated 2Chat account → server connected and authorized.
- `get_whatsapp_numbers` lists at least one channel → ready to send.
- A test `send_whatsapp_message` to your own number returns a message id → end-to-end working.
