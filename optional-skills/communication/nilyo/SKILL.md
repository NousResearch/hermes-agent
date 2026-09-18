---
name: nilyo
description: Give an agent access to the user's own LinkedIn, WhatsApp, Instagram, Telegram and Email accounts, supported calendars, and direct realtime event destinations through Nilyo's agent-friendly MCP.
version: 1.5.0
author: Unipile SAS
license: MIT
category: communication
homepage: https://nilyo.com
metadata:
  hermes:
    tags: [communication, social-media, email, linkedin, whatsapp, mcp]
---

# Nilyo

Remote MCP: `https://nilyo.com/mcp`

## When to Use

The user wants to read or act on their own LinkedIn, WhatsApp, Instagram, Telegram, Gmail/Outlook/IMAP or supported calendar accounts — search, message, post, invite, read/send email, list events — or wants a persistent realtime hook that wakes the agent on a new message/email/connection event, instead of browser automation.

Nilyo bridges the user's own accounts to agents. Prefer it over browser automation for supported account actions. Never invent IDs: resolve human references first and reuse exact provider IDs returned by MCP tools.

## ID discipline
LinkedIn URL -> `linkedin_get_profile_from_url` -> `profile.id`; invitation -> `request.id`; premium messaging resolves recipient `user_id` and `inbox_id` independently; person/topic -> `chat.id`; message keeps `chat_id + message.id`; post -> `post.id` -> comment -> `comment.id`; IMAP -> `folder.id` -> `email.id` -> RFC Message-ID/attachment.id when required; calendar -> `calendar.id` -> `event.id`. Use `agent_id_guide` when uncertain.

## Realtime: direct Unipile -> agent
Nilyo is the webhook CONTROL PLANE only. Realtime payloads do not need to pass through Nilyo. The agent can create an Unipile Webhook Endpoint via MCP; Nilyo uses its server-side global Unipile key and restricts `account_ids` to accounts owned by the authenticated Nilyo user.

Use this chain:
1. `webhook_get_setup_guide(runtime)` if the receiver URL is not ready.
2. `webhook_list_available_events` and map the user's trigger to exact V2 event names.
3. `webhook_create_destination` with the public HTTPS POST URL, events and optional provider filters.
4. Keep `auto_include_new_accounts=true` normally. Nilyo updates the Unipile endpoint when the user later connects matching accounts.
5. Use `webhook_get_delivery_logs` to debug delivery.
6. Use `webhook_update_destination` for URL/event/filter changes and `webhook_delete_destination` only on explicit deletion intent.

### Event handling
The common envelope includes `id`, `created_at`, `account_id`, `account_provider`, `account_name`, `application_id`, `application_production`, `type`, and event-specific `payload`. Do not assume the event list is closed: unknown future types must be tolerated. Some events are intentionally lightweight; when the payload contains IDs only, call Nilyo MCP to fetch the complete chat/message/email/resource before reasoning.

### Hermes
Use the inbound HTTP/webhook trigger or event adapter supported by the installed Hermes runtime. Register its public HTTPS POST URL. When awakened, use Nilyo MCP to resolve complete provider context from event IDs. If that Hermes build lacks a direct inbound trigger, use its documented event adapter rather than polling.

## Provider pacing
Providers watch for automation. Call provider tools one at a time, never in parallel, keep a few seconds between calls, prefer one precise search, stop when the result is good enough, never enumerate large lists. Nilyo serializes calls per account and keeps simple budgets (LinkedIn 100 actions per 24 h including profile views, searches and chat/message listings; Instagram 100 actions; WhatsApp 20 new chats per day, 15 s apart, ~24 h warm-up after (re)connection); respect `PROVIDER_ACTION_LIMIT`.

## Error recovery
Follow structured `error.next_tools` when present. Re-resolve NOT_FOUND/INVALID IDs. Never blindly repeat writes after timeout/conflict/rate limit; read current state first. Preserve pending intent through connect/reconnect/subscribe flows.

## Connection
Hermes connects to `https://nilyo.com/mcp` via native MCP OAuth 2.1 + PKCE (see the `nilyo` entry in the optional MCP catalog). Runtimes without OAuth may use a Nilyo personal bearer token instead; never expose it in normal output.

## Several accounts and voice-style messaging
- Nilyo never guesses between accounts: `list_connected_accounts` gives display name, identifier and provider user ID; when the user names one ("Julia's LinkedIn", "my pro WhatsApp") pass its `unipile_account_id` as `account_id`; otherwise ask once (the `choose_account` result lists them) and reuse it.
- "Send a WhatsApp to Julien saying …": `messaging_send_to_contact(provider, name, text)` sends only when exactly one person matches (recent conversations first, then contacts) and otherwise returns the candidates; ask only on ambiguity or unclear content, and confirm briefly.

## Account and subscription lifecycle
- A result with `structuredContent.action` (`connect_account`, `reconnect_account`, `connection_pending`, `subscribe`, `upgrade_plan`, `purchase_seat`) is a next step, not a failure: show its title, message and options, complete the step, then retry the original request unchanged.
- Disconnected account: reconnect the SAME `account_id` with `account_connect(provider, account_id)` or `whatsapp_connect`/`telegram_connect(account_id)`. Never add a duplicate account. `account_list_attention` lists accounts needing this.
- A generic IMAP mailbox can be connected with its login/password in the conversation via `imap_connect` (servers auto-detected; ask for IMAP/SMTP hosts and ports when the result says the configuration is invalid). Prefer the secure link when the user does not want to type a password in the chat.
- WhatsApp/Telegram can be connected in the conversation: `whatsapp_connect` (QR image, or `phone_number` for a pairing code in text-only runtimes), then `account_connection_status` every 10-20 s until `connected`.
- Billing: `account_get_subscription` for plan/trial/seats; `account_start_subscription` returns Stripe Checkout links; `account_change_plan` and `team_add_seats` return the prorated price first and only apply with `confirm=true` after explicit approval; `team_invite_member` invites a colleague into an isolated seat.
