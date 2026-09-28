# Connecting Messaging Platforms

Who is on the other end decides the flow, not the platform. The
organization's own people reach you through a channel an admin connects
in the dashboard; people outside it — customers, suppliers, partners —
are reached through the provider's API and a webhook, an ordinary
connection built as `{guides_root}/connections/guide.md` describes. The same platform
can be both at once: the team's Telegram is a dashboard channel while a
customer-facing Telegram bot is its own outside connection.

## The Team's Platform Is a Channel, Not a Connection

Where the organization's own people talk to you — Slack or Telegram — an
admin configures the native Hermes gateway in its dashboard. Telegram uses a BotFather-created bot token; access rules and group/topic mention policy are separate settings. The Config editor exposes those native settings. Messages there reach you natively, so never build an API or webhook connection to stand in for a configured channel.

## Outside the Organization: API Out, Webhook In

Two pieces, always both. Outbound is the provider's send API, its
credential in native secret configuration with the organization as owner. Inbound is a
webhook authored under the responsibility that owns the work — format,
ownership, and registration live in `{guides_root}/responsibility-authoring/references/webhooks.md` — with `key` on
the sender's identifier so each customer is one continuous conversation.
Register the minted URL with the provider; a test message arriving is
what proves the connection live.

The particulars are the provider's to define — research its official
docs with `web_search` at setup time, never from memory. WhatsApp goes
through Twilio's WhatsApp Business platform; a customer-facing Telegram
bot is created with BotFather and pointed at the webhook with
`setWebhook`. Every other platform follows the same shape: the official
bot or business-messaging API, credential in native secret configuration, webhook in, API
out. A platform offering no official API for automated messaging has no
legitimate surface — say so rather than reach for an unofficial client.
