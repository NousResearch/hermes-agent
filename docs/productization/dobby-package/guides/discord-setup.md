# Discord Setup

Reader: the customer operator who owns the Discord app. Next action: create a
bot account, invite it with minimum permissions, and restrict where Dobby can
listen and respond.

## Ownership Model

Use a Discord application owned by the customer. Dobby should authenticate only
as that bot account. Do not automate a normal user account, do not use a
self-bot, and do not scrape channels passively.

Official Discord references:

- Discord bot setup: <https://discord.com/developers/docs/bots>
- Gateway intents: <https://docs.discord.com/developers/topics/gateway>
- Message content caveat: <https://docs.discord.com/developers/resources/message>
- Self-bot policy: <https://support.discord.com/hc/en-us/articles/115002192352-Automated-user-accounts-self-bots->

## Create The App

1. Open the Discord Developer Portal.
2. Create a new application for this package.
3. Add a bot user on the Bot page.
4. Reset and copy the bot token once, then store it only in the package env
   file or secret manager.
5. Keep the bot install restricted to the customer workspace.

## OAuth Scopes

Use these scopes for the invite URL:

- `bot`
- `applications.commands`

Do not grant `Administrator`.

## Minimum Permissions

Start with the smallest permissions that support the selected channel:

- View Channels
- Send Messages
- Read Message History
- Use Slash Commands
- Attach Files, only if attachment review or file replies are enabled
- Add Reactions, only if approval reactions are enabled

If the bot needs thread or forum behavior, add the matching thread permissions
only for the target channels.

## Gateway Intents

Configure intents deliberately:

- `MESSAGE_CONTENT` may be needed for free-text commands in guild channels.
- Prefer slash commands, DMs, direct mentions, and allowlisted free-response
  channels to reduce broad message access.
- `GUILD_MEMBERS` is only needed if the deployment resolves or gates users by
  member data that Discord does not otherwise provide.
- `GUILD_PRESENCES` is not part of the V1 package.

Discord treats some intents as privileged. Verified apps, and apps eligible for
verification, may need approval before those intents work. If message content
is not configured or approved where required, Discord can deliver message
events with content-like fields empty.

## Allowlist Policy

Set allowlists before the first gateway start.

```env
DISCORD_HOME_CHANNEL=<DISCORD_CHANNEL_ID>
DISCORD_ALLOWED_USERS=<DISCORD_USER_ID_LIST>
DISCORD_ALLOWED_CHANNELS=<DISCORD_CHANNEL_ID_LIST>
DISCORD_REQUIRE_MENTION=true
DISCORD_FREE_RESPONSE_CHANNELS=<OPTIONAL_CHANNEL_ID_LIST>
```

Rules:

- Use numeric Discord IDs, not display names.
- Start with one staging channel and one operator user.
- Keep free-response channels empty until staging proves the bot stays quiet in
  ordinary conversation.
- Review allowlists before inviting the bot to any additional server.

## Safe Channel Pattern

Recommended staging layout:

- `#dobby-staging`: allowed, mention required.
- `#dobby-demo`: allowed, synthetic demo data only.
- `#general`: not allowed.

Recommended live layout:

- One private operator channel for commands and status.
- One webhook inbox delivery channel if signed webhooks are enabled.
- No passive ingestion channels.

## Validation Checklist

- Bot is invited only to the intended server.
- Bot lacks `Administrator`.
- Token is not printed in terminal history, logs, or tickets.
- User and channel allowlists are present.
- Message content behavior is tested in staging.
- A message in a non-allowed channel receives no response.
- An allowed user mention in the allowed channel receives one response.
