# Quickstart

Reader: a buyer or operator installing the V1 Dobby/Hermes package for the
first time. Next action: stand up a staging Discord bot, verify the package
with synthetic data, then promote the same configuration pattern to live.

## Scope

V1 is a self-hosted, Discord-first Hermes operator. The customer brings:

- A customer-owned Discord application and bot token.
- A model endpoint and API key compatible with Hermes provider configuration.
- A fresh package-owned `HERMES_HOME`.
- Optional webhook senders that can sign HMAC requests.

The package includes no model artifacts, does not start an external memory
server, and keeps the default path limited to Discord plus signed webhooks.

## Assumptions

- You can install and run Hermes on a host you control.
- You have a private staging Discord server or staging channel.
- You can create a new Discord app in the Discord Developer Portal.
- You can create a model endpoint API key for this package.
- You will not copy an existing `~/.hermes`, logs, sessions, or secrets into
  this installation.

## Fresh Home

Create a new runtime home for staging. Use a path that is clearly separate from
any personal Hermes install.

```bash
export HERMES_HOME="$HOME/.hermes-dobby-staging"
mkdir -p "$HERMES_HOME"
chmod 700 "$HERMES_HOME"
```

Create the package env file from the release example, then replace only the
angle-bracket placeholders.

```env
HERMES_HOME=<FRESH_HERMES_HOME>
DISCORD_BOT_TOKEN=<DISCORD_BOT_TOKEN>
DISCORD_CLIENT_ID=<DISCORD_CLIENT_ID>
DISCORD_HOME_CHANNEL=<DISCORD_CHANNEL_ID>
DISCORD_ALLOWED_USERS=<DISCORD_USER_ID_LIST>
DISCORD_ALLOWED_CHANNELS=<DISCORD_CHANNEL_ID_LIST>
DISCORD_REQUIRE_MENTION=true
OPENAI_BASE_URL=<MODEL_ENDPOINT_URL>
OPENAI_API_KEY=<MODEL_API_KEY>
WEBHOOK_SECRET=<WEBHOOK_HMAC_SECRET>
HERMES_REDACT_SECRETS=true
```

Do not paste real secret values into tickets, docs, chat transcripts, or demo
fixtures. Store the real env file only on the staging host.

## Discord Staging

1. Create a customer-owned Discord application and bot.
2. Invite it only to the private staging server or staging channel.
3. Grant the minimum permissions from `guides/discord-setup.md`.
4. Enable only the intents the package needs.
5. Set user and channel allowlists before starting the gateway.

Keep `DISCORD_REQUIRE_MENTION=true` in staging until the operator verifies that
the bot only responds where intended.

## Model Endpoint

Configure a model endpoint dedicated to staging. Use a quota low enough that an
accidental loop is visible before it becomes expensive.

The package should reach the endpoint only through the configured provider
settings. Verification should use mock or low-risk prompts before any customer
data is introduced.

## Preflight

Run the package preflight from the release. It must fail closed on:

- Placeholder env values.
- Missing Discord token, client ID, home channel, user allowlist, or channel allowlist.
- Missing model endpoint URL or API key.
- Weak webhook secret.
- `HERMES_HOME` pointing at an existing personal or live runtime.
- `GATEWAY_ALLOW_ALL_USERS=true`, broad user/channel values, or disabled mention requirement.
- Disabled redaction.

Do not start the gateway until preflight passes.

## First Staging Run

Start only Discord and the signed webhook inbox. Leave other platform env vars
unset.

Use safe prompts:

```text
/dobby status
/dobby help
Summarize this synthetic incident: ACME staging build failed because tests timed out.
Remind me in 10 minutes to review the staging checklist.
```

Expected result:

- The bot responds only in the allowed staging channel or DM.
- Status output redacts secrets.
- Research and attachment flows ask for explicit approval before reading data.
- Reminders deliver to the configured Discord home channel.
- Webhooks without a valid signature are rejected.

## Promote To Live

Promote only after `runbooks/verify.md` passes in staging.

1. Create or select the live Discord app/channel.
2. Create a separate live `HERMES_HOME`.
3. Copy configuration shape, not staging secrets or runtime data.
4. Run preflight again against live placeholders replaced on the live host.
5. Run the demo script with synthetic prompts in the live channel.
6. Keep the rollback runbook open during the first live hour.

Success means the operator can control Dobby from Discord, use the six core
use cases safely, and roll back without deleting user data.
