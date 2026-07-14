# Hermes Cron

Scheduled-job engine for Hermes Agent. Jobs run on a 60-second tick loop inside the gateway process.

## Morning Brief — Discord Delivery

Delivers a daily morning brief to a Discord channel at 06:00 Asia/Bangkok via `cron/scripts/morning_brief_discord.py`.

### Required config

Add the following to `~/.hermes/config.yaml`:

```yaml
discord:
  # Target channel ID (right-click the channel in Discord → Copy Channel ID).
  # The bot must have the Send Messages permission in this channel.
  morning_brief_channel_id: "1234567890"

  # Optional: list of text/markdown files whose content is included in the brief.
  morning_brief_contracts:
    - ~/my-daily-contract.md
    - ~/goals.txt
```

### Required env var

Add to `~/.hermes/.env` (this file is for secrets only — never put it in config.yaml):

```bash
DISCORD_BOT_TOKEN=Bot_token_from_discord_developer_portal
```

### Discord bot permissions

The bot must have the **Send Messages** permission in the target channel.

To verify:
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Select your application → Bot → copy the token
3. Under OAuth2 → URL Generator, grant **Send Messages** scope and add the bot to your server

### Register the job

```bash
python3 -c "from cron.jobs import register_morning_brief_discord_job; print(register_morning_brief_discord_job())"
```

### Verify the channel ID

Right-click the target channel in Discord (Developer Mode must be on: User Settings → Advanced → Developer Mode) and select **Copy Channel ID**.

### List all registered jobs

```bash
python -m cron.scheduler --list
```

Expected output includes a `morning_brief_discord` row with `0 6 * * *` and `Asia/Bangkok`.

### Dry-run without posting

Test the brief generation and Discord delivery logic without actually posting:

```bash
MORNING_BRIEF_DRY_RUN=1 python -m cron.scheduler --run-job morning_brief_discord --dry-run
```

Or run the script directly:

```bash
MORNING_BRIEF_DRY_RUN=1 HERMES_HOME=~/.hermes python cron/scripts/morning_brief_discord.py
```

### Error handling

| Scenario | Exit code | Logged message |
|----------|-----------|----------------|
| `discord.morning_brief_channel_id` missing | 1 | `Missing required config key: discord.morning_brief_channel_id` |
| `DISCORD_BOT_TOKEN` unset | 1 | `Missing required env var: DISCORD_BOT_TOKEN` |
| Contract file not found | 2 | `Contract file not found: <path>` |
| Discord 401 (bad token) | 3 | `Discord authentication failed (401)` |
| Discord 404 (bad channel) | 3 | `Discord channel not found (404)` |
| Network error / timeout | 3 | `Discord delivery network error: …` |

All errors are logged to stderr and cause a non-zero exit. The scheduler records the failure and (if `deliver` points somewhere) delivers an error alert.

## General cron commands

```bash
# List jobs
hermes cron list

# Add a job interactively
hermes cron add

# Pause / resume
hermes cron pause <job-id>
hermes cron resume <job-id>

# Run a job immediately
hermes cron run <job-id-or-name>

# View output
hermes cron output <job-id>
```
