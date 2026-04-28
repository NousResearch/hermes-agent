### Start Gateway and Configure Voice Mode

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/features/voice-mode.md

Start the Hermes gateway to connect to messaging platforms and enable voice mode. The gateway command connects to configured Telegram and Discord bots.

```bash
hermes gateway        # Start the gateway (connects to configured platforms)
hermes gateway setup  # Interactive setup wizard for first-time configuration
```

--------------------------------

### Set Up Hermes Gateway for Messaging Platforms

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/getting-started/quickstart.md

Configure Hermes to connect with messaging platforms like Telegram, Discord, Slack, WhatsApp, Signal, Email, or Home Assistant.

```bash
hermes gateway setup    # Interactive platform configuration
```

--------------------------------

### Docker Compose Configuration for Hermes Matrix Proxy

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

This `docker-compose.yml` defines the Hermes Matrix proxy service, configuring Matrix credentials and the URL for forwarding requests to the host agent. It uses environment variables for sensitive data.

```yaml
services:
  hermes-matrix:
    build: .
    environment:
      # Matrix credentials
      MATRIX_HOMESERVER: "https://matrix.example.org"
      MATRIX_ACCESS_TOKEN: "syt_..."
      MATRIX_ALLOWED_USERS: "@you:matrix.example.org"
      MATRIX_ENCRYPTION: "true"
      MATRIX_DEVICE_ID: "HERMES_BOT"

      # Proxy mode — forward to host agent
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "your-secret-key-here"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

--------------------------------

### Manual Configuration in .env File

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/discord.md

Add Discord credentials directly to ~/.hermes/.env. DISCORD_BOT_TOKEN is required; DISCORD_ALLOWED_USERS accepts a single user ID or comma-separated list of user IDs.

```bash
# Required
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=284102345871466496

# Multiple allowed users (comma-separated)
# DISCORD_ALLOWED_USERS=284102345871466496,198765432109876543
```

--------------------------------

### Configure Hermes with Access Token

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

Environment variables for ~/.hermes/.env using token-based authentication. MATRIX_USER_ID is optional and auto-detected from the token if omitted. MATRIX_ALLOWED_USERS restricts bot interaction to specified users.

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# Optional: user ID (auto-detected from token if omitted)
# MATRIX_USER_ID=@hermes:matrix.example.org

# Security: restrict who can interact with the bot
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# Multiple allowed users (comma-separated)
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```