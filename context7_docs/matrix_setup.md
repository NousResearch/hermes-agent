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

### Generate new Matrix access token via login API

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

Obtain a fresh access token and device ID by authenticating with the Matrix homeserver. Required as the first step of E2EE migration when upgrading Hermes to the new SQLite crypto store.

```bash
curl -X POST https://your-server/_matrix/client/v3/login \
  -H "Content-Type: application/json" \
  -d '{
    "type": "m.login.password",
    "identifier": {"type": "m.id.user", "user": "@hermes:your-server.org"},
    "password": "***",
    "initial_device_display_name": "Hermes Agent"
  }'
```

--------------------------------

### Install `hermes-agent` with Matrix Extras

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

Install the `hermes-agent` package with the `matrix` extra, which includes `mautrix[encryption]` and other Matrix-specific dependencies.

```bash
pip install 'hermes-agent[matrix]'
```

### Step 4: Configure Hermes Agent > Option B: Manual Configuration

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/messaging/matrix.md

Manual configuration involves adding Matrix settings to the ~/.hermes/.env file, including the homeserver URL, authentication credentials (either access token or user ID and password), and a list of allowed user IDs to restrict bot interactions. The MATRIX_ALLOWED_USERS setting supports multiple users as a comma-separated list.