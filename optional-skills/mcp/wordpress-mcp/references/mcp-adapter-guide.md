# MCP Adapter (Path A) — complete guide

The official WordPress MCP Adapter (`wordpress/mcp-adapter`). Bridges the **Abilities API** to the **Model Context Protocol**, exposing WordPress abilities as MCP tools, resources, and prompts.

**Current version:** v0.6.1 (August 2026)
**License:** GPL-2.0-or-later
**Repository:** https://github.com/WordPress/mcp-adapter

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| WordPress | 6.9+ (Abilities API is built into core) |
| PHP | 7.4+ |
| Plugin | `mcp-adapter` from GitHub releases |
| Composer | **Not needed** if using the release ZIP (v0.6.0+ ships with vendor/) |
| Application Passwords | Required for HTTP transport (Basic Auth) |
| WP-CLI | Required for STDIO transport |

> **WP < 6.9:** The Abilities API is not in core. Install the [Abilities API plugin](https://github.com/WordPress/abilities-api) separately, or use [wp-mcp-ultimate](../wp-mcp-ultimate.md) which includes a polyfill.

---

## Installation

### Option 1: WP-CLI with release ZIP (recommended — no composer)

```bash
wp plugin install https://github.com/WordPress/mcp-adapter/releases/latest/download/mcp-adapter.zip --activate --path=$WP_PATH
wp rewrite flush --path=$WP_PATH
```

The release ZIP (v0.6.0+) ships with `vendor/` pre-built. No `composer install` needed.

### Option 2: Manual download + copy (when wp plugin install fails)

```bash
cd /tmp
curl -L -o mcp-adapter.zip "https://github.com/WordPress/mcp-adapter/releases/latest/download/mcp-adapter.zip"
unzip -q mcp-adapter.zip -d /tmp/mcp-adapter-extract
sudo cp -r /tmp/mcp-adapter-extract/mcp-adapter $WP_PATH/wp-content/plugins/mcp-adapter
sudo chown -R www:www $WP_PATH/wp-content/plugins/mcp-adapter
wp plugin activate mcp-adapter --path=$WP_PATH
wp rewrite flush --path=$WP_PATH
```

### Option 3: From source (developers — requires composer)

```bash
cd /tmp
curl -L -o mcp-adapter.zip "https://github.com/WordPress/mcp-adapter/archive/refs/heads/trunk.zip"
unzip -q mcp-adapter.zip -d /tmp/mcp-adapter-src
mv /tmp/mcp-adapter-src/mcp-adapter-trunk $WP_PATH/wp-content/plugins/mcp-adapter
cd $WP_PATH/wp-content/plugins/mcp-adapter
composer install --no-dev --no-interaction --optimize-autoloader
wp plugin activate mcp-adapter --path=$WP_PATH
wp rewrite flush --path=$WP_PATH
```

### Option 4: As a Composer dependency (plugin developers)

```bash
composer require wordpress/mcp-adapter
```

> **Warning:** Bundling MCP Adapter as a Composer library can conflict with the standalone plugin. Use the plugin dependency method (`Requires Plugins: mcp-adapter` in your plugin header) when possible.

### aaPanel / BT-Panel notes

```bash
WP_PATH=/www/wwwroot/yourdomain.com
WP_PHP=/www/server/php/85/bin/php

# Use the right PHP binary
$WP_PHP /usr/local/bin/wp plugin install \
  https://github.com/WordPress/mcp-adapter/releases/latest/download/mcp-adapter.zip \
  --activate --path=$WP_PATH --allow-root

# Fix permissions
sudo chown -R www:www $WP_PATH/wp-content/plugins/mcp-adapter
```

---

## Application Passwords (HTTP transport)

Application Passwords must be enabled for HTTP transport (Basic Auth).

```bash
# Check if enabled
wp eval 'echo wp_is_application_passwords_available() ? "enabled" : "disabled";' --path=$WP_PATH --allow-root

# Create application password
wp user application-password create 1 "mcp-clients" --path=$WP_PATH --allow-root
# Output: Password: xxxx xxxx xxxx xxxx xxxx xxxx

# Generate Basic Auth value
echo -n "admin:xxxx xxxx xxxx xxxx xxxx xxxx" | base64
```

If Application Passwords are disabled:

```bash
sudo tee $WP_PATH/wp-content/mu-plugins/enable-app-passwords.php > /dev/null <<'EOF'
<?php
add_filter('wp_is_application_passwords_available', '__return_true');
add_filter('wp_is_application_passwords_api_available', '__return_true');
EOF
```

---

## Transports

The MCP Adapter supports **two transports**:

### HTTP transport (remote — for MCP clients over network)

```
POST https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server
Authorization: Basic <base64(user:app_password)>
Content-Type: application/json
```

Uses **Streamable HTTP** with session management:

1. `initialize` → capture `Mcp-Session-Id` from response headers
2. `notifications/initialized` → HTTP 202 (must send before tools/list)
3. `tools/list` → returns 3 meta-tools

```bash
USER="admin"
PASS="xxxx xxxx xxxx xxxx xxxx xxxx"
AUTH=$(echo -n "$USER:$PASS" | base64)
URL="https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server"

# 1. Initialize
SESSION=$(curl -s -i -X POST "$URL" \
  -H "Authorization: Basic $AUTH" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' \
  | grep -i "mcp-session-id" | awk '{print $2}' | tr -d '\r\n')

# 2. Initialized notification
curl -s -X POST "$URL" \
  -H "Authorization: Basic $AUTH" \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' -o /dev/null

# 3. List tools
curl -s -X POST "$URL" \
  -H "Authorization: Basic $AUTH" \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

### STDIO transport (local — for MCP clients that launch subprocesses)

```bash
# Serve via WP-CLI
wp mcp-adapter serve --user=admin --path=$WP_PATH

# List available servers
wp mcp-adapter list --path=$WP_PATH

# Test with a JSON-RPC request
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
  wp mcp-adapter serve --user=admin --path=$WP_PATH
```

MCP client config for STDIO:

```jsonc
{
  "mcpServers": {
    "wordpress": {
      "command": "wp",
      "args": [
        "--path=/path/to/wordpress",
        "mcp-adapter",
        "serve",
        "--server=mcp-adapter-default-server",
        "--user=admin"
      ]
    }
  }
}
```

> **STDIO is for local sites only.** The `wp` binary must be on the same machine as the MCP client. For remote sites, use HTTP transport or the npm proxy.

### HTTP proxy (remote sites via STDIO clients)

The [`@automattic/mcp-wordpress-remote`](https://www.npmjs.com/package/@automattic/mcp-wordpress-remote) npm package bridges STDIO-based MCP clients to remote WordPress HTTP endpoints:

```jsonc
{
  "mcpServers": {
    "wordpress": {
      "command": "npx",
      "args": ["-y", "@automattic/mcp-wordpress-remote@latest"],
      "env": {
        "WP_API_URL": "https://your-site.com/wp-json/mcp/mcp-adapter-default-server",
        "WP_API_USERNAME": "your-username",
        "WP_API_PASSWORD": "your-application-password",
        "LOG_FILE": "/path/to/logs/mcp-adapter.log"
      }
    }
  }
}
```

---

## Available tools (3 meta-tools)

| Tool | Description |
|------|-------------|
| `mcp-adapter/discover-abilities` | Lists all publicly available WordPress abilities |
| `mcp-adapter/get-ability-info` | Gets detailed info about a specific ability |
| `mcp-adapter/execute-ability` | Executes a WordPress ability with provided parameters |

> The mcp-adapter exposes a **meta-API**: you discover and execute abilities dynamically. This is more flexible than AI Engine's fixed tool set, but requires the agent to call `discover-abilities` first.

### Discover abilities

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"mcp-adapter/discover-abilities","arguments":{}}}
```

### Get ability info

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"mcp-adapter/get-ability-info","arguments":{"ability_id":"my-plugin/get-site-info"}}}
```

### Execute ability

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"mcp-adapter/execute-ability","arguments":{"ability_id":"my-plugin/get-site-info","parameters":{"include_stats":true}}}}
```

---

## Abilities API

The Abilities API is the foundation of mcp-adapter. It allows plugins to register "abilities" (executable functions) that the adapter exposes as MCP tools.

### WP 6.9+ (built into core)

No separate plugin needed. The `wp_register_ability()` function is available globally.

### WP < 6.9 (needs plugin)

Install the [Abilities API plugin](https://github.com/WordPress/abilities-api):

```bash
wp plugin install https://github.com/WordPress/abilities-api/releases/latest/download/abilities-api.zip --activate --path=$WP_PATH
```

Or use [wp-mcp-ultimate](../wp-mcp-ultimate.md) which includes an Abilities API polyfill.

### Creating a custom ability

```php
// Register a custom ability exposed via MCP
add_action( 'wp_abilities_api_init', function() {
    wp_register_ability( 'my-plugin/get-site-info', [
        'label'             => 'Get Site Information',
        'description'       => 'Retrieves basic information about the current WordPress site',
        'category'          => 'site',
        'input_schema'      => [
            'type' => 'object',
            'properties' => [
                'include_stats' => [
                    'type'        => 'boolean',
                    'description' => 'Whether to include post/page statistics',
                    'default'     => false,
                ],
            ],
        ],
        'execute_callback'  => function( $input ) {
            $result = [
                'site_name'    => get_bloginfo( 'name' ),
                'site_url'     => get_site_url(),
                'description'  => get_bloginfo( 'description' ),
            ];

            if ( $input['include_stats'] ?? false ) {
                $result['stats'] = [
                    'post_count' => wp_count_posts( 'post' )->publish,
                    'page_count' => wp_count_posts( 'page' )->publish,
                ];
            }

            return $result;
        },
        'permission_callback' => function() {
            return current_user_can( 'read' );
        },
        'meta' => [
            'public' => true,  // REQUIRED to expose via MCP (abilities are private by default)
        ],
    ] );
} );
```

> **Critical:** Abilities are **private by default**. Set `meta.public` (or `meta.mcp.public`) to `true` to expose them via MCP. Without this, the ability exists but `discover-abilities` won't list it.

### Creating a custom MCP server

```php
add_action( 'mcp_adapter_init', function( $adapter ) {
    $adapter->create_server(
        'my-server',           // Unique server ID
        'my-plugin',           // REST API namespace
        'mcp',                 // REST API route
        'My MCP Server',       // Human-readable name
        'Description',         // Description
        '1.0.0',               // Version
        [ \WP\MCP\Transport\HttpTransport::class ],  // Transport methods
        \WP\MCP\Infrastructure\ErrorHandling\ErrorLogMcpErrorHandler::class,
        null,                  // Observability handler (null = default)
        [ 'my-plugin/get-site-info' ]  // Abilities to expose
    );
} );
```

---

## WP-CLI commands

| Command | Description |
|---------|-------------|
| `wp mcp-adapter serve [--server=<id>] [--user=<user>]` | Serve MCP server via STDIO |
| `wp mcp-adapter list [--format=<format>]` | List all MCP servers |

### Examples

```bash
# Serve default server as admin
wp mcp-adapter serve --user=admin --path=$WP_PATH

# Serve specific server
wp mcp-adapter serve --server=my-custom-server --user=1 --path=$WP_PATH

# List servers in JSON
wp mcp-adapter list --format=json --path=$WP_PATH

# Debug mode
wp mcp-adapter serve --user=admin --debug --path=$WP_PATH
```

---

## Migration from Automattic/wordpress-mcp

The `Automattic/wordpress-mcp` repository is **archived and deprecated**. Migrate to `wordpress/mcp-adapter`:

```bash
# Deactivate old plugin
wp plugin deactivate wordpress-mcp --path=$WP_PATH

# Install mcp-adapter
wp plugin install https://github.com/WordPress/mcp-adapter/releases/latest/download/mcp-adapter.zip --activate --path=$WP_PATH

# Update MCP client configs to point to the new endpoint
# Old: /wp-json/automattic-mcp/v1
# New: /wp-json/mcp/mcp-adapter-default-server
```

> **Breaking change:** The endpoint path changed. Update all MCP client configurations.

---

## Comparison with AI Engine (Path B)

| Feature | mcp-adapter (Path A) | AI Engine (Path B) |
|---------|---------------------|---------------------|
| **Type** | Official WordPress plugin | Community plugin |
| **Tools** | 3 meta-tools (dynamic discovery) | 43-109+ fixed tools |
| **Transport** | HTTP + STDIO | HTTP only |
| **Auth** | Application Passwords (Basic Auth) | Bearer Token |
| **Session** | Required (Streamable HTTP) | Not required |
| **Extensibility** | Register custom abilities | Feature flags only |
| **WP version** | 6.9+ (or Abilities API plugin) | 6.0+ |
| **PHP** | 7.4+ | 8.1+ |
| **Composer** | Not needed (release ZIP) | Not needed |
| **AI image gen** | No | Yes (`mwai_image`, `mwai_vision`) |
| **Block editor** | No | Yes (`wp_write_blocks`) |
| **Best for** | Custom abilities, official support | Ready-to-use admin tools |

> **Both can coexist** on the same site (different endpoints, different auth). Use mcp-adapter for custom abilities and AI Engine for standard WordPress admin tasks.
