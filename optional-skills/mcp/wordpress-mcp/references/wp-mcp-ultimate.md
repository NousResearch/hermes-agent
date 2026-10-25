# WP MCP Ultimate (Path C) — community plugin

A self-contained MCP server plugin with **58 WordPress abilities** — manage posts, pages, media, users, plugins, menus, comments, and more through any MCP-compatible AI client. No other plugins needed.

**Repository:** https://github.com/AgriciDaniel/wp-mcp-ultimate
**License:** GPL-2.0-or-later
**Author:** Agrici Daniel

---

## When to use

- You want a **ready-to-use** MCP server with 58 pre-built abilities (no custom PHP needed)
- You need **OAuth 2.1** support (for Claude Web/Mobile connectors)
- You're on **WordPress 6.7+** (not 6.9+ required by mcp-adapter)
- You want a **self-contained** plugin (no composer, no separate Abilities API plugin)
- You want an **admin dashboard** with one-click API key generation and config export

## When NOT to use

- You need AI image generation → use AI Engine (Path B)
- You need Gutenberg block manipulation (`wp_write_blocks`) → use AI Engine (Path B)
- You need to register **custom abilities** → use mcp-adapter (Path A)
- You need STDIO transport → use mcp-adapter (Path A)

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| WordPress | 6.7+ |
| PHP | 8.0+ |
| Plugin | `wp-mcp-ultimate` from GitHub |
| Application Passwords | Required (Basic Auth) — or OAuth 2.1 for Claude Web/Mobile |
| Abilities API | Not required (includes polyfill for WP < 6.9) |

---

## Installation (requires explicit approval)

`wp-mcp-ultimate` is published from a personal GitHub repository. Do not install it automatically. Use a pinned release, verify the source, and get user approval before proceeding.

### Option 1: Upload via WordPress admin (recommended)

1. Pick a pinned release from [GitHub](https://github.com/AgriciDaniel/wp-mcp-ultimate/releases) (replace `<VERSION>` with the release tag).
2. Verify the checksum when the release provides one.
3. WordPress Admin → Plugins → Add New → Upload Plugin.
4. Choose the ZIP file → Install Now → Activate.

### Option 2: WP-CLI (pinned release only)

```bash
WP_MCP_ULTIMATE_VERSION="<VERSION>"  # e.g. "v1.0.0"
DOWNLOAD_URL="https://github.com/AgriciDaniel/wp-mcp-ultimate/releases/download/${WP_MCP_ULTIMATE_VERSION}/wp-mcp-ultimate.zip"

# Download locally, verify checksum when available, then install
wp plugin install "$DOWNLOAD_URL" --activate --path="$WP_PATH"
wp rewrite flush --path="$WP_PATH"
```

**Do not use `https://github.com/AgriciDaniel/wp-mcp-ultimate/releases/latest/download/...` or `git clone` for production.** Always use a pinned release artifact and verify the checksum.

---

## Authentication

WP MCP Ultimate supports **two authentication methods**:

### 1. Application Passwords (Basic Auth)

Works with all MCP clients (Claude Code, Cursor, Devin, etc.):

```bash
# Generate via admin UI: Tools → MCP Ultimate → Generate
# Or via WP-CLI:
wp user application-password create 1 "mcp-ultimate" --path=$WP_PATH
```

Config:
```json
{
  "mcpServers": {
    "wordpress-ultimate": {
      "type": "http",
      "url": "https://yourdomain.com/wp-json/mcp-ultimate/v1",
      "headers": {
        "Authorization": "Basic <base64(user:app_password)>"
      }
    }
  }
}
```

### 2. OAuth 2.1 (for Claude Web/Mobile)

The plugin bundles an OAuth 2.1 authorization server with PKCE and dynamic client registration. Point the Claude Web/Mobile connector at the MCP endpoint and it discovers everything automatically.

> **Capability requirement:** By default, reaching the MCP endpoint requires the `edit_posts` capability. This is filterable.

---

## Available abilities (58 across 9 domains)

All abilities are exposed via the **3 meta-tools pattern** (same as mcp-adapter):

- `discover-abilities` — list all 58 abilities
- `get-ability-info` — get details about a specific ability
- `execute-ability` — execute an ability

### Content — Posts (6)

| Ability | Label | Capability |
|---------|-------|------------|
| `content/list-posts` | List Posts | `edit_posts` |
| `content/get-post` | Get Post | `edit_posts` |
| `content/create-post` | Create Post | `edit_posts` |
| `content/update-post` | Update Post | `edit_posts` |
| `content/delete-post` | Delete Post | `delete_posts` |
| `content/patch-post` | Patch Post Content | `edit_posts` |

> `content/create-post` and `content/update-post` accept `featured_media` (image attachment ID) to set the post's featured image.

### Content — Pages (6)

| Ability | Label | Capability |
|---------|-------|------------|
| `content/list-pages` | List Pages | `edit_pages` |
| `content/get-page` | Get Page | `edit_pages` |
| `content/create-page` | Create Page | `edit_pages` |
| `content/update-page` | Update Page | `edit_pages` |
| `content/delete-page` | Delete Page | `delete_pages` |
| `content/patch-page` | Patch Page Content | `edit_pages` |

### Content — Taxonomy (4)

| Ability | Label | Capability |
|---------|-------|------------|
| `content/list-categories` | List Categories | `manage_categories` |
| `content/create-category` | Create Category | `manage_categories` |
| `content/list-tags` | List Tags | `manage_categories` |
| `content/create-tag` | Create Tag | `manage_categories` |

### Content — Search & Revisions (3)

| Ability | Label | Capability |
|---------|-------|------------|
| `content/search` | Search Content | `edit_posts` |
| `content/list-revisions` | List Revisions | `edit_posts` |
| `content/get-revision` | Get Revision | `edit_posts` |

### Media (5)

| Ability | Label | Capability |
|---------|-------|------------|
| `content/list-media` | List Media | `upload_files` |
| `media/upload` | Upload Media | `upload_files` |
| `media/get` | Get Media Item | `upload_files` |
| `media/update` | Update Media Item | `upload_files` |
| `media/delete` | Delete Media Item | `delete_posts` |

### Users (6)

| Ability | Label | Capability |
|---------|-------|------------|
| `content/list-users` | List Users | `list_users` |
| `users/list` | List Users (Extended) | `list_users` |
| `users/get` | Get User | `list_users` |
| `users/create` | Create User | `create_users` |
| `users/update` | Update User | `edit_users` |
| `users/delete` | Delete User | `delete_users` |

### Plugins (6)

| Ability | Label | Capability |
|---------|-------|------------|
| `plugins/upload` | Upload Plugin | `install_plugins` |
| `plugins/upload-base64` | Upload Plugin (Base64/Zip Path) | `install_plugins` |
| `plugins/list` | List Plugins | `activate_plugins` |
| `plugins/delete` | Delete Plugin | `delete_plugins` |
| `plugins/activate` | Activate Plugin | `activate_plugins` |
| `plugins/deactivate` | Deactivate Plugin | `activate_plugins` |

> **Note:** Plugin installs no longer auto-activate (security measure). Use `plugins/activate` separately.

### Menus (7)

| Ability | Label | Capability |
|---------|-------|------------|
| `menus/list` | List Menus | `edit_theme_options` |
| `menus/get-items` | Get Menu Items | `edit_theme_options` |
| `menus/create` | Create Menu | `edit_theme_options` |
| `menus/add-item` | Add Menu Item | `edit_theme_options` |
| `menus/update-item` | Update Menu Item | `edit_theme_options` |
| `menus/delete-item` | Delete Menu Item | `edit_theme_options` |
| `menus/assign-location` | Assign Menu to Location | `edit_theme_options` |

### Widgets (3)

| Ability | Label | Capability |
|---------|-------|------------|
| `widgets/list-sidebars` | List Widget Sidebars | `edit_theme_options` |
| `widgets/get-sidebar` | Get Sidebar Widgets | `edit_theme_options` |
| `widgets/list-available` | List Available Widgets | `edit_theme_options` |

### Comments (6)

| Ability | Label | Capability |
|---------|-------|------------|
| `comments/list` | List Comments | `moderate_comments` |
| `comments/get` | Get Comment | `moderate_comments` |
| `comments/update-status` | Update Comment Status | `moderate_comments` |
| `comments/reply` | Reply to Comment | `moderate_comments` |
| `comments/create` | Create Comment | `moderate_comments` |
| `comments/delete` | Delete Comment | `moderate_comments` |

### Options (3)

| Ability | Label | Capability |
|---------|-------|------------|
| `options/get` | Get Option | `manage_options` |
| `options/update` | Update Option | `manage_options` |
| `options/list` | List Options | `manage_options` |

### System (3)

| Ability | Label | Capability |
|---------|-------|------------|
| `system/get-transient` | Get Transient | `manage_options` |
| `system/debug-log` | Read Debug Log | `manage_options` |
| `system/toggle-debug` | Toggle Debug Mode | `manage_options` |

---

## Admin dashboard

The plugin includes a React-based admin dashboard at **Tools → MCP Ultimate**:

- API key management (one-click generation)
- Connection testing
- Config export snippets for Claude Code, Claude Desktop, and Cursor
- Conflict detection for legacy plugins (MCP Adapter, MCP Expose Abilities, Abilities API)

---

## MCP client configuration

### Claude Code

```json
{
  "mcpServers": {
    "wordpress-ultimate": {
      "type": "http",
      "url": "https://yourdomain.com/wp-json/mcp-ultimate/v1",
      "headers": {
        "Authorization": "Basic <base64>"
      }
    }
  }
}
```

### Cursor

```json
{
  "mcpServers": {
    "wordpress-ultimate": {
      "url": "https://yourdomain.com/wp-json/mcp-ultimate/v1",
      "headers": {
        "Authorization": "Basic <base64>"
      }
    }
  }
}
```

> The admin dashboard provides ready-to-copy config snippets for each supported client.

---

## Comparison with other paths

| Feature | wp-mcp-ultimate (Path C) | mcp-adapter (Path A) | AI Engine (Path B) |
|---------|-------------------------|---------------------|---------------------|
| **Type** | Community plugin | Official WordPress | Community plugin |
| **Tools** | 58 fixed abilities | 3 meta-tools (dynamic) | 43-109+ fixed tools |
| **Transport** | HTTP only | HTTP + STDIO | HTTP only |
| **Auth** | App Passwords + OAuth 2.1 | Application Passwords | Bearer Token |
| **WP version** | 6.7+ | 6.9+ | 6.0+ |
| **PHP** | 8.0+ | 7.4+ | 8.1+ |
| **Abilities API** | Polyfill included | Required (core in 6.9) | Not used |
| **OAuth 2.1** | Yes | No | No |
| **Admin UI** | Yes (dashboard) | No | Yes (settings) |
| **Custom abilities** | No | Yes | No |
| **AI image gen** | No | No | Yes |
| **Block editor** | No | No | Yes |
| **Best for** | Ready-to-use, OAuth, lower WP version | Custom abilities, official | Standard admin + AI |

> **All three can coexist** on the same site (different endpoints, different auth). Use the one that fits your needs, or combine them.
