# AI Engine MCP — complete tool reference

This is the full set of tools exposed by the AI Engine MCP server (`/wp-json/mcp/v1/http`) as of AI Engine **3.7.x**. The exact tools available depend on enabled feature flags (see `references/troubleshooting.md` → "Tools missing").

Tool count: **109+** when all features are enabled (core + plugins + themes + WooCommerce + SEO + social + Polylang). Core-only (default) is ~43 tools.

> **Discovery:** Always run `tools/list` on the live endpoint to see what's available on your site. The list below is a reference, not a contract.

---

## System

| Tool | Description | Key args |
|------|-------------|----------|
| `mcp_ping` | Connectivity check — returns GMT time + site name | — |

---

## Posts & Pages

| Tool | Description | Key args |
|------|-------------|----------|
| `wp_get_posts` | List posts/pages by post_type, status, search, limit, offset, paged | `post_type`, `status`, `search`, `limit`, `offset`, `paged` |
| `wp_get_post` | Get a single post by ID | `ID` |
| `wp_get_post_snapshot` | Get a post snapshot (content + meta + terms at a point in time) | `ID` |
| `wp_create_post` | Create a post/page/CPT | `post_title`, `post_content`, `post_status`, `post_type`, `post_author` |
| `wp_update_post` | Update a post — pass `ID` + `fields` object | `ID`, `fields: { post_content, post_title, post_status, ... }` |
| `wp_alter_post` | Alter a post (status change, trash, restore) | `ID`, `action` |
| `wp_delete_post` | Delete a post (force=true bypasses trash) | `ID`, `force` |
| `wp_write_blocks` | Replace or append Gutenberg blocks on a post | `ID`, `mode: "replace"\|"append"`, `blocks: [...]` |

### `wp_write_blocks` block schema

Each entry in the `blocks` array is an object:

```json
{
  "type": "heading|paragraph|list|buttons|image|separator|html|columns|group|quote|code|table",
  "content": "text content (for text blocks)",
  "level": 1,
  "items": ["item 1", "item 2"],
  "ordered": false,
  "buttons": [{ "text": "Label", "url": "https://..." }],
  "id": 123,
  "alt": "alt text",
  "size": "thumbnail|medium|large|full",
  "align": "left|center|right",
  "columns": [[{ "type": "paragraph", "content": "col 1" }], [{ "type": "paragraph", "content": "col 2" }]],
  "blocks": [{ "type": "paragraph", "content": "nested" }]
}
```

> **Gotcha:** `wp_write_blocks` with very large payloads (many blocks) can cause the MCP connection to drop. For large rewrites, use `wp_update_post` with raw Gutenberg HTML (`<!-- wp:heading -->...<!-- /wp:heading -->`) instead — it's more robust.

> **Gotcha:** `wp_write_blocks` `mode: "replace"` replaces ALL content. Use `"append"` to add blocks after existing content.

---

## Users

| Tool | Description | Key args |
|------|-------------|----------|
| `wp_get_users` | List users (search, role, limit, offset, paged) | `search`, `role`, `limit`, `offset` |
| `wp_create_user` | Create a user | `user_login`, `user_email`, `user_pass`, `display_name`, `role` |
| `wp_update_user` | Update a user — pass `ID` + `fields` object | `ID`, `fields: { user_email, display_name, user_pass, role }` |
| `wp_delete_user` | Delete a user | `ID` |

---

## Comments

| Tool | Description | Key args |
|------|-------------|----------|
| `wp_get_comments` | List comments (filter by post_id, status, type, search, user_id, author_email) | `post_id`, `status`, `type`, `search`, `limit` |
| `wp_create_comment` | Create a comment | `post_id`, `comment_content`, `comment_author`, `comment_author_email` |
| `wp_update_comment` | Update a comment — pass `comment_ID` + `fields` | `comment_ID`, `fields: { comment_content, comment_approved }` |
| `wp_delete_comment` | Delete a comment (force=true bypasses trash) | `comment_ID`, `force` |

> **WP 6.9 Notes:** Pass `type: "note"` to read editor Notes (block-level feedback). `comment_approved "0"` = open/unresolved, `"1"` = resolved.

---

## Taxonomy

| Tool | Description | Key args |
|------|-------------|----------|
| `wp_get_terms` | List terms in a taxonomy (search, hide_empty, parent) | `taxonomy`, `search`, `hide_empty`, `parent` |
| `wp_create_term` | Create a term | `taxonomy`, `name`, `slug`, `parent` |
| `wp_add_post_terms` | Assign terms to a post | `post_id`, `terms: [id, ...]`, `taxonomy` |
| `wp_count_terms` | Count terms in a taxonomy | `taxonomy` |

---

## Media

| Tool | Description | Key args |
|------|-------------|----------|
| `wp_list_media` | List media attachments (search, limit, offset, paged) | `search`, `limit`, `offset` |
| `wp_get_media` | Get a single media item | `ID` |
| `wp_upload_media` | Upload a media file from URL or base64 | `url` or `base64`, `title`, `alt`, `filename` |
| `wp_set_featured_image` | Set featured image on a post | `post_id`, `media_id` |
| `wp_count_media` | Count attachments (optional after/before date) | `after`, `before` |

> **Gotcha:** `wp_upload_media` requires the upload directory to be writable by the web user. On aaPanel, the web user is `www:www`. If WP-CLI runs as `ubuntu`, uploads fail with "O arquivo enviado não pode ser movido". See `references/troubleshooting.md` → "Media upload fails with permission error".

---

## AI / Image generation

| Tool | Description | Key args |
|------|-------------|----------|
| `mwai_image` | Generate an image via AI Engine's image provider | `message` (prompt), `title`, `alt` |
| `mwai_vision` | Analyze an image with a vision model | `image` (URL or base64), `message` (question) |

> **Gotcha:** `mwai_image` requires an API key configured in AI Engine settings (OpenAI, Stability, etc.). If no key is set, returns: `"No API Key provided. Please visit the Settings."`. Configure in WP Admin → AI Engine → Settings → API Keys.

---

## Options & Settings

| Tool | Description | Key args |
|------|-------------|----------|
| `wp_get_option` | Get a single option value (scalar or array). `raw: true` bypasses cache/filters | `key`, `raw` |
| `wp_update_option` | Create or update an option (arrays/objects stored natively) | `key`, `value` |
| `wp_get_settings` | Get site settings (general, reading, writing, etc.) | `section` |
| `wp_update_settings` | Update site settings | `section`, `fields: { ... }` |

> **Gotcha:** `wp_update_option` does NOT purge full-page caches (Cloudflare, WP Rocket, Varnish). Use cache-busting query strings (`?nocache=N`) or purge the cache manually after changes.

---

## Plugins & Themes

| Tool | Description | Key args |
|------|-------------|----------|
| `wp_list_plugins` | List installed plugins (search filter) | `search` |
| `wp_activate_plugin` | Activate a plugin | `slug` |
| `wp_deactivate_plugin` | Deactivate a plugin | `slug` |

> **Note:** There is no `wp_install_plugin` or `wp_list_themes` MCP tool. Use WP-CLI for those operations:
> ```bash
> wp plugin install <slug> --activate --path=$WP_PATH
> wp theme install <slug> --activate --path=$WP_PATH
> wp theme list --path=$WP_PATH
> ```

---

## Post types & counts

| Tool | Description | Key args |
|------|-------------|----------|
| `wp_list_post_types` | List all registered post types | — |
| `wp_count_posts` | Count posts by status (optional post_type) | `post_type` |

---

## Feature-gated tools (require enabling in mwai_options)

These tools only appear in `tools/list` when the corresponding feature flag is enabled:

### Plugins feature (`mcp_feature_plugins`)
Install, activate, update plugins via MCP.

### Themes feature (`mcp_feature_themes`)
Install, activate, switch themes via MCP.

### Database feature (`mcp_feature_database`)
Execute SQL queries. **DANGEROUS** — enable with caution.

### Polylang feature (`mcp_feature_polylang`)
Multilingual operations (requires Polylang plugin).

### WooCommerce feature (`mcp_feature_woocommerce`)
Products, orders, coupons (requires WooCommerce).

### SEO Engine feature (`mcp_feature_seo_engine`)
SEO metadata operations (requires SEO Engine add-on).

### Social Engine feature (`mcp_feature_social_engine`)
Social media scheduling (requires Social Engine add-on).

### Dynamic REST feature (`mcp_feature_dynamic_rest`)
Expose custom REST endpoints as MCP tools.

---

## Enabling feature flags

```bash
wp eval '
$o = get_option("mwai_options");
$o["mcp_feature_plugins"] = true;
$o["mcp_feature_themes"] = true;
// $o["mcp_feature_database"] = true;    // DANGEROUS
// $o["mcp_feature_polylang"] = true;
// $o["mcp_feature_woocommerce"] = true;
// $o["mcp_feature_seo_engine"] = true;
// $o["mcp_feature_social_engine"] = true;
// $o["mcp_feature_dynamic_rest"] = true;
update_option("mwai_options", $o);
' --path=$WP_PATH --allow-root
```

---

## Tool discovery snippet

```bash
TOKEN="your_bearer_token"
URL="https://yourdomain.com/wp-json/mcp/v1/http"

# List all available tools (names + descriptions)
curl -s -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | python3 -c "import sys,json; tools=json.load(sys.stdin)['result']['tools']; print(f'{len(tools)} tools:'); [print(f'  {t[\"name\"]}') for t in tools]"
```
