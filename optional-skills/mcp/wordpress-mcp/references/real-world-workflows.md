# WordPress MCP — real-world workflows

Practical workflows learned from managing a live WordPress site (afonsoft.dev) via MCP + WP-CLI. These complement the tool reference with end-to-end recipes.

---

## Workflow 1: Convert posts homepage to static front page

**Scenario:** Default WordPress shows posts on the homepage. Convert to a static page + separate blog page.

```bash
WP_PATH=/www/wwwroot/yourdomain.com

# 1. Create Home page
HOME_ID=$(wp post create --post_type=page --post_status=publish \
  --post_title="Home" --post_content="<!-- wp:paragraph --><p>Content here.</p><!-- /wp:paragraph -->" \
  --porcelain --path=$WP_PATH 2>/dev/null | tail -1)

# 2. Create Blog page (empty — WP will fill with posts list)
BLOG_ID=$(wp post create --post_type=page --post_status=publish \
  --post_title="Blog" --post_content="" \
  --porcelain --path=$WP_PATH 2>/dev/null | tail -1)

# 3. Configure WordPress to use them
wp option update show_on_front page --path=$WP_PATH
wp option update page_on_front $HOME_ID --path=$WP_PATH
wp option update page_for_posts $BLOG_ID --path=$WP_PATH

# 4. Update tagline
wp option update blogdescription "Your tagline here" --path=$WP_PATH

# 5. Flush
wp rewrite flush --path=$WP_PATH
wp cache flush --path=$WP_PATH
```

**Via MCP (AI Engine):**
```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"wp_update_option","arguments":{"key":"show_on_front","value":"page"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"wp_update_option","arguments":{"key":"page_on_front","value":10}}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"wp_update_option","arguments":{"key":"page_for_posts","value":11}}}
```

---

## Workflow 2: Rewrite homepage with Gutenberg blocks

**Scenario:** Replace the entire homepage content with structured Gutenberg blocks.

### Option A: `wp_write_blocks` (structured, clean)

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"wp_write_blocks","arguments":{
  "ID": 10,
  "mode": "replace",
  "blocks": [
    {"type": "heading", "level": 1, "content": "Hello World"},
    {"type": "paragraph", "content": "My homepage content."},
    {"type": "buttons", "buttons": [
      {"text": "GitHub", "url": "https://github.com/user"},
      {"text": "LinkedIn", "url": "https://linkedin.com/in/user"}
    ]},
    {"type": "columns", "columns": [
      [{"type": "heading", "level": 3, "content": "Col 1"}, {"type": "paragraph", "content": "Content 1"}],
      [{"type": "heading", "level": 3, "content": "Col 2"}, {"type": "paragraph", "content": "Content 2"}]
    ]},
    {"type": "separator"},
    {"type": "list", "ordered": false, "items": ["Item 1", "Item 2"]}
  ]
}}}
```

### Option B: `wp_update_post` with raw Gutenberg HTML (more robust for large content)

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"wp_update_post","arguments":{
  "ID": 10,
  "fields": {
    "post_content": "<!-- wp:heading --><h1>Hello World</h1><!-- /wp:heading -->\n<!-- wp:paragraph --><p>My homepage content.</p><!-- /wp:paragraph -->\n<!-- wp:buttons --><div class=\"wp-block-buttons\"><!-- wp:button --><div class=\"wp-block-button\"><a class=\"wp-block-button__link\" href=\"https://github.com/user\">GitHub</a></div><!-- /wp:button --></div><!-- /wp:buttons -->"
  }
}}}
```

> **When to use which:**
> - `wp_write_blocks` — clean, structured, good for < 20 blocks
> - `wp_update_post` with raw HTML — more robust for large content (30+ blocks), handles `wp:html` blocks with inline styles, doesn't drop the MCP connection

---

## Workflow 3: Switch to a professional theme

**Scenario:** Replace the default WordPress theme with a professional one (e.g., Blocksy).

```bash
WP_PATH=/www/wwwroot/yourdomain.com

# 1. Check current theme
wp theme list --path=$WP_PATH 2>/dev/null | grep -v Deprecated

# 2. Search for professional themes
wp theme search "block" --per-page=10 --path=$WP_PATH 2>/dev/null | grep -v Deprecated

# 3. Install + activate (if wp theme install works)
wp theme install blocksy --activate --path=$WP_PATH

# 4. If wp theme install fails (permission denied on upgrade dir):
#    Download manually and copy
cd /tmp
curl -sL "https://downloads.wordpress.org/theme/blocksy.2.1.56.zip" -o blocksy.zip
unzip -q blocksy.zip -d /tmp/blocksy-extract
sudo cp -r /tmp/blocksy-extract/blocksy $WP_PATH/wp-content/themes/blocksy
sudo chown -R www:www $WP_PATH/wp-content/themes/blocksy
wp theme activate blocksy --path=$WP_PATH

# 5. Configure theme (Blocksy example — dark palette)
wp theme mod set blocksy_palette dark --path=$WP_PATH
wp theme mod set blocksy_container_width 1200 --path=$WP_PATH

# 6. Flush
wp cache flush --path=$WP_PATH
wp rewrite flush --path=$WP_PATH
```

> **Rollback:** `wp theme activate twentytwentyfive --path=$WP_PATH` — the old theme is never deleted, just deactivated.

---

## Workflow 4: Create navigation menu

**Scenario:** Create a professional nav menu with pages and external links.

```bash
WP_PATH=/www/wwwroot/yourdomain.com

# 1. Create menu
MENU_ID=$(wp menu create "Principal" --porcelain --path=$WP_PATH 2>/dev/null | tail -1)

# 2. Add page items (use add-post, NOT add-post-type)
wp menu item add-post $MENU_ID 10 --title="Início" --position=0 --path=$WP_PATH
wp menu item add-post $MENU_ID 11 --title="Blog" --position=1 --path=$WP_PATH

# 3. Add custom (external) links
wp menu item add-custom $MENU_ID "GitHub" "https://github.com/user" --position=2 --path=$WP_PATH
wp menu item add-custom $MENU_ID "LinkedIn" "https://linkedin.com/in/user" --position=3 --path=$WP_PATH

# 4. Assign to menu locations (check available locations first)
wp menu location list --path=$WP_PATH
wp menu location assign $MENU_ID menu_1 --path=$WP_PATH  # header
wp menu location assign $MENU_ID footer --path=$WP_PATH  # footer
```

> **Gotcha:** The WP-CLI subcommand is `wp menu item add-post` (not `add-post-type`). Using `add-post-type` returns: `'add-post-type' is not a registered subcommand of 'menu item'`.

> **Check locations:** `wp menu location list` shows available slots. Blocksy uses `menu_1`, `menu_2`, `menu_mobile`, `footer`. Twenty Twenty-Five uses `primary`, `footer`.

---

## Workflow 5: Upload media when wp_upload_media fails

**Scenario:** `wp_upload_media` or `wp media import` fails with "O arquivo enviado não pode ser movido para wp-content/uploads/YYYY/MM".

**Cause:** The web user (`www:www` on aaPanel) owns the uploads directory. WP-CLI runs as `ubuntu` (or root). PHP can't move the file.

**Fix — manual copy + register as attachment:**

```bash
WP_PATH=/www/wwwroot/yourdomain.com
UPLOAD_DIR=$WP_PATH/wp-content/uploads/2026/09

# 1. Copy file with sudo
sudo cp /tmp/myimage.png $UPLOAD_DIR/myimage.png
sudo chown www:www $UPLOAD_DIR/myimage.png
sudo chmod 644 $UPLOAD_DIR/myimage.png

# 2. Register as WordPress attachment (creates the DB entry)
wp post create \
  --post_type=attachment \
  --post_status=inherit \
  --post_title="My Image" \
  --post_mime_type="image/png" \
  --guid="https://yourdomain.com/wp-content/uploads/2026/09/myimage.png" \
  --porcelain --path=$WP_PATH

# 3. Set the _wp_attached_file meta (critical — without this, WP doesn't know the file path)
ATTACHMENT_ID=<id from step 2>
wp post meta update $ATTACHMENT_ID _wp_attached_file "2026/09/myimage.png" --path=$WP_PATH
```

**Then use in Gutenberg via `wp_update_post`:**
```html
<!-- wp:image {"id":<ATTACHMENT_ID>,"sizeSlug":"thumbnail"} -->
<figure class="wp-block-image size-thumbnail">
  <img src="https://yourdomain.com/wp-content/uploads/2026/09/myimage.png" alt="My Image" class="wp-image-<ATTACHMENT_ID>"/>
</figure>
<!-- /wp:image -->
```

Or inline in an HTML block:
```html
<img src="https://yourdomain.com/wp-content/uploads/2026/09/myimage.png" alt="My Image" style="width:48px;height:48px"/>
```

---

## Workflow 6: Add custom CSS

**Scenario:** Add custom CSS to refine the theme's appearance.

```bash
WP_PATH=/www/wwwroot/yourdomain.com

# Create a custom_css post (WordPress Customizer CSS)
wp post create \
  --post_type=custom_css \
  --post_status=publish \
  --post_title="Theme Custom CSS" \
  --post_content='/* Custom CSS */
h1 { font-weight: 800; }
h2 { font-weight: 700; margin-top: 2.5em; }
.entry-content p { line-height: 1.7; }
.wp-block-buttons .wp-block-button__link { border-radius: 6px; font-weight: 600; }' \
  --porcelain --path=$WP_PATH
```

> **Note:** The `custom_css` post type is registered by WordPress core (since 4.7). Each theme gets its own custom CSS post. The post is linked to the active theme via the `custom_css_post_id` theme mod.

---

## Workflow 7: Create blog posts with categories

**Scenario:** Create multiple blog posts and assign categories.

```bash
WP_PATH=/www/wwwroot/yourdomain.com

# 1. Create categories
CAT_AI=$(wp term create category "Inteligência Artificial" --porcelain --path=$WP_PATH 2>/dev/null | tail -1)
CAT_NET=$(wp term create category ".NET" --porcelain --path=$WP_PATH 2>/dev/null | tail -1)
CAT_PROJ=$(wp term create category "Projetos" --porcelain --path=$WP_PATH 2>/dev/null | tail -1)

# 2. Create posts
POST_ID=$(wp post create \
  --post_type=post \
  --post_status=publish \
  --post_title="My Article Title" \
  --post_content="<!-- wp:paragraph --><p>Article content.</p><!-- /wp:paragraph -->" \
  --porcelain --path=$WP_PATH 2>/dev/null | tail -1)

# 3. Assign categories
wp post term set $POST_ID category $CAT_AI $CAT_PROJ --path=$WP_PATH
```

**Via MCP (AI Engine):**
```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"wp_create_term","arguments":{"taxonomy":"category","name":"Inteligência Artificial"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"wp_create_post","arguments":{"post_title":"My Article","post_content":"<p>Content</p>","post_status":"publish","post_type":"post"}}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"wp_add_post_terms","arguments":{"post_id":28,"terms":[5,7],"taxonomy":"category"}}}
```

---

## Workflow 8: Verify changes bypassing Cloudflare cache

**Scenario:** After updating content via MCP, the live site still shows old content because Cloudflare cached it.

```bash
# Cache-bust with query string
curl -s "https://yourdomain.com/?nocache=1" | grep "new content"

# Increment the nocache param each time
curl -s "https://yourdomain.com/?nocache=2"
curl -s "https://yourdomain.com/?nocache=3"

# For specific assets
curl -sI "https://yourdomain.com/wp-content/uploads/2026/09/image.png?nocache=1"
```

> **Note:** Cloudflare cache can persist for hours. The `?nocache=N` query string bypasses the cache for verification. For permanent cache purge, use the Cloudflare API (requires API token + zone ID) or wait for natural expiration.

> **WP-CLI cache flush does NOT purge Cloudflare.** It only clears the WordPress object cache. Full-page caches (Cloudflare, WP Rocket, Varnish) need their own purge mechanism.

---

## Workflow 9: Convert SVG icons to PNG and upload

**Scenario:** Need technology logo icons on the site. SimpleIcons provides SVGs, but WordPress media library works better with PNGs.

```bash
# 1. Download SVGs from SimpleIcons CDN
mkdir -p /tmp/tech-icons && cd /tmp/tech-icons
curl -sL "https://cdn.simpleicons.org/dotnet/512BD4" -o dotnet.svg
curl -sL "https://cdn.simpleicons.org/angular/DD0031" -o angular.svg

# 2. Convert to PNG (ImageMagick)
convert -background none -density 300 dotnet.svg -resize 256x256 dotnet.png

# 3. If ImageMagick produces empty PNG (SVG without fill on root element):
#    Use cairosvg instead
pip3 install cairosvg --break-system-packages
python3 -c "import cairosvg; cairosvg.svg2png(url='icon.svg', write_to='icon.png', output_width=256, output_height=256)"

# 4. Upload (see Workflow 5 if wp media import fails)
```

> **SimpleIcons CDN format:** `https://cdn.simpleicons.org/<slug>/<hex-color>` returns an SVG with the fill on the root `<svg>` element. `https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/<slug>.svg` returns SVG with fill on the `<path>` element (or no fill at all). The CDN version converts more reliably.

> **Docker SVG gotcha:** The SimpleIcons Docker SVG ships without a `fill` attribute. ImageMagick renders it as empty (332 bytes). Use `cairosvg` or add `fill="#2496ED"` to the `<svg>` root element before converting.
