---
name: wordpress-static-front-page
description: "Set a WordPress static front page over XML-RPC, verified."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [WordPress, XML-RPC, FrontPage, cPanel, Verification]
---

# WordPress Static Front Page (XML-RPC, verified)

Making a page the front page is three WordPress options that only work
together: `show_on_front=page`, `page_on_front=<page id>` and
`page_for_posts=<blog page id>`. Creating the page itself with `wp.newPost`
(`post_type=page`) works fine on shared hosts; the option write is the part
that silently breaks.

## Why a "success" proves nothing (#121361)

On hardened shared hosts (cPanel / Softaculous installs; reproduced on three
sites of one host family) the XML-RPC option route is filtered:

- `wp.setOptions` returns an empty `[]` - no error - while discarding the
  write. The homepage still renders the blog index.
- `wp.getOptions` never returns `show_on_front`, `page_on_front` or
  `page_for_posts`, so the state cannot be read back either.

An agent that trusts the return value reports "static front page set" while
the site is wrong. Treat an empty or unchanged read-back as a failure -
the same class as a provider that accepts unknown fields and answers HTTP
200 while quietly dropping them.

## Route 1: verified XML-RPC write (try first)

The bundled script sends the full option trio in one `wp.setOptions` call,
then reads every member back with `wp.getOptions`. It exits 1 with
`FrontPageNotApplied` when any value is missing or unchanged, and exits 0
only when the read-back confirms the combo:

```bash
WP_PASSWORD=... python3 scripts/set_front_page.py \
  --url https://example.com/xmlrpc.php \
  --user admin --page-id 12 --page-for-posts 14
```

Exit 0 plus `{"ok": true, "options": ...}` means the options verified.
Exit 1 means the host discarded the write - go to Route 2. Never report
success on exit 1.

## Route 2: mu-plugin file write (works where XML-RPC is filtered)

The database write itself is fine; only the XML-RPC route is filtered. Drop
a small must-use plugin at `wp-content/mu-plugins/set-front-page.php` via
the file manager or SFTP. The ids below are placeholders - replace them
with the created page ids:

```php
<?php // mu-plugins/set-front-page.php
add_action('init', function () {
    if (get_option('show_on_front') !== 'page') {
        update_option('show_on_front', 'page');
        update_option('page_on_front', 12);   // PAGE_ID placeholder
        update_option('page_for_posts', 14);  // BLOG_PAGE_ID placeholder
    }
});
```

Verify this route by fetching the homepage and checking it renders the
static page instead of the post list - `wp.getOptions` cannot confirm the
state on these hosts.

## Read-back discipline for every option write

1. Send the complete option set in one call whenever options only make
   sense together.
2. Read the options back after the write; never trust the write's return
   value alone.
3. Empty, missing or unchanged read-back = failure. Escalate (mu-plugin
   file write, admin UI) instead of reporting success.
