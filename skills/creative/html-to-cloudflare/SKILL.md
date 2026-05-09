---
name: html-to-cloudflare
description: "Generate HTML content and publish it to Gordon's Cloudflare Pages site (hermes-pages). Covers the full workflow from HTML generation to live URL."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [html, cloudflare, publishing, static-site]
    homepage: https://github.com/rousegordon-ops/hermes-pages
---

# HTML → Cloudflare Pages

Generate HTML content and publish it to Gordon's personal Cloudflare Pages deployment. **The canonical workflow for any generated HTML that Gordon wants to view live without copy-pasting.**

## Workflow

1. **Generate HTML** — write the content to a local file (typically in `/opt/data/repo/` or wherever the work is happening)
2. **Publish** — use `publish_html` tool via Python (since the tool isn't auto-discovered as a direct tool call):
   ```python
   import sys, os
   sys.path.insert(0, '/opt/data/repo')
   os.environ['GITHUB_TOKEN'] = read_token()
   from tools.publish_html import publish_html
   with open('/path/to/file.html', 'r') as f:
       html = f.read()
   result = publish_html('slug-name', html)
   print(result)  # {"success": true, "url": "https://hermes-pages.rouse-gordon.workers.dev/<hash>-slug-name.html", ...}
   ```
3. **Update the index** — `publish_html` always creates a new file (new hash each time). The index at `/opt/data/hermes-pages-repo/index.html` needs to be updated with the new filename so the hub page links to the latest version.

## Git credentials setup (critical)

The `publish_html` tool requires `GITHUB_TOKEN` in the environment. It's NOT automatically set — read it from the credentials file:

```python
import re
with open('/opt/data/.git-credentials') as f:
    cred = f.read()
m = re.search(r'x-access-token:([^\@]+)\@', cred)
token = m.group(1) if m else ''
os.environ['GITHUB_TOKEN'] = token
```

The credentials file has the format: `https://x-access-token:{github_pat_...}@github.com`

## Pushing to hermes-pages repo directly

If `publish_html` isn't available or you need to push manually:

```bash
cd /opt/data/hermes-pages-repo
git config user.email "hermes@hermes-agent.local"
git config user.name "Hermes"
git add <file>
git commit -m "<message>"
GIT_TERMINAL_PROMPT=0 git push origin main
```

**Always use `GIT_TERMINAL_PROMPT=0`** — avoids interactive SSH/GitHub prompts that would hang.

## Key repos and URLs

| Repo | URL |
|------|-----|
| Pages site | `https://github.com/rousegordon-ops/hermes-pages` |
| Pages deployment | `https://hermes-pages.rouse-gordon.workers.dev` |
| Hermes-agent repo | `https://github.com/rousegordon-ops/hermes-agent` |
| Gordon's GitHub org | `https://github.com/rousegordon-ops` |

## Index page maintenance

The hub at `https://hermes-pages.rouse-gordon.workers.dev/` is backed by `/opt/data/hermes-pages-repo/index.html`. Every time a new page is published:
1. Note the new filename from the `publish_html` result
2. Update the `href` in the index page's `.page-card` links
3. Push the index update to `hermes-pages`

Cloudflare Pages auto-deploys on push — typically live within 30 seconds.

## Design preferences (from Gordon's feedback)

- **Email display:** Show as plain text `gordon.rouse@gmail.com` NOT as a `mailto:` link. `mailto:` links trigger browser/app picker dialogs which users find annoying on a static page. Use `<span>✉️ gordon.rouse@gmail.com</span>` instead.
- **GitHub link:** Use `⚙️ GitHub` button in the hero actions area. Don't put it in the footer alongside the email.
- **Hero layout:** Avatar + name + role + company + tenure + email (plain text) + status badge + action buttons.

## Common page types Gordon publishes

- **Profession/career** — landing page with work history, targets, contact info (email as plain text, GitHub in hero)
- **Hobbies** — future: interests, projects outside work, GitHub link goes here per Gordon's preference
- **Reports** — one-off generated content (job search summaries, analysis, etc.)

Each gets its own entry in the hub index.

## Wiki as HTML pages

Gordon maintains his personal wiki in markdown (`/opt/data/wiki/`) and wants it rendered as HTML on Cloudflare. Workflow:

1. **Write markdown** — entities, concepts, comparisons in the wiki dir
2. **Convert to HTML** — use `/opt/data/scripts/md2html.py`:
   - Reads all `.md` files under the wiki directory
   - Applies dark GitHub-style theme (matching the landing page aesthetic)
   - Adds a top navigation bar with links to key wiki pages
   - Outputs `.html` files mirroring the wiki structure under `/opt/data/hermes-pages-repo/wiki/`
   - Strips YAML frontmatter before rendering
   - Converts wikilinks `[[Page Name]]` to `<a href="Page-Name.html">Page Name</a>`
3. **Push** — commit to `hermes-pages` repo: `cd /opt/data/hermes-pages-repo && git add wiki/ && git commit -m "Update wiki" && GIT_TERMINAL_PROMPT=0 git push origin main`
4. **Access control** — Cloudflare Zero Trust Access (free for 50 users) secures the wiki:
   - Go to **dash.cloudflare.com → Zero Trust → Settings → Authentication → Add identity provider → GitHub** (easiest since Gordon uses GitHub)
   - Create an Access Policy for `hermes-pages.rouse-gordon.workers.dev/wiki/*`
   - Include Gordon's email or GitHub identity; action: Allow
   - Note: this locks the entire `workers.dev` subdomain. If landing pages should stay public, need separate subdomains.
   - If Gordon can't find Settings: try **cloudflare.com/zero-trust** directly, or in Pages: **Workers & Pages → hermes-pages → Settings → Restrictions**

## Cloudflare Access setup troubleshooting

Gordon couldn't find Settings at cloudflare.com/zero-trust. Possible paths:
- **Zero Trust dashboard**: dash.cloudflare.com → Left sidebar → Zero Trust → Settings tab
- **Pages project**: Workers & Pages → hermes-pages project → Settings → General → Access Policy
- **Direct URL**: cloudflare.com/zero-trust
- The key setting is under **Access → Applications** to create a policy targeting `/wiki/*`

## Pitfalls

- **Forgetting to update the index** — results in stale links on the hub page pointing to old filenames
- **`GITHUB_TOKEN` not set** — `publish_html` returns `{"success": false, "error": "GITHUB_TOKEN is not set"}`. Fix: read from `/opt/data/.git-credentials` as shown above
- **Interactive git prompts** — never run `git push` without `GIT_TERMINAL_PROMPT=0` in this environment
- **Git author identity unknown** — always configure `user.email` and `user.name` before committing to hermes-pages repo (different from hermes-agent repo which has its own gitconfig)
- **Wiki subdirectory in hermes-pages** — the `.git` directory from the original wiki clone can cause `git add` failures. Always `rm -rf /opt/data/hermes-pages-repo/gordons-llm-wiki/.git` before adding.
