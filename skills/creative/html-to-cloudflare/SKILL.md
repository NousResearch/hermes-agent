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
| Current public Pages deployment | `https://hermes-pages-d55.pages.dev/` |
| Cloudflare Pages project name | `hermes-pages` |
| Preview deploy host pattern | `https://<hash>.hermes-pages-d55.pages.dev` |
| Hermes-agent repo | `https://github.com/rousegordon-ops/hermes-agent` |
| Gordon's GitHub org/user | `https://github.com/rousegordon-ops` |

## Current direct deploy workflow for `/opt/data/hermes-pages`

When editing Gordon's live homepage, wiki, Pivotal Systems site, or public career/vocation pages, the canonical working tree is usually `/opt/data/hermes-pages` and deploys are direct Cloudflare Pages uploads — not markdown regeneration and not the old workers.dev publish flow.

1. Edit static HTML directly under `/opt/data/hermes-pages`.
2. Deploy with Wrangler to the **project name `hermes-pages`**:
   ```bash
   npx -y -p node@22 -p wrangler wrangler pages deploy /opt/data/hermes-pages --project-name hermes-pages --commit-dirty=true
   ```
   Use this `node@22` wrapper because the base system node may be v20 while current Wrangler requires Node 22+.
3. Verify the canonical URL, not just the preview URL:
   ```bash
   python3 - <<'PY'
   import urllib.request
   url='https://hermes-pages-d55.pages.dev/'
   html=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30).read().decode('utf-8','replace')
   print('ok', len(html), html[:80])
   PY
   ```

Pitfalls:
- `--project-name hermes-pages-d55` fails with "Project not found"; use `hermes-pages`.
- Avoid `curl | python` verification patterns; security tooling may block or require approval. Fetch inside Python instead.
- A successful deploy prints a preview URL, but Gordon's preferred public URL remains `https://hermes-pages-d55.pages.dev/`.

## Index page maintenance

The hub at `https://hermes-pages.rouse-gordon.workers.dev/` is backed by `/opt/data/hermes-pages-repo/index.html`. Every time a new page is published:
1. Note the new filename from the `publish_html` result
2. Update the `href` in the index page's `.page-card` links
3. Push the index update to `hermes-pages`

Cloudflare Pages auto-deploys on push — typically live within 30 seconds.

## Design preferences (from Gordon's feedback)

- **High contrast:** Gordon dislikes low-contrast formatting. Use readable high-contrast text; do not put substantive content in dim gray.
- **Concise public-site CTAs:** Avoid noisy CTA clusters and marketing jargon. For Pivotal Systems, keep the primary CTA simple (e.g. "Book a free consult") and avoid piling on parallel offers like "request an audit," trust pills, "human-in-the-loop," "SMB and semicap," etc. unless explicitly requested.
- **Public email exposure:** Do not expose Gordon's personal email in public site UI or client-side fallback text. For contact forms, route privately through Cloudflare Pages Functions/secrets such as `CONTACT_TO` and `RESEND_API_KEY`.
- **GitHub link:** Use `⚙️ GitHub` button in the hero actions area when appropriate. Don't put it in the footer alongside contact info.
- **Hero layout:** Avatar + name + role + company + tenure + status badge + action buttons.

## Common page types Gordon publishes

- **Profession/career** — landing page with work history, targets, contact info (email as plain text, GitHub in hero)
- **Hobbies** — future: interests, projects outside work, GitHub link goes here per Gordon's preference
- **Reports** — one-off generated content (job search summaries, analysis, etc.)

Each gets its own entry in the hub index.

## Wiki as HTML pages (md2html)

Gordon maintains his personal wiki in markdown (`/opt/data/wiki/`). Convert and publish as auth-protected HTML at `https://hermes-pages.rouse-gordon.workers.dev/wiki/`.

### CRITICAL: Cloudflare Pages .html redirect quirk

**Cloudflare Pages serves static files from a directory but strips `.html` from URLs** — requesting `/wiki/entities/gordon-rouse.html` gets a 307 redirect to `/wiki/entities/gordon-rouse`. This means:

1. **Never use `.html` in any href** used in nav, hub, or any link on wiki pages. Always link to the path WITHOUT the extension. The files exist as `.html` on disk but URLs are extensionless.
2. **Auth check redirect target must also be extensionless**: `window.location.href='/wiki/login?dst='+...` not `/wiki/login.html?...`
3. The `md2html.py` script handles this correctly — do NOT hand-write wiki HTML links.

### Workflow

**Gordon retired the markdown pipeline (2026-05-10). Wiki pages are now written directly in HTML.** The old md2html.py script exists but is no longer used. Do NOT write markdown and convert — edit HTML files directly and push.

**Current workflow:**
1. Edit HTML source directly (e.g. `/opt/data/hermes-pages-repo/wiki/...`)
2. Commit and push: `cd /opt/data/hermes-pages-repo && git add <files> && git commit -m "..." && GIT_TERMINAL_PROMPT=0 git push origin main`
3. Live on Cloudflare Pages in ~30 seconds

### Wiki as HTML pages
- `wiki/index.html` → **hub page** (auth check + page links, entry point at `/wiki/`)
- `wiki/login.html` → **login page** (email + password, NO auth required, URL: `/wiki/login`)
- `wiki/entities/<name>.html` → entity pages (auth check, nav links WITHOUT .html)
- `wiki/concepts/<name>.html` → concept pages (auth check, nav links WITHOUT .html)
- `wiki/schema.html`, `wiki/log.html` → meta pages (auth check)
- `wiki/raw/articles/<name>.html` → raw sources (auth check)

### Public career/vocation section pattern

Gordon wants the career/vocation material publicly accessible while the rest of the personal wiki remains password protected. Current public label is **Vocation Ideas**.

Pattern:
- Homepage card label: `Vocation Ideas`, alongside `Resume`.
- Public hub: `/career` backed by `/opt/data/hermes-pages/career.html`.
- Public pages: `/career-opportunities`, `/business-opportunities`, and `/business-opportunities/...` drilldowns.
- Old protected wiki career URLs should redirect to the public equivalents, e.g. `/wiki/career-opportunities` → `/career-opportunities` and `/wiki/business-opportunities/...` → `/business-opportunities/...`.
- Do **not** remove auth from unrelated wiki pages. Verify a non-career wiki page still contains `wiki_auth` or redirects to `/wiki/login`.
- Update scheduled jobs that maintain this content to write to the public `/opt/data/hermes-pages/...` paths, not `/wiki/...` copies.

**3. Push to hermes-pages:**
```bash
cd /opt/data/hermes-pages-repo && git add wiki/ && git commit -m "Update wiki" && GIT_TERMINAL_PROMPT=0 git push origin main
```
Live in ~30 seconds.

**4. Update the hub index** (`/opt/data/hermes-pages-repo/index.html`) with link to `/wiki/` (or `/wiki/login` for direct login link). Push separately.

### Auth system (client-side, no Cloudflare Access needed)

| File | URL | Auth required | Purpose |
|------|-----|--------------|---------|
| `index.html` | `/wiki/` | ✅ Yes | Hub page — list of wiki pages, auth check on load |
| `login.html` | `/wiki/login` | ❌ No | Login form — email + password |
| `*.html` | `/wiki/<path>` | ✅ Yes | Content pages — inline auth check |

**Credentials:**
- Email: `rouse.gordon@gmail.com`
- Password: `GordonWiki2026!`
- Cookie: `wiki_auth=GW2026` (path=`/wiki`, max-age=1yr, SameSite=Strict)

**Login flow:**
1. Unauthenticated user → hits any `/wiki/*` page → inline script redirects to `/wiki/login?dst=<current path>`
2. Login form shows email + password fields
3. On success: sets cookie `wiki_auth=GW2026`, redirects to `?dst` param (or `/wiki/` hub if no dst)
4. If already logged in: login page auto-redirects to `/wiki/`

**Error messages:**
- Wrong email → "Email not recognized"
- Wrong password (right email) → "Incorrect password"
- Separate messages so user knows which field is wrong

**Enterprise upgrade path:** Set up **Cloudflare Zero Trust** → cloudflare.com/zero-trust → Authentication → Add GitHub identity provider → Access Policy for `/wiki/*`. Replaces client-side auth with proper OAuth/GitHub login. Remove client-side auth check scripts when that is done.

### Wiki content strategy

Gordon's wiki is a long-term knowledge base. When incorporating new information (e.g. from chat exports):

**What to add:** Things that reveal something personally significant — not random factoids:
- Interests, likes/dislikes (e.g. hiking, fishing, style)
- Career/project details (e.g. Sidekick Studio evolution)
- Health or behavioral patterns (e.g. cutting back on alcohol)
- Financial life decisions
- Travel that reveals preferences

**What to skip:** One-off factoids, recipes, restaurant tips, ephemeral advice.
- Trivial "how many minutes" questions
- Generic questions answered quickly
- Things that would be outdated in 6 months

**Content sensitivity:** Gordon corrects overstatements. E.g. Sidekick Studio is a **hobby project** — don't frame it as a career pivot or viable post-Ventura option. When in doubt, understate rather than overstate significance. Gordon will correct you if you go too far; don't make him repeat it.

**How to incorporate:**
1. Read the JSON export (`/opt/data/cache/documents/`) — scan titles first with a quick loop to identify meaty conversations
2. Read the content of significant ones
3. Write new markdown pages under `/opt/data/wiki/` in the appropriate subdirectory:
   - `hobbies/` — outdoor activities, fitness, style, travel
   - `projects/` — Sidekick Studio and similar
   - `personal/` — health, habits
   - `entities/` — update the person entity
   - `concepts/` — update concept pages (e.g. Ventura relocation) with new context
4. Re-run `python3 /opt/data/scripts/md2html.py /opt/data/wiki`
5. Git add, commit, push
6. Update nav bar in `md2html.py` if adding entirely new categories

### Wiki page category conventions

- **Day Hikes** — separate from overnight trips. Day hike content goes in `hobbies/hiking.md`
- **Backpacking** — overnight wilderness trips (hiking + camping). Goes in `hobbies/backpacking.md`. This is where Tinker Knob and training hikes belong.
- **Fishing** — specifically backcountry/trout fishing, which is a component of backpacking trips. Goes in `hobbies/backcountry-fishing.md`.
- These are three distinct categories — don't conflate them.

### Wiki content strategy (when adding from chat exports)

See `references/wiki-content-strategy.md` for the full guide. Key rules:
- **Add** personally significant things — interests, career context, health patterns, financial decisions, travel that reveals preferences.
- **Skip** trivial one-off facts, generic advice, anything outdated in 6 months.
- **Understate rather than overstate** — Gordon corrects overstatements. "A hobby project" is safe; "a viable career option" gets corrected. Use "hobby" as the load-bearing word for Sidekick Studio.

### Wiki index page structure

The index is an HTML file (`wiki/index.html`), not markdown. When adding a new page, insert a link in the appropriate section using the same pattern as existing links. Don't create a summary or description — just a link.

### Nav bar organization

**User preference: hide the top nav bar.** Set `.topbar { display: none; }` in the CSS. Gordon preferred a clean page without it.

The nav list is still generated (so `[[wikilinks]]` in content stay correct and `active` highlighting works), but the entire `.topbar` div is hidden via CSS:

```python
.topbar { display: none; }
```

If you need a visual separator between nav groups (e.g. between Ventura and Fishing to show "Hobbies" group), insert a `<li class="sep">|` after a known anchor in the nav HTML via string replacement after generation:

```python
nav_html = '\n'.join(
    f'<li><a href="{href}" class="...">{label}</a></li>'
    for href, label in nav_items
)
nav_html = nav_html.replace(
    'Ventura</a></li>',
    'Ventura</a></li>\n        <li class="sep">|</li>'
)
```

Current nav order: Home → Gordon → KLA → Ventura → Sidekick → Fishing → Hiking → Backpacking → Fitness → Style → London → Log. No "Health" separate nav item — health/alcohol content lives inline in Fitness.

### Wiki content strategy (when adding from chat exports)

- **Forgetting to update the index** — results in stale links on the hub page pointing to old filenames
- **`GITHUB_TOKEN` not set** — `publish_html` returns `{"success": false, "error": "GITHUB_TOKEN is not set"}`. Fix: read from `/opt/data/.git-credentials` as shown above
- **Interactive git prompts** — never run `git push` without `GIT_TERMINAL_PROMPT=0` in this environment
- **Git author identity unknown** — always configure `user.email` and `user.name` before committing to hermes-pages repo (different from hermes-agent repo which has its own gitconfig)
- **Wiki subdirectory in hermes-pages** — the `.git` directory from the original wiki clone can cause `git add` failures. Always `rm -rf /opt/data/hermes-pages-repo/gordons-llm-wiki/.git` before adding.
- **`md2html.py` runs on import** — the old version had a top-level for-loop that executed immediately on `import`, which is dangerous if the script is ever imported elsewhere. The script now guards all work behind `if __name__ == '__main__'`. Never add top-level side effects to this script.
- **Login redirect must NOT include `.html`** — the Cloudflare Pages static file server strips `.html` from URLs (returning a 307 to the extensionless version). The auth redirect in `build_page()` MUST use `/wiki/login?dst=...` not `/wiki/login.html?dst=...`. The script generates this inline; verify the generated HTML contains the extensionless URL.
- **Old wiki source path** — the original script hardcoded `/opt/data/hermes-pages-repo/gordons-llm-wiki` as the markdown source. Always pass the source dir as the first argument: `python3 /opt/data/scripts/md2html.py /opt/data/wiki`.
- **Wiki page URLs, display names, and nav order** — All driven by `WIKI_PATH_MAP` in `md2html.py`. This map is the single source of truth for: URL path (what appears in href), display label (what the user sees), and nav order (items appear in `nav_items` list in insertion order). Changes to page names or paths require updating the map AND the `index.md` wikilink AND rebuilding. Missing map entries = wrong links and garbled labels.

**Table rendering (md2html bug fix):** The old code emitted one `<table>` per row. The fix uses a `table_buf` list that collects consecutive pipe-delimited rows, then flushes them as a single `<table>` on the next non-table line. First row gets `<th>` cells. Current CSS: `border-collapse: separate; border-radius: 8px; overflow: hidden` on the `<table>`, header row distinct background, `tr:hover td` hover effect.

**Git commit/push:** Use `execute_code` with subprocess — `terminal` blocks compound `&` commands. Set `GIT_TERMINAL_PROMPT=0` in the subprocess environment, not as a shell string prefix.

**Overstating Gordon's projects:** Sidekick Studio is a hobby project. "A hobby project he's building" is safe; "a viable post-Ventura career option" gets corrected. Understate by default.

  ```python
  WIKI_PATH_MAP = {
      'gordon-rouse':      ('entities/gordon-rouse',      'Gordon Rouse'),
      'hobbies/backcountry-fishing': ('hobbies/backcountry-fishing', 'Backcountry Fishing'),
      ...
  }
  
  def wiki_link(name):
      name = name.replace('.md', '').replace('.html', '')
      if name in WIKI_PATH_MAP:
          path, label = WIKI_PATH_MAP[name]
      else:
          path = name
          label = name.split('/')[-1].replace('-', ' ').title()
      return f'<a href="/wiki/{path}">{label}</a>'
  ```

  Every page in `/opt/data/wiki/` must have a corresponding entry. Missing entries cause wrong URLs and garbled display names (e.g. "hobbies/backcountry-fishing" instead of "Backcountry Fishing"). The map also serves as the canonical list of all wiki pages.
- **Combining wiki pages** — Valid pattern. When combining: pull the secondary section to the bottom of the primary page. Remove the merged page's markdown file, remove it from nav, rebuild, commit, push.
- **Overstating significance triggers corrections** — Gordon corrected the Sidekick Studio framing twice. When in doubt, understate. "A hobby project he's building" is safer than "a viable post-Ventura option." Don't make him repeat it.
- **Terminal blocks `git commit && git push`** — Compound commands with `&` cause errors. Use `execute_code` with subprocess instead:
  ```python
  import subprocess, os
  os.chdir('/opt/data/hermes-pages-repo')
  os.environ['GIT_TERMINAL_PROMPT'] = '0'
  subprocess.run(['git', 'commit', '-m', 'message'])
  subprocess.run(['git', 'push', 'origin', 'main'])
  ```
- **`rousegordon-ops` is a GitHub USER, not an org** — `https://api.github.com/users/rousegordon-ops/repos` works; `https://api.github.com/orgs/rousegordon-ops/repos` returns 404. The PAT only has read access to that user's repos (hermes-agent public, gordonclaw/hermes-pages/parts-finder/SidekickStudio private).

- **Python 3.13 f-string brace collision — CRITICAL** — Literal `{` and `}` in CSS inside f-strings must be doubled to `{{` and `}}`. Python 3.13 added dict unpacking syntax `{**}` as a reserved pattern, making bare `{` in f-strings a syntax error. The script appears to run (compiles OK) but fails at runtime inside `build_page()` with `NameError: name 'border' is not defined` (or whichever CSS property word comes first after the unparsed brace). The error occurs AFTER the truncated HTML has been written to disk — so the file exists with wrong content and `git status` shows "nothing to commit." **Verification:** After running md2html.py, ALWAYS grep the actual output file for expected content before committing. Example failure: `table { border-collapse: ... }` → `NameError: name 'border'`. **Fix:** Identify the `<style>` block in the f-string and replace ALL `{` with `{{` and all `}` with `}}`:
  ```python
  style_start = content.find('  <style>\n')
  style_end = content.find('\n  </style>')
  style_block_fixed = style_block.replace('{', '{{').replace('}', '}}')
  content = before + style_block_fixed + after
  ```

- **Script sync between `/opt/data/scripts/` and `/opt/data/hermes-pages-repo/scripts/`** — Edit `/opt/data/scripts/md2html.py` first (canonical). Then `shutil.copy('/opt/data/scripts/md2html.py', '/opt/data/hermes-pages-repo/scripts/md2html.py')`. Always copy before running, and always run from the repo directory (`python3 /opt/data/hermes-pages-repo/scripts/md2html.py`). The two copies diverge over time — assume the repo copy is stale until just synced.

- **Always verify generated HTML before committing** — md2html.py may run without crashing but produce wrong output (e.g. "8 sidekicks" when source says "9 sidekicks"). `git status` shows "nothing to commit" while the file is wrong. Grep the actual output file: `grep "9 sidekicks" wiki/entities/gordon-rouse.html`. If wrong, the script errored after writing truncated output — fix script, re-sync, re-run, verify again.

- **Git clone private repos works with the PAT** — `GIT_TERMINAL_PROMPT=0 git clone https://github.com/rousegordon-ops/SidekickStudio.git` succeeds. Don't assume a repo is inaccessible without trying.

- **Source repo is ground truth over session memory** — When Gordon says "get the latest info from the source repo," clone it and read actual files. Don't rely on wiki pages that may be stale. The source repo is authoritative.
