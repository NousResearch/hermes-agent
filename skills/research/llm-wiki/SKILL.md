---
name: llm-wiki
description: "Karpathy's LLM Wiki: build/query interlinked markdown KB."
version: 2.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [wiki, knowledge-base, research, notes, markdown, rag-alternative]
    category: research
    related_skills: [obsidian, arxiv]
---

# Karpathy's LLM Wiki

Build and maintain a persistent, compounding knowledge base as interlinked markdown files.
Based on [Andrej Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

Unlike traditional RAG (which rediscovers knowledge from scratch per query), the wiki
compiles knowledge once and keeps it current. Cross-references are already there.
Contradictions have already been flagged. Synthesis reflects everything ingested.

**Division of labor:** The human curates sources and directs analysis. The agent
summarizes, cross-references, files, and maintains consistency.

## Agent Brain Wiki (separate from Gordon's wiki)

My agent brain wiki lives at `/opt/data/hermes-brain/` (or `~/wiki` if that doesn't exist).
It's markdown-based, Karpathy-style, for research synthesis and knowledge compilation.
- `SCHEMA.md`, `index.md`, `log.md`, `raw/`, `entities/`, `concepts/`, etc.
- Gordon does NOT use this — it's mine.
- Gordon's wiki is separate: `/opt/data/hermes-pages/wiki/` (pure HTML, see below).

**Do NOT conflate the two.** When I mention "the wiki" in a research context, I'm referring to my agent brain at `/opt/data/hermes-brain/`, not Gordon's HTML wiki.

## Gordon's Wiki — Pure HTML

Gordon retired the markdown pipeline. His wiki is **pure static HTML** at `/opt/data/hermes-pages/wiki/`.

**Structure:**
- `index.html` — wiki hub (auth-protected, lists all pages)
- `login.html` — auth form
- `entities/` — person/org pages
- `concepts/` — topic pages
- `hobbies/` — hobby pages
- `projects/` — project pages
- `raw/` — source articles
- `schema.html`, `log.html` — meta pages

**To add a page:**
1. Write HTML directly to the appropriate subdirectory
2. Add link in `index.html` under the correct section
3. Deploy: `cd /opt/data/hermes-pages && git add . && git commit -m "..." && GIT_TERMINAL_PROMPT=0 git push origin main`
4. Then run: `npx -y -p node@22 -p wrangler wrangler pages deploy /opt/data/hermes-pages --project-name hermes-pages --commit-dirty=true`
5. Live in ~30s at `https://hermes-pages-d55.pages.dev/wiki/...`

**⚠️ Git push alone does NOT trigger Cloudflare Pages deployment.** The push goes to GitHub but the Pages site doesn't update from git alone — you MUST run the `wrangler pages deploy` command after every push. This was discovered empirically (2026-05-16).

**Deploy verification:** `curl -L -A 'Mozilla/5.0' https://hermes-pages-d55.pages.dev/wiki/...`

**Auth:** email `rouse.gordon@gmail.com`, password `GordonWiki2026!`, cookie `wiki_auth=GW2026`

**Do NOT:** use `md2html.py`, regenerate from markdown, or touch `/opt/data/hermes-pages/gordons-llm-wiki/`.

### Public standalone knowledge bases on Hermes Pages

When Gordon asks to create a new "LLM wiki knowledge base" and also says it should appear on the Hermes Pages homepage, treat it as a **public standalone static knowledge base** unless he explicitly asks for the password-protected personal `/wiki/`. Use `/opt/data/hermes-pages/<topic-kb>/` with small interlinked HTML pages plus a homepage card in `/opt/data/hermes-pages/index.html`. This is different from updating the protected personal wiki.

Recommended pattern:
1. Research broadly first; for deep technical domains, delegate parallel research streams (fundamentals, latest research, applied/commercial context) and synthesize before writing.
2. Build a hub page (`index.html`) plus 6–10 concise child pages, not one giant article.
3. Make hub cards clickable drilldowns: the entire card should be an `<a>`/link to the relevant child page, not just a small "read more" text link. Cards should visually indicate clickability and preserve keyboard accessibility.
4. Link tool and software product references inline. When mentioning a concrete tool, vendor product, open-source project, library, SaaS, CLI, framework, or platform, link the first substantive mention to the official product/docs/repo page when available; prefer official URLs over generic search results.
5. Include a `sources.html` reading list with source titles and URLs.
6. Keep the design high-contrast and navigable with a sticky side nav.
7. Validate internal links locally before deploying, including that every hub card target resolves and that important tool/software references have outbound links.
8. Deploy with the direct Cloudflare Pages workflow from `html-to-cloudflare`, then verify homepage card and representative child pages live.

**How it works now for Gordon:**
- Preferred homepage: `https://hermes-pages-d55.pages.dev/` (links to Wiki and Profession)
- Static wiki HTML: `/opt/data/hermes-pages/wiki/`
- Deploy mirror: `/opt/data/hermes-pages-files/wiki/`
- Deploy command: `export PATH=/opt/data/.nvm/bin:$PATH && /opt/data/.npm-global/bin/wrangler pages deploy /opt/data/hermes-pages-files --project-name hermes-pages --branch main --commit-dirty=true`
- Auth: email+password login required. Only `rouse.gordon@gmail.com` / `GordonWiki2026!`
- Gordon views the rendered HTML in his browser; maintain static HTML directly unless Gordon explicitly asks for markdown regeneration.
- Important: do **not** regenerate the wiki from `/opt/data/hermes-pages/gordons-llm-wiki/` by default; that previously overwrote richer static pages.

**To update Gordon's wiki:**
1. Edit/add static HTML in `/opt/data/hermes-pages/wiki/`
2. Keep parent index/hub pages linked manually; root wiki index should list top-level hubs only, not every child page
3. For deep dives, create semantic child pages under the parent hub (e.g. `/wiki/business-opportunities/acquire-local-service-business`) and link parent ↔ child from the relevant parent item/section
4. Copy wiki into deploy mirror: `rm -rf /opt/data/hermes-pages-files/wiki && cp -a /opt/data/hermes-pages/wiki /opt/data/hermes-pages-files/wiki`
5. Run the direct Cloudflare Pages deploy command above — project name is `hermes-pages`, not the domain suffix `hermes-pages-d55`
6. Verify with browser-like curl: `curl -L -A 'Mozilla/5.0' -H 'Cookie: wiki_auth=GW2026' <live-url>` because Python `urllib` can get Cloudflare 403s
7. Live in ~30 seconds at `https://hermes-pages-d55.pages.dev/wiki/`

Detailed Gordon-specific notes: see `references/gordon-static-html-wiki-maintenance.md`. For reducing duplication by refactoring hub/child topology, see `references/gordon-static-html-wiki-topology-refactors.md`. For public standalone KBs linked from the Hermes Pages homepage, see `references/public-static-kb-on-hermes-pages.md`. For large internal engineering KB hierarchy strategy (LLM-derived vs programmatic graph/community detection), see `references/hierarchy-strategy-for-large-engineering-kbs.md`. For hierarchy trade-offs and the recommended hybrid model for large engineering/team KBs, see `references/hierarchy-strategy-for-large-engineering-kbs.md`.

**To add the wiki to Obsidian later:** Clone `https://github.com/rousegordon-ops/hermes-pages`, point Obsidian at `gordons-llm-wiki/` subdirectory.

## When This Skill Activates

Use this skill when the user:
- Asks to create, build, or start a wiki or knowledge base
- Asks to ingest, add, or process a source into their wiki
- Asks a question and an existing wiki is present at the configured path
- Asks to lint, audit, or health-check their wiki
- References their wiki, knowledge base, or "notes" in a research context

## Wiki Location

**Location:** Set via `WIKI_PATH` environment variable (e.g. in `~/.hermes/.env`).

If unset, defaults to `~/wiki`.

```bash
WIKI_PATH=/opt/data/memories
WIKI="${WIKI_PATH:-/opt/data/memories}"
```

The wiki is just a directory of markdown files — open it in Obsidian, VS Code, or
any editor. No database, no special tooling required.

## Architecture: Three Layers

```
wiki/
├── SCHEMA.md           # Conventions, structure rules, domain config
├── index.md            # Sectioned content catalog with one-line summaries
├── log.md              # Chronological action log (append-only, rotated yearly)
├── raw/                # Layer 1: Immutable source material
│   ├── articles/       # Web articles, clippings
│   ├── papers/         # PDFs, arxiv papers
│   ├── transcripts/    # Meeting notes, interviews
│   └── assets/         # Images, diagrams referenced by sources
├── entities/           # Layer 2: Entity pages (people, orgs, products, models)
├── concepts/           # Layer 2: Concept/topic pages
├── comparisons/        # Layer 2: Side-by-side analyses
└── queries/            # Layer 2: Filed query results worth keeping
```

**Layer 1 — Raw Sources:** Immutable. The agent reads but never modifies these.
**Layer 2 — The Wiki:** Agent-owned markdown files. Created, updated, and
cross-referenced by the agent.
**Layer 3 — The Schema:** `SCHEMA.md` defines structure, conventions, and tag taxonomy.

## Resuming an Existing Wiki (CRITICAL — do this every session)

When the user has an existing wiki, **always orient yourself before doing anything**:

① **Read `SCHEMA.md`** — understand the domain, conventions, and tag taxonomy.
② **Read `index.md`** — learn what pages exist and their summaries.
③ **Scan recent `log.md`** — read the last 20-30 entries to understand recent activity.

```bash
WIKI_PATH=/opt/data/memories
WIKI="${WIKI_PATH:-/opt/data/memories}"
# Orientation reads at session start
read_file "$WIKI/SCHEMA.md"
read_file "$WIKI/index.md"
read_file "$WIKI/log.md" offset=<last 30 lines>
```

Only after orientation should you ingest, query, or lint. This prevents:
- Creating duplicate pages for entities that already exist
- Missing cross-references to existing content
- Contradicting the schema's conventions
- Repeating work already logged

For large wikis (100+ pages), also run a quick `search_files` for the topic
at hand before creating anything new.

## Initializing a New Wiki

When the user asks to create or start a wiki:

1. Determine the wiki path (from `$WIKI_PATH` env var, or ask the user; default `~/wiki`)
2. Create the directory structure above
3. Ask the user what domain the wiki covers — be specific
4. Write `SCHEMA.md` customized to the domain (see template below)
5. Write initial `index.md` with sectioned header
6. Write initial `log.md` with creation entry
7. Confirm the wiki is ready and suggest first sources to ingest

### SCHEMA.md Template

Adapt to the user's domain. The schema constrains agent behavior and ensures consistency:

```markdown
# Wiki Schema

## Domain
[What this wiki covers — e.g., "AI/ML research", "personal health", "startup intelligence"]

## Conventions
- File names: lowercase, hyphens, no spaces (e.g., `transformer-architecture.md`)
- Every wiki page starts with YAML frontmatter (see below)
- Use `[[wikilinks]]` to link between pages (minimum 2 outbound links per page)
- When updating a page, always bump the `updated` date
- Every new page must be added to `index.md` under the correct section
- Every action must be appended to `log.md`
- **Provenance markers:** On pages that synthesize 3+ sources, append `^[raw/articles/source-file.md]`
  at the end of paragraphs whose claims come from a specific source. This lets a reader trace each
  claim back without re-reading the whole raw file. Optional on single-source pages where the
  `sources:` frontmatter is enough.

## Frontmatter
  ```yaml
  ---
  title: Page Title
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  type: entity | concept | comparison | query | summary
  tags: [from taxonomy below]
  sources: [raw/articles/source-name.md]
  # Optional quality signals:
  confidence: high | medium | low        # how well-supported the claims are
  contested: true                        # set when the page has unresolved contradictions
  contradictions: [other-page-slug]      # pages this one conflicts with
  ---
  ```

`confidence` and `contested` are optional but recommended for opinion-heavy or fast-moving
topics. Lint surfaces `contested: true` and `confidence: low` pages for review so weak claims
don't silently harden into accepted wiki fact.

### raw/ Frontmatter

Raw sources ALSO get a small frontmatter block so re-ingests can detect drift:

```yaml
---
source_url: https://example.com/article   # original URL, if applicable
ingested: YYYY-MM-DD
sha256: <hex digest of the raw content below the frontmatter>
---
```

The `sha256:` lets a future re-ingest of the same URL skip processing when content is unchanged,
and flag drift when it has changed. Compute over the body only (everything after the closing
`---`), not the frontmatter itself.

## Tag Taxonomy
[Define 10-20 top-level tags for the domain. Add new tags here BEFORE using them.]

Example for AI/ML:
- Models: model, architecture, benchmark, training
- People/Orgs: person, company, lab, open-source
- Techniques: optimization, fine-tuning, inference, alignment, data
- Meta: comparison, timeline, controversy, prediction

Rule: every tag on a page must appear in this taxonomy. If a new tag is needed,
add it here first, then use it. This prevents tag sprawl.

## Page Thresholds
- **Create a page** when an entity/concept appears in 2+ sources OR is central to one source
- **Add to existing page** when a source mentions something already covered
- **DON'T create a page** for passing mentions, minor details, or things outside the domain
- **Split a page** when it exceeds ~200 lines — break into sub-topics with cross-links
- **Archive a page** when its content is fully superseded — move to `_archive/`, remove from index

## Entity Pages
One page per notable entity. Include:
- Overview / what it is
- Key facts and dates
- Relationships to other entities ([[wikilinks]])
- Source references

## Concept Pages
One page per concept or topic. Include:
- Definition / explanation
- Current state of knowledge
- Open questions or debates
- Related concepts ([[wikilinks]])

## Comparison Pages
Side-by-side analyses. Include:
- What is being compared and why
- Dimensions of comparison (table format preferred)
- Verdict or synthesis
- Sources

## Update Policy
When new information conflicts with existing content:
1. Check the dates — newer sources generally supersede older ones
2. If genuinely contradictory, note both positions with dates and sources
3. Mark the contradiction in frontmatter: `contradictions: [page-name]`
4. Flag for user review in the lint report
```

### index.md Template

The index is sectioned by type. Each entry is one line: wikilink + summary.

```markdown
# Wiki Index

> Content catalog. Every wiki page listed under its type with a one-line summary.
> Read this first to find relevant pages for any query.
> Last updated: YYYY-MM-DD | Total pages: N

## Entities
<!-- Alphabetical within section -->

## Concepts

## Comparisons

## Queries
```

**Scaling rule:** When any section exceeds 50 entries, split it into sub-sections
by first letter or sub-domain. When the index exceeds 200 entries total, create
a `_meta/topic-map.md` that groups pages by theme for faster navigation.

### log.md Template

```markdown
# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: `## [YYYY-MM-DD] action | subject`
> Actions: ingest, update, query, lint, create, archive, delete
> When this file exceeds 500 entries, rotate: rename to log-YYYY.md, start fresh.

## [YYYY-MM-DD] create | Wiki initialized
- Domain: [domain]
- Structure created with SCHEMA.md, index.md, log.md
```

## Wiki Gardener Policy — Agent-Inferred Taxonomy & Hierarchy

Do not make the user define the taxonomy. Infer a reasonable structure from the content, then refactor as the wiki grows. "Human maintained" means human-curated/overseen, not manual hierarchy upkeep by the user: for substantial updates, the agent should automatically re-evaluate whether the hierarchy needs to evolve.

Principles:
- **Parent pages are maps, not dumping grounds.** Keep hub/index pages concise and navigable. If a hub starts accumulating repeated details also covered by child pages, refactor it back into a map/portfolio page and move shared analysis into an intermediate child hub.
- **Promote repeated or central topics to pages.** If a topic appears repeatedly, becomes a decision area, or the user asks for a deep dive, create a dedicated page.
- **Use semantic containment for hierarchy:** broad domain → hub page; major bucket → child hub; specific idea/opportunity/company/source → child page. When two child pages share the same thesis/market framing, add a middle parent for the shared material instead of duplicating it in each page.
- **No hard max-child rule yet, but use review thresholds.** There is no deterministic max children/depth optimizer in the current prompt-driven workflow. As a soft heuristic, keep direct children around 7±2 when possible; if a parent reaches 10+ direct children, consider introducing intermediate hubs. Keep root/homepage navigation around 6–10 major hubs for scannability.
- **Create child pages for deep dives by default.** Example: `business-opportunities` → `business-opportunities/ai-consulting-workflow-automation` → `business-opportunities/ai-consulting-workflow-automation/manufacturing-semicap-workflow-automation`.
- **Split large sections.** If a page section exceeds ~300–500 words, includes multiple independent subtopics, duplicates another page, or would be painful to skim in 30 seconds, split it into child pages.
- **Preserve navigation.** Add links from parent to child, child back to parent, and related links between sibling/adjacent pages.
- **Do not promote child pages into the top-level wiki index as peers.** Deep-dive child pages should be linked from the relevant parent hub section/item unless they are major standalone hubs themselves.
- **Prefer small composable pages over long monoliths.** The wiki should become more navigable over time, not just longer.
- **Refactor without asking when the structure is obvious.** If uncertain, make a reasonable structure and note it can be adjusted later.
- **Keep raw/source facts distinct from synthesized conclusions.** Sources are evidence; wiki pages are curated synthesis.

## Core Operations

### 1. Ingest

When the user provides a source (URL, file, paste), integrate it into the wiki:

① **Capture the raw source:**
   - URL → use `web_extract` to get markdown, save to `raw/articles/`
   - PDF → use `web_extract` (handles PDFs), save to `raw/papers/`
   - Pasted text → save to appropriate `raw/` subdirectory
   - Name the file descriptively: `raw/articles/karpathy-llm-wiki-2026.md`
   - **Add raw frontmatter** (`source_url`, `ingested`, `sha256` of the body).
     On re-ingest of the same URL: recompute the sha256, compare to the stored value —
     skip if identical, flag drift and update if different. This is cheap enough to
     do on every re-ingest and catches silent source changes.

② **Discuss takeaways** with the user — what's interesting, what matters for
   the domain. (Skip this in automated/cron contexts — proceed directly.)

③ **Check what already exists** — search index.md and use `search_files` to find
   existing pages for mentioned entities/concepts. This is the difference between
   a growing wiki and a pile of duplicates.

④ **Write or update wiki pages:**
   - **New entities/concepts:** Create pages only if they meet the Page Thresholds
     in SCHEMA.md (2+ source mentions, or central to one source)
   - **Existing pages:** Add new information, update facts, bump `updated` date.
     When new info contradicts existing content, follow the Update Policy.
   - **Cross-reference:** Every new or updated page must link to at least 2 other
     pages via `[[wikilinks]]`. Check that existing pages link back.
   - **Tags:** Only use tags from the taxonomy in SCHEMA.md
   - **Provenance:** On pages synthesizing 3+ sources, append `^[raw/articles/source.md]`
     markers to paragraphs whose claims trace to a specific source.
   - **Confidence:** For opinion-heavy, fast-moving, or single-source claims, set
     `confidence: medium` or `low` in frontmatter. Don't mark `high` unless the
     claim is well-supported across multiple sources.

⑤ **Update navigation:**
   - First classify the page as **root-level hub**, **child/deep-dive**, or **leaf/source note**.
   - Add only root-level hubs to the root `index.md`.
   - For child/deep-dive pages, do **not** add them as peers in the root index; link them from the relevant parent hub section/item and add a breadcrumb/back-link from child to parent.
   - In static HTML hubs, make cards clickable drilldowns: wrap the whole card in an anchor (`<a class="card" href="child-page.html">…</a>` or equivalent), not just the title or CTA. Preserve visible focus states and meaningful accessible labels.
   - Link references to concrete tools and software products on first substantive mention, using official product/docs/repo URLs when available.
   - Update the "Total pages" count and "Last updated" date in index header when using markdown index files.
   - Append to `log.md`: `## [YYYY-MM-DD] ingest | Source Title`
   - List every file created or updated in the log entry

⑥ **Report what changed** — list every file created or updated to the user.

A single source can trigger updates across 5-15 wiki pages. This is normal
and desired — it's the compounding effect.

### 2. Query

When the user asks a question about the wiki's domain:

① **Read `index.md`** to identify relevant pages.
② **For wikis with 100+ pages**, also `search_files` across all `.md` files
   for key terms — the index alone may miss relevant content.
③ **Read the relevant pages** using `read_file`.
④ **Synthesize an answer** from the compiled knowledge. Cite the wiki pages
   you drew from: "Based on [[page-a]] and [[page-b]]..."
⑤ **File valuable answers back** — if the answer is a substantial comparison,
   deep dive, or novel synthesis, create a page in `queries/` or `comparisons/`.
   Don't file trivial lookups — only answers that would be painful to re-derive.
⑥ **Update log.md** with the query and whether it was filed.

### 3. Lint

When the user asks to lint, health-check, or audit the wiki:

① **Orphan pages:** Find pages with no inbound `[[wikilinks]]` from other pages.
```python
# Use execute_code for this — programmatic scan across all wiki pages
import os, re
from collections import defaultdict
wiki = "<WIKI_PATH>"
# Scan all .md files in entities/, concepts/, comparisons/, queries/
# Extract all [[wikilinks]] — build inbound link map
# Pages with zero inbound links are orphans
```

② **Broken wikilinks:** Find `[[links]]` that point to pages that don't exist.

③ **Index completeness:** Every wiki page should appear in `index.md`. Compare
   the filesystem against index entries.

④ **Frontmatter validation:** Every wiki page must have all required fields
   (title, created, updated, type, tags, sources). Tags must be in the taxonomy.

⑤ **Stale content:** Pages whose `updated` date is >90 days older than the most
   recent source that mentions the same entities.

⑥ **Contradictions:** Pages on the same topic with conflicting claims. Look for
   pages that share tags/entities but state different facts. Surface all pages
   with `contested: true` or `contradictions:` frontmatter for user review.

⑦ **Quality signals:** List pages with `confidence: low` and any page that cites
   only a single source but has no confidence field set — these are candidates
   for either finding corroboration or demoting to `confidence: medium`.

⑧ **Source drift:** For each file in `raw/` with a `sha256:` frontmatter, recompute
   the hash and flag mismatches. Mismatches indicate the raw file was edited
   (shouldn't happen — raw/ is immutable) or ingested from a URL that has since
   changed. Not a hard error, but worth reporting.

⑨ **Page size:** Flag pages over 200 lines — candidates for splitting.

⑩ **Tag audit:** List all tags in use, flag any not in the SCHEMA.md taxonomy.

⑪ **Log rotation:** If log.md exceeds 500 entries, rotate it.

⑫ **Report findings** with specific file paths and suggested actions, grouped by
   severity (broken links > orphans > source drift > contested pages > stale content > style issues).

⑬ **Append to log.md:** `## [YYYY-MM-DD] lint | N issues found`

## Working with the Wiki

### Searching

```bash
# Find pages by content
search_files "transformer" path="$WIKI" file_glob="*.md"

# Find pages by filename
search_files "*.md" target="files" path="$WIKI"

# Find pages by tag
search_files "tags:.*alignment" path="$WIKI" file_glob="*.md"

# Recent activity
read_file "$WIKI/log.md" offset=<last 20 lines>
```

### Bulk Ingest

When ingesting multiple sources at once, batch the updates:
1. Read all sources first
2. Identify all entities and concepts across all sources
3. Check existing pages for all of them (one search pass, not N)
4. Create/update pages in one pass (avoids redundant updates)
5. Update index.md once at the end
6. Write a single log entry covering the batch

### Archiving

When content is fully superseded or the domain scope changes:
1. Create `_archive/` directory if it doesn't exist
2. Move the page to `_archive/` with its original path (e.g., `_archive/entities/old-page.md`)
3. Remove from `index.md`
4. Update any pages that linked to it — replace wikilink with plain text + "(archived)"
5. Log the archive action

### Obsidian Integration

The wiki directory works as an Obsidian vault out of the box:
- `[[wikilinks]]` render as clickable links
- Graph View visualizes the knowledge network
- YAML frontmatter powers Dataview queries
- The `raw/assets/` folder holds images referenced via `![[image.png]]`

For best results:
- Set Obsidian's attachment folder to `raw/assets/`
- Enable "Wikilinks" in Obsidian settings (usually on by default)
- Install Dataview plugin for queries like `TABLE tags FROM "entities" WHERE contains(tags, "company")`

If using the Obsidian skill alongside this one, set `OBSIDIAN_VAULT_PATH` to the
same directory as the wiki path.

### Obsidian Headless (servers and headless machines)

On machines without a display, use `obsidian-headless` instead of the desktop app.
It syncs vaults via Obsidian Sync without a GUI — perfect for agents running on
servers that write to the wiki while Obsidian desktop reads it on another device.

**Setup:**
```bash
# Requires Node.js 22+
npm install -g obsidian-headless

# Login (requires Obsidian account with Sync subscription)
ob login --email <email> --password '<password>'

# Create a remote vault for the wiki
ob sync-create-remote --name "LLM Wiki"

# Connect the wiki directory to the vault
cd ~/wiki
ob sync-setup --vault "<vault-id>"

# Initial sync
ob sync

# Continuous sync (foreground — use systemd for background)
ob sync --continuous
```

**Continuous background sync via systemd:**
```ini
# ~/.config/systemd/user/obsidian-wiki-sync.service
[Unit]
Description=Obsidian LLM Wiki Sync
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/path/to/ob sync --continuous
WorkingDirectory=/home/user/wiki
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now obsidian-wiki-sync
# Enable linger so sync survives logout:
sudo loginctl enable-linger $USER
```

This lets the agent write to `~/wiki` on a server while you browse the same
vault in Obsidian on your laptop/phone — changes appear within seconds.

- **Wiki index vs nav:** The index at `index.md` is a hub page (sectioned link list). The nav bar is a separate top strip generated by `md2html.py`. They serve different purposes — index for browsing, nav for in-page navigation. Keep them consistent but don't conflate them.

### Adding new wiki pages

When you add a new page to `/opt/data/wiki/`:
1. Create the markdown file
2. Add it to `index.md` under the right section as `[[path/to/page]]`
3. **Add entry to `WIKI_PATH_MAP` in `md2html.py`** — this map drives both the wiki hub links AND the in-page nav. Missing entries cause garbled display names.
4. Run `python3 /opt/data/scripts/md2html.py`
5. Git commit and push

### Adding images to wiki pages

Use `[[assets/filename.jpg]]` to embed an inline image in any wiki page. This syntax renders
as `<img src="/wiki/assets/filename.jpg">` in the HTML, not as a link. The `md2html.py` script
handles this automatically.

**To add a photo:**
1. Create `wiki/assets/` subdirectory if it doesn't exist
2. Copy image file: `cp /path/to/image.jpg wiki/assets/`
3. Reference in markdown: `[[assets/image.jpg]]`
4. Commit image to the hermes-pages repo, push, let Cloudflare Pages pick it up

### Publishing flow (Gordons-specific)

The end-to-end pipeline uses two repos:
```
/opt/data/hermes-pages/           ← GitHub: rousegordon-ops/hermes-pages (source + images)
/opt/data/hermes-pages-repo/       ← working clone, md2html.py writes HTML output here
```

**Full publish sequence:**
1. Copy image files to `hermes-pages/wiki/assets/` (create dir if missing)
2. Edit markdown source in `hermes-pages/gordons-llm-wiki/`
3. Run `python3 hermes-pages/scripts/md2html.py hermes-pages/gordons-llm-wiki`
   (outputs to `/opt/data/hermes-pages-repo/wiki/`)
4. Copy generated HTML back: `cp -r hermes-pages-repo/wiki/* hermes-pages/wiki/`
5. `cd hermes-pages && git add -A && git commit -m "message" && git push`
6. Cloudflare Pages auto-deploys in ~30s → live at `https://hermes-pages.rouse-gordon.workers.dev/wiki/`

**`WIKI_PATH_MAP` is the single source of truth** for: URL path, display label, and nav order.
```python
WIKI_PATH_MAP = {
    'gordon-rouse':      ('entities/gordon-rouse',      'Gordon Rouse'),
    'hobbies/fitness':   ('hobbies/fitness',           'Fitness'),
    ...
}
```
If a `[[wikilink]]` in `index.md` doesn't have a map entry, `wiki_link()` falls back to auto-labeling: `name.split('/')[-1].replace('-',' ').title()` — which produces ugly results like "Ventura Renovation" from "projects/ventura-renovation" becomes "Projects / Ventura Renovation". Always add the map entry.

## Pitfalls

- **Static HTML is the source of truth for Gordon's wiki.** Do not use the older markdown-generation publishing flow unless Gordon explicitly asks for it. See `references/gordon-static-html-wiki-maintenance.md` for the current direct-edit/deploy workflow.
- **Cloudflare Pages project vs domain:** deploy with `--project-name hermes-pages`. `hermes-pages-d55` is the public Pages domain suffix; using it as the project name causes Wrangler `Project not found`.
- **Cloudflare verification:** if Python `urllib` gets a 403 while verifying a live page, retry with `curl -L -A 'Mozilla/5.0' -H 'Cookie: wiki_auth=GW2026' ...` before assuming deploy failed.
- **Never modify files in `raw/`** — sources are immutable. Corrections go in wiki pages.
- **Always orient first** — read SCHEMA + index + recent log before any operation in a new session.
  Skipping this causes duplicates and missed cross-references.
- **`WIKI_PATH_MAP` must be updated when adding pages** — missing entries cause garbled link labels
  in the index. The fallback auto-label produces names like "Projects / Ventura Renovation" instead of
  "Ventura Renovation". See [[wiki-link-display-name-bug]] for the original incident.
- **Always update index.md and log.md** — skipping this makes the wiki degrade. These are the
  navigational backbone.
- **Don't create pages for passing mentions** — follow the Page Thresholds in SCHEMA.md. A name
  appearing once in a footnote doesn't warrant an entity page.
- **Don't create pages without cross-references** — isolated pages are invisible. Every page must
  link to at least 2 other pages.
- **Don't list child deep dives as top-level peers** — if a page lives under a parent hub path (e.g. `/wiki/business-opportunities/acquire-local-service-business`), link it from the relevant parent hub item/section, not as a peer in the root wiki index unless it has become a major standalone hub.
- **Don't let scheduled jobs undo hierarchy refactors** — when changing page topology, update any cron/webhook prompts that write to those pages so future automated updates target the correct hub or child page and do not reintroduce duplicated content.
- **Preserve old URLs when moving child pages** — use lightweight static HTML redirects for old child-page URLs so existing links/bookmarks keep working, while canonical links point to the new semantic hierarchy.
- **Frontmatter is required** — it enables search, filtering, and staleness detection.
- **Tags must come from the taxonomy** — freeform tags decay into noise. Add new tags to SCHEMA.md
  first, then use them.
- **Keep pages scannable** — a wiki page should be readable in 30 seconds. Split pages over
  200 lines. Move detailed analysis to dedicated deep-dive pages.
- **Avoid low-contrast formatting** — Gordon finds dim gray metadata/text unreadable. Use high-contrast text for all substantive content, including meta/trend/source notes; do not hide important claims in muted styles.
- **Ask before mass-updating** — if an ingest would touch 10+ existing pages, confirm
  the scope with the user first.
- **Rotate the log** — when log.md exceeds 500 entries, rename it `log-YYYY.md` and start fresh.
  The agent should check log size during lint.
- **Handle contradictions explicitly** — don't silently overwrite. Note both claims with dates,
  mark in frontmatter, flag for user review.

## Related Tools

[llm-wiki-compiler](https://github.com/atomicmemory/llm-wiki-compiler) is a Node.js CLI that
compiles sources into a concept wiki with the same Karpathy inspiration. It's Obsidian-compatible,
so users who want a scheduled/CLI-driven compile pipeline can point it at the same vault this
skill maintains. Trade-offs: it owns page generation (replaces the agent's judgment on page
creation) and is tuned for small corpora. Use this skill when you want agent-in-the-loop curation;
use llmwiki when you want batch compile of a source directory.
