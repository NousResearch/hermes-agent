# Defuddle Integration Patterns for Hermes Agents

## Comparison: Defuddle vs existing research skills

| Skill | Layer | What it does | When to use |
|-------|-------|-------------|-------------|
| `walled-web-research` | Access | Bypasses Cloudflare, bot-walls, JS shells via Wayback, r.jina.ai, sitemaps | Page is blocked or returns JS shell to curl |
| `headless-web-research` | Access | curl + JSON APIs for data extraction when browsers unavailable | Catalogs, benchmarks, structured data, user sentiment |
| `web-content-extraction` | Access+Cleanup | browser_console + manual CSS selectors for SPA DOM extraction | JS-rendered SPAs where you need the rendered DOM |
| **`defuddle`** | **Cleanup** | **Purpose-built content extraction from HTML/URL → clean Markdown + metadata** | **You have HTML (from any source) and need clean article content** |

Key distinction: the first three skills solve *getting to the content*. Defuddle solves *cleaning the content once you have it*. They are complementary, not competing.

## Defuddle vs Mozilla Readability

Defuddle was explicitly designed as a Readability replacement:

| Feature | Defuddle | Readability |
|---------|----------|-------------|
| Clutter removal | More forgiving, removes fewer uncertain elements | Aggressive removal |
| Footnotes | Consistent standardized format | Inconsistent |
| Math | MathML standardization + LaTeX conversion | No math support |
| Code blocks | Standardized with language preserved | Basic |
| Mobile styles | Uses mobile CSS to detect unnecessary elements | No |
| Metadata | Rich (schema.org, meta tags, favicon, image, published) | Basic (title, byline, length) |
| Output format | HTML, Markdown, JSON | HTML only |
| Frontmatter | YAML frontmatter for Obsidian | No |
| CLI | Built-in | None (library only) |
| Callouts | GitHub/Obsidian/Bootstrap standardized | No |

## Full integration workflow

### Scenario 1: Simple article extraction

```
User: "Extract the content from https://example.com/article"

Agent:
1. Run: npx -y defuddle@latest parse "https://example.com/article" --markdown
2. Receive clean Markdown
3. Present to user or save to file
```

### Scenario 2: Walled site research

```
User: "Get the content from this Cloudflare-protected page"

Agent:
1. Load walled-web-research skill for access strategy
2. Fetch HTML: curl -sL "https://web.archive.org/web/2026/https://walled-site.com/article" > /tmp/raw.html
   (Use Wayback Machine — it returns full HTML. Do NOT use r.jina.ai here, it returns Markdown not HTML.)
3. Load defuddle skill for cleanup
4. Clean: npx -y defuddle@latest parse /tmp/raw.html --markdown
5. Present clean content
```

### Scenario 3: Research report with multiple sources

```
User: "Research X from these 3 articles"

Agent:
1. For each URL:
   a. curl -sL <url> | npx -y defuddle@latest parse --markdown --output /tmp/article_N.md
   b. Also extract metadata: npx -y defuddle@latest parse <url> --json --output /tmp/meta_N.json
2. Read all /tmp/article_N.md files
3. Synthesize research report from clean content
```

### Scenario 4: Obsidian note creation

```
User: "Save this article to my Obsidian vault"

Agent:
1. npx -y defuddle@latest parse "https://example.com/article" --markdown --frontmatter --output /tmp/note.md
2. Load obsidian skill
3. Save /tmp/note.md to the vault with proper frontmatter already included
```

### Scenario 5: SPA content extraction (browser + Defuddle)

```
User: "Extract content from this React site"

Agent:
1. browser_navigate to the URL
2. browser_console with expression: document.documentElement.outerHTML
3. Save HTML to /tmp/rendered.html
4. npx -y defuddle@latest parse /tmp/rendered.html --markdown
```

## Programmatic usage (Node.js library)

For agents that need to integrate Defuddle into a Node.js script rather than the CLI:

```javascript
import { parseHTML } from 'linkedom';
import { Defuddle } from 'defuddle/node';

const { document } = parseHTML(html);
const result = await Defuddle(document, 'https://example.com/article', {
  markdown: true,
  // Options:
  // debug: false,
  // url: 'https://example.com/article',
  // markdown: true,
  // separateMarkdown: false,
  // removeExactSelectors: true,
  // removePartialSelectors: true,
  // removeHiddenElements: true,
  // removeLowScoring: true,
  // removeSmallImages: true,
  // removeImages: false,
  // standardize: true,
  // contentSelector: 'css-selector',  // bypass auto-detection
  // useAsync: true,   // allow third-party API fallback for SPAs
  // language: 'en',
  // includeReplies: 'extractors',
});

console.log(result.content);    // cleaned content
console.log(result.title);      // article title
console.log(result.author);     // author
console.log(result.wordCount);  // word count
```

Requires a DOM implementation:
```bash
npm install linkedom   # lightweight, recommended
# or
npm install jsdom       # heavier, more complete
```

## Fleet deployment recommendations

### Which profiles should have this skill

The `research` category exists across 14 profiles. Defuddle is most valuable for:

| Profile | Priority | Reason |
|---------|----------|--------|
| `u365-orchestrator` | **Done** (installed here) | Central coordination, research synthesis |
| `u365-research` | High | Primary research profile |
| `u365-intelligence` | High | Already has `terminal-web-research` skill |
| `u365-strategy` | Medium | Strategic research needs clean content |
| `u365-academics` | Medium | Academic content extraction |
| `coding` | Medium | Already has `web-content-extraction` peer |
| `u365-engagement` | Low | Occasional content needs |
| Others | Low | Can delegate to orchestrator or research |

### Deployment options

**Option A: Per-profile install (current)**
- Skill lives at `~/.hermes/profiles/<profile>/skills/research/defuddle/SKILL.md`
- Each profile that needs it gets its own copy
- Pro: simple, isolated
- Con: duplication, maintenance across profiles

**Option B: Workspace-external (recommended for fleet-wide)**
- Skill lives at `/workspace/u365-skills/defuddle/SKILL.md`
- All profiles load from `skills.external_dirs` in config.yaml
- Pro: single source of truth, one update propagates everywhere
- Con: requires config change on each profile

**Option C: In-repo (if contributing upstream)**
- Skill lives at `skills/research/defuddle/SKILL.md` in the hermes-agent repo
- Ships with every clone
- Pro: maximum distribution
- Con: requires upstream PR and merge

### Recommendation

Start with Option A (current — installed in u365-orchestrator). When the skill is validated through real use, promote to Option B (workspace-external) for fleet-wide access. Option C is a future possibility if the Hermes Agent project accepts community skills.

## Version history and maturity

| Version | Date | Notes |
|---------|------|-------|
| 0.19.2 | 2026-07-22 | Latest (3 days ago) |
| 0.19.0 | 2026-06-16 | Recent feature release |
| 0.18.0 | 2026-04-21 | |
| 0.17.0 | 2026-04-15 | |
| 0.16.0 | 2026-04-09 | |
| 0.15.0 | 2026-03-31 | |
| 0.14.0 | 2026-03-17 | |
| 0.9.0 | early 2025 | Fixed XSS advisory GHSA-5mq8-78gm-pjmq |
| 0.1.0 | 2025-02-27 | Initial release |

44 versions in 5 months = active development, weekly release cadence.

## Security assessment

- **GHSA-5mq8-78gm-pjmq**: XSS via unescaped string interpolation in `_findContentBySchemaText` image tag. Affected versions < 0.9.0. Fixed in 0.9.0. Current version (0.19.2) is not affected.
- **Dependency surface**: Single runtime dependency (`commander`), a well-established CLI framework. No transitive dependency chain risk.
- **No network calls beyond the target URL**: Defuddle fetches only the URL you specify (or reads from stdin/file). The `useAsync` option can make third-party API calls for SPA fallback, but this is disabled by default in the CLI.
- **No credential handling**: Defuddle does not process or store any credentials, tokens, or secrets.