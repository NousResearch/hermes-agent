# Defuddle Quick-Reference Templates

Copy-paste command templates for common agent workflows.

## Basic extraction

```bash
# URL → Markdown
npx -y defuddle@latest parse "URL" --markdown

# URL → JSON (full metadata)
npx -y defuddle@latest parse "URL" --json

# URL → Markdown with YAML frontmatter
npx -y defuddle@latest parse "URL" --markdown --frontmatter

# URL → single property (title, author, description, domain)
npx -y defuddle@latest parse "URL" --property title
```

## Pipe from curl (agent workflow)

```bash
# Fetch + clean in one pipe
curl -sL "URL" | npx -y defuddle@latest parse --markdown

# With custom User-Agent (for 403 sites)
curl -sL -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15" "URL" | npx -y defuddle@latest parse --markdown

# Save to file
curl -sL "URL" | npx -y defuddle@latest parse --markdown --output /tmp/article.md
```

## Pipe from access tools

```bash
# Wayback Machine → Defuddle (returns HTML)
curl -sL "https://web.archive.org/web/2026/URL" | npx -y defuddle@latest parse --markdown
```

Note: r.jina.ai returns Markdown, not HTML. Do not pipe r.jina.ai output into Defuddle — it will return empty. When you use r.jina.ai, you already have clean content.

## Save to file

```bash
# Markdown to file
npx -y defuddle@latest parse "URL" --markdown --output /tmp/article.md

# JSON to file
npx -y defuddle@latest parse "URL" --json --output /tmp/metadata.json
```

## Language preference

```bash
# French
npx -y defuddle@latest parse "URL" --markdown --lang fr

# Japanese
npx -y defuddle@latest parse "URL" --markdown --lang ja
```

## Debug mode

```bash
# Show debug info (content selector, removals)
npx -y defuddle@latest parse "URL" --markdown --debug
```

## Local HTML file

```bash
# Parse saved HTML
npx -y defuddle@latest parse /tmp/page.html --markdown

# Parse saved HTML with frontmatter
npx -y defuddle@latest parse /tmp/page.html --markdown --frontmatter
```

## Pinned version (reproducibility)

```bash
# Pin to specific version
npx -y defuddle@0.19.2 parse "URL" --markdown
```