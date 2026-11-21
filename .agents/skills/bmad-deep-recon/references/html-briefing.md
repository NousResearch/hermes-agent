# HTML Briefing

Generate `research-briefing.html` in `{doc_workspace}` after `research.md` is final, when `{workflow.output_format}` calls for it: `"auto"` renders on interactive runs and skips on headless/skill-invoked runs (the md is always there to render from later), `"html"`/`"both"` always render, `"md"` never. The page is a full-fidelity presentation of the report, never a second source of truth — same claims, same numbers, same citations; nothing is lost by reading it instead of the markdown.

## Requirements

- **Self-contained single file**: inline CSS and JS, no external requests of any kind (no CDN, no fonts, no remote images). It must render from a `file://` open, offline, forever.
- **Structure**: a header (topic, type, decision, date, depth, verification level) → the executive summary as the opening card → sticky table of contents → dimension sections → contrary evidence (when present) → recommendations → collapsible source appendix → staleness map.
- **Confidence is visual**: every claim carries its badge — verified / medium / low / `unverified` / disputed — color-coded with the status text always present (never color alone). Unverified and disputed must be *more* prominent than verified, not less.
- **Sources are live**: inline `[n]` markers link to the appendix row; appendix rows link out to the source URL. Source URLs are untrusted content — never hand-escape them: generate the appendix table with `uv run scripts/recon_kit.py escape-sources {doc_workspace}/research.md` and embed its `html` output, which escapes every cell, anchors each row (`id="src-n"`), and links only validated `http(s)` URLs (anything else renders as plain text; the script lists it in `invalid_urls`). Apply the same escape discipline to any other source-derived text you place in attributes.
- **Charts sparingly**: only where the data genuinely benefits (market size trajectory, decision matrix scores) — simple inline SVG, labeled axes, no library.
- **Responsive and theme-aware**: readable on a phone; respect `prefers-color-scheme` for light/dark.

## Theme

`{workflow.html_theme}` governs: a `file:` path loads a theme/brand spec to follow; inline text is applied as directives; empty means the shipped default — neutral, professional, generous whitespace, system font stack, one restrained accent color. Whatever the theme, the confidence-badge semantics above are non-negotiable.
