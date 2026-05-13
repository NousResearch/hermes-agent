---
name: kami
description: 'Typeset professional documents: resumes, one-pagers, white papers, letters, portfolios, slide decks. Warm parchment, ink-blue accent, serif-led hierarchy. CN uses TsangerJinKai02, EN uses Charter, JA uses YuMincho (best-effort). Triggers on "做 PDF / 排版 / 一页纸 / 白皮书 / 作品集 / 简历 / PPT / slides", or "build me a resume / make a one-pager / design a slide deck / turn this into a PDF / make this presentable".'
---

# kami · 紙

**紙 · かみ** - the paper your deliverables land on.

Good content deserves good paper. One design language across eight document types: warm parchment canvas, ink-blue accent, serif-led hierarchy, tight editorial rhythm.

Part of `Kaku · Waza · Kami` - Kaku writes code, Waza drills habits, **Kami delivers documents**.

## Step 0 · Load brand profile (if exists)

Check `~/.config/kami/brand.md` (preferred) or `~/.kami/brand.md` (legacy fallback). If found, read `references/brand-profile.md` for the full four-layer application spec (placeholder substitution, session defaults, visual customization, habit notes) and its six guardrails. If no profile exists, continue without interruption.

Key rule: explicit prompt > editorial judgment > habit notes > frontmatter defaults > built-in defaults. Profile fills gaps silently; it never overrides the current conversation.

---

## Step 1 · Decide the language

**Match the user's language.** Chinese -> `*.html` / `slides.py`. English -> `*-en.html` / `slides-en.py`. Japanese -> CJK path (`.html` / `slides.py`) as best-effort, JP Mincho first, visual QA before shipping. Reference docs are shared English specs.

When ambiguous (e.g. a one-word command like "resume"), ask a one-liner rather than guess.

| User language | HTML templates | Slides template |
|---|---|---|
| Chinese (primary) | `*.html` | `slides.py` |
| English | `*-en.html` | `slides-en.py` |
| Japanese (best-effort) | `*.html` | `slides.py` |
| Other languages (best-effort) | choose CJK or EN path by script coverage, then verify manually | choose `slides.py` or `slides-en.py`, then verify manually |

Always use `CHEATSHEET.md` and `references/*.md` for design, writing, production, and diagram guidance.

## Step 1.5 · Intent extraction (silent checklist)

Before choosing a template, verify these four dimensions are clear. Do not ask unless 2+ are missing and cannot be inferred from context.

| Dimension | What to extract | Example |
|---|---|---|
| **Purpose** | Why this document exists | Persuade investor vs. align internal team vs. close a candidate |
| **Audience** | Who reads it, what they already know | Technical CTO (skip basics) vs. non-technical board (explain terms) |
| **Constraint** | Hard limits on length, format, tone, or delivery | "One page max", "formal English", "print-ready A4" |
| **Success** | What outcome counts as success | They schedule a meeting / they approve the budget / they understand the architecture |

Rules:
- If the conversation already answered a dimension, skip it silently.
- If a dimension can be inferred from the document type (e.g. resume purpose is always "get an interview"), skip it.
- If 2+ dimensions are genuinely unclear, ask in a single compact question (max 2 sub-questions).
- Never ask all four as a checklist. This is a background verification, not a form.

---

## Step 2 · Pick the document type

| User says | Document | CN template | EN template |
|---|---|---|---|
| "one-pager / 方案 / 执行摘要 / exec summary" | One-Pager | `one-pager.html` | `one-pager-en.html` |
| "white paper / 白皮书 / 长文 / 年度总结 / technical report" | Long Doc | `long-doc.html` | `long-doc-en.html` |
| "formal letter / 信件 / 辞职信 / 推荐信 / memo" | Letter | `letter.html` | `letter-en.html` |
| "portfolio / 作品集 / case studies" | Portfolio | `portfolio.html` | `portfolio-en.html` |
| "resume / resume / CV / 简历" | Resume | `resume.html` | `resume-en.html` |
| "slides / PPT / deck / 演示" | Slides | `slides-weasy.html` | `slides-weasy-en.html` |
| "个股研报 / equity report / 估值分析 / investment memo / 股票分析" | Equity Report | `equity-report.html` | `equity-report-en.html` |
| "更新日志 / changelog / release notes / 版本记录" | Changelog | `changelog.html` | `changelog-en.html` |
| "facture / invoice / facturation" | Invoice | `invoice-fr.html` | — |

> **Changelog vs. release notes**: The changelog template above is for styled document output. GitHub release notes are a separate deliverable; use `/write` with Release Note Template Mode.

> **Invoice**: French invoice template (`invoice-fr.html`) includes SIREN, TVA intracommunautaire, RIB, and legal mentions per French accounting requirements. Use `dingbat-invoice` skill for the full workflow (data filling, placeholder cleanup, Facture.net sync).

> Slides: default to `slides-weasy.html` / `slides-weasy-en.html` (WeasyPrint HTML → PDF). Use `slides.py` / `slides-en.py` only when the user explicitly requires an editable PPTX file.

> Deck recipe: read design.md Section 8 before drafting slides.

If unsure, ask a one-liner about the scenario rather than guess.

### Diagrams (primitives, not a 7th doc type)

When the user asks for **a diagram inside** a long-doc / portfolio / slide (not a standalone document), route to `assets/diagrams/` rather than a template:

| User says | Diagram | Template |
|---|---|---|
| "架构图 / architecture / 系统图 / components diagram" | Architecture | `assets/diagrams/architecture.html` |
| "流程图 / flowchart / 决策流 / branching logic" | Flowchart | `assets/diagrams/flowchart.html` |
| "象限图 / quadrant / 优先级矩阵 / 2×2 matrix" | Quadrant | `assets/diagrams/quadrant.html` |
| "柱状图 / bar chart / 分类对比 / grouped bars" | Bar Chart | `assets/diagrams/bar-chart.html` |
| "折线图 / line chart / 趋势 / 股价 / time series" | Line Chart | `assets/diagrams/line-chart.html` |
| "环形图 / donut / pie / 占比 / 分布结构" | Donut Chart | `assets/diagrams/donut-chart.html` |
| "状态机 / state machine / 状态图 / lifecycle" | State Machine | `assets/diagrams/state-machine.html` |
| "时间线 / timeline / 里程碑 / milestones / roadmap" | Timeline | `assets/diagrams/timeline.html` |
| "泳道图 / swimlane / 跨角色流程 / cross-team flow" | Swimlane | `assets/diagrams/swimlane.html` |
| "树状图 / tree / hierarchy / 层级 / 组织架构" | Tree | `assets/diagrams/tree.html` |
| "分层图 / layer stack / 分层架构 / OSI / stack" | Layer Stack | `assets/diagrams/layer-stack.html` |
| "维恩图 / venn / 交集 / overlap / 集合关系" | Venn | `assets/diagrams/venn.html` |
| "K 线 / candlestick / OHLC / 股价走势 / price history" | Candlestick | `assets/diagrams/candlestick.html` |
| "瀑布图 / waterfall / 收入桥 / revenue bridge / decomposition" | Waterfall | `assets/diagrams/waterfall.html` |

Read `references/diagrams.md` before drawing - it has the selection guide, kami token map, and the AI-slop anti-pattern table. Extract the `<svg>` block from the template and drop it into a `<figure>` inside long-doc / portfolio.

Before drawing, always ask: **would a well-written paragraph teach the reader less than this diagram?** If no, don't draw.

**Auto-select charts from data.** When content contains numerical data, choose the chart type and embed it without waiting for the user to specify. Decision tree (first match wins):

| Data shape | Chart |
|---|---|
| Has open/high/low/close fields, or per-day price | Candlestick |
| Has + and - contributions that sum to a total (bridge, waterfall, P&L) | Waterfall |
| One series, values sum to ~100%, items ≤ 6 | Donut |
| One series, values sum to ~100%, items ≥ 7 | Horizontal bar |
| Two or more series across time (months, quarters, years) | Line |
| One series across time, large count changes dominate (not rate) | Bar |
| Multiple categories, same time snapshot, 2+ series | Grouped bar |
| 2×2 strategic or priority positioning | Quadrant |
| Hierarchical data with depth ≥ 2 | Tree |
| Process with decision branches | Flowchart |
| Cross-team or cross-role process with ≥ 3 actors | Swimlane |
| Set overlaps or shared attributes between 2-3 groups | Venn |
| Category comparison, single series, no time axis | Bar |

When data fits multiple types, prefer the one that shows variance most clearly. Always embed inside a `<figure>` with a caption that states the insight, not just the data range.

## Step 2.1 · Source and material pass

Run this before distilling or filling content when the document depends on facts or materials outside the user's draft. Skip it only for personal drafts where the user already supplied everything needed.

### Source check

Trigger when the document mentions a specific company, product, person, release date, version, funding round, metric, market fact, technical spec, or any current fact likely to change.

- Use primary sources before writing: user-provided material, official site, docs, filings, press release, app store page, or repo release
- Keep a short note of source names and dates for facts that drive the document
- If sources conflict or a fact cannot be checked quickly, ask the user instead of choosing silently
- Avoid current-sounding claims such as "latest", "recent", "new", version numbers, launch dates, or financial figures unless they are checked

### Material check

Trigger when the document is about a company, product, project, venue, or personal brand.

Confirm the materials that make the subject recognizable before layout:

| Need | Required when | Accept |
|---|---|---|
| Logo | Any branded document | User file or official SVG/PNG |
| Product image | Physical product / venue / object | Official image, user image, or marked gap |
| UI screenshot | App / SaaS / website / tool | Current screenshot, official product image, or user capture |
| Brand colors | Branded one-pager / portfolio / deck | Official value, extracted asset value, or keep kami ink-blue |
| Fonts | Only if brand typography matters | Official font, close system fallback, or kami default |

If a required item is missing, use a compact gap table and ask once. Do not replace missing material with generic imagery, approximate logo drawings, or invented values.

### Materials status block

After the material check, output a structured status block before continuing. This is a one-shot transparency display, not a question:

```
Materials status:
- Logo: OK assets/client-logo.svg
- Brand colors: OK #1B365D mapped to --brand
- Product screenshot: MISSING (proceeding with kami default placeholder)
- UI screenshot: not required for this doc type
```

Use `OK`, `MISSING`, or `not required`. If a required item is missing and no user input arrived, ask once with the gap table; otherwise continue silently.

## Step 2.5 · Distill raw content (if applicable)

**Auto-detect whether to distill.** Do not ask the user; judge from the input:

| Skip distill (fill directly) | Run distill |
|---|---|
| Content has explicit section labels matching template structure | Raw prose without section structure |
| Metrics already quantified with units in place | Numbers scattered or implied, not extracted |
| User wrote "use this as-is" / "直接用这个" / "原封不动" | User pasted multi-source dump (chat / email thread / multiple docs) |
| Content count matches template (e.g. 4 metrics for 4 metric cards) | Content count mismatches template (too many or too few items) |
| One coherent voice with consistent claims | Conflicting claims or duplicate facts across sources |

When in doubt, run distill. Distill is cheap; rebuilding a misaligned doc is not.

When the user hands over **raw material** (meeting notes, brain dump, existing doc in different format, chat transcript, scattered points):

1. **Extract**: pull out every factual claim, number, date, name, source, material reference, and action item
2. **Classify**: map each extract to the target template's sections (see `references/writing.md` for section structure per doc type)
3. **Gap-check**: list what the template needs but the raw content doesn't have - include missing facts, missing proof, and missing materials
4. **Ask once**: share the gap table with the user. Do not guess to fill gaps.

Example gap-check:

| Template needs | Found | Missing |
|---|---|---|
| 4 metric cards | "8 years", "50-person team" | 2 more quantifiable results |
| 3-5 core projects | 2 mentioned | at least 1 more with outcome |
| Materials | logo file provided | product screenshot source |

Then proceed to Step 2.6 (slides) or the layout note (all other doc types) with structured, distilled content.

## Step 2.6 · Deck pre-flight (slides only)

Skip this step for every doc type except slides.

### Path selection

Default to the WeasyPrint HTML path. Switch to pptx only if the user explicitly requires an editable PPTX file.

| Path | Template | When |
|---|---|---|
| WeasyPrint HTML → PDF (default) | `slides-weasy.html` / `slides-weasy-en.html` | All cases unless PPTX is required |
| python-pptx → PPTX (fallback) | `slides.py` / `slides-en.py` | User explicitly requires editable PPTX |

### Page size

Default is `280mm 158mm`. Ask only if the user has mentioned length or density constraints.

| Size | When |
|---|---|
| `280mm 158mm` | Default; fits most decks |
| `297mm 167mm` | User wants a bit more room |
| `338mm 190mm` | Heavy content slide or many data points per page |

### Content pre-flight

Before drafting any slide, confirm these points with the user. Ask all at once, skip any already answered:

| # | Question |
|---|---|
| 1 | **Audience + venue** - who is in the room, and is it live keynote, investor 1:1, or async share link? |
| 2 | **Length target** - presentation time or slide count? (15 min: ~10 slides / 30 min: ~20 slides / 45 min: ~25-30 slides) |
| 3 | **Source material** - what content is already ready: outline, doc, notes, data? |
| 4 | **Images** - are screenshots, charts, logos, or product images available, or are gaps expected? |
| 5 | **Hard constraints** - brand colors, required logo, PPTX required, any slides that must exist? |
| 6 | **Format confirmation** - slides deck, or a one-pager that looks like a deck? |

### Content rules for slides

- No section divider slides: use `.eyebrow` for section numbering, not a dedicated blue-background page
- No CJK parentheses: replace `（...）` with `·` or `,`
- Each bullet fits one line: trim until it does
- 2×2 layouts: use `table.t2x2`, not CSS Grid
- Pinned conclusions: use `.co` at `position: absolute; bottom: 12mm`

## Step 2.7 · Layout note (transparent, non-blocking)

Before loading specs and filling the template, write a short editor-style note stating the layout intent: template choice, length target, narrative arc, embedded diagrams, material status, and output formats. Match the document's language. Keep it under 80 words, written as prose, not a status panel. Continue immediately after; do not wait.

Example (CN):

> 排版意图：Equity Report 中文版，2 页 A4。先立论与目标价，进入估值 (DCF 与可比公司)，落于催化剂与风险。中段嵌一张营收趋势折线和 FY26 收入桥瀑布。Logo 已就位，产品图暂缺，header 改走纯文字。输出 HTML 与 PDF。

Example (EN):

> Layout intent: Equity Report (EN), two pages A4. Open with thesis and price target, run through valuation (DCF and comparables), close on catalysts and risks. A revenue line chart and an FY26 waterfall sit mid-doc. Logo is in hand; product image is absent, so the header stays text-only. Output: HTML and PDF.

The note is for transparency, not approval. If the user pushes back, adjust; otherwise proceed to Step 3.

---

## Step 3 · Load the right amount of spec

Pick the tier that matches the task. Default to the lowest tier that covers the work.

| Tier | When | Read |
|---|---|---|
| **Content-only** | Updating text, swapping bullets, translating an existing doc. CSS stays untouched. | `CHEATSHEET.md` only |
| **Layout tweak** | Adjusting spacing, moving sections, changing font size within spec. CSS touched. | `CHEATSHEET.md` + template (tokens already inline) |
| **New document** | Building from scratch or from raw content. | Full design spec + writing spec + template |
| **Sources / materials** | Company, product, market, launch, funding, specs, or branded subject. | `writing.md` source rules + user/source material |
| **Deck (>20 slides)** | Long presentation needing Part Divider, Code Cards, section headers. | Full design spec + Deck Recipe (design.md section 8) |
| **Troubleshoot** | Rendering bug, font issue, page overflow. | `production.md` (+ design spec if CSS is the cause) |
| **Diagram** | Embedding SVG in a doc. | `diagrams.md` only (has its own token map) |

You can always escalate mid-task if the work turns out to need more than the initial tier.

The full spec files for reference:
- Design: `references/design.md`
- Writing: `references/writing.md`
- Production: `references/production.md`
- Diagrams: `references/diagrams.md`

## Step 4 · Fill content into the template

- Copy the template into your working directory; don't write HTML from scratch
- **CSS stays untouched**, only edit the body
- Content follows `writing.md`: data over adjectives, distinctive phrasing over industry clichés
- Avoid patterns listed in `references/anti-patterns.md`: emptiness, fabrication, mimicry, excess, source gaps, tone contamination
- **Before filling, read the quality bar for your document type** in `writing.md` section "Quality bars by document type". Structure is necessary but not sufficient: a resume bullet needs Action + Scope + Result + Business Outcome; an equity report needs variant perception + quantified catalysts; slides need assertion-evidence titles. Meeting the quality bar is as important as filling every placeholder.

### Do not generate

These are the most common AI document failures. Cross-reference `references/anti-patterns.md` for the full list.

- Do not leave placeholder text in the final document ("Lorem ipsum", "[Insert here]", "TBD")
- Do not invent metrics, financial data, or statistics; mark gaps with `[DATA NEEDED: description]`
- Do not use stock-image descriptions as image placeholders ("A diverse team collaborating in a modern office")
- Do not pad content to fill template slots (a resume with 3 real projects does not need 5 fabricated ones)
- Do not write a paragraph that merely restates its own heading in sentence form

### Fill PDF metadata (WeasyPrint reads these into the PDF)

Every template has meta placeholders in `<head>`. Fill all four before building:

| Placeholder (CN) | Placeholder (EN) | Rule |
|---|---|---|
| `{{作者}}` | `{{AUTHOR}}` | Resume/letter/portfolio: use the person's name from the doc. All others: leave as-is (build script infers from git config or env) |
| `{{摘要}}` | `{{DESCRIPTION}}` | Extract one sentence (≤150 chars) from the first 2 paragraphs |
| `{{关键词}}` | `{{KEYWORDS}}` | 3-5 keywords from the title + section headings, comma-separated |
| `{{文档标题}}` / `{{信件主题}}` etc. | `{{DOC_TITLE}}` / `{{LETTER_SUBJECT}}` etc. | Infer from the H1 or `.header .title` text |

`<meta name="generator" content="Kami">` is already fixed in the template; do not change it.

**Author inference**: `build.py` automatically sets PDF `/Author` metadata from:
1. `git config user.name` (primary)
2. `KAMI_AUTHOR` environment variable (fallback)
3. `"Kami"` (final fallback)

For personal documents (resume/letter/portfolio), the HTML `<meta name="author">` should match the person's name in the content. For non-personal documents (one-pager/long-doc), leave the placeholder as-is and let the build script infer it.

## Step 4.5 · Auto-select output format

Do not ask the user which format to export. Decide from context:

| Signal | Output | Why |
|---|---|---|
| Any document request | HTML + PDF | PDF is the default deliverable, HTML is the source |
| Slides / PPT / deck | HTML + PDF + PPTX | Presentations need a projectable format |
| "分享" / "发朋友圈" / "share" / "post" / "preview" | + PNG | Social platforms and messaging need images |
| "嵌入" / "插图" / "embed in another doc" | PNG only | Used as material inside other documents |
| User explicitly says a format | Follow the user | Explicit request overrides auto-selection |

PDF always ships. PPTX follows slides. PNG follows sharing context. The user should never need to think about formats.

## Step 5 · Build & verify

```bash
python3 scripts/build.py --verify           # build all templates + page count + font check + slides
python3 scripts/build.py --verify resume-en # single target full verification
python3 scripts/build.py --verify slides    # single slide deck verification
python3 scripts/build.py --check-placeholders path/to/filled.html
python3 scripts/build.py --check-density              # page whitespace scanner (skips cover)
python3 scripts/build.py --check            # CSS rule violations only (fast, no build)
```

**macOS Apple Silicon pitfall**: WeasyPrint requires system libraries (glib, pango, cairo, harfbuzz) that aren't found by default. Use the convenience wrapper that sets `DYLD_FALLBACK_LIBRARY_PATH`:

```bash
bash scripts/kami-build.sh --verify resume-en   # preferred on macOS
```

Or set manually:
```bash
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:/opt/homebrew/opt/glib/lib:/opt/homebrew/opt/pango/lib:/opt/homebrew/opt/harfbuzz/lib:/opt/homebrew/opt/cairo/lib:/opt/homebrew/opt/gdk-pixbuf/lib"
```

Without this, WeasyPrint crashes with `OSError: cannot load library 'pango-1.0'` or similar missing library errors.

Source templates intentionally keep `{{...}}` fields. Run placeholder checks on completed documents, not on the template library.

Visual anomalies (tag double rectangle, font fallback, page break issues) -> `production.md` Part 4.

## Fonts

**Chinese**
- Main serif: TsangerJinKai02-W04.ttf (400 weight) + TsangerJinKai02-W05.ttf (500 weight, real bold)
- Templates use dual @font-face declarations: W04 for body text, W05 for headings
- Both files are commercial fonts. Keep them available in the repository for local preview and CDN fallback, but do not bundle them inside Claude Desktop skill ZIPs
- Fallback chain baked into templates: Source Han Serif SC -> Noto Serif CJK SC -> Songti SC -> STSong -> Georgia

**Japanese (best-effort)**
- Uses CJK template path, no dedicated `-ja` templates yet
- JP Mincho-first stack: YuMincho -> Hiragino Mincho ProN -> Noto Serif CJK JP -> Source Han Serif JP -> TsangerJinKai02 -> serif
- Visually verify line breaks, punctuation rhythm, and emphasis weight before shipping

**English**
- Single serif: Charter (system-bundled, macOS/iOS), used for both headlines and body
- No separate sans: `--sans: var(--serif)`, one font per page
- Fallback: Georgia (cross-platform) / Palatino / Times New Roman

Font files next to HTML with relative `@font-face` paths is the most stable setup. `scripts/package-skill.sh` excludes TsangerJinKai TTFs from the Claude Desktop ZIP.

**Font auto-recovery (Claude Desktop)**

Before building Chinese documents, ensure fonts are present. The script tries multiple CDN sources with retry and size validation:

```bash
bash scripts/ensure-fonts.sh
```

Run once before building. If all sources fail, the script suggests installing Source Han Serif SC as fallback.

## Feedback protocol

When the user gives **vague visual feedback** ("looks off", "太挤了", "not elegant"), do not guess. Ask back with current values:

| User says | Ask about |
|---|---|
| "太挤了" / "too cramped" | Which element? Line-height (current: X)? Padding (current: Y)? Page margin? |
| "太松了" / "too loose" | Same direction, reversed |
| "颜色不对" / "color feels wrong" | Which element? Brand blue overused? A gray reading too cool? |
| "不够好看" / "not polished" | Font rendering? Alignment? Whitespace distribution? Hierarchy unclear? |
| "看着不专业" / "unprofessional" | Content wording? Or layout (alignment, consistency)? |

Template response: "X is currently set to Y. Would you like (a) [specific alternative within spec] or (b) [another option]?"

Never say "I'll adjust the spacing" without naming the exact property and its new value.

---

## When not to use this skill

- User explicitly wants Material / Fluent / Tailwind default - different design language
- Need dark / cyberpunk / futurist aesthetic (this is deliberately anti-future)
- Need saturated multi-color (this has one accent)
- Need cartoon / animation / illustration style (this is editorial)
- Web dynamic app UI (this is for print / static documents)

---

Next: **apply Step 3's tier table to decide what to read**, then copy the matching template and start filling.
