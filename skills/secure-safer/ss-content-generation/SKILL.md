---
name: ss-content-generation
description: "Content generation pipeline for Secure Safer Insurance — takes compact research briefs and produces effective, concise, conversion-focused content with SEO keywords, source links, and word-temperature labels."
version: 2.0.0
author: Hermes Agent via Rafiul
license: MIT
metadata:
  hermes:
    tags: [secure-safer, content, generation, copywriting, seo, research-to-content]
    related_skills: [ss-research-engine, copywriting, copy-editing, ai-seo, content-strategy]
---

# Secure Safer Content Generation Pipeline

## Overview

This skill bridges research → content. It defines:

1. **The Compact Research Format** — how the research engine condenses findings into a dense, actionable brief
2. **The Content Generation Process** — how the content writer uses that brief to produce effective copy
3. **Word Temperature Labels** — a system for marking the emotional weight of words and sentences
4. **Brand Identity** — Vibrant Professionalism applied to every content piece

## Part 1: Compact Research Format

Every research brief from `ss-research-engine` uses this format:

```
TOPIC: [One-line topic]
AUDIENCE: [Who this affects]
CONCERN: [The actual fear or pain point in plain English — 1-2 sentences]
SEARCH INTENT: [informational / commercial / transactional / navigational]
SOURCE: [URL of the key source]
ADDITIONAL SOURCES: [URL2, URL3]

--- KEYWORDS ---
PRIMARY: [main keyword]
SECONDARY: [kw2, kw3, kw4]
LONG-TAIL: ["question phrase", "question phrase"]
AEO TARGET: [Featured snippet question to answer]

--- WORD TEMPERATURE ---
HOT WORDS (urgent/scary): word1, word2, word3
WARM WORDS (concerned): word1, word2, word3
NEUTRAL WORDS (informational): word1, word2, word3
COLD WORDS (reassuring): word1, word2, word3

--- COMPETITOR ANGLE ---
[1-2 sentences on how competitors cover this or don't]
CONTENT GAP: [What Secure Safer can say that competitors miss]

--- SELLING POINTS ---
• [One benefit per bullet]
• [Second benefit]
• [Third benefit]

--- RISKS OF NOT ACTING ---
[One sentence — worst case]
```

## Part 2: Brand Identity — Vibrant Professionalism

This is the brand identity for Secure Safer. Every piece of content MUST follow this.

### Brand Palette

| Element | Value | Usage |
|---------|-------|-------|
| Primary Orange | #fc820c | CTAs, highlights, energy — the singular driver of action |
| Surface | #ffffff | Clean white backgrounds — maximal breathing room |
| Surface Dim | #d3daef | Subtle background shifts, cards, containers |
| Text | #141b2b | Body copy — sharp, readable |
| Text Secondary | #574235 | Labels, captions, metadata |
| Outline | #8b7263 | Borders, dividers, separators |

### Typography

| Role | Font | Weight | Size |
|------|------|--------|------|
| Display (desktop) | Plus Jakarta Sans | 700 | 48px, -0.02em tracking |
| Display (mobile) | Plus Jakarta Sans | 700 | 32px, -0.02em tracking |
| Headlines (H2-H3) | Plus Jakarta Sans | 600 | 24px |
| Body | Manrope | 400 | 16-18px |
| Labels | Manrope | 600 | 13px, 0.05em tracking |

### Visual Style for Images

When generating banners and social graphics:
- **Background:** Clean white (#f9f9ff) or light gray surface
- **Accent:** Orange (#fc820c) for headlines, CTAs, divider lines
- **Typography:** Clean sans-serif (Plus Jakarta Sans for display)
- **Shapes:** Rounded corners (0.5rem standard, 1rem-1.5rem for containers)
- **Elevation:** No heavy shadows. Use tonal layers (subtle gray shifts) instead.
- **Mood:** "Energetic precision" — fast, responsive, trustworthy. Not decorative, not stock-photo.

**Image gen prompt base:**
```
Create a professional insurance marketing image with a clean white/light gray background,
using vibrant orange (#fc820c) as the accent color. Clean sans-serif typography.
Minimal and modern. Abstract geometric elements. No heavy shadows.
[Specific subject/context for the piece]
```

### Verbal Identity — The SMA Voice

All content speaks as the Secure Safer Marketing Assistant:

- **Energetic precision** — sharp thinking, warm delivery, zero fluff
- **Approachable authority** — knows insurance inside out, never sounds like a textbook
- **Short sentences.** Average 14-18 words. Active voice. "You" before "we."
- **Specific beats vague.** "$847/year" not "affordable."
- **Jargon?** Explain it once. Then own it.
- **Sources?** Always. .gov > .edu > trade > blog.

### Temperature Arc Tied to Brand

| Phase | Brand Color | Temperatures | Emotional Job |
|-------|-------------|--------------|---------------|
| Open | Light gray (#d3daef) | WARM | Identify the concern, show empathy |
| Peak | Orange (#fc820c) | HOT | Show the risk — 1-2 sentences max |
| Resolve | White (#ffffff) | COLD | Provide the solution, reassurance |
| CTA | Orange (#fc820c) | ACTION | Clear next step, low friction |

Orange is used sparingly — it's the "singular driver of action and focus." In content: most of the piece is neutral/warm, with orange hitting only at the peak tension point and the final CTA.

## Part 3: Content Generation Process

### Step 1: Load the Compact Research Brief
Read the research from `_research/` or receive it inline.

### Step 2: Select Content Type

| Search Intent | Best Content Type |
|---------------|------------------|
| Informational | Blog post, guide, explainer |
| Commercial | Landing page, comparison, checklist |
| Transactional | Quote page, service page, calculator |
| Navigational | Location page, "about us", reviews |

### Step 3: Write Using Word Temperature

- **HOT words** — 1-2 per piece. Creates urgency. Overuse feels manipulative.
- **WARM words** — main emotional driver. ~60% of emotional language.
- **NEUTRAL words** — backbone of content. Most of the piece.
- **COLD words** — solution section. Reassures after identifying the problem.

**Arc:** Start WARM → peak HOT → resolve COLD

### Step 4: Apply Copywriting Principles
- Lead with the reader's concern (from the research brief)
- Answer the AEO target question in the first 60 words
- Primary keyword in H1 and first paragraph
- 14-18 word average sentences
- More "you/your" than "we/our"
- End with a specific, low-friction CTA

### Step 5: SEO + AEO
- Structured for featured snippets (list/table/paragraph)
- Internal links to related Secure Safer content
- External links to .gov sources (builds EEAT)
- Schema markup (FAQ, HowTo, Article)

### Step 6: Edit (per copy-editing skill)
Pass 1: Structure → Pass 2: Clarity → Pass 3: Conciseness → Pass 4: Persuasion → Pass 5: Proofreading

## Output Formats

### Blog Post
```
TITLE: [SEO-optimized title with primary keyword]
AEO ANSWER: [60-word direct answer — targets featured snippet]
BODY: [Full post following the temperature arc]
META DESCRIPTION: [150-160 chars with keyword + CTA]
SEO KEYWORDS: [primary + 3-5 secondary]
SOURCES: [URLs cited]
CTA: [Specific action with link]
```

### Social Post
```
PLATFORM: [LinkedIn / Facebook / X]
ANGLE: [One sentence hook]
POST: [Platform-optimized copy with temp labels]
VISUAL: [Image/video concept]
CTA: [One action]
HASHTAGS: [3-5]
```

### Landing Page
```
HEADLINE: [Benefit-driven, includes keyword]
SUBHEADLINE: [Specificity + objection handler]
BODY: [Problem → solution → proof → action]
KEY SELLING POINTS: [3-5 bullets from research brief]
SOCIAL PROOF: [Testimonials, stats, trust signals]
CTA: [Button text + destination]
SCHEMA: [FAQ / HowTo / LocalBusiness as applicable]
```

## References
- `ss-research-engine` — produces the compact research briefs
- `copywriting` — full copywriting methodology
- `copy-editing` — editing passes for polish
- `ai-seo` — AI search optimization
- `content-strategy` — broader content planning
- `_architecture/Brand Design System.md` — full brand spec
- `_architecture/Content Style Guide.md` — writing guidelines
- `_templates/Article Template.md`
- `_templates/Social Post Template.md`
