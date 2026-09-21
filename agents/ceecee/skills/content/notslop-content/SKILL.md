---
name: notslop-content
description: Draft grounded social content using notslop's real-time signal pipeline. Wraps notslop digest, trending, find-related into content drafting workflows for Sahil's brands (LinkedIn, Twitter, MatchdayMaestro, Plenishd). Uses real current signal — not training data hallucinations.
version: 1.0.0
author: Sahil Saghir
license: MIT
metadata:
  hermes:
    tags: [content-drafting, social-content, grounding, notslop, brand-voice]
    related_skills: [notslop-digest, sahil-linkedin-voice, sahil-twitter-voice, matchdaymaestro-voice, plenishd-voice]
---

# notslop-content — Grounded Content Drafting

Draft social content (LinkedIn, Twitter/X, MatchdayMaestro, Plenishd) grounded in what people are actually discussing right now. Uses notslop to pull real signal before drafting, so every post references real recent conversations — not fabricated examples.

This skill is a **thick wrapper** — it orchestrates notslop digests, brand voice skills, and content lead drafting standards into one workflow.

## Prerequisites

- notslop CLI installed (global, confirmed working)
- ZeroEntropy API key configured (`notslop init` or `ZEROENTROPY_API_KEY`)
- The relevant brand voice skill loaded for tone

## Workflow: Draft Grounded Content

### Step 1: Identify brand and topic

Ask one question if not already clear: **"Which brand is this for?"**

Brands:
- **Personal LinkedIn** — load `sahil-linkedin-voice` skill
- **Personal Twitter/X** — load `sahil-twitter-voice` skill
- **MatchdayMaestro** — load `matchdaymaestro-voice` skill
- **Plenishd** — load `plenishd-voice` skill
- **CoachOS** (beta Aug 2026) — no voice skill yet, default to warm authoritative UK English

Extract the topic they want to post about. If vague ("something about AI"), push for specificity in one question.

### Step 2: Pull fresh signal

```bash
npx notslop@latest digest "<TOPIC>" --since 7d --top 10 --for-content
```

Pick the window based on content type:
- **24h** — Twitter/X tweets (current-day urgency)
- **7d** — LinkedIn posts (slightly longer horizon)
- **14d** — Weekly recap or analysis posts
- **30d** — Slow-moving brand content (Plenishd, evergreen)

For existing content repurposing:
```bash
npx notslop@latest find-related "<URL_OR_TEXT>" --since 14d --top 10
```

### Step 3: Extract grounding points

From the digest output, identify:
- **2-3 specific data points** — real numbers, observations, recent discussions
- **Source URLs** to cite in the draft or below it
- **Themes** — what's actually being discussed, not what you guessed

### Step 4: Draft using brand voice

Apply the loaded brand voice skill. Key rules per brand:

| Brand | Register | Length | Hook style | Signs off with |
|---|---|---|---|---|
| Sahil LinkedIn | Authoritative + specific | 150-250 words | Pattern observation, counter-intuitive number, personal proof | Quotable closer, no "what do you think?" |
| Sahil Twitter/X | Direct + wry | 280 chars or thread | Just-shipped, real number, triple-punch | One-liner, no engagement bait |
| MMaestro | Friendly + gamified | 1-4 tweets | Question-as-engagement, streak marker, group chat energy | "Your move." / "Tap to play." |
| Plenishd | Warm + practical + witty | Varies | Real moment, list promise, rescue arc | "Sorted." / "Less waste. Less faff." |

### Step 5: Show draft + sources

Every draft MUST include:
1. The draft content itself
2. Source URLs used for grounding (1-3 lines below draft)
3. Character/word count
4. Voice used

### Step 6: Offer variants

Ask: "Want a safe, sharp, or experimental version?"
- **Safe** — standard brand voice, no edge
- **Sharp** — stronger opinion, more provocative angle
- **Experimental** — different hook type or structure

## Commands reference

All run via terminal, all use `npx notslop@latest`:

```bash
# Standard digest (most common)
notslop digest "<TOPIC>" --since 7d --top 10 --for-content

# Trending — what's hot right now
notslop trending "<NICHE>" --since 6h --top 10 --format md

# Semantic find-related — match a draft or URL
notslop find-related "<URL>" --since 14d --top 10

# Sources status
notslop sources
```

## Common content types and their windows

| Content type | Window | Top N | Flag |
|---|---|---|---|
| Twitter/X single tweet | 24h | 5-10 | `--for-content` |
| Twitter/X thread (4-8 tweets) | 7d | 10-15 | `--for-content` |
| LinkedIn post (150-250 words) | 7d | 10-15 | `--for-content` |
| MMaestro match-week content | 24h | 5-10 | `--format md` |
| Plenishd kitchen tips | 14d | 5-10 | `--format md` |
| Long-form blog/linkedin article | 14-30d | 15-20 | `--format json` |
| Repurpose existing content | 14d | 10 | (use `find-related`) |

## Pitfalls

- **Never fabricate sources.** If the digest doesn't return enough signal (too narrow a topic), say so. Don't make up data points.
- **British English throughout.** Digest output is whatever people wrote — don't copy phrasing, adapt to brand voice.
- **The digest returns raw posts.** They may contain American English, slang, or AI-generated slop. Filter mentally — surface the signal, not the noise.
- **X signal requires Orthogonal key.** If it's not configured, the digest still works on Reddit + HN + blogs. That's enough for most content.
- **Don't ground in a single source.** Pick 2-3 data points so the post feels like a synthesis, not a retweet.
- **Source URLs to user:** below the draft, not in the draft body (LinkedIn penalises external links in body).

## Example workflow

> **User:** "Draft me a LinkedIn post about vibe coding safety"
>
> → **Load:** `sahil-linkedin-voice`, `notslop-digest`
> → **Run:** `notslop digest "vibe coding security vulnerabilities" --since 7d --top 10 --for-content`
> → **Extract:** Veracode 45% failure rate, Escape.tech 65% vuln rate, Georgia Tech CVE surge data
> → **Draft:** LinkedIn post in Authoritative register, Counter-Intuitive Numbers hook
> → **Show:** Draft (198 words) + 3 source URLs + "Safe/sharp/experimental?"
