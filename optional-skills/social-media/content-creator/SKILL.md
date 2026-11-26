---
name: content-creator
description: |
  Write platform-native social media content — X/Twitter threads, LinkedIn
  posts, Instagram captions — with the right hook, structure, tone, and
  hashtags for each platform. Turns an idea or a link into ready-to-post copy.
version: 0.1.0
author: HeLLGURD
license: MIT
platforms: [linux, macos, windows]
category: social-media
triggers:
  - "write a tweet about [topic]"
  - "write a thread about [topic]"
  - "turn this into a thread"
  - "write a LinkedIn post about [topic]"
  - "write an Instagram caption for [topic]"
  - "make social media content from [link/article]"
  - "draft a post announcing [thing]"
  - "repurpose this for social media"
toolsets:
  - terminal
  - web
  - file
metadata:
  hermes:
    tags: [Social-Media, Content, Twitter, X, LinkedIn, Instagram, Copywriting, Marketing]
    related_skills: [creative-ideation, meme-generation]
---

# Content Creator

Turn an idea, an article, or a product update into platform-native social media
copy. Each platform has its own rhythm — what works as an X thread dies on
LinkedIn and vice versa. This skill writes for the platform you're targeting,
with the correct hook, length, structure, and hashtag strategy.

Works from a topic, a pasted draft, or a URL (fetched via `web`). No external
services beyond optional link fetching.

---

## When to Use

- User wants a tweet, thread, LinkedIn post, or Instagram caption
- User has an article/blog/PR and wants to promote it on social
- User wants to repurpose one piece of content across multiple platforms

Do NOT use for:
- Long-form blog posts or articles — that's a different format
- Replying to specific comments/DMs — just answer inline
- Generating images — use `meme-generation` or an image skill

---

## Prerequisites

- None required for topic-based writing.
- `web` toolset only if pulling content from a URL.

---

## Platform Cheat Sheet

| Platform | Sweet spot | Hook style | Hashtags | Tone |
|---|---|---|---|---|
| **X / Twitter (single)** | ≤ 280 chars | Punchy first line, no wind-up | 0–2, inline | Conversational, opinionated |
| **X / Twitter (thread)** | 5–12 tweets | Tweet 1 = the whole promise | 1–2 in last tweet | Build tension, payoff per tweet |
| **LinkedIn** | 1,300–2,000 chars | First 2 lines (before "see more") must hook | 3–5 at the end | Professional but human, story-driven |
| **Instagram** | 125–2,200 chars | First line + emoji | 5–15, in a block at the end | Warm, visual, lifestyle |

---

## Procedure

### Step 1 — Gather the input

- **Topic only:** ask one clarifying question if the angle is unclear
  (audience? goal — awareness, clicks, signups?).
- **URL given:** fetch it with `web` and extract the core message, key points,
  and any quotable lines.
- **Draft given:** identify the strongest idea buried in it — that becomes the
  hook.

### Step 2 — Find the hook

The hook is everything. Before writing the body, draft 3 candidate hooks and
pick the strongest. Strong hooks do one of:
- State a surprising result or number
- Take a clear, slightly contrarian stance
- Promise a specific payoff ("here's how we cut X by Y")
- Open a curiosity gap (without clickbait dishonesty)

Weak hooks to avoid: "I'm excited to share…", "Check out our new…", anything
that buries the point below the fold.

### Step 3 — Write for the target platform

**X single tweet:**
- Lead with the hook. Cut every filler word.
- One idea per tweet. End with a light CTA or question if it fits naturally.

**X thread:**
```
1/ [The complete promise — readable standalone. This tweet alone should make
   someone want the rest.]

2/ [Context or the first concrete point]

3–N/ [One idea per tweet, each a self-contained payoff]

Last/ [Recap + CTA: follow, link, or a question]
```
- Number tweets (`1/`, `2/`) or use a clear visual break.
- Each tweet must earn the next — no "as I said above" filler.

**LinkedIn:**
- First 2 lines are the hook (everything else is hidden behind "see more").
- Short paragraphs, lots of line breaks (mobile-first).
- Tell a small story or share a concrete lesson.
- Hashtags (3–5) at the very end.

**Instagram:**
- Hook line + emoji up top.
- Body in scannable chunks.
- Hashtag block (5–15) separated from the caption by line breaks or dots.

### Step 4 — Optimize hashtags and mentions

- Mix broad (#AI) and niche (#LLMOps) tags — niche tags reach engaged
  audiences with less competition.
- Mention relevant accounts/brands only when genuinely relevant (not for reach
  farming — it reads as spammy).
- X: keep hashtags minimal; they suppress reach if overused.

### Step 5 — Deliver options

Provide the post(s) ready to copy-paste. When useful, offer:
- 2–3 alternative hooks the user can swap in
- A shorter and a longer variant
- The same content adapted for a second platform if they mentioned cross-posting

Save to a file if the user wants a content batch:
```
~/.hermes/content/<topic>-<platform>.md
```

---

## Repurposing One Idea Across Platforms

When asked to cross-post, don't copy-paste the same text — adapt the *format*:

| Source | → X thread | → LinkedIn | → Instagram |
|---|---|---|---|
| Blog post | Pull the 5 key takeaways into 5 tweets | Lead with the core insight + a story | Visual quote + caption |
| Product update | Hook on the headline benefit | Story of why you built it | Before/after framing |
| Data/result | Lead with the number | Explain the method + takeaway | Bold stat graphic caption |

---

## Voice and Authenticity Guardrails

- **Match the user's voice** if they provide examples of past posts. Mirror
  their sentence length, formality, and emoji usage.
- **No fabricated claims** — never invent statistics, testimonials, or results.
  If the user gives a number, use it; don't make one up.
- **Avoid engagement-bait** — "comment YES if you agree", fake urgency, and
  manufactured controversy read as cheap and hurt credibility.
- **Disclose appropriately** — if it's an ad/sponsored, include the disclosure.

---

## Edge Cases

**Topic is too broad** ("write about AI"): narrow it with one question — what's
the specific angle, audience, or goal? Generic posts perform poorly.

**Sensitive/controversial topic:** stay factual, avoid inflammatory framing,
and flag to the user that the topic may draw strong reactions.

**Character limit overflow** (thread tweet > 280): split it or tighten — never
ship a tweet that gets truncated mid-sentence.

---

## What This Skill Does NOT Cover

- Scheduling/posting to platforms — this produces copy; use a scheduler
  (Buffer, native tools) to publish
- Image/video creation — pair with an image or video skill
- Paid ad copy with conversion tracking — that's a specialized discipline
- Community management / replying at scale
