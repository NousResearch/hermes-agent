---
name: content-week
description: Use when the user wants a week of social content (e.g. "plan my week", "7 posts for next week", "a content calendar"). Produces a simple day-by-day plan and generates the matching assets inline, one per day, on-brand and ready to post.
version: 1.0.0
author: Avocado AI
license: MIT
metadata:
  hermes:
    tags: [marketing, social, calendar, content, image, video, template]
    related_skills: [product-ad-set]
---

# Content Week

Plan and generate a week of on-brand social content, delivered inline in the chat.

## When to use

- "Plan a week of content for me"
- "I need 7 posts for next week"
- The user wants a recurring-cadence content batch, not a single asset.

## What you produce

1. A short **day-by-day plan** (default 7 days; ask if they want 5). Each day: a content pillar, a
   one-line hook/caption, and the asset type (image or short video).
2. The **matching assets**, generated in chat, one per day, presented with the plan.

## How to run it

1. **Anchor on the profile + a tiny brief.** Use the saved profile for business, audience, tone.
   Ask at most 1-2 things you genuinely don't know (e.g. a current promo or theme for the week).

2. **Draft the plan first, get a nod.** Propose the 7 days as a compact list (pillar + hook + asset
   type) before generating anything. Mix pillars so the week isn't repetitive: educate, show product,
   social proof / testimonial, behind-the-scenes, offer/CTA, engagement/question, lifestyle.

3. **Generate per day.** For image days use `generate_image`; for a video day use `generate_video`
   (keep videos short). Hold tone, palette, and subject consistent across the week so it reads as one
   brand. Use the platform default model and a vertical aspect ratio (`9:16`) unless told otherwise.

4. **Respect the approval / auto-run protocol.** This batch costs real credits and a video day costs
   more than an image day. If approval mode is on, present the full plan with a per-day and total
   credit estimate in one approval request, then wait. If auto-run is on, generate directly and report
   total credits at the end.

5. **Deliver.** Present the plan with each day's asset URL inline next to its hook/caption, so the user
   can copy-paste straight into a scheduler. End by offering to adjust any day or regenerate a single one.

## Guardrails

- Keep the week visually coherent: same brand identity, varied scenes.
- Captions are starting points; keep them short and in the user's voice, never use em dashes.
- Be honest about credit cost up front, especially when a video day is included.
