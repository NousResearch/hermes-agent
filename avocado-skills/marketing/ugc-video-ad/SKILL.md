---
name: ugc-video-ad
description: Use when the user wants a short, authentic creator-style video ad for a product (e.g. "make me a UGC ad", "a TikTok-style video for my product"). Produces one short vertical video ad generated in chat, native and human, not polished-corporate.
version: 1.0.0
author: Avocado AI
license: MIT
metadata:
  hermes:
    tags: [marketing, ugc, video, ads, social, template]
    related_skills: [product-ad-set, content-week]
---

# UGC Video Ad

Produce one short, authentic, creator-style video ad for the user's product, generated inline.

## When to use

- "Make me a UGC ad for X"
- "A TikTok / Reels style video for my product"
- The user wants a native, human-feeling short video, not a polished brand film.

## What you produce

A single short vertical video (default `9:16`), built around one hook, that feels like a real
creator made it. Generate it in chat and share the asset URL inline.

## How to run it

1. **Tiny brief (1-2 questions, lean on the profile):** the product, the one hook or benefit to
   lead with, the vibe (casual / energetic / calm), and the platform (default short, `9:16`).

2. **Draft the concept + hook line.** One simple shot idea and a spoken or on-screen hook in the
   first second. Confirm it in a sentence, don't over-plan.

3. **Generate** a short vertical video with `generate_video`. Keep it native and human: real-feeling
   setting, direct-to-camera energy, not a glossy commercial. Keep duration short.

4. **Respect the approval / auto-run protocol.** Video costs more credits than an image. If approval
   mode is on, state the concept and the estimated credits, then wait. If auto-run is on, generate
   and report the cost.

5. **Deliver** the video inline. Offer one alternate hook or a captions pass as the next step.

## Guardrails

- Authentic over polished. If it looks like a TV ad, it's wrong for this format.
- Don't fabricate product claims. Use only what the user told you.
- Keep captions/hooks short and in the user's voice, never use em dashes.
