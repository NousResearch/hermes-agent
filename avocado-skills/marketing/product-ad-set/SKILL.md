---
name: product-ad-set
description: Use when the user wants a set of ad creatives for a product (e.g. "make me 3 ad variations", "ads for my new launch"). Produces a small batch of on-brand static ad images generated directly in chat, each with a distinct angle, ready to post.
version: 1.0.0
author: Avocado AI
license: MIT
metadata:
  hermes:
    tags: [marketing, ads, product, creative, image, template]
    related_skills: [content-week]
---

# Product Ad Set

Turn one product into a small set of distinct, on-brand ad creatives, generated inline in the chat.

## When to use

- "Make me a few ads for X"
- "I need ad variations for my launch"
- The user has a product and wants ready-to-post static creatives (not a storyboard, not a flow).

## What you produce

A batch of **3 ad images by default** (ask if they want more or fewer), each a different angle so the
user can test which resonates. Generate each one in chat and present them together with the asset URLs
inline. Do not build a storyboard or a flow unless explicitly asked.

## How to run it

1. **Gather the brief (1-2 short questions max, lean on the saved profile first):**
   - Product + the single benefit each ad should sell.
   - Platform / aspect ratio (default `1:1` for feed, `9:16` for stories/reels).
   - Any brand cues (colors, mood, must-include text). Pull from the user's profile if known.

2. **Pick 3 distinct angles** so the set is a real test, not 3 near-duplicates. Good default mix:
   - Hero / product-front: clean studio shot, benefit as a short headline.
   - Lifestyle / in-use: product in a real context with a person or setting.
   - Bold / pattern-interrupt: high-contrast, punchy, scroll-stopping.

3. **Generate.** Use the `generate_image` tool once per angle. Keep the same product description and
   brand cues across all three; vary only the scene + composition. Use the requested aspect ratio.
   Default to the platform's enforced model unless the user asked for a specific one.

4. **Respect the approval / auto-run protocol.** Each generation costs credits. If approval mode is on,
   list all 3 planned generations and the total estimated credits in one request, then wait. If
   auto-run is on, generate directly and state the total credits spent.

5. **Present.** Return the 3 images inline (asset URLs in the message text), each labelled with its
   angle and the one-line headline you wrote for it. Offer a quick next step: "Want variations of the
   winner, or copy to go with these?"

## Guardrails

- Keep all three visibly the same product and brand. The variation is in angle, not identity.
- If the user gave must-include text, put it in the prompt and verify it reads correctly.
- Don't invent claims about the product. Use only what the user told you.
