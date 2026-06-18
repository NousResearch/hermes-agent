---
name: brand-photoshoot
description: Use when the user wants a cohesive set of product photos as if from one shoot (e.g. "brand photoshoot", "product photos for my store/site"). Produces a set of on-brand product images across varied scenes, generated in chat.
version: 1.0.0
author: Avocado AI
license: MIT
metadata:
  hermes:
    tags: [marketing, product, photography, image, brand, template]
    related_skills: [product-ad-set]
---

# Brand Photoshoot

Generate a cohesive set of product photos across varied scenes, as if from one shoot, inline.

## When to use

- "Do a brand photoshoot for my product"
- "I need product photos for my site / store"
- The user wants a coordinated set, not a single image and not ad creatives with headlines.

## What you produce

A set of **4 photos by default** that share one lighting and palette (so they read as one shoot)
but cover different setups. Generate each in chat and present them together with the asset URLs.

## How to run it

1. **Tiny brief (lean on the profile):** the product, the brand mood (clean / luxury / playful /
   natural), and where the shots will be used.

2. **Pick 4 complementary setups** with a shared look: studio on seamless, lifestyle in context,
   detail / macro, and in-use or styled-with-props. The throughline is one consistent brand
   identity, lighting, and color story.

3. **Generate** each shot with `generate_image`, holding palette and lighting constant and varying
   only the setup and composition. Default to a clean, photographic aspect ratio for the use case.

4. **Respect the approval / auto-run protocol** before generating (it costs credits). If approval
   mode is on, list all 4 planned shots and the total estimate, then wait.

5. **Deliver** the set inline, labelled by setup. Offer to regenerate any single shot or push a
   favorite into an ad (hand off to the product ad set skill).

## Guardrails

- Coherence is the point: same product, same brand look across all four.
- Don't invent product details or claims. Use only what the user provided.
