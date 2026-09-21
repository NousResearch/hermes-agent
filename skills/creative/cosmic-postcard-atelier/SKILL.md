---
name: cosmic-postcard-atelier
description: |
  Universal illustration style for blog posts, LinkedIn articles, and X posts.
  Produces museum-quality mid-century retro-futurism — 1958-1978 sci-fi travel posters,
  vintage magazine covers, painted editorial artwork, analogue gouache/airbrush.
  Use in rotation alongside Mythic Tech Codex, cyberpunk HUD, and other image styles.
  Variables: {CONCEPT}, {REFERENCE} (optional), {EDITORIAL_TONE} (optional).
version: 1.0.0
category: creative
---

# Cosmic Postcard Atelier — Universal Illustration Style

## When to Use

- Blog post hero images (SahilBlog, agency site, personal site)
- LinkedIn article header images
- X/Twitter post images and article banners
- Any content image where an optimistic, cinematic, retro-futurist aesthetic is required
- **Rotate with other image styles** (Mythic Tech Codex, cyberpunk HUD, claude-design) — don't use for every image. Alternate based on content tone and audience.

## What This Style Is

A fictional illustration studio that produced iconic science-fiction travel posters, editorial artwork, magazine covers, educational illustrations, advertising campaigns, and gallery prints between 1958-1978.

Every image should feel like it belongs to the same legendary publishing house — painted by master illustrators whose work inspired generations to imagine impossible futures.

## How to Use

1. Build a concept prompt from the content (blog post title, article theme, social post topic)
2. Fill the template variables: {CONCEPT}, {REFERENCE} (optional), {EDITORIAL_TONE} (optional)
3. Feed the full prompt (style definition + concept) to the image generation tool
4. The style definition is always the same — only the concept inputs change

## Generation Pipeline

This skill works with any image generation backend:

- **Codex CLI** (`codex exec`) — primary on VPS, ChatGPT subscription OAuth, £0 cost
- **OpenAI API** (`gpt-image-2`) — if API credits available
- **FAL.ai** — if key configured
- **Pollinations** — last-resort free fallback (lower quality, sanitize prompts)

For Codex CLI specifically:
```bash
codex exec --skip-git-repo-check -s workspace-write --add-dir "{output_dir}" \
  "{FULL_PROMPT} Save the generated image to {output_path}"
```

Then copy from `~/.codex/generated_images/` if Codex can't write to target directly.

## Prompt Template

The full prompt is the style definition below with the concept inputs filled in.
Always include the complete style definition — it's what makes the style consistent.

```
Create an ultra-detailed illustration that follows the visual language of the **Cosmic Postcard Atelier**—a fictional illustration studio that produced iconic science-fiction travel posters, editorial artwork, magazine covers, educational illustrations, advertising campaigns, and gallery prints between **1958–1978**.

The purpose of this prompt is **not** to define the image itself.

Instead, it establishes a timeless artistic identity that can be consistently applied to any subject, article, person, technology, historical event, business story, scientific discovery, political issue, or imagined world.

---

# Image Inputs

## Concept

{CONCEPT}

The primary subject, narrative or story being illustrated.

This determines **what** the artwork depicts.

---

## Optional Reference

{REFERENCE}

Use this only as inspiration for:

- composition
- colour harmony
- atmosphere
- lighting
- visual motifs
- framing
- emotional tone

Never reproduce or imitate the reference literally.

Instead reinterpret it entirely through the visual language of the Cosmic Postcard Atelier.

---

## Optional Editorial Tone

{EDITORIAL_TONE}

Examples include:

- Optimistic
- Hopeful
- Inspirational
- Cautionary
- Competitive
- Urgent
- Celebratory
- Mysterious
- Satirical
- Reflective
- Dramatic

This influences only the mood, symbolism, lighting and composition.

The artistic identity always remains consistent.

---

Everything else in this prompt defines only the artistic identity.

---

# The Cosmic Postcard Atelier

Every illustration should feel like it belongs to the same legendary publishing house.

Regardless of subject matter, every artwork should appear to have been painted by one of a handful of master illustrators whose work inspired generations to imagine impossible futures.

The collection should feel cohesive while allowing unlimited creative freedom.

Every illustration should immediately feel recognisable without becoming repetitive.

---

# Artistic Philosophy

Celebrate curiosity.

Celebrate imagination.

Celebrate exploration.

Celebrate optimism.

Technology should feel human.

The future should feel hopeful, mysterious and full of possibility.

Even when illustrating conflict, political tension, risk, or existential challenges, preserve an underlying sense of wonder.

The artwork should invite viewers to explore ideas rather than simply observe them.

---

# Style

Blend together influences from:

- 1950s–1970s retro-futurism
- mid-century commercial illustration
- vintage travel posters
- classic science-fiction paperback covers
- painted editorial magazine artwork
- cinematic movie posters
- analogue matte paintings
- surreal landscape painting
- optimistic Space Age industrial design
- Golden Age illustration

The result should feel entirely hand-painted.

Never photorealistic.

Never CGI.

Never 3D rendered.

Never glossy.

Never modern concept art.

Never generic AI art.

---

# Painting Technique

Paint using the behaviour of traditional analogue media.

Use:

- gouache
- acrylic
- subtle airbrush transitions
- visible brush texture
- painterly blending
- layered paint
- dry brush details
- soft glazing
- textured highlights
- hand-painted imperfections

Edges should remain naturally softened.

Brushwork should remain visible.

Every image should feel physically painted by an expert illustrator.

---

# Colour Philosophy

Use bold but nostalgic colour harmonies.

Primary colours include:

- coral
- vermilion
- salmon
- dusty turquoise
- teal
- ultramarine
- midnight blue
- sandstone
- ochre
- burnt orange
- cream
- warm ivory

Accent colours include:

- glowing gold
- pale cyan
- faded magenta
- lavender
- mint
- rich crimson

Colours should resemble aged offset printing and vintage lithography rather than modern digital RGB.

Avoid HDR colour grading.

Avoid neon.

Avoid fluorescent colours.

Avoid glossy digital rendering.

---

# Lighting

Use cinematic lighting driven by atmosphere rather than realism.

Possible light sources include:

- glowing horizons
- oversized moons
- distant suns
- nebulae
- auroras
- atmospheric haze
- illuminated clouds
- reflective deserts
- futuristic city lights
- spacecraft illumination

Lighting should create emotional scale rather than technical accuracy.

---

# Landscapes & Environments

Every world should feel expansive.

Whether depicting cities, boardrooms, scientific breakthroughs, history, mythology, politics or abstract ideas, environments should possess cinematic scale.

Use:

- vast horizons
- monumental architecture
- impossible mountain ranges
- alien deserts
- lush valleys
- flowering foregrounds
- futuristic skylines
- enormous celestial bodies
- dramatic skies
- layered atmospheric perspective

Scale should feel emotionally true rather than physically realistic.

---

# Composition

Compose every illustration like an iconic magazine cover or legendary travel poster.

Build images using:

Foreground

↓

Primary narrative

↓

Expansive middle distance

↓

Monumental background

↓

Epic celestial sky

The artwork should remain visually powerful even when viewed as a small thumbnail.

Larger viewing should continuously reward exploration with additional details.

---

# Storytelling

Avoid literal illustration whenever possible.

Translate ideas into visual metaphors.

Examples include:

- AI competition becoming an interstellar race
- diplomacy becoming a galactic summit
- cybersecurity becoming planetary fortresses
- innovation becoming space exploration
- education becoming voyages into unknown galaxies
- financial markets becoming orbital trade routes
- climate change becoming planetary restoration
- healthcare becoming celestial ecosystems
- regulation becoming navigation through dangerous star systems

Invent fresh symbolism appropriate to the supplied Concept.

Avoid repeating the same metaphors across multiple artworks.

---

# Human Presence

People should feel expressive and purposeful.

Characters should display confidence, curiosity, intelligence and emotion.

Facial expressions should remain natural.

Body language should reinforce the narrative.

When illustrating real public figures, depict them respectfully while integrating them naturally into the imaginative world.

They should feel like explorers, diplomats, scientists, pioneers or visionaries rather than superheroes.

---

# Recurring Atelier Motifs

Across every illustration, subtly reuse visual language that creates a shared artistic universe.

Examples include:

- elegant retro typography
- fictional space agency insignias
- interstellar travel posters
- orbital diagrams
- planetary maps
- celestial navigation lines
- mission patches
- destination markers
- observatories
- spacecraft silhouettes
- geometric star charts
- decorative transport emblems
- futuristic tourism logos
- subtle astronomical graphics

These recurring motifs should never distract from the primary artwork.

Instead they should quietly establish a recognisable artistic identity.

---

# Surface & Print Quality

Every illustration should resemble a beautifully preserved vintage print.

Include subtle characteristics such as:

- paper grain
- analogue paint texture
- soft lithographic printing
- slight ink inconsistencies
- gentle colour fading
- printed imperfections
- delicate edge wear
- subtle ageing
- painterly surface variation

Avoid excessive distressing.

The artwork should feel cherished rather than damaged.

---

# Information Density

Reward careful observation.

Fill every image with meaningful craftsmanship.

Include:

- hidden visual stories
- subtle symbolism
- background activity
- tiny architectural details
- layered environments
- atmospheric depth
- carefully designed props
- elegant graphic elements

Each viewing should reveal something new.

---

# Atmosphere

Every illustration should evoke:

- wonder
- nostalgia
- adventure
- exploration
- optimism
- possibility
- discovery
- curiosity
- beautiful isolation
- quiet intelligence
- timeless imagination
- romantic futurism

The emotional atmosphere should remain consistent regardless of subject matter.

---

# Flexibility

This prompt defines only the artistic identity.

The supplied **Concept** determines what the artwork depicts.

The optional **Reference** influences composition, atmosphere and mood without ever being copied.

The optional **Editorial Tone** adjusts emotional presentation while preserving the style.

No recurring architecture, vehicle, planet, character, object or visual metaphor should become permanently associated with this style.

Every illustration should feel completely original while remaining unmistakably part of the Cosmic Postcard Atelier collection.

The viewer should recognise the artist—not predict the image.

---

# Final Objective

Create an extraordinary museum-quality illustration that feels like an authentic work from the **Cosmic Postcard Atelier**—a legendary mid-century illustration studio whose paintings inspired generations to dream about impossible worlds.

The finished artwork should possess a timeless artistic identity that remains instantly recognisable across an unlimited variety of completely different subjects, while never becoming repetitive.

Consistency should come from the painterly craftsmanship, cinematic composition, nostalgic colour language, optimistic atmosphere, editorial storytelling, analogue materials and romantic vision of the future—not from reusing the same imagery.
```

## Concept Extraction Guide

When generating an image for a piece of content, extract the concept from the content itself:

### Blog posts
- Read the post title and first 2-3 paragraphs
- Identify the core theme or central argument
- Build a concept that captures the idea as a retro-futurist visual metaphor
- Example: "Agent memory scoring system" → "A cosmic observatory where a lone scientist catalogues drifting memory crystals orbiting a distant star, each crystal glowing with stored knowledge, a vast star chart mapping which memories matter most"

### LinkedIn articles
- Read the article hook and key thesis
- Find the professional theme (leadership, product strategy, AI adoption, etc.)
- Elevate to a Space Age visual metaphor — exploration, discovery, new frontiers
- Example: "Why most AI demos aren't production-ready" → "A retro-futurist launchpad where a sleek showpiece rocket gleams under spotlights while a weathered, battle-scarred vessel sits in the background under a vast starry sky — the contrast between the show and the real mission"

### X/Twitter posts
- Read the post text
- Keep concepts punchy — single dominant element works best for small images
- The editorial tone variable is especially useful here (urgent, celebratory, reflective, etc.)
- Example: "Context engineering is the new prompt engineering" → "A mid-century mission control room where navigators plot trajectories through a cosmic map, charting courses between knowledge stars, old paper star charts being replaced by glowing orbital projections"

## Rotation Policy

This style should be used in rotation with other image generation approaches:

1. **Mythic Tech Codex** — editorial, scholarly, timeless, Edwardian/Victorian
2. **Cosmic Postcard Atelier** (this skill) — optimistic, cinematic, retro-futurist, mid-century
3. **Cyberpunk HUD** — technical, modern, dark-on-dark neon
4. **Claude Design / popular-web-designs** — clean UI mockups, product shots
5. **Other styles as added** — rotate based on content tone and audience

### Selection heuristics

- Technical deep-dive blog → Cyberpunk HUD or Cosmic Postcard (alternate)
- Product/PM essay → Cosmic Postcard or Mythic Tech Codex (alternate)
- LinkedIn article → Cosmic Postcard (optimistic, stands out in feed, professional)
- X post with code/technical content → Cyberpunk HUD
- X post with opinion/essay → Cosmic Postcard or Mythic Tech Codex
- Future-focused / AI / technology topic → Cosmic Postcard (Space Age optimism fits naturally)
- Historical / research / scholarship topic → Mythic Tech Codex (archaeological feel)
- Agency/portfolio imagery → Claude Design or custom
- Celebratory / launch / milestone content → Cosmic Postcard (editorial tone: celebratory)
- Cautionary / risk / security topic → Mythic Tech Codex or Cyberpunk HUD

## Pitfalls

- **Don't use for every image** — rotation prevents visual fatigue
- **Don't simplify the prompt** — the full style definition is what produces quality
- **Don't add modern digital aesthetics** — no screens, no neon, no HDR, no photorealism
- **Codex CLI can't always copy to target** — check `~/.codex/generated_images/` and copy manually
- **Pollinations fallback** — sanitize markdown/newlines in prompts before URL encoding or you get 404s
- **Aspect ratios vary** — Codex gpt-image-2 produces ~1672x941 or ~1024x1024. No reliable aspect ratio control via Codex CLI. Accept this or crop post-generation.
- **{EDITORIAL_TONE} is powerful** — use it to vary the mood across posts even when the style stays the same. A "cautionary" tone vs a "celebratory" tone produces very different images from the same style.

## Verification

After generation, verify:
- File exists at target path
- File size > 500KB (gpt-image-2 quality threshold)
- Image dimensions are reasonable (1000px+ on both sides)
- Image opens without error (PIL.Image.open)