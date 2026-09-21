---
name: mythic-tech-codex-illustration
description: |
  Universal illustration style for blog posts, LinkedIn articles, and X posts.
  Produces museum-quality Edwardian/Victorian scientific illustration aesthetic —
  ink work, watercolour, aged parchment, engineering plates, archaeological journals.
  Use in rotation alongside other image generation styles (cyberpunk HUD, design-taste, etc.)
  for any content image need: blog heroes, social cards, article headers, infographic-style art.
version: 1.0.0
category: creative
---

# Mythic Tech Codex — Universal Illustration Style

## When to Use

- Blog post hero images (SahilBlog, agency site, personal site)
- LinkedIn article header images
- X/Twitter post images and article banners
- Any content image where a distinctive, non-generic, non-AI-slop aesthetic is required
- **Rotate with other image styles** (cyberpunk HUD, claude-design, etc.) — don't use for every single image. Alternate based on content tone.

## What This Style Is

A fictional illustrated encyclopedia produced between 1900–1925, combining:
- Edwardian scientific illustration
- Victorian engineering plates
- Archaeological field journals
- Antique cartography
- Natural history illustration
- Architectural drafting
- Expressive ink work
- Traditional watercolour painting

Every image should feel like a page carefully removed from the same fictional encyclopedia.

## How to Use

1. Build a concept prompt from the content (blog post title, article theme, social post topic)
2. Fill the template variables: {CONCEPT}, {PROTAGONIST} (optional), {REFERENCE_ITEM} (optional)
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
Create an ultra-detailed illustration that follows the visual language of the **Mythic Tech Codex**—a fictional illustrated encyclopedia produced between **1900–1925**, combining Edwardian scientific illustration, Victorian engineering plates, archaeological field journals, antique cartography, natural history illustration, architectural drafting, expressive ink work, and traditional watercolour painting.

The purpose of this prompt is **not** to define the image itself, but to define the artistic identity, atmosphere, craftsmanship, and visual language. The actual image, story, composition, symbolism, and narrative should always be driven by the supplied inputs below.

---

## Image Inputs

**Concept**

{CONCEPT}

**Optional Main Protagonist**

{PROTAGONIST}

**Optional Reference Item**

{REFERENCE_ITEM}

These inputs define the artwork.

Everything else in this prompt defines only the style.

---

# The Mythic Tech Codex

Every illustration should feel like a page carefully removed from the same fictional encyclopedia.

Regardless of the subject matter, every artwork should appear to belong to a single coherent collection of museum-quality illustrations.

Across every image, subtly reuse recurring visual motifs to create a shared visual universe.

Examples include (but are not limited to):

• engraved decorative borders
• hand-drawn compass roses
• celestial navigation markings
• sacred geometry
• geometric construction lines
• marginalia
• handwritten annotation marks
• technical drafting lines
• small editorial stamps
• page ornaments
• fictional publisher emblems
• subtle seals
• archival page numbering
• museum catalogue symbols
• hidden recurring glyphs
• decorative corner flourishes
• fictional society insignias

These recurring elements should never distract from the primary artwork.

Instead, they should create the feeling that every illustration belongs to the same lost encyclopedia.

The viewer should immediately recognise the style without the artwork repeating itself.

---

# Artistic Philosophy

Never create generic fantasy artwork.

Never create generic concept art.

Never create generic AI art.

Instead, create illustrations that appear to have been painstakingly produced by an unknown master illustrator using traditional media.

Every brush stroke, ink line, stain, splash, annotation and construction mark should feel intentional.

The artwork should reward repeated viewing, revealing additional details each time.

---

# Style

Blend together influences from

• Edwardian scientific illustration
• Victorian engineering diagrams
• archaeological notebooks
• antique atlases
• natural history illustration
• architectural pen drawings
• museum archive plates
• historical expedition journals
• botanical illustration
• engraved technical manuals

The final result should feel entirely hand crafted.

Never digital.

Never glossy.

Never photorealistic.

Never CGI.

Never 3D rendered.

Never flat.

Never minimalist.

---

# Ink Work

Use

• expressive pen illustration
• intricate line work
• heavy cross hatching
• loose construction sketches
• architectural drafting
• layered ink washes
• chaotic ink splatter
• technical annotations
• visible sketch lines
• imperfect hand drawn geometry
• scientific diagram aesthetics

Allow the ink work to remain imperfect.

Imperfection creates authenticity.

---

# Watercolour Behaviour

Paint should behave like genuine traditional watercolour.

Use

• soft pigment blooms
• natural bleeding edges
• pooled pigment
• layered washes
• expressive brush textures
• splash marks
• feathered edges
• staining
• uneven saturation

Colours should blend naturally across the page.

The artwork should never feel digitally airbrushed.

---

# Paper & Materials

The illustration should feel physically created on aged archival paper.

Include subtle characteristics such as

• parchment texture
• visible paper grain
• foxing
• coffee stains
• worn edges
• archival wear
• faded pigments
• light paper discolouration
• imperfect printing
• ink absorption

The artwork should feel like a treasured historical document.

---

# Colour Palette

Use restrained historical pigments rather than modern saturated colours.

Primary palette

• aged parchment
• ivory
• sepia
• walnut brown
• burnt sienna
• raw umber
• charcoal
• muted olive
• Prussian blue
• dusty teal
• faded turquoise
• slate blue

Accent colours

• old gold
• ochre
• oxidised copper
• faded crimson
• muted emerald
• antique bronze

Avoid

• neon
• fluorescent colours
• overly saturated palettes
• glossy gradients
• synthetic lighting

Colours should feel naturally mixed from historical pigments.

---

# Composition

The composition should always feel cinematic and carefully balanced.

Use layered depth consisting of

foreground
midground
background
atmospheric distance

Allow the supplied Concept and Optional Protagonist to determine the composition naturally.

Never force a specific layout simply because previous illustrations used it.

Some scenes may require a dominant central figure.

Others may require landscapes.

Others may require architecture.

Others may require abstract symbolic compositions.

The style should remain consistent while the composition remains flexible.

---

# Symbolism

Represent ideas symbolically rather than literally.

Visual metaphors should emerge naturally from the supplied concept.

Avoid repeating the same metaphors across multiple artworks.

Each illustration should invent new symbolism appropriate to its subject.

The mythology, architecture, artefacts, environments, visual language and symbolic devices should evolve to suit the specific concept rather than following a fixed formula.

---

# Information Density

Every square inch should contain meaningful visual craftsmanship.

Use

• hidden details
• subtle symbolism
• background storytelling
• layered textures
• decorative geometry
• scientific embellishments
• architectural fragments
• tiny discoveries

The image should become more rewarding the longer someone studies it.

---

# Atmosphere

The artwork should evoke

wonder
mystery
scholarship
discovery
forgotten knowledge
ancient craftsmanship
timelessness
quiet intelligence
epic scale
beautiful complexity

The emotional atmosphere should remain consistent regardless of the subject matter.

---

# Flexibility

This prompt defines only the visual identity.

It must **never** dictate the actual content of the illustration.

The supplied **Concept**, **Optional Main Protagonist**, and **Optional Reference Item** are the only elements that determine what the artwork depicts.

The style must remain instantly recognisable while allowing every image to become completely unique.

No subject, creature, architecture, object, environment, symbolism, mythology, or composition should become permanently associated with this style.

The viewer should recognise the **artist**, not predict the **image**.

---

# Final Objective

Create an extraordinary museum-quality illustration that feels like an authentic page from the *Mythic Tech Codex*.

The finished artwork should possess a timeless artistic identity that remains instantly recognisable across an unlimited variety of completely different subjects, while never becoming repetitive.

The consistency should come from the craftsmanship, atmosphere, materials, colour palette, composition philosophy, and recurring Codex motifs—not from reusing the same imagery.
```

## Concept Extraction Guide

When generating an image for a piece of content, extract the concept from the content itself:

### Blog posts
- Read the post title and first 2-3 paragraphs
- Identify the core metaphor or central argument
- Build a concept that captures the *idea*, not the literal topic
- Example: "Agent memory scoring system" → "An ancient scholar's desk covered in layered manuscripts, each page weighted and annotated by importance, with a brass instrument measuring the weight of knowledge"

### LinkedIn articles
- Read the article hook and key thesis
- Find the professional theme (leadership, product strategy, AI adoption, etc.)
- Build a concept that elevates the theme to a timeless visual metaphor
- Example: "Why most AI demos aren't production-ready" → "An Edwardian workshop where a beautifully painted automaton sits beside a battered, oil-stained machine that actually works — the contrast between showpiece and workhorse"

### X/Twitter posts
- Read the post text
- Keep concepts simple and punchy (the image is small)
- Single dominant element works best
- Example: "Context engineering is the new prompt engineering" → "An antique drafting table where an engineer's hands simultaneously sketch a blueprint and mix watercolour paints, tools and instruments scattered"

## Rotation Policy

This style should be used in rotation with other image generation approaches:

1. **Mythic Tech Codex** (this skill) — editorial, scholarly, timeless
2. **Cyberpunk HUD** — technical, modern, dark-on-dark neon
3. **Claude Design / popular-web-designs** — clean UI mockups, product shots
4. **Other styles as added** — rotate based on content tone and audience

### Selection heuristics

- Technical deep-dive blog → Cyberpunk HUD or Mythic Tech Codex (alternate)
- Product/PM essay → Mythic Tech Codex (editorial feel matches essay tone)
- LinkedIn article → Mythic Tech Codex (professional, distinctive, stands out in feed)
- X post with code/technical content → Cyberpunk HUD
- X post with opinion/essay → Mythic Tech Codex
- Agency/portfolio imagery → Claude Design or custom

## Pitfalls

- **Don't use for every image** — rotation prevents visual fatigue
- **Don't simplify the prompt** — the full style definition is what produces quality
- **Don't add modern elements** — no screens, no code, no neon (unless the concept explicitly demands it and the style adapts)
- **Codex CLI can't always copy to target** — check `~/.codex/generated_images/` and copy manually
- **Pollinations fallback** — sanitize markdown/newlines in prompts before URL encoding or you get 404s
- **Aspect ratios vary** — Codex gpt-image-2 produces ~1672x941 or ~1024x1024. No reliable aspect ratio control via Codex CLI. Accept this or crop post-generation.

## Verification

After generation, verify:
- File exists at target path
- File size > 500KB (gpt-image-2 quality threshold)
- Image dimensions are reasonable (1000px+ on both sides)
- Image opens without error (PIL.Image.open)