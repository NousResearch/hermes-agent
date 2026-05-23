# Slide Deck Builder Guide

Build interactive HTML decks that render beautifully in Town and support container-controlled editing.

---

## Routine

Follow these steps when creating or editing a slide deck:

**Step 1 — Understand the request.** Before building anything, make sure you have enough context. If the user's request is vague (e.g. "make me a presentation about cheese"), ask 1–2 clarifying questions about audience, purpose, key points, or tone. If the request already has enough detail, skip this and proceed. If the user says "just make it" or similar, proceed with reasonable defaults — don't block them. See the "Conversation Before Building" section below for more guidance.

**Step 2 — Choose a style.** For a new deck, if the user has not already specified a deck style, call `choose_deck_style` to present the style picker. Wait for their selection (or a custom description) before generating. The user may also describe a custom style instead of picking one — that's valid too.

**Step 3 — Load the styles.** Copy the companion `styles.json` file into the sandbox using `town_cp` with source `skills://slide-deck-guide/styles.json`. This contains the deck style definitions with scaffolds and generation instructions.

**Step 4 — Build the deck.** Use the selected style's `generationInstructions` and `scaffoldHtml` from `styles.json` as the starting point. Adapt the content to the user's topic, but preserve the selected style's composition system unless the user asks for a deviation.

## Tool Rules

- The deck must support both vertical and horizontal layouts; vertical is the default unless the user asks otherwise.
- Once a style is selected, use it as the primary structural recipe for the deck — the style defines both the visual language and the composition system.
- If visuals would materially improve the deck, you may call `generate_image` and insert the returned Town-hosted image into the HTML deck.
- When using generated images in decks, prefer the `deckImageSrc` or `deckImageHtml` returned by `generate_image`.
- Do NOT use external image URLs in decks when a Town-hosted generated image is available.
- If editing an existing deck, first call `read_slide_deck` to get the latest HTML, then call `update_slide_deck` with the full revised HTML.
- After creating or updating a deck file, call `show_slide_deck(file_id=...)` so the deck opens for the user in the UI.
- If the deck was saved via `town_cp`, use the returned `fileId` field as the `show_slide_deck(file_id=...)` argument.
- Do NOT pass a filename like `deck.html`, a `destinationUri`, or a `content://...` URI to `show_slide_deck`.
- Do NOT use `create_document` or `read_document` as the user-facing preview for slide decks. The slide deck should be presented via `show_slide_deck`.
- If the user asks for revisions, keep following this guide and update the deck accordingly.

---

## Conversation Before Building

Before generating any deck HTML, have a short conversation with the user to understand what they need. A great deck requires context — don't jump straight to building.

**Ask clarifying questions** when any of the following are unclear:

- **Audience** — Who is this for? (investors, team, clients, class, general audience)
- **Purpose** — What should the deck achieve? (persuade, inform, showcase, teach)
- **Key content** — What are the main points, sections, or story beats?
- **Tone / mood** — Formal, playful, premium, bold, understated?
- **Length** — Roughly how many slides? (a quick 3-slide overview vs. a 12-slide narrative)
- **Assets** — Does the user have specific images, data, or text to include?

You do NOT need to ask all of these — use judgment. If the user's request already provides enough context (e.g. "make a 5-slide pitch deck about our Q4 results for the board"), go ahead. If the request is vague (e.g. "make me a presentation about cheese"), ask 1–2 focused questions to fill in the gaps.

**The user can skip this.** If they say something like "just make it" or "surprise me", proceed with reasonable defaults. Don't block them from getting a deck.

---

## Non-Negotiable Rules

1. The deck must be a single self-contained HTML file.
2. Do not include a visible in-deck toolbar/edit UI (for example: #toolbar, #toolbarTrigger, #editToggle).
3. The embedding container owns Edit/Save controls and keyboard save handling.
4. Keep deck edit primitives (.eb, .eb-handle, .eb-resize, drag/resize logic, text editing logic) so container edit mode works.
5. The deck's visual style is set once at creation via the chosen deck style template. Do not include multiple theme CSS overrides or a theme switcher — each deck has a single look.
6. The deck must support both layout modes: vertical and horizontal. Default layout is vertical with html data-layout set to vertical.

---

## Deck Style System

When generating a new deck, choose a **deck style** first.

- A **theme** controls tokens like colors, fonts, shadows, radii, and accents.
- A **deck style** controls the actual HTML composition system: hero treatment, image placement, text density, card usage, and slide structure.

For new decks in interactive chat:

- If the user has not specified a deck style, present the style picker so they can choose one before generating the HTML.
- Use the chosen style as a hard constraint for the generated composition, not just the palette.
- Keep the deck visually consistent with that style across all slides unless the user asks for a deliberate mix.

### Custom styles ("Other")

If the user doesn't pick a predefined style and instead **describes their own** (e.g. "I want a retro 80s neon look" or "something like a Japanese woodblock print"), treat this as a valid choice:

1. Use `minimal_story` as the **structural base** (clean single-idea-per-slide layout, generous whitespace).
2. Adapt the CSS variables (colors, fonts, radii, shadows) and generation instructions to match the user's description.
3. If the description implies illustrations, call `generate_image` with a style prompt that matches what they asked for — just like the illustrated style does.
4. Confirm the interpretation briefly with the user before generating ("I'll build a deck with a retro neon aesthetic — dark backgrounds, hot pink and cyan accents, pixel-style headings. Sound right?").
5. Do NOT force the user to pick from the predefined list. The picker is a convenience, not a gate.

Available styles:

### `cinematic_noir`

- Default theme: `noir`
- Core look: full-bleed imagery, grayscale or dimmed backgrounds, oversized typography, stark monochrome contrast, and magazine-like composition
- Use when: the deck should feel dramatic, premium, image-led, or narrative
- Structural rules:
  - Prefer a full-screen hero with typography over image
  - Use split editorial slides with text on one side and image on the other
  - Keep cards sparse; rely more on image planes and typographic hierarchy
  - When using images, favor large background or half-slide treatments over small thumbnails

### `editorial_cards`

- Default theme: `editorial`
- Core look: balanced storytelling with stat grids, cards, featured boxes, timelines, and modular information design
- Use when: the deck should explain a product, company, plan, or story with clear structure
- Structural rules:
  - Mix hero slides with stat grids, chapter cards, featured boxes, and supporting sections
  - Use cards as a primary layout primitive
  - Good default when the user says "make a deck" without a stronger stylistic cue

### `minimal_story`

- Default theme: `minimal`
- Core look: restrained typography, large whitespace, fewer elements per slide, simple premium layouts
- Use when: the deck should feel clean, calm, executive, or keynote-like
- Structural rules:
  - Keep one core idea per slide
  - Use fewer cards and fewer decorative elements
  - Prefer large margins, simple image placements, and short text blocks

### `illustrated`

- Default theme: `illustrated`
- Core look: editorial card structure enriched with AI-generated hand-painted illustrations in a Studio Ghibli-inspired style — soft watercolour textures, warm natural lighting, storybook charm
- Use when: the deck should feel warm, creative, whimsical, or educational; when the user wants illustrations instead of photos
- Structural rules:
  - Uses the same card/grid/featured-box structure as editorial_cards
  - Call `generate_image` for EVERY content slide to create a supporting illustration
  - All illustrations MUST use a consistent hand-painted style: "watercolour illustration in the style of Studio Ghibli, soft light, pastel tones, hand-painted texture, rounded organic forms"
  - Place illustrations prominently: large hero images, card header images, or half-slide visuals — never small thumbnails
  - Warm parchment canvas (#faf6f0), sage green accent (#6b8f71), soft card shadows, generous rounding (20px)
  - Keep text warm and inviting

---

## What to Generate

A single HTML file that:

- Supports both vertical and horizontal full-screen slide paging with scroll snap.
- Allows text editing, drag, and resize while in edit mode.
- Has no build step and no local assets required.
- May include Town-hosted generated images using normal `<img>` tags that point at same-origin image URLs from `generate_image`.
- Uses the selected deck style as the primary composition system for slide structure.

External dependencies:

- Interact.js for drag/resize:
  - https://cdn.jsdelivr.net/npm/interactjs@1.10.27/dist/interact.min.js
- Google Fonts (Inter, Playfair Display, Space Grotesk, DM Sans, Sora)

---

## Images In Decks

You MAY include images generated by Town's `generate_image` tool inside decks.

Preferred flow:

1. Call `generate_image` when the slide would benefit from a visual.
2. Use the returned `deckImageSrc` or `deckImageHtml` in the deck HTML.
3. Prefer Town-hosted same-origin image URLs like `/api/content/image/{imageId}`.

Rules:

- Prefer generated images only when they materially improve the deck.
- Do NOT embed base64 data URLs for generated images unless absolutely necessary.
- Do NOT depend on third-party image hosts when Town-hosted URLs are available.
- Always provide meaningful `alt` text.
- Size images with normal HTML/CSS inside the deck.

Recommended patterns:

```html
<img
  src="/api/content/image/img_abc123"
  alt="Town product hero illustration"
  data-town-image-id="img_abc123"
  style="width: 100%; height: auto; object-fit: cover; border-radius: 16px;"
/>
```

```html
<div class="chapter-card">
  <img
    src="/api/content/image/img_abc123"
    alt="Routine automation concept"
    data-town-image-id="img_abc123"
    style="width: 100%; height: 220px; object-fit: cover; border-radius: 12px; margin-bottom: 16px;"
  />
  <h3>Automate Repetitive Work</h3>
  <p>...</p>
</div>
```

If you generate multiple images for a deck, keep visual style consistent across slides.

---

## Required HTML Skeleton

Use this shape:

```html
<!doctype html>
<html lang="en" data-layout="vertical">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Deck Title</title>
    <script src="https://cdn.jsdelivr.net/npm/interactjs@1.10.27/dist/interact.min.js"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;0,900;1,400;1,700&family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Sora:wght@300;400;500;600;700;800&display=swap" />
    <style>
      /* Theme vars, slide layout, components, edit-block styles, responsive rules */
    </style>
  </head>
  <body>
    <div id="progressBar"></div>
    <div id="navBar" class="nav-bar"></div>

    <section class="slide" id="slide-0">
      <div class="slide-inner fade-up">
        <!-- .eb blocks -->
      </div>
    </section>

    <script>
      // Single IIFE with navigation, edit primitives, interact.js wiring.
    </script>
  </body>
</html>
```

---

## CSS Variables

All visual properties should use CSS custom properties on `:root`. Each deck style template defines its own colors, fonts, shadows, and radii directly — there is no separate "theme" layer. The layout system uses `html[data-layout="vertical"]` and `html[data-layout="horizontal"]` selectors. Vertical is the default.

---

## Edit-Block Contract

Every movable/editable item must follow:

```html
<div class="eb">
  <span class="eb-handle">&#9783;</span>
  <div class="eb-content">Editable content</div>
</div>
```

Resizable cards:

```html
<div class="eb chapter-card">
  <span class="eb-handle">&#9783;</span>
  <span class="eb-resize"></span>
  <div class="eb-content">...</div>
</div>
```

Behavior expectations:

- Single click selects block.
- Double click enters text edit mode for content.
- Drag starts only from .eb-handle.
- Resize starts only from .eb-resize.
- Escape exits text editing first, then selection.

---

## Container-Controlled Editing

The generated deck should support these assumptions:

- Container can toggle edit mode by adding/removing body.edit-mode or by invoking deck-level toggle logic.
- Container can switch layouts by updating html[data-layout] or by invoking deck-level layout toggle logic.
- Container handles save and persistence. Deck should not store files itself.
- Cmd/Ctrl+S may be intercepted by container; deck must not force its own save UX.

If legacy toolbar markup exists, keep it hidden by default and non-essential.

---

## Slide Authoring Conventions

- Structure slides as section elements with sequential slide ids.
- Add slide-inner fade-up wrappers for reveal animation.
- Use semantic classes such as kicker, h2, subtitle, body-text, big-number, and big-quote.
- Use reusable components:
  - stat grid
  - chapter cards
  - timeline
  - pivot before/after
  - founder cards

---

## JavaScript Requirements

Include one IIFE that provides:

- nav dot generation and active-state tracking
- scroll progress bar updates
- keyboard slide navigation (when not editing) for both vertical and horizontal layouts
- edit-block selection and text editing
- Interact.js drag/resize setup for handles
- layout switching that reacts cleanly when html[data-layout] changes between vertical and horizontal
- teardown/reset when edit mode turns off

Do not include deck-specific save buttons, edit toolbar, or in-deck file persistence.

---

## Responsive Requirements

At max-width 700px:

- reduce slide padding
- collapse multi-column grids to one column
- rotate pivot arrows for vertical layout
- ensure horizontal layout still pages one slide at a time without clipping
- tighten nav-dot spacing

Use clamp(...) for major type sizes.

---

## Quality Checklist

1. Deck opens with Editorial template by default.
2. Deck opens with Vertical layout by default.
3. No visible in-deck edit toolbar UI.
4. All key content blocks are wrapped in .eb.
5. Drag, resize, and text editing function in edit mode.
6. Slide navigation works by scroll, keyboard, and nav dots in both vertical and horizontal layouts.
7. Mobile layout remains readable at <=700px.
8. Output is a single valid HTML file.
