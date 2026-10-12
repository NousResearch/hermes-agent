# Visual Language: Color, Image Treatment, Typography, Tone

When to load: while resolving the manifest's substrate/palette/plate roles, writing image-treatment or typography prompt paragraphs, or inventing display text. Exact hex values and IDs live in `design-system/`; the catalog wins over this prose.

## Color System

Default to controlled two-ink. Give the dominant and accent plates separate content roles before composing; never use the second ink merely to decorate the page. Switch to pure one-ink only when the user explicitly requests one ink, monochrome, or one named ink without a second color. The paper substrate does not count as an ink.

- **Substrate:** Neutral White `#FAFAF7` suits crisp cultural, social, event, and colorful image-led work; Cool Gray `#E9E9E5` suits architecture, technology, charcoal-led systems, restrained branding; Pale Beige `#F5F1E8` suits tactile, food, travel, intimate, archival, or explicitly nostalgic subjects.
- **Contemporary default:** the substrate is clean and neutral, not yellowed. Halftone and plate logic describe reproduction, not an era. No fading, sepia, antique props, distressed borders, or aged-paper staining unless the user asks for retro/vintage/archival aging.
- **Plate limit:** two assigned printing plates by default, never more than two; explicit one-ink requests use one plate.
- **Ink density:** darker coverage may appear near-black and sparse halftones may appear pale — density changes, not extra inks.
- **Paper exposure:** keep the paper visible. Never tint the whole page into a digital monochrome wash.

### One-Ink Palette

- **Cobalt / Ultramarine** `#2148B8` — default for technology, knowledge, cities, music, cultural subjects.
- **Royal Blue** `#2058D4` — youth culture, fashion, movement, energetic editorial.
- **Botanical Green** `#008A4B` — botanical, ecological, archival, explicitly green subjects.
- **Mint Green** `#5EB783` — observation journals, soft natural subjects, quiet editorial photography.
- **Terracotta Orange** `#C65F38` — classical art, food, travel, summer, tactile objects.
- **Signal Red** `#C83232` — declarations, music, events, civic or public culture.
- **Aubergine** `#63365F` — literature, cinema, night, intimate cultural subjects.
- **Charcoal** `#30343A` — architecture, photography, research, restrained publications.

### Two-Ink Recipes

Use a known pair rather than improvising arbitrary colors:

- **Powder Blue + Signal Red** `#9EB8D3` + `#C83232` — guides, announcements, information-heavy pages.
- **Cobalt + Terracotta** `#2148B8` + `#C65F38` — travel, summer, food, lifestyle.
- **Botanical Green + Oxblood** `#008A4B` + `#8F3434` — plants, natural wine, bookstores, archives.
- **Charcoal + Signal Red** `#30343A` + `#C83232` — architecture, exhibitions, reports, conceptual work.
- **Electric Blue + Carbon** `#173AE3` + `#242321` — high-contrast cultural events, image-led pages.
- **Mint Green + Charcoal** `#5EB783` + `#302D2E` — journals, essays, observations, long-form reading.
- **Ultramarine + Safety Orange** `#263E99` + `#E55D2B` — movement, objects, youth culture, active urban subjects.
- **Cyan + Brick Red** `#159DDA` + `#B64032` — repeated products, exhibitions, playful information systems.
- **Tangerine + Slate Blue** `#E46C2D` + `#4773A5` — markets, festivals, illustrated notices, large typographic compositions.

Assign each plate a role before composing. These constraints keep the result mechanically printed rather than digitally color-graded.

## Image Treatment

Convert photographs and illustrations into the selected ink plate(s) plus substrate. Choose reproduction intensity from the subject instead of automatically aging every image:

- crisp screening or clean plate separation for contemporary work; coarse halftone, risograph grain, cyanotype-like exposure, photocopy breakup, or newspaper screening only when materially useful or requested;
- visible dots at close range, recognizable subject at thumbnail scale;
- clipped highlights where paper shows through, dense shadows where ink pools;
- optional mild ink bleed, uneven coverage, scan noise, paper fibers, or 1–2 mm registration drift between plates; fewer imperfections for contemporary/clean work;
- medium contrast; no glossy photographic depth.

When no source image is supplied, do not default to a polished photorealistic hero person or complete stock-photo figure. Prefer 2–4 identifying anchors (a hand on a handlebar, one bent leg, a wheel arc, loose fabric, hair direction). Build the subject from a partial editorial crop, simplified screened fragment, and one ordinary in-between gesture. Avoid advertising poses, victory gestures, athletic hero angles, catalog-style full bodies, and the safe headline-left/photo-right split unless asked.

### Abstract Looseness

When representation is `abstract symbol extraction`, transform the supplied image into a small visual vocabulary rather than filtering the whole photograph:

1. Name 2–4 **identity anchors** that keep the subject recognizable; preserve their relationship, not their photographic detail.
2. Convert the anchors into **one dominant mass**, **one structural contour**, and **one repeated rhythm** — flat plate shapes, broken hand-drawn lines, short strokes, dots, or paper cutouts; omit incidental scenery.
3. Let paper replace at least 35% of the source scene. Crop one anchor at a page edge; let one type or line element cross it. Abstraction must create active space, not merely blur or posterize.
4. Keep the abstract geometry deterministic; looseness comes only from slightly irregular contours, uneven repeated marks, and seeded controlled imperfections. Do not randomly move anchors between retries.
5. At thumbnail scale, at least two identity anchors must still communicate the original subject without the caption.

For complementary duotone abstraction: dominant ink carries structure and rhythm; accent ink is reserved for one identity anchor or one annotation — never distributed evenly.

### Controlled Chance

Keep composition, wording, palette, and hierarchy deterministic; introduce looseness only in the reproduction layer. Contemporary work: 0–2 restrained effects; tactile/vintage/archival work: 2–3 effects from `design-system/imperfections.json`, using a stable seed hashed from subject + exact text + palette + layout, preserved across retries.

- Uneven ink density, dry-edge breakup, halftone drift, registration drift, or one broken manual gesture create the analog variation.
- Apply variation to large type, image plates, solid shapes, or the single gesture family; never distort microcopy or factual text.
- Keep all effect values inside catalog ranges; the same resolved input reproduces the same marks and offsets.
- In one-ink work, registration drift may appear only as a pale second impression of the same ink — never another color.
- Never use controlled chance to move the dominant object, change line breaks, alter the grid, or paper over an unresolved composition.

## Typography

Typography is a responsive cast, not a fixed house font. Read `design-system/typography.json` and choose one primary display skeleton from the subject, wording, and information structure — literary serif, wide cultural grotesk, compressed civic sans, engineered program type, rotated display, or word-as-object. Consistency across a set comes from ink, spacing, plate logic, and disciplined microtype, not from forcing every image into the same serif-plus-mono treatment.

Role guide: Literary for intimate/quiet subjects; Cultural Grotesk for music and contemporary culture; Condensed Civic for public events; Programmatic when dates or structured facts lead; Rotated Display for bold covers; Handwritten Interjection only as a secondary human voice (never dates, locations, essential facts, or long copy); Typographic Object when the phrase itself is the image.

Build each image with one primary display voice and one functional support voice; a third voice only as one short handwritten interjection. Choose at most one typographic behavior per image: natural lowercase sentence breaks (intimate); wide/interlocked capitals (music, movement, culture); compressed stacked lines (public events); tabular numerals with unequal ruled blocks (programs); one 90° rotation or vertical title (bold cover); one circled handwritten aside (invitation/personal note); oversized cropped letterforms (words as the dominant object). Do not repeat the same display category across every item of a multi-scene request unless the user asks for a unified campaign.

Rules:

- One dramatic scale jump: largest text 5–12x the microcopy size.
- Lowercase for intimate statements, uppercase for public declarations.
- Display copy 2–8 words; all other copy sparse.
- Invented words default to natural English even when the request is in another language. Preserve user-supplied wording exactly; do not translate unless asked.
- Exact readable wording only when supplied or concept-carrying; otherwise plausible microtype as texture — never invented organizations, URLs, sponsors, or event facts.
- No gradient type, outline effects, drop shadows, inflated 3D letters, or generic luxury-fashion spacing.
- Oversized type is valid only as the selected focal event; otherwise it stays subordinate to the image/object/crop/overprint event.
- In relaxed work, one typographic move may be audacious while all supporting type becomes sparse and functional.
- Never copy a reference's distinctive lettering, exact line breaks, or word arrangement — translate only the broader contrast, orientation, and voice relationship into an original solution.

## Communication Tone

Write like an independent cultural poster, field journal, or community print notice: terse, observant, romantic, free-spirited without sentimentality; human and specific rather than inspirational; quiet confidence, dry wit, or factual clarity; no sales language, CTA, hype, productivity slogans, or brand-manifesto voice.

For summer, movement, travel, leisure, music, and night subjects, romantic freedom is the default register — expressed through a physical sensation, an open direction, an unhurried gesture, or a small relationship between subject and space. Favor fresh English fragments (an observation or invitation), never a generic motivational slogan. For factual, civic, scientific, or archival subjects, clarity overrides this romantic default.

For romantic, intimate, nostalgic, or poetic prompts, express feeling through one observable relationship: two figures sharing one edge, an object carrying signs of use, a crop implying closeness, a small distance between forms. Do not default to string lights, wine glasses, fluttering fabric, stars, flowers, sunset silhouettes, or cinematic haze — those describe a romance category; a specific relationship creates romance while preserving graphic directness. Never reuse wording visible in reference images or repeat a stock phrase across unrelated outputs.
