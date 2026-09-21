# Sahil personal-brand imagery standard (sahil_twitter + sahil_linkedin)

## VALIDATED MODEL (2026-06-16) — supersedes single-palette DNA below

Proven end-to-end on FAL (~£2.70 of trials). The reliable quality mechanism:

**1. DNA = the CRAFT thread, NOT a colour scheme.** The cohesive brand thread is:
analog texture (grain/halftone/print/scanline), strong type hierarchy (distressed
condensed display + clean labels + mono), designed information DENSITY, restrained
composition, "designed not generated" feel. Colour is NOT fixed.

**2. Palette SYSTEM (rotated for variety — fixes the "one-trick pony" problem).**
A bank of distinct on-brand palettes the selector rotates so consecutive posts look
different. Validated set (output/palette/): cyber-neon (near-black + #3847FF +
magenta), acid-duotone (black + cobalt #1D4ED8 + acid yellow #E3FF00), synthwave
(purple #1A0B2E + pink #FF6AC1 + orange #FF8A3D), blueprint-mono (navy #0A1A2F +
cyan #5FE0FF), warm-editorial (parchment #F2E8D5 + charcoal + red #C0392B, LIGHT).
Expandable. Light palettes are allowed (warm-editorial is light) — DNA is NOT
mandatory-dark.

**3. LAYOUT = baoyu infographic-layouts** (the dense, designed templates that make
output "post-worthy"): iceberg, bridge, funnel, pyramid, comparison-table,
scale-balance, timeline-horizontal, mind-map, venn, fishbone, journey-path,
layers-stack, etc. (~/content-references/baoyu-screenshots/infographic-layouts/).

**4. STYLE = baoyu styles** for treatment (cyberpunk-neon, technical-schematic,
bold-graphic, aged-academia, retro/synthwave, etc.).

**5. TECHNIQUE = dual-anchor TRANSPLANT (the keystone).** Never let the model
invent a layout — it defaults to cheap "glow on black". Feed nano-banana-pro/edit
TWO anchors: (a) a baoyu LAYOUT exemplar for structure/density, (b) a STYLE/palette
exemplar (baoyu or Sahil ref) for the look — then swap in the topic + exact labels +
explicit palette hexes. Aspect 4:5, resolution 2K.

**6. Mandatory post-process** per palette (dark: scanline+grain+vignette; light:
paper-texture+grain) via `postprocess.py` effect chain.

Anchor pool = Sahil's 109 curated standouts (`~/content-references/Referencecontent/`,
ALL gold) + 124 baoyu exemplars. Mix-it-up = rotate palette × layout × style + the
non-infographic baoyu families (cover, comic, article-illustrator) and Sahil's
hero/tarot/poster refs. LinkedIn = lean cleaner palettes (blueprint, warm-editorial,
mono) + calmer layouts; same DNA, never sterile.

Demo tools: tools/bakeoff*.py, validate_standard.py, transplant_demo.py, palette_demo.py.

---


> Modelled on the nous-branding skill structure. Built from Sahil's 109-image
> reference set (`~/content-references/Referencecontent/`). ONE visual DNA, a
> bank of style LANES for variety, reference-anchored generation, mandatory
> post-processing, and a compliance checklist. Workhorse = skill infographics,
> but the lanes exist so the feed never reads as "infographic after infographic".

## 1. Visual DNA (constant across both brands, every lane)

> NOTE: the single fixed palette below is SUPERSEDED by the VALIDATED MODEL at
> the top — DNA is the craft thread; colour is a rotated palette system. The
> essence/typography/texture/composition guidance here still holds.

**Essence:** "cyber-classical research lab" — myth + cyberpunk + retro-tech +
analog print. Intellectual, gritty, underground, scroll-stopping. NOT corporate,
NOT clean SaaS, NOT formal.

| Token | Value |
|-------|-------|
| Background | near-black `#00000E` / very dark, deep space |
| Signature accent | electric blue `#3847FF` |
| Secondary accents | violet/lavender `#BDA6FF`, magenta/pink `#FC57A0`-ish, cyan |
| Warm accents (sparing) | burnt orange `#D6825A`, gold `#E6C666` for HUD/constellation lines |
| Text on dark | off-white `#E6E6E6` |
| Texture (required) | grain + halftone + scanlines + paper fiber + ink bleed — "raw, analog, imperfect" |
| Typography | heavy distressed/condensed display (uppercase) → clean sans labels (Inter/IBM Plex) → mono for code (JetBrains Mono) |
| Lighting | high-contrast chiaroscuro, dramatic spot, neon edge glow, light beams |
| Composition | restrained — subject breathes, deliberate negative space, 1-2 accent colours per piece, limited text blocks |

Sahil's set skews **more violet/magenta** than Nous's blue-dominant palette — keep
that. Gold/amber appears on mystical + propaganda lanes.

## 2. Style lanes (the variety engine)

Each lane = a reusable skeleton. The selector rotates lanes (no repeat within the
recency window — reuses `topic_usage_log`) so the feed stays fresh. Anchor refs
cited by contact-sheet index (see `~/content-references/_sheets/`).

### Lane A — Cyber-classical hero scene
Atmospheric character/agent or symbolic object in a neon-lit scene; volumetric
light, grain. Refs: [0-3], [29], [99], [104-105], [100], [108].
Text: minimal (title only, or textless + post overlay). Use: launches, manifestos,
big-idea posts. **No "Nous Girl"** — use abstract agents / symbolic subjects.

### Lane B — Tiered "levels / ladder" infographic
Dark panel stack, neon, numbered LEVEL 1..5 or rungs. Refs: [27] Autonomy Ladder,
[21-25] LEVEL series, [103] Five Levels, "Dark Factory".
Text: heavy, structured, must be crisp. Use: maturity models, progressions,
comparisons. **High reuse — a signature series format.**

### Lane C — Dense technical dashboard / skill sheet
Blueprint or dark dashboard with modules, labels, mono callouts. Refs: [4] Simplify
Skill (light blueprint), [24] Self-hosted Doc Dashboard, [86][87] feature cards,
[106] problem/fix/flow/robust 4-panel, [107] false-signal/why/fix/validation table.
Text: dense, granular — the hardest text test. Use: how-I-built-X, architecture,
skill breakdowns. **Workhorse for teaching content.**

### Lane D — Retro propaganda / vintage poster
WWII-poster / vintage-packaging pastiche. Refs: [80] GOAL MODE, [84] OLD BAY,
[41] Native Windows. Text: bold display headline. Use: opinionated takes,
announcements, "shipped" moments. High scroll-stop.

### Lane E — Comic / manga panel
Retro manga or pop-art comic strip. Refs: [88] Ask Hermes (90s manga), [81] pop-art,
[42] strip. Text: speech bubbles / captions. Use: narrative "how it went" posts,
humour, build-in-public storytelling.

### Lane F — Product mockup / packaging
Brand-as-product: cereal box, tin, sneaker, etc. Refs: [54][55] ChatGPT/Gemini
cereal, [56] BYTE, [84] OLD BAY, [74] shoes. Text: packaging copy. Use: playful
product/feature framing, memes-with-class. High scroll-stop.

### Lane G — Tarot / mystical card
Ornate border, celestial figure, symbolic. Refs: [33][34] Library of Babel,
[81] Synapse, [102] NOUS card, [108] celestial. Text: title + small caption. Use:
philosophy, "the ghost in the machine" abstract essays. Striking, rare.

### Lane H — Clean data viz
Minimal chart/badge on dark. Refs: [19][37][70][94] charts, [25] DSS badge.
Text: chart labels. Use: results, benchmarks, metrics. **Most LinkedIn-friendly.**
Candidate for £0 deterministic code-render later (Pillow/SVG, ARM-safe).

## 3. Brand split — ONE DNA, different lane weighting

Same DNA, same palette, same texture. The difference is lane mix + restraint,
NOT formality (Sahil: formal/corporate is the wrong target in the AI era; the
edgy aesthetic IS the differentiator and the scroll-stopper).

- **sahil_twitter** — full range. Heavier on D/E/F/G (playful, punchy, weird),
  plus B/C/A. Loud, fast, irreverent.
- **sahil_linkedin** — same DNA with "a dash more polish": lean on C/B/H/A
  (skill sheets, ladders, data, hero), lighter on cereal-box/comic. Slightly
  calmer composition, tighter type, but still grainy and edgy — never sterile.

## 4. Reference-anchored generation (the airtight + low-spend mechanism)

- Model: **nano-banana-pro** (`fal-ai/nano-banana-pro`). Reference/edit endpoint
  for img2img anchoring (verify exact id: `fal-ai/nano-banana-pro/edit` or image
  input param) — feed 1-3 lane anchors from `~/content-references/`.
- Prompt pattern (from nous-branding): **state what to PRESERVE, then what to ADD.**
  A reference is NOT a style embedding — always restate DNA tokens + lane skeleton
  + the post's real content in text.
- Always pass the palette hexes inline. Always name the texture set.

## 5. Mandatory post-processing (deterministic, £0)

"The raw generated image is never the final deliverable." Apply analog grain /
halftone / scanline / ink-bleed pass after generation (extend existing
`postprocess.py`; upstream ships `scripts/postprocess.py` with imprint/nous/
standard modes, intensity 0.45-0.8). This is what makes output feel on-brand and
is reproducible at zero cost.

## 6. "What is NOT on-brand" (anti-patterns)

- Flat, clean, glossy corporate/SaaS look (no grain) — wrong.
- Generic stock-photo or clip-art iconography (the old seedream output) — wrong.
- Literal "Nous Girl" mascot — do not use.
- Cluttered composition, >2 accent colours, walls of text — wrong.
- Gibberish / garbled labels — fails the text gate; regen or escalate.
- Formal/sterile LinkedIn corporate stock — explicitly wrong.

## 7. nano-banana / FAL pitfalls (OUR addition — upstream covers only OpenAI/DALL-E/ComfyUI)

- nano-banana is preset-size only for some aspects (`_PRESET_FALLBACK` in
  fal_client) — don't send custom WxH where it rejects it.
- It ignores `negative_prompt` (reasoning model) — bake exclusions into the prompt.
- Slow model → sync timeout 220s; queue fallback exists.
- £0.12/img — hero-only (1/post) keeps runtime ~£7/mo under the £10 cap.
- Text renders well at headline + short-label scale; for VERY dense sheets, split
  into fewer labels or consider code-render (Lane H).

## 8. Compliance checklist (airtight gate, per image)

- [ ] Near-black/dark background, restricted palette, electric-blue or violet accent
- [ ] At least one analog texture visibly applied (post-processed)
- [ ] High contrast, subject breathes, ≤2 accents, limited text blocks
- [ ] All baked text legible, no gibberish (OCR via text_integrity)
- [ ] Correct lane skeleton for the post intent
- [ ] No Nous Girl, no corporate-stock look
- [ ] LinkedIn: a touch more restraint, still edgy

## 9. Open / next
- Organise the 109 refs into per-lane folders so img2img can pick anchors.
- Verify nano-banana reference/edit endpoint id; wire reference-anchored path.
- Build lane selector + prompt-composer; wire post-process pass.
- Validate a handful of anchored generations (within £0.62 now, or after 1 Jul
  ledger rollover) before going live.
