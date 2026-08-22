# taste.md — the anti-slop system

Slop is not ugliness; it is *predictability*. A model asked for "a modern landing page" reaches for the statistical mode of its training data — and so does every other model, which is why AI pages look alike. Taste, operationally, is refusing the mode: every visible choice (color, type, layout, motion, copy) must be traceable to *this* brand and *this* story, not to the category average. This file turns that refusal into procedure.

## 1. The bans, with escape routes

The table in SKILL.md is the law; this section is jurisprudence — what to do instead, so the fix doesn't become a second cliché.

- **Side-stripe borders** → If the element needs categorical marking, use a full 1px border in a meaningful hue, a background tint at 4–8% alpha of the semantic color, or a leading glyph. If it needs *emphasis*, use scale or position, not decoration.
- **Gradient text** → Weight, size, or a single committed color. If the heading feels flat, the problem is the composition around it, not missing rainbow.
- **Glassmorphism** → Earn blur: it's justified when real content passes *under* a persistent surface (sticky nav over imagery). A static card with `backdrop-filter` over a flat background is costume jewelry.
- **Hero-metric / identical card grids** → Break the loop: one large element + two small; a real screenshot/artifact instead of an icon; a claim proven in a sentence instead of a stat. If three items genuinely deserve equal weight, differentiate their *content* (image vs number vs quote).
- **Eyebrows & numbered sections** → Sections can open with: an oversized first word, a hairline rule + heading, a change of background, an inline marginal note, a full-bleed image band. Rotate openings; repetition of any single opening is the tell.
- **Uniform reveals** → Bind each reveal to its content: text rises a few px and unblurs; an image scales from 1.04; a chart draws in; a list staggers. Same *system*, different *expression*.
- **Copy tells** → Replace abstraction with mechanism: not "Seamless integration" but "Connects to Stripe in one webhook". Never let an em dash chain replace sentence structure. Adjectives must be checkable.

## 2. Category-reflex test (run at two altitudes)

Run this during the commit-sheet, before code:

1. **First order:** Complete the sentence honestly — "An AI told to design a ⟨category⟩ site would produce: ⟨palette, type, layout⟩." If your current plan matches, discard it.
2. **Second order:** "An AI told to *avoid* that would produce: ⟨…⟩." If your plan matches *that*, discard it too. The second reflex is saturated for every major category as of 2026.
3. Commit to a deviation that is motivated by the brand (not random weirdness — an *argued* left turn).

Reference table of saturated reflexes (both orders are FORBIDDEN as landing spots; the escapes are directions, not templates — pick one and make it the brand's own):

| Category | 1st-order reflex (dead) | 2nd-order reflex (also dead) | Live escape directions |
|---|---|---|---|
| AI / dev tool | dark bg, purple-blue glow, terminal type | editorial serif on off-white "anti-SaaS" | physical-material metaphor (metal, paper, film); single drenched brand hue; diagrammatic/blueprint language with real density |
| Fintech | navy + gold, trust badges, glass cards | terminal-native dark mode | ledger/print heritage with modern motion; warm daylight photography; brutalist clarity with oversized numerals that mean something |
| SaaS B2B | blue gradient hero, 3-card features, hero-metric | cream editorial with serif headlines | product-as-hero (real UI, annotated); mono-hue drench; dense utilitarian grid done beautifully |
| Wellness | sage + beige + airy serif | clinical ultra-white minimal | saturated botanical color; dark, calm, candle-lit mood; documentary photography-led |
| E-commerce / product | white bg, symmetric product grid | full-bleed lifestyle blur | product macro-photography as texture; catalogue-as-editorial; color pulled *from the product itself* |
| Portfolio / agency | huge display type, dark, marquee | raw-HTML brutalism | one signature interactive object; film-like case-study scenes; typographic system with real editorial rules |
| Luxury | black + gold + thin serif | ultra-minimal white void | drenched jewel tone; cinematic photography with letterbox; tactile paper/foil texture |

If the project's category isn't listed, derive the two reflexes yourself — the procedure matters more than the table.

## 2.5 House tells — the third-order reflex, and the one you cannot see

§2 catches what a generic AI does for a *category*. It cannot catch what **this skill** does regardless of category, because that reflex does not feel like a reflex from the inside: each project argues its way to the choice honestly, and every project argues its way to the same one. It shows up only when you line the finished work up side by side.

Measured across the nine showcase sites, which share no subject, no client and no palette brief: **eight of nine are dark** (three landed within 0.002 of each other at mean L ≈ 0.177, at 97–98% dark pixels); a mono service font recurred to the point where the same face, Martian Mono, was independently chosen twice; amber (hue ≈ 30°) was the accent three times; the "logo left / status centre / action right" header appeared in seven of nine, and a scroll-instruction footer with a 01/05-style counter in six.

None of those is a mistake. All of them together are a signature — and a signature is exactly what a client did not order.

| # | The tell | What it looks like | Break it by |
|---|---|---|---|
| 1 | **Near-black by default** | body L < 0.25, "premium = dark" | committing to a lit page: L 0.5–0.9 with the drama in shadow, material and contrast |
| 2 | **Mono service type** | tiny mono labels in the corners, technical-drawing voice | no chrome at all, or service type in the display family at small size |
| 3 | **The status bar** | logo left · live-dot / status centre · action right | let the hero own the top edge; put navigation somewhere that costs a decision |
| 4 | **Scroll-instruction footer** | "SCROLL TO DIVE" + `01 / 05` counter | trust the page; if the affordance is genuinely needed, make it part of the art direction |
| 5 | **Amber or acid as the one accent** | hue ≈30 warm glow, or lime/neon on black | any committed hue whose reason is in the brief rather than in the palette's comfort zone |
| 6 | **Wordmark-as-hero** | the brand name set enormous, centred, filling the viewport | an image, an object, a diagram, a sentence — the peak carries meaning, not letterforms |
| 7 | **Glow as depth** | emissive bloom doing the work of lighting | real light: direction, falloff, shadow, material response |

**The rule: break at least two, deliberately, and write which two in the commit-sheet (§7).** Breaking one is coincidence; two is a decision. If a tell genuinely belongs in this project — a broadcast brand really does want the on-air dot — keep it and say why, the same way `auteur-allow` works for a ban. What is not allowed is arriving at all seven again without noticing.

The test is mechanical: put your hero beside the last thing this skill built. If a stranger could tell they came from the same studio, you have found the signature, not the direction.

## 3. Color

**Strategy before swatches.** Choose a commitment tier in the commit-sheet:

- **Restrained** — tinted neutrals + one accent ≤10% of surface. Default for product UI.
- **Committed** — one saturated color carries 30–60% of the surface. Default for identity-driven pages. *The most underused tier and the fastest way to not look AI.*
- **Full palette** — 3–4 named roles used deliberately. Campaigns, data-rich brands.
- **Drenched** — the surface IS the color. Brand heroes, launch pages, cinema scenes.

**Rules:**
- Work in OKLCH. Tinted neutrals: 0.005–0.015 chroma *toward the brand hue* — never "warm by default".
- The warm-cream band (L 0.84–0.97, C <0.06, hue 40–100) is the saturated AI default of 2026. "Warm brand" is expressed via accent, typography, imagery — not via beige body.
- Dark vs light is never a default. Write one sentence of physical scene ("who uses this, where, under what light, in what mood") and let the scene force the answer. If it doesn't force it, the sentence isn't concrete enough.
- Gray text on colored background looks washed out → use a darker shade of the background's own hue, or text-color at reduced alpha.
- Contrast: body ≥4.5:1, large ≥3:1, placeholders ≥4.5:1. When close, darken toward ink. "Light gray for elegance" is the #1 readability failure.
- Gradients: only within one hue family or between adjacent hues that both belong to the brand; both-stops-purple-blue (hue 250–290) is banned; grays don't gradient.
- **The subject must not dissolve into its own field.** A light hero on a light ground — ice on snow, a white product on a white sweep, pale type over a pale wash — reads as a smudge, and no amount of light fixes it. Put the subject *below* the field in value and let only its edges, seams and highlights carry brightness. On a scene that has this problem, this one correction is worth more than every other fix combined; check it on the greyscaled screenshot, where it is unmissable.
- **Cheerful drift is the reflex you will not notice.** Handed a near-monochrome reference, a model still returns a friendlier, bluer, more saturated version of it — saturated sky-blue reads as game graphics, not as a photograph, and the drift survives even with the reference on screen. When the reference is desaturated, record its chroma in the commit-sheet as a number (mean OKLCH C) and check your own screenshot against it, because "looks about right" is exactly the judgement that drifted.

## 4. Typography

**Pair on a contrast axis, never on similarity.** Two similar-but-not-identical sans faces read as a mistake. Working axes: serif display + sans text · geometric + humanist · mono + serif · high-contrast display serif + grotesque. One family in 3+ weights is always a safe committed choice.

**Selection procedure** (don't keep a fixed shortlist — shortlists become the next Inter):
1. Name the voice in 2 adjectives from the brief (e.g. "engineered, warm").
2. On Google Fonts, filter by classification matching the *display* adjective; pick 3 candidates you can defend; test the actual headline copy at target size.
3. Text face: prioritize x-height, open apertures, and a real italic. Verify tabular figures if numbers matter.
4. Variable font when you need >2 weights (one file, animatable weight for kinetic type).

**Numbers:** display clamp max ≤6rem **for headings inside prose flow** — a wordmark, a type-led hero, or a scene where oversized type IS the subject is exempt and routinely runs 100–140px at 1440 (refscout the top tier and you will measure exactly that). The ceiling exists to stop 200px of Inter standing in for an idea, not to stop a typographic hero; if you exceed it, the commit-sheet has to say the type is the signature. letter-spacing ≥−0.04em on display, slightly positive on small caps; body 65–75ch; line-height: display 0.95–1.1, body 1.4–1.6; `text-wrap: balance` on h1–h3, `text-wrap: pretty` on prose. Fluid type via clamp() with a rem base so zoom works.

**Kinetic typography is architecture, not decoration:** oversized text may BE the hero (cheapest wow that exists — zero asset weight). If type is the hero, assets can wait; see direct.md.

## 5. Layout

- Grid is a starting field, not a cage. Commit ONE named grid-break per page minimum (from the commit-sheet): an overlap (image crosses a section boundary), an asymmetric split (5/7, 4/8 — not 6/6), a diagonal flow, a full-bleed interruption between contained sections, an element that escapes its column.
- Whitespace is a material: vary section padding meaningfully (a tight dense section makes the following airy one land). Uniform `py-24` everywhere is rhythmless.
- Cards are the last resort, not the first. Ask: would a table, a list with strong typography, an annotated image, or plain prose serve better? Nested cards are always wrong.
- Flexbox for 1D, Grid for 2D; `repeat(auto-fit, minmax(280px, 1fr))` for breakpointless grids.
- Semantic z-index scale (dropdown < sticky < backdrop < modal < toast < tooltip); never 999.
- Container queries for components that live in variable-width slots; viewport queries for page chrome.

## 6. Texture and depth

Flat-solid-everything and blur-everything are both defaults. The live middle: film grain / noise at 2–4% opacity over large color fields (kills the "vector emptiness" of AI pages); hairline rules (1px, low-contrast) as structure; shadows only when something truly floats (one consistent light direction, larger blur than offset); real photographic/generated texture over CSS-only decoration.

## 7. Copy

Headlines state a mechanism or an image, not a superlative. Subheads carry the proof. Microcopy is a butler: quiet, precise, present-tense. Buttons say what happens ("Get the report", not "Learn more"). Error text says what to do next. No "Revolutionize / Seamless / Effortless / Unleash / Elevate", no em-dash chains, no "BRAND. MOTION. SPATIAL." strips. If a sentence survives with an adjective deleted, delete the adjective.

## 8. Self-check before verify

Before running the pipeline in verify.md, answer honestly:
1. Could a stranger guess the category from palette alone? (If yes — reflex won.)
2. Is there ONE element a visitor would describe to a friend? (If no — no signature.)
3. Do any two sections open identically? (If yes — vary one.)
4. Does every animated thing have a reason from motion.md's list? (If no — cut it.)
5. Would deleting the third font/color/pattern hurt? (If no — delete it.)
