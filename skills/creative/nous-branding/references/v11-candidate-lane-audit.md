# v11 Candidate Lane Audit — Unexpanded Nous Branding References

## Purpose

This audit inspects `~/Desktop/Nous-branding-refs/` for coherent visual lanes that are present in the reference folder but not yet fully expanded into prompt templates/reference-ordering rules.

Current expanded systems:

- **v9 Underground Technical-Art Imprint** — xerox poster, minimal stipple, industrial duotone grid, manual/letterpress, blue registration character
- **v10 NOUS Analog Cyber-Manga** — Acid Signal, Portal Minimal, Future Halftone, Blueprint Scene, Op-Art Poster
- **Legacy luminous PNW/Hermetic/Cyberpunk** — forest/mist, portals/orbs, neon energy, human-AI unity

This file is a **candidate backlog**, not a fully validated generation pipeline. Expand these into proper v11/v12 lanes only after generating and QAing examples.

---

## Audit Artifacts

Generated contact sheets:

- Full folder audit: `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-branding-refs-audit-contact-sheet.png`
- Unique unexpanded subset: `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-branding-refs-unexpanded-unique.png`

Duplicate groups found:

- `content.PNG` = `anime-nous-acid-mascot-card.png`
- `image 2.PNG` = `anime-nous-manga-xerox-portrait.png`
- `image.PNG` = `anime-nous-opart-poster.png`
- `Nous-girl-portal-reference .png` = `anime-nous-portal-silhouette.png`
- `Nous-bar-meet-reference.png` = `anime-nous-future-halftone.png`
- `style-blue-registration-character.jpg` = `anime-nous-blueprint-scene.jpg`

---

## High-Priority Candidate Lanes

### 1. Retro Media Interface Relic

**Primary reference:** `Hermes-player-reference.png`

**Visual cues:**
- old desktop/media-player UI chrome
- playback controls, menu bars, utility-window framing
- monochrome or limited-palette collage inside interface space
- degraded photocopy / CRT / scanner texture
- software nostalgia, media artifact, executable relic

**Why it is new:**
Distinct from v9 technical imprint because the interface itself is the artifact. Distinct from legacy cyberpunk because it is not neon worldbuilding; it is software-window archaeology.

**Prompt cues:**
`retro desktop media player interface`, `degraded software relic`, `menu bar chrome`, `playback controls`, `photocopied UI screenshot`, `CRT scan haze`, `NOUS as player title or window label`, `collage inside application frame`

**Priority:** High

---

### 2. Analog Newspaper Direct Response

**Status:** Expanded in `references/v11-analog-newspaper-direct-response.md`.

**Primary reference:** `Hermes4.3-reference .png`

**Visual cues:**
- newspaper page / mail-order ad layout
- serif headlines, price blocks, coupons, dense ad copy
- washed paper texture and gray newsprint halftone
- product photo or cutout embedded into editorial copy
- ironic direct-response / classifieds energy

**Why it is new:**
Related to v9 manual/letterpress, but the grammar is newspaper advertising and direct-response copy, not technical manuals or punk posters.

**Prompt cues:**
`analog newspaper ad`, `mail-order direct response layout`, `serif headline`, `price block`, `coupon box`, `washed newsprint`, `dense microcopy`, `NOUS product announcement as classified advertisement`

**Priority:** High

---

### 3. Ornate Hermetic Print Artifact

**Status:** Expanded in `references/v11-ornate-hermetic-print-artifact.md`.

**Primary reference:** `Hermes-jam-reference.png`

**Support refs:** `Hermes-classical-reference .png`, `Hermes-agent-ollama-reference .png`, `Nous-psyche-reference.png`

**Visual cues:**
- Victorian engraving / occult print / alchemical broadside
- seals, stamps, ornamental frames, medallions
- parchment, aged paper, layered institutional marks
- centered emblem or mythic figure
- dense border architecture

**Why it is new:**
Legacy Hermetic covers the subject matter, but not the print-system: ornate engraving, seals, and archival artifact construction.

**Prompt cues:**
`ornate hermetic print artifact`, `Victorian engraving`, `alchemical broadside`, `institutional seals`, `ornamental border`, `aged parchment`, `NOUS as stamped archival mark`, `mythic emblem centered in dense frame architecture`

**Priority:** High

---

### 4. Cyanotype Surveillance City

**Status:** Expanded in `references/v11-cyanotype-surveillance-city.md`.

**Primary reference:** `Computersaysno-reference .png`

**Visual cues:**
- blue monochrome / cyanotype photographic look
- foggy city or night skyline
- blown-out lights and washed contrast
- Polaroid / film border
- CRT/object silhouette, surveillance still, urban anomaly

**Why it is new:**
Adjacent to legacy cyberpunk, but more photographic, urban, and analog than luminous illustration.

**Prompt cues:**
`cyanotype surveillance city`, `foggy urban night`, `blue monochrome film still`, `Polaroid border`, `washed CRT contrast`, `NOUS as stamped evidence label`, `analog surveillance photograph`, `urban anomaly documentation`

**Priority:** High

---

### 5. Network Diagram Primitive

**Status:** Expanded in `references/v11-network-diagram-primitive.md`.

**Primary reference:** `Hermes-together-reference.png`

**Support refs:** `Nous-Discord-reference .png`, `Hermes-mind-reference.png`

**Visual cues:**
- electric blue field
- white/pixel human or agent glyphs
- dense connection graph / social network lattice
- primitive system-diagram figures rather than polished illustration
- low-res civic/infrastructure diagram energy

**Why it is new:**
Related to legacy human-AI unity, but the style is primitive diagrammatic network language, not cinematic mysticism.

**Prompt cues:**
`primitive network diagram`, `white pixel figures on electric blue`, `dense connection lattice`, `low-res system map`, `agent society diagram`, `NOUS as infrastructure label`, `screenprinted civic network poster`

**Priority:** High

---

### 6. Veiled Classical Artifact

**Status:** Expanded in `references/v11-veiled-classical-artifact.md`.

**Primary reference:** `image 3.PNG`

**Visual cues:**
- close-cropped classical marble/statue face
- partial concealment by veil, mesh, or scrim
- museum-fragment / archeological scan mood
- cold green-blue monochrome
- damaged photographic surface, halftone grain

**Why it is new:**
Not simply legacy Hermetic. It is intimate, fragmentary, archeological, and artifact-scanned rather than radiant/celestial.

**Prompt cues:**
`classical marble bust close-up`, `veil-like xerox scrim`, `museum artifact scan`, `weathered stone face`, `blue-green photocopy texture`, `occult archival fragment`, `high contrast halftone grain`

**Priority:** High

---

## Medium-Priority Candidate Lanes

### 7. Planetary Broadcast Identity

**Status:** Expanded in `references/v11-planetary-broadcast-identity.md`.

**Primary reference:** `image0 2.JPG`

**Support refs:** `Hermes-together-reference.png`, `image0.JPG`, `Nous-Discord-reference .png`

**Visual cues:**
- cropped Earth/globe/planet disk
- large institutional vertical or sideways `NOUS` typography
- teal/navy/ice palette
- space-age public-service identity
- satellite-era transmission-card framing

**Why it is new:**
Extends the brand from internal manuals/posters to planetary-scale public identity. It is currently under-supported by only one strong direct ref.

**Prompt cues:**
`planetary communications poster`, `cold war broadcast identity`, `cropped earth disk`, `oversized vertical NOUS wordmark`, `teal institutional print`, `satellite-era graphic design`, `halftone globe`

**Priority:** Medium-high

---

### 8. Manufactured Multiples

**Status:** Expanded in `references/v11-manufactured-multiples.md`.

**Primary reference:** `image0 3.JPG`

**Support refs:** `style-industrial-duotone-grid.jpg`

**Visual cues:**
- repeated branded capsules/cylinders/canisters
- industrial product array / production-line rhythm
- blue/yellow halftone print
- commodity object meets ritual object
- product-world / packaging / merch language

**Why it is new:**
Could be treated as a sub-lane of v9 Industrial Duotone Grid, but if the brand needs packaging/product/merch prompts this should split out.

**Prompt cues:**
`rows of branded capsules`, `industrial product array`, `repeating NOUS seals`, `blue yellow halftone print`, `factory documentation`, `mass-produced ritual objects`, `grainy flash photograph`

**Priority:** Medium-high

---

### 9. Motorsport Sponsor Field Research

**Status:** Expanded in `references/v11-motorsport-sponsor-field-research.md`.

**Primary reference:** `Nous-car-reference.png`

**Visual cues:**
- rally/motorsport sponsor livery, not generic car photography
- dust, speed, real-world motion, trackside or field-site atmosphere
- `NOUS` treated as a credible team/sponsor mark on doors, hood, quarter panels, pit boards, crew jackets, vinyl banners
- sponsor-placement energy: decals, number plates, auxiliary logo stack, safety markings, technical inspection stickers
- field research / expedition team tone, but framed through motorsport sponsorship language

**Why it is new:**
Completely separate from the poster/print systems. The lane should communicate **sponsor energy** — NOUS as the backer/team identity in a kinetic field operation — rather than merely “a car with a logo.” It is currently under-supported by only one strong image.

**Prompt cues:**
`NOUS as motorsport sponsor`, `rally car livery with credible decal placement`, `team sponsor graphics on doors and hood`, `number plate and auxiliary logo stack`, `dusty motorsport field photography`, `trackside banner with NOUS`, `research expedition rally team`, `analog film grain`, `not a luxury car ad`

**Priority:** Medium

---

### 10. Broadcast Glitch Summons

**Primary references:** `Nous-Discord-reference .png`, `Hermes-player-reference.png`

**Visual cues:**
- anonymous suited figure / numeric or obscured face
- VHS/glitch noise, dark event-promo aesthetic
- broadcast title card, paranormal transmission
- event/community announcement language

**Why it is new:**
It is adjacent to v9 imprint and retro media, but focuses on ominous broadcast/event-promo identity.

**Prompt cues:**
`broadcast glitch summons`, `anonymous figure with obscured numeric face`, `VHS transmission poster`, `dark event promo`, `NOUS community signal`, `glitch noise and scanline decay`

**Priority:** Medium

---

### 11. Halftone Office-Hours Heraldry

**Primary reference:** `Nous-psyche-reference.png`

**Visual cues:**
- armored/heraldic figure
- dense bitmap/risograph texture
- navy-white duotone
- minimal event copy
- stark emblematic tone

**Why it is new:**
Bridges Hermetic + print promo, but narrower than Ornate Hermetic Print Artifact. Could become an event-card lane.

**Prompt cues:**
`halftone office-hours heraldry`, `armored emblem figure`, `navy white risograph`, `minimal event copy`, `NOUS as heraldic event mark`, `stark institutional promo card`

**Priority:** Medium

---

## Low-Priority / Existing Sub-Lanes

These are useful references, but not independent lanes yet:

- `image 4.PNG` — v9 Minimal Stipple Field / specimen framing
- `image 5.PNG` — v9 technical plotter / measurement-grid sub-lane
- `image 6.PNG` — v9 blueprint technical diagram / annotation sub-lane
- `image0 4.JPG` — v9 manual + legacy Hermetic hybrid, but not new
- `image0.JPG` — legacy luminous PNW forest/canopy lane
- `IMG_8810.JPG` — strong v9 Manual / Letterpress Cover ref, not new
- `Hermes-reference-light.png` — legacy luminous mythic blueprint, not new
- `Hermes-mind-reference.png` — legacy neural occult interface, not new
- `Hermes-agent-ollama-reference .png` — could support Hermetic Event Poster, but not enough to separate from Ornate Hermetic Print Artifact yet
- `Hermes-classical-reference .png` — supports Classical/Hermetic artifact lanes, but alone is a sub-lane

---

## Recommended Expansion Order

1. **Retro Media Interface Relic** — high utility for software/product/agent outputs.
2. **Analog Newspaper Direct Response** — strong release/announcement format; distinct and useful.
3. **Ornate Hermetic Print Artifact** — strengthens Nous/Hermes occult print identity.
4. **Cyanotype Surveillance City** — strong photographic/urban analog lane.
5. **Network Diagram Primitive** — useful for community/agent/network visuals.
6. **Veiled Classical Artifact** — strong mythic-human artifact lane.
7. **Planetary Broadcast Identity** — expand after adding more supporting refs.
8. **Manufactured Multiples** — split from Industrial Duotone Grid only if product/packaging/merch requests appear.
9. **Motorsport Sponsor Field Research** — keep as seed until more sponsor/livery/field refs exist.

---

## Expansion Note

When expanding any of these, follow the v9/v10 pattern:

1. Name the lane.
2. Define exact reference order for generation.
3. Define constrained palette.
4. Write prompt block + 2-4 templates.
5. Generate at least one proof image.
6. QA with Codex vision.
7. Patch prompt language based on defects.
