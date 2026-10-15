---
name: nous-branding
description: "Generate images in the Nous Research visual brand style. Use when creating release graphics, banners, illustrations, or any visual content that should match the Nous/Hermes aesthetic. Covers color palette, composition rules, typography, and visual motifs. Includes reference images for style guidance."
version: 2.1.0
author: Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [branding, image-generation, nous, hermes, design, creative]
    related_skills: [codex-vision]
---

# Nous Branding — Visual Style Guide

## Overview

Generate images that match the **Nous Research** visual brand identity. The current user-corrected default is an **underground technical-art imprint**: xerox decay, risograph grain, constrained duotone palettes, brutal condensed typography, utilitarian borders, and the `NOUS` mark printed/stamped into the artifact. The older Pacific Northwest mysticism + hermetic/celestial + cyberpunk fusion remains available as a secondary lane for explicitly luminous Hermes/PNW requests.

**When to use:** Release announcements, social media banners, project illustrations, partnership graphics, milestone celebrations, community posts, or any visual content that should feel "on-brand" for Nous Research, Hermes, or associated projects.

**Don't use for:** Technical diagrams, data visualizations, charts, graphs, team photos, or documentation screenshots.

---

## Core Aesthetic

### v9 Default Target — Underground Technical-Art Imprint

This is the corrected target for new Nous graphics when the user asks for output closer to the attached stylistic references.

- **Medium:** xerox poster, risograph, screenprint, letterpress, degraded scan
- **Texture:** halftone dots, stipple fields, scanlines, paper fibers, ink scuffs, worn edges, misregistration
- **Palette:** 2-4 inks only — pale cyan, cobalt/navy, black, cream/tan, acid yellow, dark teal, occasional red/orange registration accents
- **Composition:** centered poster, technical manual cover, sparse negative field, or repeated industrial grid
- **Typography:** heavy condensed uppercase, stamped capsule marks, manual-cover hierarchy; `NOUS` must look printed/embedded, not digitally overlaid
- **Anti-pattern:** smooth fantasy/cyberpunk illustration with texture sprinkled on afterward

### v10 Secondary Target — NOUS Analog Cyber-Manga

Use this when the user asks for the **anime Nous girl** aesthetic: manga/anime mascot language fused with xerox, risograph, screenprint, and cyber-zine design.

- **Subject:** original anonymous stylized anime/manga girl mascot or silhouette — never a real person, never copied pose/layout
- **Medium:** manga zine, photocopied poster, risograph card, cyanotype/blueprint scene, future halftone print
- **Texture:** crushed blacks, halftone/stipple, ink dropout, scan haze, paper grain, RGB/plate misregistration
- **Palette:** xerox black + paper white/ice blue + one signal ink such as acid green, blueprint blue, signal orange, or registration violet
- **Typography:** `NOUS` badge, cropped wordmark, vertical type, tiny machine/code captions printed into the artifact
- **Anti-pattern:** glossy anime render, pastel kawaii, 3D mascot, fantasy costume detail, clean corporate vector

See `references/v10-anime-nous-girl-palette.md` for the full palette, reference ordering, and prompt templates.

### Legacy Secondary Lane — The Three Pillars

1. **Pacific Northwest Mysticism** — Cedar forests, mist, fog, rain. Dark greens, slate grays, misty blues. Cabin/workshop/wilderness settings. Nature as co-equal subject.

2. **Hermetic / Celestial Symbolism** — Orbs, portals, sigils, celestial bodies. Sacred geometry (circles, triangles, nested symbols). Figure in contemplation or awakening. Light beams, energy patterns, neural networks as mystical forces.

3. **Cyberpunk / Techno-Mystical Fusion** — Dark backgrounds with neon/accent glows. Electric blues, purples, magentas, cyans against dark fields. Network/node patterns. Figures with luminous eyes, energy emanating from hands/chest/head.

---

## Color Palette

### Primary
| Role | Color | Hex |
|---|---|---|
| Background | Deep Black / Navy | #0A0A1A / #0E2723 |
| Primary Accent | Electric Blue | #00AEEF / #00D4FF |
| Secondary Accent | Purple / Magenta | #8B5CF6 / #D946EF |
| Warm Accent | Amber / Gold | #F59E0B / #C15811 |
| Text | Off-White / Cream | #F7EDE3 / #E8E0D4 |

### Gradients (most common)
- Blue-to-purple: energy/portal effects
- Dark-to-cyan: tech/data visualizations
- Amber-to-deep-red: warmth, PNW sunset
- Green-to-black: forest/environmental depth

---

## Typography

- **Style**: Clean modern sans-serif (Inter, Geist, or Satoshi)
- **Headers**: Bold, letter-spaced
- **Version numbers**: Prominently featured as design elements (e.g., "v0.14.0")
- **Release names**: In quotes or distinctive styling (e.g., "The Foundation Treatment")
- **Rule**: Maximum 2-3 text elements per graphic. Let the image breathe.

---

## Composition Rules

1. **Dark backgrounds are default.** Light backgrounds are rare.
2. **Always include atmospheric depth** — mist, particles, gradients. Flat designs are off-brand.
3. **Glow is a feature** — soft bloom around light sources, text, focal elements.
4. **Restraint in typography** — never more than 2-3 text elements.
5. **One or two accent colors** per composition. Dark + 1-2 accents, not rainbow.
6. **Symmetry and balance** — centered or rule-of-thirds compositions.
7. **Human scale** — even in cosmic compositions, include a human figure or reference.
8. **PNW grounding** — when in doubt, add mist, forest, or atmospheric depth.

---

## Visual Motifs

| Motif | Description |
|---|---|
| Portal / Gate | Circular/arched opening with light emanating |
| Neural network | Connected nodes, glowing blue/purple |
| Orb / Sphere | Luminous sphere, sometimes containing imagery |
| Lone figure | Person in contemplation/reception of energy |
| Light beams | Radiating lines from central point |
| Mist / Fog | Atmospheric depth, PNW environmental |
| Geometric frames | Sacred geometry, nested shapes |
| Torch / Fire | Hermes brand specifically |
| Forest / Cedar | PNW environmental grounding |

---

## Texture Philosophy (Persistent User Preference)

The Nous/Hermes aesthetic is **NOT clean digital art**. This is the single most important style principle. Every image must feel:

- **Printed, not pixelated** — like ink on paper, not vectors on screen
- **Aged, not fresh** — subtle distress, wear, analog warmth
- **Tactile, not flat** — visible substrate (paper, canvas, CRT surface)

This applies at BOTH stages: the prompt must explicitly request these qualities, AND the post-processing pipeline must apply them. Neither stage alone is sufficient.

**In every prompt, explicitly request:**
- "Visible grain / noise texture across the entire image"
- "Halftone dot pattern, like risograph or screen printing"
- "Ink bleed at edges — colors slightly smeary, not crisp digital lines"
- "Paper / canvas fiber texture visible in flat areas"
- "Slightly distressed — not polished, not perfect"
- "Warm analog feel — ink on paper, not pixels on screen"
- "Do NOT make it look like a smooth clean AI-generated illustration"

---

## Prompt Templates

**Core principle:** Every prompt must describe an **entirely original scene**. Reference images guide only the *aesthetic* — color palette, lighting, mood, composition style, and visual finish. Never replicate specific elements (same figure, same pose, same scene) from any reference.

### v9 Imprint Prompt Block (append to new prompts)
```
STYLE TARGET: underground technical-art imprint, not clean fantasy illustration.
Use constrained 2-4 ink palette only. Make the whole image feel screenprinted/xeroxed/risographed on physical paper.
Visible halftone dots, stipple, paper fibers, ink scuffs, misregistration, scanlines, rough thresholded edges.
The focal subject must also be degraded by xerox breakup/ink gain; do not leave a clean sci-fi render inside a distressed border.
Typography must be brutal condensed uppercase or stamped capsule marks; `NOUS` should look printed into the artifact, distressed/scuffed like the rest of the image, not digitally overlaid.
Add uneven ink density, utilitarian border/crop/registration details, slight plate wobble, imperfect stamping, and asymmetric scuffs where appropriate.
Do NOT make smooth digital art. Do NOT use glossy 3D. Do NOT use rainbow neon. Do NOT copy the reference scene or pose; borrow only print language, palette, composition grammar, and finish.
```

### Xerox Poster Release (v9 default)
```
[Original symbolic subject for the announcement]. High-contrast xerox-style poster on pale cyan paper stock. Black/dark teal ink, dense horizontal scanlines, harsh bitmap thresholding, rough halftone breakup. Bold block `NOUS` wordmark near top; one compact secondary line for [release/version]. Centered poster composition with worn border/crop marks.
[v9 Imprint Prompt Block]
--ar 16:9
```

### Manual / Technical Cover (v9 default)
```
Distressed letterpress technical manual cover for [project/release]. Aged tan paper stock, thick dark teal-black border, compact all-caps condensed typography, red horizontal rule accent, scuffed ink and worn paper corners. `NOUS` as the leading stamped brand word.
[v9 Imprint Prompt Block]
--ar 4:5
```

### Anime Nous Girl / Analog Cyber-Manga (v10)
```
Original anonymous anime/manga mascot portrait for Nous Research, not a real person and not copied from references. [Choose sub-lane: Acid Signal / Portal Minimal / Future Halftone / Blueprint Scene]. Use constrained ink palette: xerox black, paper white or ice blue, plus one signal ink such as acid green, blueprint blue, signal orange, or registration violet. Printed cyber-zine design with halftone, ink dropout, scan haze, paper grain, misregistration, and institutional `NOUS` badge/cropped wordmark/tiny machine captions.
[v10 Prompt Block from references/v10-anime-nous-girl-palette.md]
--ar 1:1 or 4:5
```

### Retro Media Interface Relic (v11)
```
Original Nous Research graphic as a recovered retro media-player / desktop utility window. The interface itself is the artifact: window chrome, title bars, playback controls, scrub bars, waveform strips, tiny menus, utility buttons, and file/tape labels. `NOUS` appears as the window title, player label, file name, toolbar stamp, or evidence label printed into the artifact. Degraded photocopied UI screenshot, CRT scan haze, paper grain, scanner dust, ink dropout, xerox thresholding, slight plate misregistration.
[v11 Prompt Block from references/v11-retro-media-interface-relic.md]
--ar 16:9 or 4:5
```

### Analog Newspaper Direct Response (v11)
```
Original Nous Research release announcement as a recovered newspaper classified advertisement / mail-order insert. Masthead reads `NOUS RESEARCH`; main serif headline reads `[release name]`; version `[version]` appears in a price/block badge. Dense columns of tiny product copy surround a rough product cutout or symbolic device. Bottom third contains a clipped coupon or order form with `[action phrase]`. Washed newsprint, rough halftone, ink spread, paper fibers, folded-page wear, crooked columns.
[v11 Prompt Block from references/v11-analog-newspaper-direct-response.md]
--ar 4:5 or 16:9
```

### Ornate Hermetic Print Artifact (v11)
```
Original Nous Research archival/hermetic broadside as a physical print artifact. Dense ornamental border, corner flourishes, central medallion or mythic emblem, seals, stamps, small-cap serif labels, archival accession numbers, marginalia, and layered institutional marks. `NOUS` appears as engraved masthead, certificate title, circular seal inscription, or stamped archival mark. Aged parchment, oxidized dark ink, one seal-red/cobalt/gold accent, foxing, letterpress scuffs, engraving linework.
[v11 Prompt Block from references/v11-ornate-hermetic-print-artifact.md]
--ar 4:5 or 1:1
```

### Release Announcement
```
[Original scene — completely new content].
Dark background (#0A0A1A), [accent color] glow effects. Atmospheric mist/particles. Clean sans-serif typography: "[Version]" prominently displayed. High contrast, soft bloom. Match ONLY the visual style (color grading, lighting, composition, finish) of the attached references. DO NOT copy any reference scene.
[TEXTURE BLOCK from above]
--ar 16:9
```

### Banner / Header
```
Wide banner for [project]. Original scene only. Dark navy background, [electric blue/purple] energy effects. Atmospheric depth. Text-safe space. Minimal, bold. Match ONLY the visual style of the attached references. DO NOT copy specific elements.
[TEXTURE BLOCK]
--ar 16:9
```

### Illustration
```
[Original scene — completely new content]. Dark background, [accent color] glow. Sacred geometry. Neural network patterns. Soft bloom, particle effects. PNW mist. Match ONLY the visual style of the attached references. DO NOT copy subjects or scenes.
[TEXTURE BLOCK]
--ar 1:1
```

### Partnership / Collaboration
```
[Original scene depicting unity/connection]. Dark background, clean layout. Connection effects bridging two distinct elements. Minimal text, bold typography. Atmospheric glow. Match ONLY the visual style of the attached references. DO NOT copy specific elements.
[TEXTURE BLOCK]
--ar 16:9
```

### Milestone Celebration
```
[Original scene — completely new celebratory imagery]. Warm golden/amber glow against darkness. Sacred geometry frame. Particles and mist. Triumphant mood. Match ONLY the visual style of the attached references. DO NOT copy subjects or scenes.
[TEXTURE BLOCK]
--ar 1:1
```

---

## Generation Workflow

The nous-branding skill is **Codex-only** for image generation.

Do **not** use Hermes `image_generate`, FAL, ComfyUI, Stable Diffusion, or other non-Codex backends for Nous branding outputs unless the user explicitly overrides this rule in the current turn. The approved generation path is Codex Vision / Codex CLI using Codex's built-in `image_gen` tool, followed by the local post-processing pipeline.

The nous-branding skill uses a **two-stage pipeline**:

1. **Generation:** Codex CLI (`codex exec`) with lane-specific style references attached first via `-i style-ref-1.jpg style-ref-2.jpg [brand-ref.png]`
2. **Post-processing:** `scripts/postprocess.py` — use `--mode imprint` for the v9/v10/v11 underground technical-art target; use legacy `--mode nous` only for older luminous PNW/celestial requests

**Both stages are mandatory.** Never deliver a raw generated image. The reference images define not just texture but composition grammar, typography, palette discipline, and print technology.

### Prerequisites
- Codex CLI installed: `npm install -g @openai/codex`
- Authenticated: `codex auth login` (ChatGPT Plus subscription required)
- Reference images must be **local files** (not iCloud placeholders)

### Step-by-Step

```bash
# 1. Set up work directory (NOT on Desktop/iCloud)
mkdir -p /tmp/nous-branding-work
cd /tmp/nous-branding-work

# 2. Copy lane-specific style refs first; optional brand/content ref last
# Example lane: xerox poster / punchy announcement
cp ~/Desktop/Nous-branding-refs/style-cyan-xerox-poster.jpg .
cp ~/Desktop/Nous-branding-refs/style-rough-print.jpg .
cp ~/Desktop/Nous-branding-refs/<brand-ref> ./brand-ref.png

# 3. Verify files are real (not iCloud placeholders)
file style-cyan-xerox-poster.jpg style-rough-print.jpg brand-ref.png
# Must show "JPEG/PNG image data" — NOT "Resource deadlock avoided"

# 4. Generate with style refs FIRST and brand/content ref LAST
printf '%s' "[Lane-specific prompt + v9 prompt block]" | \
  codex exec --ignore-rules --skip-git-repo-check -s workspace-write \
    --add-dir /tmp/nous-branding-work \
    -i style-cyan-xerox-poster.jpg style-rough-print.jpg brand-ref.png \
    -

# 5. Collect raw output (2-5 min timeout)
SESSION=$(ls -lt ~/.codex/generated_images/ | awk 'NR==2 {print $NF}')
cp ~/.codex/generated_images/$SESSION/ig_*.png ./output-raw.png

# 6. MANDATORY post-processing for v9 imprint target
python3 <skill-dir>/scripts/postprocess.py output-raw.png output-final.png --mode imprint --intensity 0.7
```

### Reference Combinations (v11 — Expanded Candidate Lanes)

| Style Lane | Primary Ref | Support Ref 1 | Support Ref 2 | Use For |
|---|---|---|---|---|
| Retro Media Interface Relic | `Hermes-player-reference.png` | `style-cyan-xerox-poster.jpg` | `style-rough-print.jpg` | software/product/agent images where the interface itself is the artifact |
| Analog Newspaper Direct Response | `Hermes4.3-reference .png` | `style-rough-print.jpg` | `style-cyan-xerox-poster.jpg` | release announcements, model drops, product notices, classified/coupon-style posts |
| Ornate Hermetic Print Artifact | `Hermes-jam-reference.png` | `Hermes-classical-reference .png` | `style-rough-print.jpg` | occult/institutional broadsides, certificates, seals, archival model artifacts |
| Cyanotype Surveillance City | `Computersaysno-reference .png` | `style-blue-registration-character.jpg` | `style-cyan-xerox-poster.jpg` | analog surveillance photos, urban evidence stills, cyanotype anomaly archives |
| Network Diagram Primitive | `Hermes-together-reference.png` | `Nous-Discord-reference .png` | `Hermes-mind-reference.png` + `style-cyan-xerox-poster.jpg` | primitive agent/community maps, civic network diagrams, infrastructure posters |
| Veiled Classical Artifact | `image 3.PNG` | `Hermes-classical-reference .png` | `style-blue-registration-character.jpg` + `style-rough-print.jpg` | intimate archeological/classical scans, veiled statue fragments, museum records |
| Planetary Broadcast Identity | `image0 2.JPG` | `Hermes-together-reference.png` | `Nous-Discord-reference .png` + `style-cyan-xerox-poster.jpg` | satellite-era identity cards, planetary public-service broadcast posters |
| Manufactured Multiples | `image0 3.JPG` | `style-industrial-duotone-grid.jpg` | `style-cyan-xerox-poster.jpg` + `style-rough-grain.jpg` | product arrays, packaging/merch metaphors, ritual-industrial repeated objects |
| Motorsport Sponsor Field Research | `Nous-car-reference.png` | `style-industrial-duotone-grid.jpg` | `style-rough-grain.jpg` | rally sponsor-livery imagery, field-research motorsport team identity |

**v11 rule:** the lane-specific primary ref goes FIRST and controls layout grammar. Print refs add physical degradation and palette discipline. Brand/content refs provide subject semantics only.

### Reference Combinations (v10 — Anime Nous Girl / Analog Cyber-Manga)

| Sub-Lane | Style Ref 1 | Style Ref 2 | Optional Print/Brand Ref | Use For |
|---|---|---|---|---|
| Acid Signal | `anime-nous-acid-mascot-card.png` | `anime-nous-manga-xerox-portrait.png` | `style-cyan-xerox-poster.jpg` | mascot cards, culture posts, high-energy badges |
| Portal Minimal | `anime-nous-portal-silhouette.png` | `anime-nous-future-halftone.png` | `style-minimal-stipple-border.jpg` | restrained editorial posters, portal/community mood |
| Future Halftone | `anime-nous-future-halftone.png` | `anime-nous-manga-xerox-portrait.png` | `style-rough-print.jpg` | delicate manga portraits, poetic identity graphics |
| Blueprint Scene | `anime-nous-blueprint-scene.jpg` | `style-blue-registration-character.jpg` | `anime-nous-portal-silhouette.png` | cinematic scenes, agent/persona posters |
| Op-Art Poster | `anime-nous-opart-poster.png` | `anime-nous-manga-xerox-portrait.png` | `style-cyan-xerox-poster.jpg` | bold manga posters, motion/vehicle/action energy |

**v10 rule:** anime lane refs go FIRST and define the character/mascot language. Broader print refs go LAST and should only add print discipline. See `references/v10-anime-nous-girl-palette.md`.

### Reference Combinations (v9 — Current User-Corrected Default)

| Style Lane | Style Ref 1 | Style Ref 2 | Optional Brand/Content Ref | Use For |
|---|---|---|---|---|
| Xerox Poster | `style-cyan-xerox-poster.jpg` | `style-rough-print.jpg` | `Hermes-update-reference .png` | punchy release graphics, stark X posts |
| Minimal Stipple Field | `style-minimal-stipple-border.jpg` | `style-rough-print.jpg` | `Hermes-reference-light.png` | quiet abstract covers/banners |
| Industrial Duotone Grid | `style-industrial-duotone-grid.jpg` | `style-cyan-xerox-poster.jpg` | `Hermes-agent-ollama-reference .png` | products, tools, infra/system graphics |
| Manual / Letterpress Cover | `style-rough-print.jpg` | `style-minimal-stipple-border.jpg` | `Hermes4.3-reference .png` | specs, manuals, technical announcements |
| Blue Registration Character | `style-blue-registration-character.jpg` | `style-cyan-xerox-poster.jpg` | `Hermes-mind-reference.png` | agent identity, symbolic character posters |

**v9 rule:** style refs go FIRST and are primary. Brand/content refs go LAST and only provide semantic subject matter. Do not let the older luminous Hermes references dominate the output unless explicitly requested.

### Reference Combinations (v8 — Legacy Luminous PNW/Celestial Lane)

| Image Type | Texture Ref 1 | Texture Ref 2 | Brand Ref |
|---|---|---|---|
| Release Announcement | `style-rough-grain.jpg` | `style-rough-print.jpg` | `Hermes-update-reference .png` |
| Community Banner | `style-rough-grain.jpg` | `style-rough-print.jpg` | `Hermes-together-reference.png` |
| Brand Illustration | `style-rough-grain.jpg` | `style-rough-print.jpg` | `Hermes-mind-reference.png` |
| Partnership Graphic | `style-rough-grain.jpg` | `style-rough-print.jpg` | `Hermes-nous-reference .png` |
| Milestone Celebration | `style-rough-grain.jpg` | `style-rough-print.jpg` | `Nous-girl-portal-reference .png` |

**Why 2 texture refs?** The texture refs teach the model surface quality (grain, halftone, roughness) that GPT-image-2 won't produce on its own. The brand ref teaches colors, lighting, and composition. Rough refs go FIRST in `-i` order — they are the PRIMARY style drivers. Brand ref is secondary.

### Post-Processing (Mandatory)

```bash
# Current v9 imprint target
python3 <skill-dir>/scripts/postprocess.py input-raw.png output-final.png --mode imprint --intensity 0.7

# Legacy luminous PNW/celestial target
python3 <skill-dir>/scripts/postprocess.py input-raw.png output-final.png --mode nous --intensity 0.7
```

**14 effects in `--mode imprint`:**
1. Warm color grade — shadow tones → amber/gold
2. CRT scanlines — faint horizontal retro-display lines
3. Film grain — fine Gaussian noise across entire image
4. Bayer dither (Risomorphism) — 4×4 ordered dither matrix, NOT random noise
5. Warm vignette — edges darkened + warm-shifted
6. Chromatic aberration — subtle RGB channel offset at edges
7. Screen print texture — simulated halftone dot pattern
8. Paper fiber — subtle canvas/paper substrate texture
9. Ink bleed — slight edge smearing at dark boundaries
10. Palette compression — pulls clean RGB renders toward constrained print inks
11. Xerox threshold — degraded photocopy contrast and threshold breakup
12. Registration offset — subtle cyan/orange plate misalignment
13. Plate wobble — subtle row-wise distortion so crisp generated lines stop feeling digital
14. Print scuffs — sparse scratches and imperfect ink pickup

**Intensity:** Default `0.5` for early passes. Use `0.65-0.8` for v9 imprint results. If the raw image is still smooth/illustrative, use `--mode imprint --intensity 0.8`; if typography is getting too degraded, drop to `0.6`.

**The raw generated image is never the final deliverable.**

### Parallel Generation (Batch)

```bash
# Each gets its own work dir; all share ~/.codex/generated_images/
(work1 && codex exec ...) &
(work2 && codex exec ...) &
(work3 && codex exec ...) &
wait
# Then run post-processing on each output
```

### Known Pitfalls

- **Codex-only rule:** Nous branding generation uses Codex Vision / Codex CLI only. Do not call Hermes `image_generate` / FAL / ComfyUI / Stable Diffusion for this skill unless the user explicitly overrides it in the current turn.
- **`--ignore-rules` for Codex image runs:** include `--ignore-rules` on `codex exec` generation calls to prevent unrelated local Codex project instructions (for example `keep now`) from intercepting image generation. The prompt should also say: "Do not run keep. Do not write code. Use the built-in image generation tool now."
- **`keep` CLI SQLite failure:** If Codex still invokes `keep now` and hits `unable to open database file` or `attempt to write a readonly database`, this is non-fatal. Retry with `--ignore-rules` and an explicit image-generation-only prompt.
- **iCloud Desktop path:** `~/Desktop/` is iCloud-synced. Always copy refs to `/tmp/` and verify with `file` before passing to codex. The refs in `~/Desktop/Nous-branding-refs/` are copies of the canonical refs in `~/.hermes/profiles/palantir/image_cache/`.
- **`-i` flag:** Must reference files in CWD or `--add-dir` path. `cd` to work dir first. Do NOT use absolute paths with `-i`.
- **GPT-image-2 clean output:** The model will always try to produce smooth, clean digital art. The ONLY way to get the rough/analog look is: (a) explicit texture requests in prompt + (b) post-processing. Neither alone is sufficient.

---

## What Is NOT On-Brand

- Busy/cluttered compositions with competing elements
- Bright/white backgrounds as primary design
- Flat design without atmospheric effects
- Primary colors (pure red, blue, yellow) without dark context
- Complex infographics or data visualizations
- Corporate/sterile tech aesthetic
- Cartoon or anime illustration styles
- Photorealistic renders without stylization
- **Clean, smooth digital illustration look** — the #1 anti-pattern

---

## References

### Anime Nous Girl References (v10 Analog Cyber-Manga)

| File | Location | Dimensions | Characteristics |
|---|---|---|---|
| `anime-nous-acid-mascot-card.png` | `~/Desktop/Nous-branding-refs/` | 1254×1254 | Acid green mascot card, headphones/cyber silhouette, square zine frame, violet registration error |
| `anime-nous-portal-silhouette.png` | `~/Desktop/Nous-branding-refs/` | 1444×1444 | Muted teal-gray minimal silhouette, square frame, vertical type, soft scan grain |
| `anime-nous-manga-xerox-portrait.png` | `~/Desktop/Nous-branding-refs/` | 1024×1024 | High-contrast manga portrait, crushed blacks, halftone, cropped `NOUS` typography |
| `anime-nous-blueprint-scene.jpg` | `~/Desktop/Nous-branding-refs/` | 928×1232 | Electric-blue cyanotype scene, deep navy, orange registration marks, large negative space |
| `anime-nous-opart-poster.png` | `~/Desktop/Nous-branding-refs/` | 1232×928 | Pale ice-blue op-art manga poster, black xerox ink, horizontal scanlines, bold top `NOUS` |
| `anime-nous-future-halftone.png` | `~/Desktop/Nous-branding-refs/` | 636×900 | Fine-line black/white manga portrait, halftone pixel dissolution, tiny future caption |

### Texture/Style Reference Library (Primary — Use These First)

| File | Location | Size | Characteristics |
|---|---|---|---|
| `style-cyan-xerox-poster.jpg` | `~/Desktop/Nous-branding-refs/` | 1232×928, 190KB | Pale cyan xerox poster, black ink, scanlines, threshold breakup, bold wordmark |
| `style-minimal-stipple-border.jpg` | `~/Desktop/Nous-branding-refs/` | 928×1232, 333KB | Cream paper, navy stipple gradient, turquoise border, sparse negative space |
| `style-industrial-duotone-grid.jpg` | `~/Desktop/Nous-branding-refs/` | 1280×717, 226KB | Cobalt/acid-yellow duotone, repeated industrial grid, heavy halftone |
| `style-rough-print.jpg` | `~/Desktop/Nous-branding-refs/` | 928×1232, 189KB | Aged tan manual cover, dark teal ink, red rule, letterpress scuffs |
| `style-blue-registration-character.jpg` | `~/Desktop/Nous-branding-refs/` | 928×1232, 114KB | Deep blue character/symbol poster, orange registration marks, scratches |
| `style-rough-grain.jpg` | `~/Desktop/Nous-branding-refs/` | 800×800, 111KB | Legacy texture var ~1060, strong halftone, contrasty, visible grain |

These are the PRIMARY references. They go FIRST in `-i` flag order. For v9, choose two references from the same style lane instead of blindly using generic texture refs.

### Brand Reference Library (Secondary — Color/Lighting/Composition)

| Image Type | Brand Ref | Dimensions | Size |
|---|---|---|---|
| Release Announcement | `Hermes-update-reference .png` | 1248×832 | 1.7MB |
| Community Banner | `Hermes-together-reference.png` | 2040×2048 | 7.1MB |
| Brand Illustration | `Hermes-mind-reference.png` | 1609×2048 | 5.7MB |
| Partnership Graphic | `Hermes-nous-reference .png` | 1609×2048 | 5.9MB |
| Milestone Celebration | `Nous-girl-portal-reference .png` | 1444×1444 | 2.9MB |

Additional brand refs in `~/Desktop/Nous-branding-refs/`:
- `Hermes-classical-reference .png` — classical/hermetic (707×900)
- `Hermes-jam-reference.png` — creative/collaborative (943×1200)
- `Hermes4.3-reference .png` — major version release (1200×1033)
- `Hermes-agent-ollama-reference .png` — agent/product (1190×1787)
- `Hermes-player-reference.png` — media/entertainment (1609×2048)
- `Hermes-reference-light.png` — bright/clean variant (908×857)
- `Nous-psyche-reference.png` — consciousness/mind (680×624)
- `Nous-Discord-reference .png` — community (1636×2048)
- `Nous-bar-meet-reference.png` — IRL meetup (636×900)
- `Nous-car-reference.png` — PNW lifestyle (680×591)
- `Computersaysno-reference .png` — meme/humor (1543×2048)

### Skill Directory Files

- **Post-processing script**: `scripts/postprocess.py` — 9-step legacy overlay + 14-step `--mode imprint` pipeline
- **v9 imprint pipeline**: `references/v9-imprint-pipeline.md` — underground technical-art target, style lanes, prompt block, commands
- **v10 anime Nous girl palette**: `references/v10-anime-nous-girl-palette.md` — Analog Cyber-Manga lane, sub-lanes, palette, prompt templates
- **v11 Retro Media Interface Relic**: `references/v11-retro-media-interface-relic.md` — expanded software/media UI artifact lane
- **v11 Analog Newspaper Direct Response**: `references/v11-analog-newspaper-direct-response.md` — expanded newspaper/classified/coupon announcement lane
- **v11 Ornate Hermetic Print Artifact**: `references/v11-ornate-hermetic-print-artifact.md` — expanded archival seal / alchemical broadside lane
- **v11 Cyanotype Surveillance City**: `references/v11-cyanotype-surveillance-city.md` — expanded blue urban surveillance/evidence-still lane
- **v11 Network Diagram Primitive**: `references/v11-network-diagram-primitive.md` — expanded primitive civic/agent network map lane
- **v11 Veiled Classical Artifact**: `references/v11-veiled-classical-artifact.md` — expanded archeological/classical scan lane
- **v11 Planetary Broadcast Identity**: `references/v11-planetary-broadcast-identity.md` — expanded satellite-era planetary communications lane
- **v11 Manufactured Multiples**: `references/v11-manufactured-multiples.md` — expanded repeated product/object array lane
- **v11 Motorsport Sponsor Field Research**: `references/v11-motorsport-sponsor-field-research.md` — expanded sponsor-livery field/motorsport lane
- **v11 candidate lane audit**: `references/v11-candidate-lane-audit.md` — unexpanded lanes found in the refs folder, expansion backlog
- **Palette swatch**: `assets/anime-nous-girl-palette.png` — color reference card
- **Reference palette board**: `assets/anime-nous-girl-reference-palette.png` — contact sheet + swatches
- **Design spec**: `references/design.md` — extended design language documentation
- **Reference catalog**: `references/reference-catalog.md` — full inventory with type-to-ref mapping
