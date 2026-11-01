# v9 Imprint Pipeline — User-Corrected Nous Style Target

## Why v9 Exists

The previous v8 pipeline moved in the right direction on texture, but the user-corrected target is **not** primarily lush cyberpunk/fantasy illustration with texture added afterward. The closer reference family is an **underground technical-art imprint**:

- xerox/poster decay
- risograph and screenprint grain
- constrained duotone or three-ink palettes
- brutal condensed typography
- utilitarian borders, crop marks, registration marks
- `NOUS` as stamped/printed/embedded ink, not clean overlay text

Treat this as the default target for new Nous brand graphics unless the user explicitly asks for the older PNW/celestial illustration mode.

---

## New Style Reference Files

Canonical local directory: `~/Desktop/Nous-branding-refs/`

| File | Source image | Use for | Characteristics |
|---|---|---|---|
| `style-cyan-xerox-poster.jpg` | user ref 1 | xerox posters, bold announcements | pale cyan paper, black ink, scanlines, top-heavy `NOUS`, harsh thresholding |
| `style-minimal-stipple-border.jpg` | user ref 2 | quiet abstract banners/covers | cream paper, navy stipple gradient, turquoise border, sparse negative space |
| `style-industrial-duotone-grid.jpg` | user ref 3 | product/system grids | cobalt/acid-yellow duotone, repeated artifacts, high halftone density |
| `style-rough-print.jpg` | existing/user ref 4 | manuals/spec covers | aged tan stock, dark teal ink, heavy border, red rule, letterpress scuffs |
| `style-blue-registration-character.jpg` | user ref 5 | character/symbol posters | deep blue monochrome, orange registration marks, frame lines, scratches |

---

## Style Lanes

Choose **one** lane per generation. Do not mix all lanes into one prompt.

### 1. Xerox Poster
Use for release announcements, punchy X images, stark visual identity.

Prompt cues:
- high-contrast xerox-style poster
- pale cyan paper stock
- black or dark teal ink
- dense horizontal scanlines
- bitmap thresholding and rough halftone edges
- bold block `NOUS` wordmark near top
- one anonymous symbolic subject in the center

Refs: `style-cyan-xerox-poster.jpg`, `style-rough-print.jpg`, optional brand ref last.

### 2. Minimal Stipple Field
Use for refined, quiet, abstract graphics.

Prompt cues:
- abstract risograph/screenprint poster
- cream paper base
- navy stipple field fading downward
- thin turquoise double border
- small centered `NOUS` capsule mark
- lots of negative space

Refs: `style-minimal-stipple-border.jpg`, `style-rough-print.jpg`, optional brand ref last.

### 3. Industrial Duotone Grid
Use for model releases, tools, infra/product systems.

Prompt cues:
- edge-to-edge industrial duotone print
- repeated rounded branded artifacts or symbolic components
- saturated cobalt blue, acid yellow, black
- blown-out ink highlights
- dense halftone dots and print noise
- small embedded `NOUS` marks on artifacts

Refs: `style-industrial-duotone-grid.jpg`, `style-cyan-xerox-poster.jpg`, optional product/brand ref last.

### 4. Manual / Letterpress Cover
Use for specs, manuals, design docs, technical announcements.

Prompt cues:
- distressed letterpress technical manual cover
- aged tan paper stock
- thick dark teal-black border
- compact all-caps condensed typography
- red horizontal rule accent
- scuffed ink and worn paper corners
- `NOUS` as leading brand word

Refs: `style-rough-print.jpg`, `style-minimal-stipple-border.jpg`, optional brand ref last.

### 5. Blue Registration Character Poster
Use for agent identity, symbolic personas, illustrated posts.

Prompt cues:
- moody blue screenprint poster
- anonymous illustrated character or symbolic subject
- electric-blue posterized lighting
- dark cyan-black field
- orange registration marks and marginal notations
- thin frame lines, scratches, analog grain

Refs: `style-blue-registration-character.jpg`, `style-cyan-xerox-poster.jpg`, optional character/brand ref last.

---

## v9 Prompt Block

Append this block to generation prompts:

```text
STYLE TARGET: underground technical-art imprint, not clean fantasy illustration.
Use constrained 2-4 ink palette only. Make the whole image feel screenprinted/xeroxed/risographed on physical paper.
Visible halftone dots, stipple, paper fibers, ink scuffs, misregistration, scanlines, rough thresholded edges.
The focal subject must also be degraded by xerox breakup/ink gain; do not leave a clean sci-fi render inside a distressed border.
Typography must be brutal condensed uppercase or stamped capsule marks; `NOUS` should look printed into the artifact, distressed/scuffed like the rest of the image, not digitally overlaid.
Add uneven ink density, utilitarian border/crop/registration details, slight plate wobble, imperfect stamping, and asymmetric scuffs where appropriate.
Do NOT make smooth digital art. Do NOT use glossy 3D. Do NOT use rainbow neon. Do NOT copy the reference scene or pose; borrow only print language, palette, composition grammar, and finish.
```

---

## Recommended v9 Workflow

```bash
WORK=/tmp/nous-branding-v9
mkdir -p "$WORK"
cd "$WORK"

# Copy lane refs first, brand/content ref last if needed.
cp ~/Desktop/Nous-branding-refs/style-cyan-xerox-poster.jpg .
cp ~/Desktop/Nous-branding-refs/style-rough-print.jpg .
cp ~/Desktop/Nous-branding-refs/Hermes-update-reference\ .png ./brand-ref.png

printf '%s' "[lane-specific prompt + v9 prompt block]" | \
  codex exec --skip-git-repo-check -s workspace-write \
    --add-dir "$WORK" \
    -i style-cyan-xerox-poster.jpg style-rough-print.jpg brand-ref.png \
    -

SESSION=$(ls -lt "$HOME/.codex/generated_images/" | awk 'NR==2 {print $NF}')
cp "$HOME/.codex/generated_images/$SESSION"/ig_*.png ./output-raw.png
python3 /Users/johann/.hermes/profiles/palantir/skills/creative/nous-branding/scripts/postprocess.py \
  output-raw.png output-final.png --mode imprint --intensity 0.7
```

## Key Change From v8

v8: **2 texture refs + 1 brand ref** to force roughness onto a still-cyberpunk/PNW image.

v9: **2 style-lane refs + optional brand/content ref last**. The reference images are not just texture; they define composition grammar, typography, palette discipline, and print technology.

Use `--mode imprint` post-processing for this target. Use legacy `--mode nous` only when the user wants the older luminous PNW/celestial Hermes look.
