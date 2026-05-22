Create a professional infographic following these specifications:

## Image Specifications

- **Type**: Infographic
- **Layout**: bento-grid
- **Style**: retro-pop-grid
- **Aspect Ratio**: 1:1 (square)
- **Language**: en

## Core Principles

- Follow the layout structure precisely for information architecture
- Apply style aesthetics consistently throughout
- Keep information concise, highlight keywords and core concepts
- Use ample whitespace for visual clarity
- Maintain clear visual hierarchy

## Text Requirements

- All text must match the specified style treatment
- Main titles should be prominent and readable
- Key concepts should be visually emphasized
- Labels should be clear and appropriately sized
- Use English for all text content

## Layout Guidelines (bento-grid)

- Grid of rectangular cells with varied sizes (1x1, 2x1, 1x2, 2x2)
- Hero cell ("ONE TOKEN, EVERY KEY") takes the largest position (top-center or upper-left, 2x2)
- Supporting cells around the hero, mixed cell sizes for rhythm
- Each cell self-contained with its own title + icon + brief content
- Title strip at the top: "BITWARDEN SECRETS MANAGER — HERMES-AGENT PR #30035"
- Footer strip at the bottom with commit SHA + repo

## Style Guidelines (retro-pop-grid)

- 1970s retro pop art with strict Swiss international grid
- Background: warm vintage cream/beige (#F5F0E6)
- Accents: salmon pink, sky blue, mustard yellow, mint green — all muted retro tones
- Pure solid black (#000000) and solid white (#FFFFFF) for extreme-contrast cells
- Uniform thick black outlines on ALL illustrations, text boxes, grid dividers
- Pure 2D flat vector aesthetic with subtle screen-print texture
- One cell inverted to black-background-with-white-text for the "NEVER BLOCKS STARTUP" warning section
- Geometric fill patterns in empty cells: checkerboards, diagonal lines, dot grids
- Flat abstract symbols: shields (security), wrenches (install), arrows (rotation), keyholes (auth), checkmarks (tests)
- Vintage comic-style smiley face for "26/26 PASSING" cell
- Bold brutalist or thick retro display fonts for headers; clean sans-serif body
- Decorative stylistic labels acceptable: "WARNING", "NEW DEFAULT", "PINNED", "VERIFIED", "ROTATE"

## Avoid

- 3D rendering, gradients, soft shadows, sketch-like lines
- Free-floating elements — everything anchored in grid cells
- Pure white background — must use warm cream/beige

---

Generate the infographic based on the content below:

### Title (top strip)
BITWARDEN SECRETS MANAGER → HERMES-AGENT
PR #30035

### HERO CELL (largest, top-center, salmon pink background with thick black border)
ONE TOKEN, EVERY KEY
Rotate once in the Bitwarden web app.
Every Hermes process picks it up on next start.
NEW DEFAULT: override_existing = true

### Cell — LAZY INSTALL (sky blue background)
~/.hermes/bin/bws
bws v2.0.0 PINNED
SHA-256 VERIFIED
No apt · no brew · no sudo
Icon: wrench + downward arrow

### Cell — CLI SURFACE (mustard yellow background, checkerboard accents)
$ hermes secrets bitwarden
  setup    wizard
  status   diagnose
  sync     fetch
  install  binary
  disable  off
Icon: terminal prompt symbol

### Cell — SOURCE OF TRUTH (mint green background)
BITWARDEN WINS
Overwrites stale .env on every start
Bootstrap token never overwritten (exception)
Icon: keyhole + arrow

### Cell — INVERTED BLACK CELL with WHITE TEXT — NEVER BLOCKS STARTUP (extreme contrast)
WARNING-FREE STARTUP
Missing binary → warn + continue
Bad token → warn + continue
Network down → warn + continue
Checksum mismatch → refuse + warn
30s timeout ceiling
Icon: white triangle warning sign

### Cell — TESTS (cream with thick black outline, vintage comic smiley face)
26 / 26
HERMETIC
subprocess + urllib mocked
linux · macos · windows
x86_64 · arm64
Icon: comic-style smiley face with checkmark

### Cell — CONFIG YAML (white background with black grid)
secrets:
  bitwarden:
    enabled: true
    project_id: ...
    override_existing: true
    cache_ttl_seconds: 300
    auto_install: true

### Footer strip (bottom, black-on-cream)
PR #30035 · commit 7f9b05668 · NousResearch/hermes-agent
10 files · +1743 / -1 · agent/secret_sources/ · hermes_cli/secrets_cli.py
