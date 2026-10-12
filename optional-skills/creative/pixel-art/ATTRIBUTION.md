# Attribution

This skill bundles code ported from a third-party MIT-licensed project.
All reuse is credited here.

## pixel-art-studio (Synero)

- Source: https://github.com/Synero/pixel-art-studio
- License: MIT
- Copyright: © Synero, MIT-licensed contributors

### What was ported

**`scripts/palettes.py`** — the `PALETTES` dict containing 23 named RGB
palettes (hardware and artistic). Values are reproduced verbatim from
`scripts/pixelart.py` of pixel-art-studio.

**`scripts/pixel_art_video.py`** — the 12 procedural animation init/draw pairs
(`stars`, `fireflies`, `leaves`, `dust_motes`, `sparkles`, `rain`,
`lightning`, `bubbles`, `embers`, `snowflakes`, `neon_pulse`, `heat_shimmer`)
and the `SCENES` → layer mapping. Ported from `scripts/pixelart_video.py`
with minor refactors:
- Names prefixed with `_` for private helpers (`_px`, `_pixel_cross`)
- `SCENE_ANIMATIONS` renamed to `SCENES` and restructured to hold layer
  names (strings) instead of function-name strings resolved via `globals()`
- `generate_video()` split: the Pollinations text-to-image call was removed
  (Hermes uses its own `image_generate` + `pixel_art()` pipeline for base
  frames). Only the overlay + ffmpeg encoding remains.
- Frame directory is now a `tempfile.TemporaryDirectory` instead of
  hand-managed cleanup.
- `ffmpeg` invocation switched from `os.system` to `subprocess.run(check=True)`
  for safety.

### What was NOT ported

- Wu's Color Quantization (PIL's built-in `quantize` suffices)
- Sobel edge-aware downsampling (requires scipy; not worth the dep)
- Bayer / Atkinson dither (would need numpy reimplementation; kept scope tight)
- Pollinations text-to-image generation (`pixelart_image.py`,
  `generate_base()` in `pixelart_video.py`) — Hermes has `image_generate`

### License compatibility

pixel-art-studio ships under the MIT License, which permits redistribution
with attribution. This skill preserves the original copyright notice here
and in the SKILL.md credits block. No code was relicensed.

---

## pixel-art skill itself

- License: MIT (inherits from hermes-agent repo)
- Original author of the skill shell: dodo-reach
- Expansion with palettes + video: Hermes Agent contributors
