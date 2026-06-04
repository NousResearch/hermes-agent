# EVA mecha: balancing canon fidelity, glossy material, and dense neon sticker-bomb

Session: 2026-05-13

## What Nick corrected

- A glossy/high-quality mecha render is not enough for the `neon-stickerbomb-character-posters` skill.
- If the dense neon sticker-bomb system disappears, the output fails even when subject material quality improves.
- For EVA subjects, subject fidelity must be preserved, but not by suppressing the skill's visual core.

Nick's rejection signal: “机体的质感强了，但是缺少了neon繁复贴纸的视觉感，这个是neon skill的核心。这张不通过”

## Working balance

Use this balance for EVA/mecha/IP subjects:

1. Subject fidelity first:
   - Keep the official silhouette, colors, head/face cues, proportions, and signature equipment readable.
   - Do not let text labels be the only reason the subject is identifiable.

2. Premium glossy material:
   - Lacquered armor, wet specular highlights, chrome edge glints, translucent luminous trim, sharp cel shading, deep black/violet shadow blocks.
   - Avoid muddy industrial grunge when the user asks for the earlier Onmyoji-era premium feel.

3. Dense neon sticker-bomb core:
   - Dense torn vinyl decals, warning labels, barcode fragments, tape strips, graffiti tags, typography, halftone cutouts, AT-field/tech diagram scraps, offset-print registration, chromatic split.
   - Stickers should fill background, borders, halo/cable arcs, and prop surfaces.
   - Allow slight overlap onto armor edges for depth, but never hide head/chest/main silhouette.

## Prompt pattern that worked

Use explicit phrasing like:

```text
Correct balance: keep premium glossy mecha material quality, but restore the core neon skill visual language: dense layered sticker-bomb typography, torn decals, tape strips, barcode fragments, graffiti labels, halftone print energy, and aggressive cyber-pop collage.

COMPOSITION — DENSE STICKER STORM + READABLE CORE SILHOUETTE:
[subject] occupies center/right with head, chest, shoulders fully visible. [signature cable/weapon/halo] creates the motion path and is covered with small stickers, torn tape, maintenance labels, barcode tags and graffiti marks. Around the outer frame, create a dense sticker-bomb wall: overlapping torn vinyl decals, warning strips, cropped subject typography, magenta/cyan graffiti tags, hazard labels, fake maintenance cards, diagram scraps, chrome shards, halftone cutouts, and screenprint noise. Stickers overlap armor edges slightly but must not hide head/chest.

MATERIAL QUALITY:
premium polished anime illustration, glossy lacquered armor, translucent glowing trim, wet white specular highlights, chrome edge glints, sharp cel-shading, thick confident manga outlines.
```

## EVA unit publication notes

Generated/published set that passed the balance:

- EVA Unit-00 / 零号机 — `MP3WQ86PYECK3`, route `eva-unit-00-yeck3.html`
- EVA Unit-02 / 二号机 — `MP3WRPFF672XY`, route `eva-unit-02-672xy.html`
- EVA Unit-03 / 三号机 — `MP3WSGI2YLBIR`, route `eva-unit-03-ylbir.html`
- EVA Unit-04 / 四号机 — `MP3WSGXZRG3GQ`, route `eva-unit-04-rg3gq.html`

## Publishing pitfall observed again

The local `image2skill-publish` wrapper still aborts after successful Eagle import because immediate `/api/item/info?id=...` can return HTTP 500. Recovery flow:

1. Capture printed Eagle ID from wrapper stdout.
2. Retry direct `/api/item/info?id=<id>` after a short delay.
3. If folder/tags/annotation are correct, do not re-import that file.
4. Import remaining files one-by-one if needed.
5. Run `rebuild_image2skill.py`.
6. Sync rebuilt static data into `frontend/src/data.ts` before building the React/Vite SPA.
7. Build with `bun run build`, stage canonical `frontend/dist` first, overlay only allowed public assets/static metadata, force SPA rewrites, deploy with Wrangler.
8. Verify the production JS bundle contains the new route slugs/titles and direct asset URLs return JPEG/PNG bytes.
