# Session 2026-05 — Neon quality drift and Onmyoji-era material recovery

## Context

Nick noticed that recent neon generations for Evangelion / Honor of Kings no longer had the same material quality as the earlier Onmyoji neon batches. The images still used `neon-stickerbomb-character-posters`, but the output felt more like dirty industrial warning posters than the polished glossy premium anime-poster feel of the Onmyoji work.

## Diagnosis

This was not a separate skill contaminating the output. It was style drift inside the neon prompt caused by:

1. **Canon/IP fidelity pressure**
   - Prompts emphasized `SUBJECT FIDELITY FIRST`, `CANON-TIGHT SILHOUETTE`, and exact mecha/Angel shapes.
   - This made the model prioritize literal shape and label-driven recognizability over rich surface material.

2. **Industrial hazard language overload**
   - Evangelion prompts leaned heavily on warning tape, barcode tags, maintenance labels, emergency posters, ruined city, A.T. Field diagrams, industrial UI, and grunge.
   - These cues pulled the look toward muddy, distressed, hard industrial collage.

3. **Compact retry prompt degradation**
   - Several long prompts timed out or returned empty responses; shorter compact retries succeeded.
   - Compact prompts preserved identity + neon/stickerbomb but dropped richer material hierarchy: decorative graphics, palette control, controlled density, prop-led layering, and premium illustration quality.

4. **Sticker density swallowing the subject**
   - Recent outputs used full-frame signboards/stickers as the dominant style layer.
   - The earlier Onmyoji batches used stickers as ornaments around subject-specific props: umbrella, talismans, wings, fan, verdict scroll, star-orbit/astrolabe.

## Corrective prompt direction

When Nick asks to restore the earlier Onmyoji-era quality, use this direction explicitly:

```text
Restore premium Onmyoji-era material quality: rich polished anime illustration, clean subject separation, ornate prop-led composition, wet specular highlights, luxurious layered surfaces, controlled sticker density, cinematic glow. Avoid dirty industrial grunge and overpacked signboards. Stickers/labels decorate borders, props, cables, and background scraps; they must not swallow the main silhouette.
```

For Evangelion Unit-01, this worked better when combined with:

- subject fidelity first, but not label dependency;
- full readable head/torso silhouette;
- umbilical cable as ornamental circular halo;
- stickers attached to cable/edges rather than covering body;
- glossy lacquered purple armor, translucent green trim, chrome glints, glassy reflections;
- explicit negatives: `no excessive dirty grunge`, `no muddy poster texture`, `no overpacked labels`, `no giant text covering the mecha`.

Successful regenerated path from the session:

`/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gptimage2_gpt-image-2-medium_20260513_171217_ba4ffdc0.png`

Manifest:

`/Users/nick/.hermes/profiles/jea/state/neon_eva_unit_01_onmyoji_quality_20260513.json`

## Reusable lesson

Do not treat every neon/stickerbomb request as maximum warning-label density. For premium character posters, preserve a clean hierarchy:

1. subject silhouette and face/head/prop readability;
2. glossy material and color-block quality;
3. subject-specific ornamental prop architecture;
4. stickerbomb graphics as framing/texture;
5. small text only as supporting detail.

If a user says the result does not match older Onmyoji-quality outputs, diagnose **material hierarchy drift**, not necessarily wrong style or wrong skill.
