# Session note — repeated EVA-02 reroll batch and verification (2026-05-13)

## Trigger

Nick asked to `重新生成4张2号机` immediately after a prior four-image EVA-02 batch had already been generated. Treat this as a true reroll/new set, not a request to resend the prior four results.

## What worked

Use the established neon stickerbomb style, but make the four composition skeletons visibly different from the previous batch:

1. **Cross-Lance Close Assault** — foreground Lance of Longinus diagonal, both hands gripping the shaft, forked spearhead in frame, weapon frames the head/torso instead of hiding them.
2. **Shield Breaker Side Split** — three-quarter side-profile on one third, cracked shield as foreground frame, opposite side controlled negative space with oversized typography/stickers.
3. **Beast Sprint Cable Trail** — dutch-angle low all-fours sprint, foreground claw/hand for depth, umbilical cable whipping diagonally, head/green visor still visible.
4. **Halo Drop Twin Blades** — circular cable/energy/tape halo, diagonal dive, compact progressive knives near edges so they do not cover helmet/visor.

Core Unit-02 fidelity cues to repeat in every prompt:

- red EVA-02 silhouette, not generic red robot
- sharp angular helmet
- visible green visor/eyes
- orange/yellow accents
- shoulder pylons
- long athletic limbs
- black mechanical joints
- non-human mecha / no human body

## Operational notes

- Direct CLIProxyAPI `/v1/responses` image generation can produce no stdout for several minutes while the first image is pending. Do not assume failure just because the process is quiet; inspect the manifest for partial `status: done` entries.
- Save a manifest before generation and after every item completes. For this batch: `/Users/nick/.hermes/profiles/jea/state/neon_eva02_reroll4_20260513_001200.json`.
- After generation, visually inspect all four images. The useful checks are: recognizable Unit-02, red mecha, visible helmet/green visor, requested prop/action present, and stickerbomb poster quality. Do not judge by text labels alone.

## Pitfall avoided

Foreground weapons can easily obscure the face/torso. Put negative constraints directly in each variant, e.g. `no weapon covering the face`, `no hidden head`, `no hidden torso`, `no cropped-away helmet`, plus prop-specific constraints such as `no off-frame spearhead` or `no missing knives`.
