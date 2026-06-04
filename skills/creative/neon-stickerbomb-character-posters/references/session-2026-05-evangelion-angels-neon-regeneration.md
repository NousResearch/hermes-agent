# Evangelion Angels 1–4 neon regeneration (2026-05-13)

## Context
Nick requested publishing Evangelion Angels 1–4, then corrected the workflow: this channel should use the `neon-stickerbomb-character-posters` skill, so the prior Silver Rebellion Angel images had to be regenerated in neon style before publishing.

## Prompting lesson
For Evangelion Angel subjects, do not assume the earlier EVA mecha/pilot prompt set covers non-humanoid Angels. Preserve Angel identity through silhouette and core motifs while keeping the reusable style generic.

A working four-subject composition set:

1. **FIRST ANGEL ADAM** — genesis seed / origin poster
   - central pale faceless angelic humanoid or seed-life giant
   - halo/containment ring, embryo/core glow, classified-biohazard collage
   - huge cropped `FIRST ANGEL ADAM` typography

2. **SECOND ANGEL LILITH** — containment / terminal restraint poster
   - pale faceless ivory giant in dark sci-fi containment chamber
   - black cables/restraints, red glowing ribbons, cracked porcelain, gold circuitry/halo
   - avoid overly sensitive crucifixion wording if provider returns empty image; use solemn open-arm ritual/containment language instead

3. **THIRD ANGEL SACHIEL** — red-core impact poster
   - dark elongated biomechanical monster
   - skull/mask-like face plates, chest red core, explosive red energy burst
   - warning labels and `THIRD ANGEL SACHIEL` typography

4. **FOURTH ANGEL SHAMSHEL** — circular energy-whip motion ring
   - segmented magenta-red insect/serpent body
   - small pale skull-like head, glowing red core
   - two luminous energy-whip arms forming a large ring around the body
   - huge `FOURTH ANGEL SHAMSHEL` typography following the ring

## Provider behavior
- Lilith prompts with more explicit restraint/crucifixion-style language repeatedly returned CLIProxyAPI `empty_response` / no `image_generation_call`.
- A shorter, safer prompt with `pale faceless ivory giant`, `open solemn arms`, `black cables`, `red glowing ribbons`, `containment chamber`, and `no gore` succeeded.
- Shamshel succeeded after compacting the prompt and preserving the motion-ring concept.
- When only one item fails, retry only that index and keep successful paths in the manifest.

## Manifest and QC
Keep a manifest with result index, title, slug, path, prompt, provider/model, status, and retry notes. This allows later `发布1234` mapping.

Generated paths in this session:

- Adam: `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gptimage2_gpt-image-2-medium_20260513_131156_a495b282.png`
- Lilith: `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gptimage2_gpt-image-2-medium_20260513_133115_93691800.png`
- Sachiel: `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gptimage2_gpt-image-2-medium_20260513_131515_3cec4f7e.png`
- Shamshel: `/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gptimage2_gpt-image-2-medium_20260513_132259_0bff97e4.png`

QC notes:
- Adam: strong neon sticker-bomb, origin/seed/white-giant read.
- Lilith: strong Second Angel/Lilith read, white faceless biomech divinity, dense labels.
- Sachiel: mask face, red core, monster silhouette, warning collage read well.
- Shamshel: energy whip + segmented serpent read; more dragon/monster-like than canon but acceptable as neon reinterpretation.

## Style/channel rule
If Nick says this channel should use neon skill, regenerate with this skill before publishing. Do not publish a prior Silver Rebellion / other-style batch to the neon channel just because the subject and numbering match.
