# Session note — Evangelion neon sticker-bomb batch retry + IP prompting

Date: 2026-05-12

## Context

Nick requested a six-image Neon Genesis Evangelion batch in the established neon sticker-bomb poster style:

1. 碇真嗣 / Shinji Ikari
2. 绫波丽 / Rei Ayanami
3. 明日香 / Asuka
4. 零号机 / Evangelion Unit-00
5. 初号机 / Evangelion Unit-01
6. 二号机 / Evangelion Unit-02

The batch used direct CLIProxyAPI `/v1/responses` with `gpt-image-2`, medium quality, `1024x1536`, and a manifest at:

`/Users/nick/.hermes/profiles/jea/state/neon_evangelion_batch_20260512_195240.json`

## What worked

- Use **identity cues + abstracted world motifs**, not official logo dependence:
  - Shinji: short dark hair, blue-white plugsuit, anxious determination, entry-plug cockpit/HUD.
  - Rei: pale blue bob, red eyes, white plugsuit, clinical blue halo/cracked-glass motif.
  - Asuka: orange-red hair, blue eyes, red plugsuit, hair clips, red cockpit handle.
  - Unit-00: yellow/orange prototype armor, single-eye visor, shield/barricade, restraint/cable details.
  - Unit-01: purple armor, neon green accents, horned head, clawed berserk motion.
  - Unit-02: red combat armor, green visor, spear/blade diagonal, agile attack stance.
- Pre-assign distinct composition archetypes for all six, including human pilots and mecha:
  - Foreground cockpit lens depth.
  - Side-profile split + halo negative space.
  - Dutch-angle action + foreground control handle.
  - Low-angle prototype shield diagonal.
  - Circular energy ring + berserk claw.
  - Foreground spear/progressive blade depth.
- For minors/youthful pilots, include explicit constraints: `covered`, `age-appropriate`, `non-sexualized`, `no nudity`, `no cleavage emphasis`, `no adult-rated presentation`.
- For mecha entries, mark `Non-human mecha` and focus on armor silhouette, weapon/prop depth, cables, restraints, sparks, warning labels, and industrial poster density.

## Retry / manifest lesson

The first long-running batch produced entries 1–2, then later 3–5, but stalled before 6. The safe continuation pattern was:

1. Kill the hung long-running process when it has clearly stalled.
2. **Read the existing manifest before writing any new manifest.**
3. Build a resume script that preserves `status: done` items whose `path` exists.
4. Skip successful indices and generate only `pending`/`failed` entries.
5. Write back to the same manifest after each completion.

This avoided losing the delivered numbering for later `发布123456` selection.

## Verification findings

Vision verification confirmed all six matched the neon sticker-bomb/cyber-pop poster style and were recognizable, but with expected generated-poster caveats:

- Small text often becomes pseudo-text, misspellings, or mixed-language noise.
- IP-world UI labels can become over-dense; treat them as decorative, not guaranteed-readable copy.
- Mecha designs are strongly style-remixed and may not be screen-accurate; describe them as stylized reinterpretations unless exact-canon fidelity was requested.
- Creator credit `NickZag` may be distorted; one Unit-01 run rendered it close to `NickLag`. If Nick cares about credit accuracy, reroll the single affected image instead of regenerating the full batch.

## Prompting pitfall for future Evangelion-like batches

Do not overuse official terms/logos as the style engine. Keep the reusable style generic, and use franchise/name cues only as subject fidelity. Prefer abstract emergency labels, sync-ratio stickers, cockpit UI, hazard stripes, entry-plug motifs, medical tags, cables, and color/silhouette cues over official logo replication.

## Good batch skeleton

For six-character IP batches that mix humans and mecha:

- Create one manifest with indices, title, semantic filename, slug, full prompt, model, size, quality, path, status.
- Use one prompt style block but a unique composition archetype per subject.
- Include explicit non-sexualization for youthful/human subjects.
- Use `Non-human mecha` for robot/armor subjects.
- Continue interrupted runs by reading/preserving the manifest first, then retrying only pending/failed indices.
- Report numbered paths and mention visual caveats succinctly.
