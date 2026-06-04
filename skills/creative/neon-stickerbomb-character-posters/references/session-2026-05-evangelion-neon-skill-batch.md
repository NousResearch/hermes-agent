# Session note — Evangelion neon-skill batch (2026-05-12)

## Context
Nick asked to use the `neon-stickerbomb-character-posters` skill to generate a four-image Evangelion set: 二号机, 初号机, 零号机, 绫波零. This followed several rerolls where Unit-02 needed the Lance of Longinus and mecha readability was more important than just text labels.

## Worked pattern
Use the neon skill explicitly and pre-assign distinct composition archetypes before generation:

1. **Unit-02 / 二号机** — `FOREGROUND PROP + DEPTH, TWO-HANDED LANCE DIAGONAL`
   - Red combat armor, orange/yellow accents, green eye visor, shoulder pylons.
   - Both hands grip a long crimson Lance of Longinus.
   - Forked/double-pronged spearhead must be visible inside frame.
   - Negative constraints: no ordinary sword, no dual blades, no spear hidden outside frame.

2. **Unit-01 / 初号机** — `CIRCULAR MOTION RING, BERSERK CABLE HALO`
   - Purple/black armor, neon green accents, single horn, angular jaw, glowing eyes.
   - Umbilical cables and acid-green arcs form a circular halo around upper body.
   - Keep head/horn/torso visible; use clawed foreground hand for depth.

3. **Unit-00 / 零号机** — `SIDE PROFILE SPLIT LAYOUT, RESTRAINT CABLE PROFILE`
   - Yellow/orange prototype armor, single cyclopean visor/eye, gray jaw, bulky shoulders.
   - Right-third mecha profile; left side carries cropped typography and hazard stickers.
   - Foreground restraint clamps add depth but must not cover the face.

4. **Rei Ayanami / 绫波零** — `DIAGONAL FLOATING / RECLINING COCKPIT HALO`
   - Youthful subject: explicitly covered, age-appropriate, non-sexualized.
   - Pale blue bob, red eyes, calm expression, white/blue plugsuit.
   - Blue cockpit halo, glass shards, interface cables, LCL bubbles; typography wraps around arc.

## Verification observations
- All four passed visual review for subject recognizability and neon sticker-bomb style.
- Mecha images remain strongly stylized and text-heavy; judge identity by silhouette/color/props first, not only labels.
- Unit-02 Lance visibility succeeds when the spear is named both in composition and negatives, with both hands gripping the shaft and the forked head inside frame.
- Rei can be dynamic without safety issues if the prompt says full plugsuit, covered, non-sexualized, age-appropriate, and avoids cleavage/adult-rated framing.

## Operational note
For long four-image batches through CLIProxyAPI, save a manifest after each item. In this run each image took ~238–247 seconds, so waiting several 180s cycles is normal. The manifest path was `/Users/nick/.hermes/profiles/jea/state/neon_evangelion_skill_batch_20260512_230232.json`.
