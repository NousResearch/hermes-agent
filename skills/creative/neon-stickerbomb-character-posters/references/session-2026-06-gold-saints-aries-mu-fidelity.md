# Session note — Gold Saints 1–4 and Aries Mu face/eyebrow fidelity

## Context
Nick generated the first four Saint Seiya Gold Saints in the glossy neon cyber-pop sticker-bomb poster style, then repeatedly refined Aries Mu and Cancer Deathmask. The main correction signal was not overall image polish; it was canon/source fidelity under the neon treatment.

## Aries Mu / 白羊座穆 correction pattern
Prioritize original-anime/manga recognizability before poster density.

Hard anchors:
- Long pale lavender-purple hair with straight bangs and side locks.
- Small forehead dot clearly visible.
- Soft, calm, refined androgynous young face; not harsh, villainous, rugged, or macho.
- Calm narrow relaxed eyes.
- Eyebrows are a first-class fidelity anchor: extremely thin, very pale, almost invisible; no thick/dark/angry/angular eyebrows.
- Gold Aries Cloth with readable ram-horn shoulder motifs.
- Crystal Wall / psychic defense / Aries constellation motifs.

Prompt pattern that improved results:
- Put `CANON FACE LOCK FIRST` before style language.
- State that stickers, typography, hair glow, and crystal effects must not cover the face, eyes, eyebrows, forehead dot, or main hair silhouette.
- Keep hands out of frame or anatomically simple when the face/eyebrow fidelity is the main target; foreground Crystal Wall hand poses often create hand/finger issues.

QC ranking from the four-Aries variant batch:
1. Side-profile / Crystal Wall close-up was strongest: subtle eyebrows, no major hand issues, strong identity and style.
2. Foreground Crystal Wall depth had strong theme but eyebrows became too pronounced and the foreground hand introduced anatomy problems.
3. Circular Crystal Wall ring had strong motion but weaker hands/body alignment and more modernized face.
4. Low-angle fashion diagonal had strong ram-horn armor identity but hand/arm perspective and sharper eyebrows reduced strict fidelity.

## Cancer Deathmask / 巨蟹座迪斯马斯克 correction pattern
The model drifted toward giant crab claws, pincer hands, or mecha appendages. Treat that as a canon-armor failure.

Hard anchors:
- Adult sinister face and controlled villain energy.
- Gold Cancer Cloth that remains integrated armor, not a crab monster/mecha redesign.
- Avoid giant crab claws, pincer hands, oversized crab limbs, extra appendages, or mechanical crab arms.
- Put sticker-bomb density in borders, torn labels, warning cards, and background rather than replacing armor anatomy.

Useful hard line:
`STRICT CANON ARMOR LOCK FIRST: preserve Gold Saint armor silhouette; no giant crab claw, no pincer hand, no mecha appendage, no crab-monster redesign.`

## Manifest practice
When a user asks for several alternatives of the same character, save a separate variant manifest instead of silently replacing the main roster manifest. Keep the roster manifest for selected/current picks, and the variant manifest for later numbered selection.

Example from this session:
- Main roster manifest: `state/neon-stickerbomb/gold_saints_1_4_20260602_manifest.json`
- Variant manifest: `state/neon-stickerbomb/aries_mu_4_variants_20260602_manifest.json`

## Delivery note
After generating/QCing variants, deliver numbered `MEDIA:` paths directly. Include short QC labels so Nick can choose: e.g. “1号最稳 / 4号冲击强但手部透视有问题.”