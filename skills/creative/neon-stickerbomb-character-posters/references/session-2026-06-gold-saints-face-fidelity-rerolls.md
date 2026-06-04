# Session 2026-06 — Gold Saints 1–4 face-fidelity rerolls

## Context
Nick generated the first four Gold Saints in the neon sticker-bomb style, then requested targeted rerolls for indices 1, 3, and 4. A later correction focused on Aries Mu and Cancer Deathmask:

- `重新生成白羊座 穆(五官特别是眉毛需要符合原著)`
- `重新生成巨蟹座`

## Durable lessons

### Aries Mu / 白羊座穆
When Nick asks for Aries Mu, especially after mentioning facial fidelity, treat the face and eyebrows as the first-class anchor before style density.

Use prompt anchors:
- original anime/manga facial fidelity first
- soft refined youthful male face
- calm narrow almond-shaped eyes
- delicate straight nose, small composed mouth
- long lavender/purple hair with straight bangs and long side locks
- distinctive very light / near-invisible eyebrow look
- thin pale lavender eyebrows
- explicitly block thick black eyebrows, angry heavy brows, bushy brows, mismatched brows, harsh villain face
- keep eyes, eyebrows, nose, mouth, and main hair silhouette unobscured

A successful reroll used a cleaner upper-body / side split portrait with Crystal Wall as a framing element rather than covering the face. Sticker-bomb density stayed mostly in the right/background area.

### Cancer Deathmask / 巨蟹座迪斯马斯克
The first reroll was visually strong but drifted into a modern fantasy / demon-knight remix: huge feathered cyber hair, giant monster-claw gauntlet, and too much mecha armor. Treat that as a canon-fidelity failure even if the neon style is strong.

Use prompt anchors:
- strict original-anime canon fidelity first, neon style second
- harsh angular adult male face
- gaunt sharper cheeks, narrow mocking eyes, cruel vicious smirk/grin
- strong dark brows
- dark blue-purple short/wavy/spiky hair of moderate size
- classic integrated Gold Cancer Cloth, not generic demon armor
- crab-inspired curves and modest claw motifs integrated into armor
- block giant monster claw hand, huge feathered cyber hair, excessive mecha redesign, soft pretty saint face
- keep face, hair, chest armor, and shoulder armor clean and readable

A better reroll used a side-profile/split underworld poster, made the spectral death mask smaller than the character, and moved underworld labels/stickers into the background.

## Manifest handling
Patch the existing numbered Gold Saints manifest rather than creating an unrelated fresh source of truth when Nick rerolls specific indices. In this session:

- index 1 = Aries Mu
- index 2 = Taurus Aldebaran
- index 3 = Gemini Saga
- index 4 = Cancer Deathmask

When a first reroll is visually strong but canon-drifted, generate another targeted reroll and update the manifest only to the accepted path. Mention any caveats plainly, e.g. text typo or minor hand-size note.
