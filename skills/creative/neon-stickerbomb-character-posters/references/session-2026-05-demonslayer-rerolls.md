# Session note — Demon Slayer neon rerolls (2026-05-21/22)

## Context
Nick requested `使用neon skill生成鬼灭之刃4个主要角色`, then repeatedly rerolled subsets: 炭治郎 / 祢豆子 / 伊之助 and later only 祢豆子.

## Durable lessons

- Treat repeated `重新生成[角色]` as a fresh reroll, not a resend. Rotate the composition skeleton each time and keep a manifest/numbered mapping for multi-image subsets.
- For Demon Slayer main-cast subjects, preserve the canon identity anchors before sticker density:
  - 炭治郎 / Tanjiro: burgundy hair, forehead scar, hanafuda earrings, black-green checkered haori, dark demon-slayer uniform, katana, kind/determined expression.
  - 祢豆子 / Nezuko: long dark hair with orange tips, pink eyes, bamboo muzzle, pink patterned kimono, dark haori; always wholesome, covered, age-appropriate, non-sexualized.
  - 善逸 / Zenitsu: yellow-orange bowl-cut hair, yellow/orange triangle-pattern haori, dark uniform, katana, scared-but-explosive thunder energy.
  - 伊之助 / Inosuke: boar-head mask, fur pelt waist, dark hakama pants, dual serrated katanas; keep boar mask clean/readable.
- For Nezuko rerolls, useful rotation skeletons are: diagonal floating protective S-curve, chrome scanner/lens foreground, side-profile split layout, and low-angle diagonal floating. Keep bamboo muzzle and kimono readable; do not let lens/sticker density cover the face.
- For Tanjiro rerolls, useful rotation skeletons are: circular water-breathing ring, side-profile split, foreground katana/checkered-haori depth, and low-angle sword diagonal. Explicitly include hanafuda earrings and checkered haori in negatives.
- For Inosuke, foreground blade depth works when the boar mask remains fully readable and the foreground blade frames rather than covers the mask.
- Safety: all four are youthful/teen-coded; keep age-appropriate, covered, non-sexualized language in the prompt and negatives. Avoid pin-up/adultification wording.

## Prompt compacting
When provider returns `empty_response`, retry only the failed index with a shorter prompt that keeps: subject anchors, one composition archetype, in-scene `NickZag`, core neon mechanics, and subject-specific negatives. Do not regenerate successful siblings.
