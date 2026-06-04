# 2026-05 — The Boys rerolls + Demon Slayer main-four neon batch

## Context
Nick repeatedly requested `使用neon skill` rerolls for live-action *The Boys* characters, especially Soldier Boy / 士兵男孩 and The Deep / 深海, then requested a four-character *Demon Slayer* main cast batch.

## Durable workflow lessons

### Repeated `使用neon skill` means fresh generation
When Nick repeats `使用neon skill 重新生成...`, treat it as a new reroll set, not a resend. Rotate composition archetypes and generate new files. Do not reuse an older path unless explicitly asked to resend.

### Partial provider failures: complete the pair without fabricating
The image provider alternated `empty_response` between Soldier Boy and The Deep across attempts. The useful pattern was:
1. Preserve whichever character succeeds.
2. Retry only the failed character with a more compact prompt.
3. If the other character later succeeds in a newer reroll but the paired item fails, report the failure plainly and never use an old image path as a new output.
4. Once both succeed in the same requested reroll, write a manifest with the current two paths.

### Live-action The Boys prompt anchors
- Soldier Boy: live-action blond adult male, square jaw, rugged smug expression, green/gold retro military superhero armor, vintage shield, fully covered. Useful layouts: side-profile split with shield/chest readable; low-angle diagonal with shield not covering face/chest; foreground shield depth if not too obstructive.
- The Deep: live-action adult male, slick dark hair, blue-green aquatic armored wetsuit, fish-scale texture, ocean chest emblem, vain awkward expression, fully covered. Useful layouts: foreground chrome diving helmet/aquarium lens; side-profile split with aquarium glass/sonar; circular water-current ring if provider accepts the prompt.

### Demon Slayer main-four batch anchors
For `鬼灭之刃4个主要角色`, a stable set is:
1. Tanjiro / 炭治郎 — burgundy hair, forehead scar, hanafuda earrings, black-green checkered haori, katana, water ring.
2. Nezuko / 祢豆子 — long dark hair with orange tips, bamboo muzzle, pink geometric kimono, wholesome covered diagonal floating pose.
3. Zenitsu / 善逸 — yellow-orange bowl-cut hair, triangle-pattern haori, lightning diagonal dash.
4. Inosuke / 伊之助 — boar-head mask, dual serrated katanas, feral dutch-angle lunge, non-sexualized.

Because these characters are youthful, keep age-appropriate / fully covered / non-sexualized constraints early in the prompt. If one slot returns `empty_response`, retry only that slot with a compact anchor-first prompt and preserve successful siblings.
