# 2026-05 KOF and known-female neon batch notes

## Trigger

Use when Nick asks for batches like:

- `生成4张kof女性动漫角色`
- `使用skill生成4张知名女性动漫角色`
- single known fighter-game characters such as `kof angel`

## What worked

Use `neon-stickerbomb-character-posters` and pre-assign visibly different poster skeletons before generation. For batches, avoid repeating characters from the immediately prior batch unless Nick names them.

Worked subject/composition sets:

### KOF women

1. **Mai Shiranui / 不知火舞** — circular folding-fan + flame ring; red/white/orange palette.
2. **Leona Heidern / 莉安娜** — side-profile split + cyan energy hand-blade; tactical blue/green/red warning palette.
3. **Shermie / 夏尔米** — low-angle stage grappler diagonal; red hair curtain, purple shadows, stage lights.
4. **Blue Mary / 布鲁玛丽** — foreground grappling glove/belt buckle + denim street poster wall; denim/chrome/orange street palette.

Single follow-up:

- **KOF Angel** — dutch-angle grappler action poster with chrome ring rope / glove-label foreground; preserve white hair, blue-white fighter language, star motifs, athletic grappler energy. Keep covered and non-explicit.

### Known female anime batch examples

1. **Akame / 赤瞳** — dutch-angle cursed katana foreground slash; black/red/cyan palette.
2. **Revy / 蕾薇** — foreground twin pistol depth + urban poster wall; gunmetal/orange/cyan-magenta palette.
3. **Yoko Littner / 优子** — low-angle rifle desert diagonal; rifle barrel as foreground composition line.
4. **Nausicaä / 娜乌西卡** — floating glider eco-spiral; wholesome covered explorer, wind/spore/insect-wing motifs.

## Batch manifest requirement

For all multi-image neon batches, especially direct CLIProxyAPI batches, save a manifest under `~/.hermes/profiles/jea/state/` and update it as each image completes. Required fields per item:

- `index`
- `title`
- `semantic`
- `slug`
- `prompt`
- `status`
- `path` when done
- `elapsed_seconds` when done

This supports Nick's later commands such as `发布1、2` by mapping visible result numbers to exact files/prompts.

## Prompting notes

- Use known identity cues directly in each prompt; do not rely on the model to infer the character.
- Keep style mechanics creator-neutral: glossy neon cyber-pop, thick black comic ink, sticker-bomb typography, halftone, chromatic split, torn vinyl decals, cyber-streetwear accessories.
- Put `NickZag` on a real in-scene object: fight-ticket sticker, ring rope label, glove label, weapon maintenance tag, glider lens service label, etc.
- For adult fighter women, keep the remix stylish but covered: technical jackets, gloves, straps, belts, patches, labels. Avoid pin-up/adult-rated framing even if source character is often drawn that way.
- Change at least four composition dimensions between images: archetype, camera, body orientation, prop foreground, typography direction, background geometry, motion path, or negative-space placement.
