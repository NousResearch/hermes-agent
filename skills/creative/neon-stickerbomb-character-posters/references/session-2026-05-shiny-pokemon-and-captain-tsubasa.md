# Session note — shiny Pokémon and Captain Tsubasa generation

## Context
Nick gave minimal category/franchise prompts:
- `生成4张闪光宝可梦`
- `生成足球小子热门角色图`
- `重新生成大空翼，日向小次郎`

The active skill was `neon-stickerbomb-character-posters`.

## Durable lessons

### Minimal franchise/category image requests
When Nick asks for a broad category without a roster, choose a recognizable/popular roster immediately rather than asking for clarification, unless the identity is genuinely ambiguous.

For batch requests:
1. Pre-assign distinct composition archetypes.
2. Generate the images directly.
3. Deliver each result as `MEDIA:/absolute/path` in numbered order.
4. Save a sidecar manifest, even for two-image rerolls, so later `发布12` or `重新生成1` maps cleanly.

### Fresh rerolls
Requests like `重新生成大空翼，日向小次郎` mean fresh generation, not resend. Rotate away from the most recent skeleton for those exact characters and save a new reroll manifest.

Worked reroll rotation from this session:
- 大空翼 / Tsubasa Ozora: previous foreground dribble -> new circular motion ring / overhead drive shot.
- 日向小次郎 / Kojiro Hyuga: previous low-angle shooting wind-up -> new dutch-angle tiger-shot follow-through.

### Shiny Pokémon roster and recovery
For `生成4张闪光宝可梦`, a successful set was:
1. Shiny Charizard / 闪光喷火龙 — black dragon-like body, cream wings, flame tail.
2. Shiny Gyarados / 闪光暴鲤龙 — red serpentine dragon-fish body, whiskers, fins, no legs.
3. Shiny Umbreon / 闪光月亮伊布 — black quadruped with blue glowing rings, red eyes.
4. Shiny Rayquaza / 闪光烈空坐 — black serpentine sky dragon, gold rings, two small clawed forearms behind head.

Shiny Metagross was attempted first but returned `empty_response` twice. Because the user asked for a count rather than an exact roster, substituting another fitting shiny Pokémon was acceptable; record the failed candidate and substitution in the manifest and report it plainly.

For shiny Pokémon and creature Pokémon, include an explicit `ANATOMY LOCK` with canon body plan and shiny palette. For Rayquaza specifically, require both small clawed forearms visible and unobscured, not replaced by fins, stickers, lightning, or typography.

### Captain Tsubasa / 足球小将 anchors
A good default four-character popular roster:
1. 大空翼 / Tsubasa Ozora — youthful soccer captain, short dark hair, determined eyes, white-blue kit, captain armband, soccer ball, drive shot / dribble genius.
2. 日向小次郎 / Kojiro Hyuga — fierce striker, dark spiky hair, intense eyes, dark/black kit with orange-yellow accents, tiger-shot force.
3. 若林源三 / Genzo Wakabayashi — serious goalkeeper, dark hair, keeper cap/headband cue, green goalkeeper jersey, large gloves, goal-net defense pose.
4. 岬太郎 / Taro Misaki — calm elegant midfielder/playmaker, soft dark hair, blue-white kit, graceful pass/combi motion.

Keep all subjects covered, sporty, non-sexualized, and age-appropriate. Preserve soccer identity first: ball, kit, boots/gloves, field lines, goal net, tactical diagrams. Put sticker-bomb density in borders, signs, tape strips, typography, and background rather than covering faces, limbs, ball, gloves, or jersey.

## Manifest paths from this session
- `/Users/nick/.hermes/profiles/jea/cache/images/neon_shiny_pokemon_manifest_20260531_1204.json`
- `/Users/nick/.hermes/profiles/jea/cache/images/neon_captain_tsubasa_popular_manifest_20260531_1259.json`
- `/Users/nick/.hermes/profiles/jea/cache/images/neon_captain_tsubasa_reroll_tsubasa_hyuga_manifest_20260531_1308.json`
