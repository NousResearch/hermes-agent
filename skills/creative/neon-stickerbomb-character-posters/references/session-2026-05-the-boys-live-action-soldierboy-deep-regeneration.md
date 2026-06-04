# The Boys live-action — Soldier Boy / The Deep regeneration notes (2026-05-21)

## Context
Nick requested repeated neon-skill generations for live-action The Boys characters, especially Soldier Boy and The Deep. The workflow produced several successful images but also exposed a useful failure-handling pattern for partial two-character batches.

## Durable lessons

### 1. Preserve `剧版` as live-action fidelity, not generic comic version
For The Boys prompts, keep `剧版` bound to live-action-recognizable anchors:
- Soldier Boy: adult blond live-action likeness, square jaw, rugged smug expression, green/gold retro military superhero armor, chest plate, gloves, vintage shield.
- The Deep: adult male live-action likeness, slick dark hair, blue-green aquatic armored wetsuit, fish-scale texture, ocean chest emblem, vain awkward expression.

Do not let neon/sticker-bomb styling turn them into generic soldier/ocean-hero archetypes.

### 2. Rotate composition on rerolls even when the same two characters repeat
Useful reroll composition swaps:
- Soldier Boy: low-angle diagonal → foreground shield depth → Dutch-angle lunge.
- The Deep: foreground fisheye/diving helmet → circular water ring → side-profile aquarium-glass split.

Keep the reroll fresh by changing the poster skeleton, not only the prompt adjectives.

### 3. Partial success handling matters
When a two-image batch partly succeeds and the other slot returns `empty_response`, do not reuse a prior image path or present old output as new. Save/patch a manifest with:
- successful item path and `status: done`
- failed item as `status: failed_empty_response`, `local_path: null`

Report clearly that one image failed and only deliver MEDIA for actual new outputs.

### 4. Retry only the failed slot, but stop after repeated empty responses
For transient empty responses, retry the failed character with a shorter compact prompt preserving:
- identity anchors
- one composition archetype
- in-scene `NickZag`
- condensed neon style keywords
- short negative block

If the same slot fails twice in the same turn, stop rather than thrashing or faking success. The useful report is: provider returned no image; no new result for that slot.

## Prompt snippets that worked

### The Deep compact prompt shape
```text
The Deep from the live-action The Boys, adult male live-action likeness, slick dark hair, blue-green aquatic armored wetsuit, fish-scale texture, ocean chest emblem, vain awkward expression, fully covered.
Vertical 3:4 glossy neon cyber-pop sticker-bomb poster. SIDE PROFILE SPLIT LAYOUT ... chrome aquarium-glass panel, sonar rings, bubbles, fish silhouettes, octopus tentacle stickers, torn waterproof labels ... NickZag on a waterproof inspection sticker ...
Negative: no photorealism, no clean official key art, no centered static pose, no watermark, no generic ocean hero, no mermaid, no ocean king.
```

### Soldier Boy compact prompt shape
```text
Soldier Boy, live-action The Boys version: blond adult male, square jaw, rugged smug stare, green/gold retro military superhero armor, chest plate, gloves, vintage shield, fully covered.
Vertical 3:4 neon cyber-pop sticker-bomb poster, Dutch angle action, off-center diagonal lunge, shield at mid-depth not covering face/chest ... NickZag on small torn badge sticker on shield strap ...
Negative: no photorealism, no clean key art, no centered pose, no watermark, no generic soldier, no nudity.
```

This compact Soldier Boy retry can still fail on provider empty-response; if so, preserve the failed status rather than substituting an old path.
