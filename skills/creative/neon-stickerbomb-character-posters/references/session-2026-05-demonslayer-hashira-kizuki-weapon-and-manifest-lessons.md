# Demon Slayer Hashira / Twelve Kizuki generation lessons — 2026-05

## Context
Nick generated multiple Demon Slayer character posters in the neon sticker-bomb style, including Hashira rerolls and Twelve Kizuki batches. The main repeated correction was character/prop fidelity under heavy sticker-bomb styling.

## Durable lessons

### Mitsuri Kanroji / Love Hashira whip-sword
- Nick rejected versions where Mitsuri's sword hilt did not physically connect to the flexible ribbon-like blade.
- A visually pretty sticker-bomb poster is not enough if the weapon mechanics are wrong.
- The prompt must keep the hilt/guard/blade junction large, unobscured, and readable.
- Specify: `katana-style handle and flower-shaped guard held by both hands; the flexible metallic whip-blade emerges from the FRONT of the guard as one continuous attached blade; first 20% of the blade connection is visible and not covered by stickers/hair/effects`.
- Push sticker density to borders/background around the weapon junction; do not place NickZag or tape directly over the junction.
- Negative block should include: `blade from handle bottom`, `detached blade`, `floating ribbon`, `hair replacing blade`, `covered junction`, `hidden guard`.
- If two correction loops still fail, stop thrashing and switch composition: half-body / three-quarter close-up with the hilt/guard/blade junction as the central readable focal point.

### Dynamic pose versus readable hands
- Nick rejected stiff/standing Hashira poses, but also rejected avoiding hands by cropping or hiding them.
- For sword users, dynamic prompts should require both hands visible and readable, clean hilt contact, and a diagonal/twisting pose rather than extreme foreshortening that breaks anatomy.

### Parallel batch mapping failure to avoid
- In one Twelve Kizuki batch, a failed slot was followed by successful sibling images. The assistant incorrectly reported the sibling paths under the wrong character labels.
- Future handling: after any parallel `image_generate` batch, map each result by its actual tool-call slot and success status. Never shift labels upward to fill a failed slot. Do not include a failed item as `MEDIA:` or as a completed manifest row.
- Save a dedicated batch manifest for the current subject family rather than appending unrelated Demon Slayer villain entries into an existing Hashira manifest.

### Twelve Kizuki batching
- For broad prompts like `生成鬼灭之刃的十二鬼月角色图`, default to a recognizable subset first if generating all twelve at once is unstable, but report partial completion accurately.
- Better first-set roster: Kokushibo, Doma, Akaza, Hantengu, Gyokko, Gyutaro/Daki, Enmu, Rui, Nakime/Kaigaku as appropriate. Keep exact generated labels aligned to the manifest.
