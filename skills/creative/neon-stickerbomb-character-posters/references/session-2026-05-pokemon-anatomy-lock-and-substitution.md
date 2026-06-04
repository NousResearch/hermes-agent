# Session note — Pokémon anatomy lock, repeated empty responses, and substitution

## Context

Nick corrected the neon sticker-bomb workflow after several Pokémon generations showed or risked wrong hands/limbs. The correction is class-level: anatomy is not a cosmetic detail; a result with wrong hands, mismatched paired limbs, extra/missing/fused limbs, or broken joints should not be treated as successful even if the style surface is strong.

## Durable lessons

- Add an `ANATOMY LOCK` before style language whenever hands, limbs, paws, wings, tails, weapon grips, or other appendages are visible.
- State the body plan positively: exact arm/leg/wing/tail count, finger/paw/claw count, matching left/right anatomy, and readable attachment points.
- Keep sticker-bomb density away from hands, joints, wing roots, paws, tail attachments, and weapon-grip junctions.
- If a dynamic pose causes anatomy drift, reduce foreshortening and use an anatomy-safe diagonal / side split / controlled motion pose rather than escalating pose drama.
- Visual acceptance must check anatomy after generation: style quality alone is not enough.

## Pokémon-specific provider pattern from this session

- Mewtwo / 超梦 repeatedly triggered `empty_response` from the image backend when prompted with long direct-IP prompts and later even compact prompts.
- The one successful Mewtwo retry came from an ultra-compact visual-anchor prompt: pale gray psychic alien monster, purple belly, huge purple tail, smooth feline head, three fingers, glowing orb, off-center dynamic pose, stickerbomb/cyan-magenta style, `NickZag` tape tag.
- Zapdos / 闪电鸟 failed twice with `empty_response`; because Nick asked for four rare Pokémon rather than an exact roster, completing the count by substituting Raikou / 雷公 was acceptable, but the substitution must be reported clearly and captured in the manifest.

## Manifest / reporting rule

- For repeated rerolls or failed slots, preserve successful siblings and retry only failed items.
- If all attempts fail, write a manifest marking the slot failed and do not reuse an older image path as if it were new.
- If substituting to satisfy a count request, mark `done_with_substitution`, record the failed subject and attempts, and explain the substitution in the final response.
