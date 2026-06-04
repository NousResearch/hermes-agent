# Gintama Neon Batch Retry Notes

Session takeaways for `neon-stickerbomb-character-posters` when generating Gintama / multi-character anime batches.

## What worked

- A full neon stickerbomb prompt with long style blocks can work for some characters, but if the provider hangs or returns text-only for a specific item, a compact retry often succeeds.
- Keep the same manifest entry and number when retrying. Do not create a new numbering scheme just because one item needed a shorter prompt.
- For known characters, preserve the smallest identity anchors that matter most:
  - Gintoki: silver hair, deadpan/tired expression, white kimono with blue trim, wooden sword, sugar/snack cues.
  - Kagura: red hair buns, blue eyes, red Chinese-style outfit, oversized purple umbrella.
  - Shinpachi: short dark hair, round glasses, blue/white dojo-style outfit.
  - Sadaharu: giant fluffy white dog, red collar.
  - Katsura: long black hair, purple kimono, Elizabeth cue.
  - Hijikata: black hair, black-and-gold Shinsengumi uniform, cigarette, katana, mayo gag.
  - Okita: light brown hair, black-and-gold Shinsengumi uniform, bazooka, sly deadpan energy.
- The compact retry can drop some decorative clauses while preserving the core style contract: neon stickerbomb, glossy highlights, thick black manga ink, halftone, graffiti, torn decals, warning labels, huge cropped typography.

## Retry pattern

1. Keep the original manifest and item index.
2. Rewrite the prompt shorter if the first attempt hangs or returns no image.
3. Preserve the identity anchors first, style second.
4. Re-run only the failed item, not the whole batch.
5. QC the result before publishing or replying.

## Practical note

For this class of batch, the best balance was often:
- short subject description,
- one composition archetype sentence,
- one compact style block,
- one negative block.

That was more reliable than an over-long prompt with too many lore or typography details.
