# Session note — The Boys / 黑袍纠察队 live-action neon batches

Use for future neon sticker-bomb character-poster batches when Nick asks for `黑袍纠察队`, especially `剧版黑袍纠察队`.

## Durable task lessons

- Treat `剧版` / live-action as a hard subject qualifier. Do not silently broaden the batch to generic comic/franchise versions; include `from the live-action TV version` early in every prompt.
- If Nick asks repeatedly for another four-character batch, avoid immediate repeats from the prior delivered set unless he names them. Rotate into characters not just already generated in the immediately preceding batch.
- If one or two named characters repeatedly return empty/no-image responses, do not report them as successful and do not reuse a duplicate path. Retry only the failed character with a shorter prompt; if it still fails, substitute only when Nick asked for a generic batch, and clearly say which item was substituted.
- When the final response lists a batch, never assign the same local path to two different indices unless the user explicitly asked for duplicates. If the provider returns a reused path or no image, mark that index as failed/retry-needed rather than pretending four unique images exist.
- For publication commands like `发布134到小红书`, use the latest completed delivered manifest/batch mapping and send only the selected indices to `discord:#rednote`. Keep the numbering from the visible delivered batch.

## Character anchors that worked or should be preserved

- Homelander / 祖国人: blond slicked-back hair, blue eyes, blue suit, red-white cape, gold eagle shoulder/chest armor, smug cold expression, faint laser-eye glow. Low-angle fashion diagonal works.
- Starlight / 星光: blond hair, white-and-gold covered suit, star chest motif, glowing hands, golden light ring. Keep explicitly non-sexualized and covered.
- Queen Maeve / 梅芙女王: long reddish-brown hair, stern expression, red cape, silver/gold warrior armor, bracers/boots. Low-angle diagonal sword/shield shapes work.
- The Deep / 深海: blue-green aquatic suit, fish-scale/wetsuit texture, ocean-emblem chest cue, slick dark hair, awkward smug/uneasy expression. If Nick says only `生成深海` immediately after a live-action The Boys batch, resolve it to The Deep rather than a generic underwater image. A long full-template prompt may return empty; compact retry worked with a huge cracked chrome diving helmet / fisheye lens in the lower-left foreground, off-center three-quarter pose, water-current spiral, sea-creature silhouettes, bubbles, torn warning labels, and `NickZag` on a waterproof inspection sticker on the helmet rim.
- Hughie / 休吉: soft brown hair, anxious earnest expression, practical jacket/plaid layer, cracked phone and evidence file. Side-profile split worked.
- Kimiko / 金子: dark hair, intense eyes, tactical streetwear, compact athletic silent-fighter pose. Circular motion ring worked.
- Mother’s Milk / M.M.: shaved head, trimmed beard, tactical jacket, evidence notebook/case file, disciplined expression. Side-profile split worked.
- Frenchie / 法兰奇: messy dark hair, stubble/mustache, tactical jacket, chemical vials/wires/detonator tools, acid green chemistry glow. Dutch-angle chemistry action worked.
- Black Noir / 玄色: full black mask, all-black tactical armor, glossy armor plates, short blades, smoke slash, no exposed face. Shorter compact prompt succeeded after longer prompts failed.
- Billy Butcher / 布彻: black trench coat, Hawaiian shirt hint, messy dark hair, heavy stubble/beard, hard glare, pistol low, Compound V vial, crowbar/evidence folder. This subject repeatedly returned empty responses in this session; the durable lesson is to retry with a shorter, visual-anchor-first prompt and only report success when a file exists.

## Prompting pattern after empty responses

When a character returns no image:

1. Retry only that character.
2. Shorten to one compact paragraph plus negatives.
3. Put live-action qualifier and identity anchors before style.
4. Keep `NickZag` as a physical prop/tape/sticker label.
5. If still empty and the user requested a generic four-character batch, substitute a different live-action character and disclose the substitution.

Avoid concurrent retries after multiple empty/API responses; serial compact retries were more reliable in this session.
