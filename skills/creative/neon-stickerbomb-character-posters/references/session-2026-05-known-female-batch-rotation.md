# 2026-05 known female anime batch rotation notes

Use when Nick asks for repeated batches like `生成4张知名女性动漫角色` or `使用skill生成4张知名女性动漫角色` in the neon sticker-bomb poster style.

## Session learning

Nick repeatedly requests batches, then often publishes selected indices (`发布 1 3 4`) to image2skill. The batch must therefore optimize for:

1. recognizable well-known female anime/game characters;
2. visible composition diversity across the four images;
3. no immediate repetition of characters from recent batches;
4. stable numbered manifest mapping for later publication.

## Characters already used heavily in this run

Avoid repeating these in the immediate next few generic `知名女性动漫角色` batches unless Nick names them explicitly:

- Saber / 阿尔托莉雅
- Rem / 蕾姆
- C.C. / C2
- Violet Evergarden / 薇尔莉特
- Nico Robin / 罗宾
- Yor Forger / 约尔
- Esdeath / 艾斯德斯
- Winry Rockbell / 温莉
- KOF Angel
- KOF: Mai Shiranui / 不知火舞, Leona Heidern / 莉安娜, Shermie / 夏尔米, Blue Mary / 布鲁玛丽
- Akame / 赤瞳
- Revy / 蕾薇
- Yoko Littner / 优子
- Nausicaä / 娜乌西卡
- Frieren / 芙莉莲
- Riza Hawkeye / 莉莎·霍克艾
- Holo / 赫萝
- Utena Tenjou / 天上欧蒂娜
- Makima / 玛奇玛
- Maka Albarn / 玛嘉
- Boa Hancock / 女帝
- Shampoo / 珊璞

## Good next-pool candidates

For future generic batches, rotate toward characters not just generated:

- Motoko Kusanagi / 草薙素子 (if not immediate-repeat from earlier session)
- Lum / 拉姆
- Meryl Stryfe / 梅丽尔
- Major Kusanagi alternatives only if not recent
- Tohsaka Rin / 远坂凛
- Asuka Langley / 明日香 (covered/non-sexualized)
- Misato Katsuragi / 葛城美里
- Integra Hellsing / 因特古拉
- Celty Sturluson / 塞尔提 (helmet/urban rider)
- San / 珊公主
- Kiki / 琪琪 (wholesome/age-appropriate)
- Motoko/Kusanagi-like cyber roles only one per batch
- Faye Valentine / 菲 (avoid if recent)
- Android 18 / 人造人18号 (avoid if recent)

## Batch composition recipe

For each four-image batch, pre-assign one of each:

1. **Foreground prop depth** — gun, blade, lens, tool, fan, microphone, card.
2. **Circular motion ring** — spell circle, tail, ribbon, hair, fire, petals, chains.
3. **Side profile split** — calm/tactical/elegant character with negative-space typography.
4. **Low-angle or dutch-angle action** — fighter, duelist, rider, dancer, shooter.

Do not place two gunfighters, two swordswomen, or two calm side-profile characters in the same batch unless Nick specifically asks.

## Manifest requirement

Every four-image batch should write/update a manifest under `~/.hermes/profiles/jea/state/` with index, title, semantic filename, path, full prompt, status, elapsed seconds, model, skill, and folder. Later `发布 1 3 4` must resolve against the latest manifest, not the visible Discord order or Eagle order.

## Publication reminder

`发布到 image2skill` and `发布 1 3 4` mean Eagle-backed frontend publication by default, not Discord channel posting. Use the latest manifest to import selected items into Eagle `neon` with `image2skill` tag, then rebuild and verify the frontend.
