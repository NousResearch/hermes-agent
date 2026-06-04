# One Piece + Overlord repeat reroll notes — 2026-05-26/27

## What happened
Nick generated repeated neon sticker-bomb batches for Overlord and One Piece, often asking for the same characters again soon after prior outputs. The successful behavior was to treat each repeat request as a fresh reroll set, rotate the composition skeleton, and save a new numbered manifest for the current batch.

## Durable workflow lessons
- Repeated named-character requests like `生成安兹乌尔恭，安兹乌尔恭(莫莫)` or `重新生成娜美，罗宾` should not resend old paths. Generate fresh images unless Nick explicitly says publish/send previous results.
- Preserve character identity anchors first, then rotate layout. For Ainz: skull face, red eye glow, black/gold/white robe, huge collar, staff. For Momon: full black armor, closed helmet, twin greatswords, no skull/robe. For Nami: orange hair, clima-tact/weather, navigator map/compass. For Robin: long black hair, calm archaeologist, book/glyph/flower-hands.
- If the same character was just generated, explicitly rotate away from the last composition: e.g. foreground prop -> side split; circular ring -> foreground prop; low-angle diagonal -> side split; dutch angle -> circular ring.
- If one item in a multi-image batch times out or returns 500, preserve successful siblings and retry only the failed character with a shorter anchor-first prompt. Robin succeeded after reducing to: subject anchors + one composition + key style mechanics + in-scene NickZag + compact negatives.
- For five-image One Piece straw-hat-style batches, a useful distinct composition set is: Nami side-profile split, Robin foreground archive prop, Luffy rubber-arm circular ring, Zoro low-angle three-sword diagonal, Sanji dutch-angle flaming kick.
- For Overlord repeat rerolls, separate Ainz and Momon strongly. Ainz should be skeletal robe/staff/collar; Momon should be black armored adventurer/twin swords/no exposed skull/no robe.

## Manifest discipline
Save a fresh manifest after every completed reroll batch, even for two-image batches, so later `发布12` maps to the latest delivered numbering rather than older outputs.
