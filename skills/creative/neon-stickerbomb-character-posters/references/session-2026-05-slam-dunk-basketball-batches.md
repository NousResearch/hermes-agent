# Slam Dunk basketball character batches — 2026-05

## Session context
Nick generated repeated neon sticker-bomb batches for *Slam Dunk* characters, first a popular set, then exact named rosters. The useful durable lesson is not the specific file paths, but how to handle basketball-anime team rosters in this style.

## Workflow lessons
- Treat a repeated `生成灌篮高手 ...` request as a **fresh generation batch**, not a resend, unless Nick explicitly asks to resend/publish previous files.
- When Nick names the roster, use those exact characters and order. Do not substitute a popular character unless a slot repeatedly fails and Nick asked only for a count rather than an exact named list.
- Save a new numbered manifest for each batch so later `发布123` or `发布2 5` maps to the latest delivered roster.
- For five-character sports-team batches, parallel generation through `image_generate` was stable enough, but still write the manifest after the outputs are known.

## Prompting anchors
Use basketball identity cues before sticker-bomb style language:

- 樱木花道 / Hanamichi Sakuragi: tall athletic Japanese high-school player, bright red cropped hair, cocky/hot-blooded expression, red basketball jersey, rebound/dunk rookie energy.
- 流川枫 / Kaede Rukawa: black hair, cool sharp eyes, lean ace scorer, white jersey or clean basketball uniform, calm fadeaway/shooting posture.
- 赤木刚宪 / Takenori Akagi: very tall muscular center/captain, shaved or buzzed dark hair, stern expression, red basketball uniform, block/rebound dominance.
- 宫城良田 / Ryota Miyagi: shorter quick point guard, dark pompadour/curly swept hair, confident eyes, red uniform, crossover/dribble playmaker energy.
- 三井寿 / Hisashi Mitsui: dark hair, intense comeback-player expression, shooting guard, three-point jump-shot energy, red/white basketball uniform.
- 赤木晴子 / Haruko Akagi: wholesome team-support/student-manager energy, youthful/covered/non-sexualized, basketball team context; avoid turning her into a pin-up sports model.

## Composition rotation that worked
For a five-image roster, pre-assign distinct skeletons:
1. Sakuragi — foreground basketball + dunk/rebound depth.
2. Rukawa — side-profile split / fadeaway.
3. Akagi — low-angle diagonal block/rebound.
4. Miyagi — dutch-angle crossover dribble with ball/sneaker foreground.
5. Mitsui — circular motion ring / three-point shot arc.

If Haruko replaces Mitsui, use a side-profile split or wholesome court-side support poster with team cards, gym tape, score fragments, and basketball-court collage; keep the subject covered and age-appropriate.

## Quality/pitfalls
- Avoid generic basketball boys: hair/body role anchors must appear before style mechanics.
- Keep hands readable around the ball; basketball prompts often create broken fingers/grips if the pose is too extreme.
- `NickZag` should be a physically attached in-scene item: athletic tape on ball, wristband label, backboard-padding sticker, sneaker label, scorecard/evidence sticker, team-card corner.
- Do not let huge typography replace identity; the image should read correctly without the name label.
