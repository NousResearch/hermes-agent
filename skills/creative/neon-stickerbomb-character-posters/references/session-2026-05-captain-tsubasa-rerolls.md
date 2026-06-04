# Captain Tsubasa / 足球小将 neon rerolls — 2026-05-31

## Scope
Session-specific notes for generating `Captain Tsubasa / 足球小将` characters in the glossy neon sticker-bomb poster style.

## Worked roster and anchors
- 大空翼 / Tsubasa Ozora: youthful soccer protagonist/captain, short dark hair, determined bright eyes, white-and-blue kit, captain armband, soccer ball, drive-shot / dribble genius. Keep age-appropriate, covered, sporty.
- 日向小次郎 / Kojiro Hyuga: fierce striker, intense dark spiky hair, sharp brows, aggressive eyes, dark/black kit with orange-yellow tiger-shot accents, powerful shot posture.
- 若林源三 / Genzo Wakabayashi: serious goalkeeper, dark hair, goalkeeper cap/headband cue, green keeper jersey, large gloves, goal-net defense pose.
- 岬太郎 / Taro Misaki: graceful playmaker/midfielder, soft dark hair, calm intelligent expression, blue-white kit, pass/combi motion.
- 石崎了 / Ryo Ishizaki: scrappy loyal defender, short dark hair, expressive determined face, blue-white kit, muddy defensive block / slide tackle / face-block energy.

## Composition rotation lessons
Repeated rerolls of 大空翼 and 日向小次郎 should be treated as fresh generations, not resends. Rotate the whole skeleton:
- 大空翼 successful variants: foreground soccer-ball dribble depth; circular overhead drive-shot ring; Dutch-angle diagonal dribble cut.
- 日向小次郎 successful variants: low-angle striker wind-up; Dutch-angle tiger-shot follow-through; foreground spinning-ball pre-shot depth; side-profile pre-strike split; diagonal airborne tiger-shot volley.

Avoid repeating a prior pose for the same character in adjacent rerolls. Use explicit negative lines such as `avoid previous foreground-ball layout`, `avoid previous circular overhead drive-shot layout`, or `avoid previous huge foreground-ball pre-shot layout`.

## Prompting specifics
- Keep soccer identity visible: one readable soccer ball, boots, kit, face, and action mechanics. Do not let typography cover face/jersey/ball.
- Use `ANATOMY LOCK` for soccer athletes: exactly two arms/two legs, matching hands if visible, correct hips/knees/ankles/shoulders, readable boot-ball relation.
- For youthful sports characters, explicitly state covered, age-appropriate, non-sexualized.
- Integrate `NickZag` as a physical match-world object: sideline tape, boot-tape sticker, glove wrist strap, match-warning sticker, shot-speed label, goal-net inspection sticker.

## Provider behavior observed
石崎了 returned `empty_response` twice with both a full prompt and compact retry. Preserve successful sibling outputs and mark the failed slot explicitly in the manifest; do not reuse an older path or pretend the failed image generated. A future retry can use an even more generic visual identity if needed: `scrappy Japanese soccer defender in blue-white kit, short dark hair, muddy face-block slide tackle`.

## Manifest lesson
Even for one- or two-character rerolls, save a fresh manifest so later `发布1/2` maps to the latest reroll batch, not an earlier Captain Tsubasa batch.
