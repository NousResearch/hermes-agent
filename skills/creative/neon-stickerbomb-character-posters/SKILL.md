---
name: neon-stickerbomb-character-posters
description: Use when generating character images in a stable glossy neon cyber-pop sticker-bomb poster style from a user-described character, pose, prop, palette, or scene element. Builds prompts without retaining third-party artist, creator, brand, or studio names; focuses on reusable style mechanics, composition diversity, and in-scene graphic integration.
version: 1.0.2
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [image-generation, character-art, prompt-engineering, cyber-pop, poster-design]
    related_skills: [imagegen-creative-direction]
---

# Neon Sticker-Bomb Character Posters

## Overview

This skill turns a user-provided character description, role, prop, palette, or visual element into a stable high-impact character-poster prompt. The style is defined by visual mechanics, not by any third-party creator, artist, brand, studio, or franchise reference.

The target look is a glossy neon cyber-pop poster: saturated color, thick manga/comic ink, sharp cel shading, wet specular highlights, sticker-bomb typography, torn poster fragments, graffiti tags, halftone texture, chromatic offset, fashion accessories, and energetic asymmetrical layout.

The user may give only a character name, or may specify elements such as “red boxing gloves”, “black wings”, “foreground fan”, “pink fox detective”, “EVA-like pilot suit”, “poison butterfly sword”, etc. Extract the character identity cues from the user, then build a prompt using the framework below.

**Important boundary:** the reusable skill must not include or depend on third-party artist names, creator names, brand names, style labels, studio names, watermark text, or franchise names. If the user supplies an IP character, you may use the user-supplied identity cues in the task prompt, but the style system itself stays generic and creator-neutral.

## When to Use

Use this skill when the user asks to:

- Generate a character poster in the established glossy neon sticker-bomb style.
- “Use the same style” from prior batches without mentioning any third-party creator/brand.
- Convert a character name or role into a stable image prompt.
- Add specific props, typography, color palettes, pose archetypes, or layout variation.
- Prevent repeated centered poses or same-looking poster layouts.
- Integrate `NickZag` as a creator-information element inside the picture.

Do **not** use this skill for:

- Photorealistic portraits.
- Minimal clean key art.
- Official merch-style mascot renders.
- Pure logo design.
- Website UI, product mockups, or slide design.
- Direct imitation of a named living artist or third-party creator style.

## Core Style Contract

Every generated prompt should preserve these mechanics unless the user asks otherwise:

1. **Poster format**
   - Vertical 3:4 character poster by default.
   - Character or mascot should be the main subject, but not always centered.
   - Use deliberate asymmetry, cropped edges, and strong visual hierarchy.

2. **Rendering**
   - Thick bold black manga/comic outlines.
   - Sticker-cut contour edges.
   - Sharp cel-shading with deep black/purple/crimson shadow blocks.
   - Wet glossy highlights on hair, skin/fur, clothing, metal, plastic, stickers, glass, and props.
   - Electric cyan and hot magenta rim lighting.
   - High saturation, high contrast, loud color clashes.

3. **Texture and print language**
   - Halftone dots.
   - Spray-paint grain.
   - Rough offset-print registration.
   - Screenprint texture.
   - Chromatic aberration / cyan-magenta edge split.
   - Torn vinyl decals and layered poster scraps.

4. **Graphic density**
   - Sticker-bomb collage.
   - Huge cropped typography.
   - Torn tape strips.
   - Graffiti tags.
   - Barcode fragments.
   - Fake warning labels or evidence labels.
   - Small icon systems tied to the character’s role and props.

5. **Fashion remix**
   - Preserve the user-specified character identity.
   - Add streetwear / cyber-fashion / punk accessory logic: straps, buckles, patches, enamel pins, tags, chains, badges, gloves, belts, utility pouches, holographic labels.
   - For mascots: use cute streetwear, explorer gear, vinyl-toy gloss, stickers, satchels, ribbons, accessories; never adultify or sexualize.

6. **In-scene creator info**
   - If appropriate, include exact readable text `NickZag`.
   - `NickZag` must be a picture-world graphic element, **not** a watermark, logo stamp, floating overlay, or corner signature.
   - Integrate it as graffiti on torn tape, a printed sticker, a clothing patch, a prop label, a card label, a barcode/evidence tag, a poster fragment, or a lens reflection.
   - It should be recognizable but not dominant, sharing the scene’s lighting, perspective, ink texture, halftone, and distortion.

## Prompt Assembly Procedure

When the user gives a character or element, build the prompt in this order.

### 1. Identify the Subject

Extract and state:

- Character/persona name or archetype.
- Species or body type if relevant: human, armored hero, mascot, creature, robot, etc.
- Core recognizability cues: hair, eyes, costume silhouette, weapons, props, color language, role.
- Tone: elegant, rebellious, cute, dangerous, calm, heroic, playful, gothic, sporty, etc.
- Safety boundary: adult/non-sexualized/wholesome/covered as appropriate.

Example structure:

```text
Vertical 3:4 glossy neon cyber-pop sticker-bomb character poster of [SUBJECT], faithful to the user-supplied identity cues, redesigned as a high-saturation [ROLE / FASHION ARCHETYPE]. [Safety/tone sentence].
```

### 2. Choose a Distinct Composition Archetype

Do not repeat the same centered layout across a batch. Pick one archetype based on the character’s prop, motion, personality, or prior outputs.

#### A. Foreground Prop + Depth

Use for characters with fans, cards, weapons, boxing gloves, lenses, microphones, masks, food, phones, or tools.

```text
COMPOSITION ARCHETYPE — FOREGROUND PROP + DEPTH, NOT CENTERED POSTER:
A huge [PROP] dominates the extreme foreground from [corner], tilted diagonally and partly framing [SUBJECT] behind it. [SUBJECT] is off-center with strong foreshortening and cropped edges. Typography and stickers follow the prop’s angle instead of sitting as a flat background title wall.
```

#### B. Circular Motion Ring

Use for fire, wings, ribbons, hair, water, butterflies, smoke, chains, tail, scarf, energy arcs.

```text
COMPOSITION ARCHETYPE — CIRCULAR MOTION RING:
[SUBJECT] moves through the frame in an S-curve while [hair / wings / flames / ribbons / smoke / butterflies] forms a large circular arc around the body. Typography follows the ring and breaks across the motion trail. Avoid centered static pose.
```

#### C. Side Profile Split Layout

Use for calm, cold, elegant, mysterious, quiet, or fashion-cover characters.

```text
COMPOSITION ARCHETYPE — SIDE PROFILE SPLIT LAYOUT:
Place [SUBJECT] on the right or left third in a three-quarter side profile / over-the-shoulder pose. The opposite two-thirds contain oversized cropped typography, torn stickers, color blocks, and controlled negative space. Avoid full-centered front pose.
```

#### D. Low-Angle Fashion Diagonal

Use for proud, rebellious, heroic, fighter, racer, armored, or dominant characters.

```text
COMPOSITION ARCHETYPE — LOW-ANGLE FASHION DIAGONAL:
A low-angle editorial camera shows [SUBJECT] leaning diagonally through the frame with confident swagger. A giant diagonal stripe / weapon / shadow / cape shape cuts across the poster, carrying typography and stickers. Avoid generic punch pose.
```

#### E. Dutch-Angle Action Poster

Use for boxers, dancers, spies, street fighters, urban characters, musicians.

```text
COMPOSITION ARCHETYPE — DUTCH-ANGLE ACTION POSTER:
Tilt the entire poster rhythm. [SUBJECT] is off-center in a twisted action/fashion stance. Foreground [prop/body part] creates depth; background ropes, signs, architecture, or street fragments create diagonal motion.
```

#### F. Floating/Reclining Diagonal

Use for elegant queens, gothic characters, dreamlike characters, winged characters, sorcerers.

```text
COMPOSITION ARCHETYPE — DIAGONAL FLOATING / RECLINING:
[SUBJECT] crosses the frame diagonally in an elegant floating S-curve, as if suspended in smoke, feathers, petals, ribbons, or light. Hair/cloak/wings trail across the canvas and wrap typography around the body. Avoid low-angle reaching-hand pose and vertical text-column layout.
```

#### G. Mascot Scanner / Toy-Poster Remix

Use for wholesome mascots and cute creatures; keep them mascot-like.

```text
COMPOSITION ARCHETYPE — CHROME SCANNER + GLITCH STICKER-BOMB:
A huge chrome/holographic [magnifying glass / camera / badge / toy prop] dominates the foreground with fisheye distortion, neon rim light, pixelated overlays, and corrupted reflections. The mascot stays plush-like and wholesome but gains sharper cyber-pop poster energy. Avoid official merchandise, scrapbook softness, and humanized adult redesign.
```

### 3. Add Creator-Info Integration

Use this unless the user says not to include `NickZag`.

```text
Important creator-credit integration:
Include the exact readable text “NickZag” as an IN-SCENE GRAPHIC DESIGN ELEMENT, not a watermark or overlay. Integrate it once as a small but recognizable handwritten tag printed on [prop label / torn sticker / clothing patch / tape strip / card / lens reflection / poster fragment]. It must share the same ink texture, halftone grain, lighting, perspective, distortion and print texture as the artwork. Recognizable but not dominant.
```

Avoid:

```text
watermark, floating signature overlay, corner logo, separate stamp outside the composition, large obvious author mark, NickZag only as a dangling charm unless the charm is narratively integrated into the prop design.
```

### 4. Preserve Character Fidelity

Use user-provided cues. If user supplies only a name, infer visually stable cues conservatively. Do not overfit the style so hard that the character becomes generic.

```text
Character fidelity:
Preserve [SUBJECT] clearly: [cue 1], [cue 2], [cue 3], [core outfit/role], [prop/weapon], [expression/personality], [color language]. The subject must read immediately as [SUBJECT], not a generic [archetype].
```

For mascots:

```text
Keep [SUBJECT] as a mascot/anthropomorphic creature with rounded plush-toy proportions. Do not humanize into an adult woman/man. Keep wholesome, playful, and non-sexualized.
```

### 5. Style Mechanics Block

Reusable block:

```text
STYLE MECHANICS:
High-saturation glossy neon cyber-pop poster, thick bold black manga/comic ink outlines, sticker-cut contour edges, sharp cel-shading, deep black/violet/crimson shadow blocks, wet glossy white highlights on hair, skin/fur, clothing, props, metal, plastic, glass and sticker surfaces, electric cyan and hot magenta rim lighting, acid accent highlights, chromatic aberration, halftone dot shadows, spray-paint grain, rough offset-print registration, screenprint texture, glitch-rim outlines, torn vinyl decals and aggressive sticker-bomb layering.
```

### 6. Fashion / Accessory Remix Block

Use a role-specific version:

```text
FASHION / ACCESSORY REMIX:
Preserve the subject’s core silhouette and color language, but remix it with glossy street-fashion details: cropped technical jacket, utility straps, metallic buckles, belts, enamel pins, patches, small chains, gloves, rings, satchel/pouches, holographic labels, torn tag ornaments, printed fabric graphics and prop labels. Stylish, character-faithful, and covered; no unnecessary nudity.
```

For mascots:

```text
Keep the mascot’s original role language but make it street-poster iconic: mini utility jacket/capelet, satchel, enamel pins, ribbons, clue cards, stickers, holographic labels, glossy toy-like highlights, cute but sharper accessories. Wholesome and plush-like, not mature or adultified.
```

### 7. Decorative Graphics Block

Tie motifs to the character role, not random clutter.

```text
DECORATIVE GRAPHICS:
Add printed/sticker motifs on clothing, props, accessories and background layers: [role motifs], [flora/fauna/weapon/element motifs], barcode strips, fake labels, small icon systems, handwritten marks, script fragments, holographic decals, warning/evidence stickers, graffiti tags, halftone cutouts and torn poster fragments. These should feel like graphic overload on fabric/poster layers, not exposed-skin sexualization.
```

### 8. Background and Typography Block

```text
BACKGROUND:
Chaotic urban street-poster collage, not clean official key art. Use huge cropped angled typography “[NAME / TITLE]”, secondary role phrases “[PHRASE 1]”, “[PHRASE 2]”, user-supplied language text if requested, torn poster fragments, diagonal tape strips, sticker labels, graffiti tags, barcode graphics, [role-specific silhouettes], [element trails], light streaks, warning/evidence labels, halftone dots and spray-paint marks. Typography should follow the chosen composition’s motion path instead of sitting as a neat centered title wall.
```

### 9. Palette Block

```text
PALETTE:
Use [subject core colors] contrasted with deep black, electric cyan, hot magenta, acid accent colors, chrome white, deep violet shadows, red/orange warning accents and pure white specular shine. Extremely saturated, high contrast, glossy, loud, stylish.
```

For cute mascots, do not let the palette become soft-only:

```text
Soft mascot colors must be contrasted with deep black, electric cyan, hot magenta, acid yellow, chrome white and high-contrast shadow blocks. Avoid pastel-only palette.
```

### 10. Anatomy Lock + Negative Constraints

Always include a tailored negative block. Anatomy correctness is mandatory, not optional: every visible hand, foot, limb, tail, wing, weapon grip, and joint must match the subject’s real/canon body plan.

For any character with visible hands or limbs, add a positive anatomy line before the negative block:

```text
ANATOMY LOCK:
Preserve the subject’s correct limb count and matching left/right anatomy. Both hands must follow the same species/canon structure and finger count; wrists, elbows, knees, shoulders, hips, tail/wing attachment points and visible joints must be physically consistent and readable. Use a slightly less extreme pose if needed to keep limbs correct.
```

```text
Negative constraints:
no photorealism, no watercolor, no muted colors, no clean official key art, no organized vertical text column, no boring centered front-facing pose, no repeated standard poster layout, no minimalist empty background, no soft children’s book illustration, no clean merchandise render, no watermark, no floating signature overlay, no separate logo stamp outside the composition, no large obvious author mark, no loss of subject identity, no deformed hands/props, no extra fingers, no mismatched left/right hands, no inconsistent hand anatomy, no wrong limb count, no duplicated limbs, no missing limbs, no fused limbs, no broken joints, no impossible elbows/knees, no asymmetrical accidental body parts, no malformed feet/tails/wings.
```

Add subject-specific negatives:

- For covered/human characters: `no sexualization, no nudity, no cleavage emphasis, no exposed intimate skin, no adult-rated presentation`.
- For mascots: `no humanized adult body, no realistic animal, no scary monster version, no pin-up pose, no sexualization, no wrong paw count, no mismatched paws, no broken tail attachment`.
- For weapons/props: `no missing [prop], no deformed [prop]`.
- For identity: `no wrong hair/color/outfit, no missing defining feature`.
- For species/body-plan fidelity: explicitly state the correct number of arms, legs, fingers/claws/paws, wings, tails, horns, or other appendages; block extra/missing/fused versions of those parts.

## Prompt Template

Use this as the canonical prompt skeleton.

```text
Vertical 3:4 glossy neon cyber-pop sticker-bomb character poster of [SUBJECT], faithful to the supplied identity cues, redesigned as a high-saturation [ROLE / FASHION ARCHETYPE]. [Safety/tone sentence].

COMPOSITION ARCHETYPE — [NAME], NOT CENTERED POSTER:
[Composition archetype paragraph customized to subject, prop, camera angle, motion path, asymmetry, typography flow and anti-repetition constraints.]

Important creator-credit integration:
Include the exact readable text “NickZag” as an IN-SCENE GRAPHIC DESIGN ELEMENT, not a watermark or overlay. Integrate it once as a small but recognizable handwritten tag printed on [specific scene object]. It must share the same ink texture, halftone grain, lighting, perspective, distortion and print texture as the artwork. Recognizable but not dominant.

Character fidelity:
Preserve [SUBJECT] clearly: [identity cues]. The subject must read immediately as [SUBJECT], not a generic [archetype].

STYLE MECHANICS:
High-saturation glossy neon cyber-pop poster, thick bold black manga/comic ink outlines, sticker-cut contour edges, sharp cel-shading, deep black/violet/crimson shadow blocks, wet glossy white highlights on hair, skin/fur, clothing, props, metal, plastic, glass and sticker surfaces, electric cyan and hot magenta rim lighting, acid accent highlights, chromatic aberration, halftone dot shadows, spray-paint grain, rough offset-print registration, screenprint texture, glitch-rim outlines, torn vinyl decals and aggressive sticker-bomb layering.

FASHION / ACCESSORY REMIX:
Preserve the subject’s core silhouette and color language, but remix it with [role-specific fashion/accessories]. Stylish, character-faithful, and covered; no unnecessary nudity.

DECORATIVE GRAPHICS:
Add printed/sticker motifs on clothing, props, accessories and background layers: [role motifs], barcode strips, fake labels, small icon systems, handwritten marks, script fragments, holographic decals, warning/evidence stickers, graffiti tags, halftone cutouts and torn poster fragments.

BACKGROUND:
Chaotic urban street-poster collage, not clean official key art. Use huge cropped angled typography “[NAME]”, secondary phrases “[PHRASE 1]”, “[PHRASE 2]”, torn poster fragments, diagonal tape strips, sticker labels, graffiti tags, barcode graphics, [role-specific silhouettes], [element trails], light streaks, warning/evidence labels, halftone dots and spray-paint marks. Typography should follow the chosen composition’s motion path instead of sitting as a neat centered title wall.

PALETTE:
Use [subject core colors] contrasted with deep black, electric cyan, hot magenta, acid accent colors, chrome white, deep violet shadows, red/orange warning accents and pure white specular shine. Extremely saturated, high contrast, glossy, loud, stylish.

QUALITY:
ultra detailed, high contrast, saturated neon colors, thick black manga/comic ink, glossy highlights, dynamic asymmetrical character poster composition, cyber-pop street fashion, pop-art character illustration, sticker-bomb typography, premium polished poster art.

Negative constraints:
[Global negatives + subject-specific negatives.]
```

## Composition Rotation Rules

To keep a batch from looking repetitive:

- For repeated reroll requests like `重新生成4张18号`, treat the request as a fresh batch, not a resend of the previous files.
- Rotate the whole poster skeleton, not just the subject name. For canon-heavy rerolls, preserve the core silhouette first and move sticker density to borders, typography, tape, and prop surfaces.
- Save a new manifest for each reroll set so later selected publishes like `发布23到小红书` map to the correct batch.


1. Never use the same archetype twice in a row unless the user asks.
2. Track the last generated layout mentally in the current conversation.
3. Change at least four of these between images:
   - Camera angle.
   - Body orientation.
   - Subject placement.
   - Main prop placement.
   - Typography direction.
   - Negative-space distribution.
   - Motion path.
   - Dominant background geometry.
4. Avoid the default “subject centered, fills 75%, big title behind, stickers at lower corner” layout.
5. If the user says “same style but different pose/layout,” explicitly state the new archetype before generating.
6. When the user requests a batch of multiple characters, pre-assign a different archetype to every image before calling generation. Do not rely on minor prompt variations; each image needs a visibly different poster skeleton.
7. Rotate where `NickZag` lives as an in-scene object: prop label, clothing patch, torn tape, poster fragment, card corner, lens reflection, evidence tag, etc. Do not repeatedly put it near the same corner or edge.
8. For silhouette-driven canon subjects, treat pose dynamism as a first-class requirement. If a result reads as stiff, static, or mannequin-like, reroll with stronger motion language: airborne diagonal, foreshortened foreground limb, tail/weapon arc, twist through the torso, or circular motion ring. Keep identity anchors readable, but do not let the subject default to a straight-on standing poster.
9. For canon-sensitive rerolls, preserve the known silhouette/costume anchors first and move stickerbomb density to background, borders, tape, and typography. If the user flags outfit/hair mismatch, tighten the prompt around those anchors instead of only adding style adjectives.
10. For parody/crossdress requests, keep the character immediately recognizable and non-explicit. Use covered stage/fashion parody framing rather than lingerie or body-focused sexualization unless the user explicitly asks for a different, policy-allowed direction.


## Handling Minimal User Inputs

If the user gives only a character name, do not ask for clarification unless identity is genuinely ambiguous. Infer a stable prompt using:

- Recognizable visual cues.
- Character role / archetype.
- A composition archetype that differs from the previous image.
- `NickZag` in-scene graphic integration unless the user disabled it.
- Non-sexualized/covered safety constraints unless the user’s request is clearly for a permitted adult fashion illustration.

If the character is not visually known or ambiguous, ask one concise clarification:

```text
这个名字可能有多个版本。你要哪类识别点：发型/服装/武器/配色/性格？给我 2-3 个关键词我就能按同风格生成。
```

## Safety and Identity Rules

- Keep minors and school-age characters non-sexualized, covered, and age-appropriate.
- For mascot or animal characters, keep them mascot-like or creature-like; do not humanize into adult pin-up bodies.

## Known-IP Creature/Mecha Canon Fidelity

When the subject identity is silhouette-driven — Evangelion Angels, EVA units, kaiju, mecha, mascots, creatures — surface style success is not sufficient. Nick explicitly rejected Angel outputs that were attractive neon posters but far from the source shapes. For these subjects, prompt and QC in this priority order:

1. Canon/source silhouette and anatomy fidelity.
2. Key colors, face/mask/core/weapon/body anchors.
3. Subject remains clean, large, and readable without relying on text labels.
4. Neon sticker-bomb density moves to background, borders, tape, UI labels, typography, and warning cards.
5. Optional in-scene NickZag tag.

Use a hard prompt line such as: `CANON-TIGHT SILHOUETTE PRIORITY: subject fidelity first, neon sticker-bomb treatment second. Keep the body clean and immediately recognizable; put sticker density in the background and borders, not covering or redesigning the subject.`

QC must answer: would this still read as the named subject if all labels were removed? If not, mark it not usable even when the neon surface style is strong. Report as `style strong, shape drift significant` instead of calling it passed.

See `references/session-2026-05-evangelion-angels-canon-fidelity.md` for the Angel-specific correction and retry pattern.
- Avoid explicit nudity, exposed genitals, exposed nipples, or adult-rated framing.
- Avoid direct named-artist imitation.
- Do not write third-party creator, artist, brand, studio, or franchise names into the style instruction.
- If the user supplies a known character, preserve identity through user-supplied cues, but do not add unauthorized brand logos unless they explicitly request and policy allows.

## Batch-generation lessons from use

- For stickerbomb + manga/comic-page hybrid requests, do not let the stickerbomb style collapse the result into one unbroken poster. If Nick says to keep the style but still needs “漫画分镜分隔” or panel separation, make the page requirement explicit and early: clear 5–7 distinct comic panels, thick irregular black/white gutters, visible hard panel borders, diagonal manga cuts, and storyboard flow. Allow stickers, torn tape, paint splatter, halftone bursts, and typography to cross borders slightly, but each panel must remain readable as a separate story moment. Add negatives such as `no single unbroken poster composition`, `no missing panel borders`, and `no unclear gutters`.
- When Nick asks that generated content/text be presented with “中英文字母”, constrain visible typography to Chinese characters and English alphabet letters only. Explicitly allow readable labels such as `小乔 XIAO QIAO`, `李白 LI BAI`, `对线 DUEL`, and alphabetic SFX like `WHOOSH`, while excluding numbers, kana, Arabic script, random symbols, watermark-like text, and messy pseudo-typography. Keep this as a prompt constraint, not an after-the-fact guarantee, because image models may still distort text.
- For repeated `generate [character]` requests, always rotate to a visibly different composition archetype. Do not only change the named character while reusing the same centered poster skeleton.
- For repeated reroll requests that say `重新生成` or `使用neon skill 重新生成`, treat the request as a fresh batch, not a resend. Keep the latest successful outputs in the manifest, but do not silently recycle an older path as a new image.
- If the provider alternates `empty_response` between sibling items in a batch, preserve the successes, retry only the failed slot with a shorter anchor-first prompt, and report the failure plainly if it still does not return an image.
- For silhouette-driven villains/monsters, explicit motion must be part of the prompt. Add phrases like `EXTREME MOTION POSE`, `airborne diagonal lunge`, `diving pounce`, `tail spiraling across the foreground`, `one giant foreground hand`, `wings flared wide`, or `energy orb thrust toward viewer`. If the image comes back statically posed, reroll with stronger action-language before changing identity cues.
- When Nick says the pose is `呆板`, treat that as a composition failure, not a style failure: keep the canon anchors and rewrite only the action skeleton toward stronger foreshortening, diagonal energy, and moving limbs/tails/wings.
- For Dragon Ball villain rerolls, Frieza should lean into compact aerial cruelty with the tail as a foreground motion device; Cell should lean into a diving pounce or attack dash with one hand in extreme foreground and wings flared; slim/Super Buu should lean into elastic twisting kick-and-beam motion rather than a static float or stance.

- Support file: `references/session-2026-05-theboys-reroll-and-demonslayer-main4.md` for live-action The Boys reroll handling and Demon Slayer four-character batch anchors.
- For Rednote/Xiaohongshu captions, avoid explicit neon/stickerbomb visual-style wording and do not use `#neonstickerbomb`. Keep captions centered on character temperament, story tension, recognizable traits, and role archetype.
- When a user asks to publish a named subset drawn from multiple recent generation batches, resolve those names back to the correct manifest entries by character title first; do not assume the latest batch only. Use `discord:#rednote` for the mirror channel unless the user explicitly requests a different destination.
- If the same turn combines publication and fresh generation, preserve prior manifests and append a new follow-up manifest with `published_to_rednote` so later numbered publication requests remain traceable.
- If the user asks for multiple known female anime characters without specifying names, choose a varied set across franchises/archetypes and vary composition at the same time: e.g. tactical fighter, navigator, ice swordswoman, punk blade fighter, magical guardian, inventor, bounty hunter. Avoid repeating characters generated in the immediately preceding batch unless requested.
- When adapting cute or child/mascot characters into this style, preserve their original age/species proportions and comedy/mascot identity. Add neon/sticker/print aggression through props, background, and texture, not through adult fashion anatomy.
- If regenerating Evangelion Angels after a non-neon or wrong-style batch, use this skill directly rather than adapting Silver Rebellion prompts. Preserve Angel identity before style surface: Adam = primordial pale seed/halo/core, Lilith = faceless ivory restrained terminal entity, Sachiel = white skull-mask + dark body + red core, Shamshel = magenta segmented aerial serpent + energy-whip arms. For fresh rerolls, use distinct composition archetypes rather than reusing a centered poster skeleton; QC should call out if Lilith becomes too humanoid/sexualized, Sachiel becomes a generic demon with too many masks, or Shamshel becomes a mechanical dragon/centipede instead of a segmented aerial serpent with energy whips.
- If Nick says Evangelion Angels differ too much from the TV/anime source, treat it as a **shape-fidelity failure**, not a style failure. Regenerate with `CANON-TIGHT SILHOUETTE PRIORITY: subject fidelity first, neon sticker-bomb treatment second`; keep the Angel body clean/readable and move sticker-bomb density to borders, background, labels, and typography. Text labels cannot substitute for silhouette. QC must explicitly judge whether each image is `style close, shape drift` or actually usable. See `references/session-2026-05-evangelion-angels-canon-fidelity.md`.
- For EVA late-unit rerolls: Unit-09 passes more reliably when the scythe is described as a physical held halberd/scythe with visible handle, crescent blade thickness, and tip, plus a single strong red-orange chest/face core and restraint cables/clamps. **Do not make Unit-09 all silver/full chrome**: use mostly white/off-white or warm ivory lacquered armor, black biomechanical undersuit, orange-gold/yellow helmet and face-plate cue, red-orange face/chest core, and only small chrome edge glints. Unit-08+02 passes more reliably when described as a `true interwoven hybrid, not clean vertical half split`: pink plates invade red zones, red/orange plates invade magenta zones, with a green-glass sniper scope and a red forked lance framing but not covering the face. See `references/session-2026-05-eva-09-and-08plus02-second-reroll.md` and `references/session-2026-05-eva-unit09-color-fidelity-and-publish.md`.
- For repeated generic prompts like `生成4张知名女性动漫角色`, avoid reusing characters from the immediate prior batches unless Nick names them. Pick four recognizably different archetypes and pre-assign four distinct composition skeletons before generation.
- Rednote mirror send details are captured in `references/rednote-discord-mirror-publication.md`: use `discord:#rednote`, retry selected sends there after closed-session errors, and preserve numbered image selection from manifests.
- For known-IP human/mascot rerolls, protect canon costume/hair anchors before applying the neon sticker-bomb remix. If Nick corrects a specific outfit or hairstyle, add a `CANON-FIDELITY PRIORITY` line and keep the corrected anchors large, clean, and unobscured. Move sticker density to background, borders, typography, tape, and prop surfaces rather than covering the identity anchors.
- For `NickZag`, prefer physically attached placements: prop labels, tape strips, stickers, clothing patches, ticket corners, lens reflections, or tool/gadget labels. Avoid post-process corner signatures and avoid making it merely a decorative dangling charm with no relation to the scene.
- If an image comes out too official/clean/soft, rewrite the next prompt to explicitly suppress official merchandise/key-art language and strengthen: deep black contrast, chrome/glass foreground prop, torn vinyl decals, graffiti tags, rough offset print, non-centered asymmetry, and text following a motion path.
- If Nick says a generated image did not arrive in chat (for example `皮卡丘没有发我`), treat it as a delivery issue rather than a generation failure: resend only the missing `MEDIA:/absolute/path` attachment plainly, without regenerating or giving a long explanation.
- For multi-image batches, especially when using direct CLIProxyAPI scripts or parallel `image_generate` calls, write a sidecar manifest as images complete. Include visible result index, title, semantic filename, slug, local path, full prompt, model, skill, intended Eagle folder, and status. This preserves exact `发布 1 3 4` / `发布124` selection mapping and prevents manual prompt reconstruction during Eagle/image2skill publication. For two-image batches, still save the manifest so `发布 1 2` works the same way.
- Nick expects generated images to be delivered directly in chat as native media attachments, not only listed as paths. In final responses after generation, include each result as `MEDIA:/absolute/path` in numbered order. If a transport cannot attach media, state the limitation and provide paths as fallback, but the default completion format is direct `MEDIA` delivery.
- For Rednote/Xiaohongshu publication from this workflow, send to the explicit mirror target `discord:#rednote` rather than bare `discord`. If bare-platform sending fails with a closed session or wrong home channel, list targets and retry the same selected images to `discord:#rednote`; do not treat it as a generation failure. Preserve the user's requested numbering in the captions, but if publishing a cross-batch selection, number the posted messages sequentially only when the user explicitly framed them as a combined set.

- When Nick names a batch directly, use those exact characters instead of substituting a generic “known female anime” set. Pre-assign distinct archetypes before generation and keep any youthful/magical-girl subjects explicitly wholesome, covered, age-appropriate, and non-sexualized; publish only the selected numbers later, not the whole manifest by default. If one parallel generation times out but the other succeeds, retry only the failed image and keep the successful image path/prompt in the manifest.
- For large named IP batches in this style, preserve the numbered manifest before generation, then prefer serial generation with compact prompts if the provider is unstable. Do a one-image smoke test before launching the full batch; if the first item hangs with no stdout/manifest update beyond the normal window, kill that process and retry only that index with a shorter prompt. Avoid concurrent retries after 502s. With CLIProxyAPI `/v1/responses`, do not force `tool_choice` for `image_generation`; provide the tool and instruct the dialogue model clearly instead. See `references/session-2026-05-eva-human-cast-generation-provider-routing.md`.
- For game-specific repeated batches such as Honor of Kings / 王者荣耀, avoid repeating characters already generated in the immediately preceding completed set unless Nick explicitly names them. Save each completed batch/add-on manifest immediately; if a later generation is interrupted, future `发布<主题>N个结果` requests should still be traceable to completed delivered manifests rather than incomplete recency.
- For Honor of Kings duel/comic-page images, especially Xiao Qiao vs Li Bai, the corrected direction is: preserve the stickerbomb poster energy while making the central character-vs-character clash oversized and frame-breaking. Manga panels should either sit mostly in the background or frame the enlarged center splash; they must not shrink the characters into small boxes. Keep faces, fan, sword, `BAM`, and the skill collision readable; concentrate sticker density on gutters/borders/background. Session notes: `references/session-2026-05-honor-of-kings-duel-manga-panel-stickerbomb.md`.
- For Honor of Kings male-character batches, a proven popular set is 李白 / Li Bai, 韩信 / Han Xin, 赵云 / Zhao Yun, 百里守约 / Baili Shouyue. Give each a distinct readable archetype: sword-calligraphy diagonal, low-angle spear diagonal, circular dragon spear ring, and foreground scope-lens sniper split. Save a numbered manifest before reporting so `发布1234` maps cleanly.
- If a multi-image batch is interrupted after some `image_generate` calls succeeded, immediately write/patch a partial manifest for the successful images before continuing. Do not rely on conversation recency alone; later `发布12` may refer to the delivered numbering across pre-interruption and post-interruption images.
- If a single image in a named batch times out or fails with a transient provider/tool-choice error (for example CLIProxyAPI `Tool choice 'image_generation' not found in 'tools' parameter`), retry only that character with a shorter prompt that preserves the same identity cues, composition archetype, `NickZag` in-scene placement, palette/style mechanics, and safety constraints. Keep the successful retry in the same batch manifest with the original intended index; do not regenerate successful siblings.
- If repeated `empty_response`, `500`, or connection-reset failures hit the same slot, compact the prompt aggressively instead of repeating the full template. Keep only subject anchors, one composition archetype, in-scene `NickZag`, style keywords, palette, and a short negative block. If the user requested a batch count rather than a strict roster, it is acceptable to swap the failed subject for another fitting character after multiple failures so the batch completes; say so in the report. See `references/session-2026-05-batch-short-prompt-recovery.md`.
- For repeated rerolls of the same Pokemon/creature slot (e.g. Mewtwo), treat every `重新生成` as a fresh image request. Preserve successful sibling outputs, retry only the failed slot, and progressively compact the prompt; if all attempts return `empty_response`, write/patch a manifest marking the slot failed and do **not** reuse a prior successful image path as the new result. If useful, provide the prior successful path clearly labeled as fallback/reference. See `references/session-2026-05-pokemon-mewtwo-empty-response-reroll.md` and `references/session-2026-05-pokemon-anatomy-lock-and-empty-response.md`.
- For small named reroll batches where one image succeeds and another repeatedly returns `empty_response`, preserve the successful new image in a fresh partial manifest immediately and mark the failed slot explicitly after compact retries. Do **not** reuse an older image path or silently pretend the failed item regenerated. Deliver the successful `MEDIA:` attachment and report the failed index/status plainly so later publish commands cannot map to a stale output.
- For Pokemon/creature anatomy, avoid optional language like `if visible` for canon appendages. If a creature canonically has small forearms, claws, paws, wings, tails, horns, or other paired appendages, require them explicitly as visible, paired, and unobscured. For Rayquaza specifically, require both small clawed forearms behind the head in a clean area; block hands being replaced by fins/spikes/lightning/stickers or covered by typography. Treat missing required appendages as a failed image even when the neon sticker-bomb surface is strong.
- Pokémon/creature anatomy lock and substitution lesson: Nick explicitly corrected that wrong hands, mismatched left/right hands, and wrong limbs are unacceptable. Always add positive body-plan constraints (`ANATOMY LOCK`) and negative anatomy constraints for Pokémon/creatures with visible appendages. For count-based rare-Pokémon batches, if a specific candidate repeatedly returns `empty_response`, substituting another fitting rare Pokémon can be acceptable only when the user did not name an exact roster; record the failed candidate and substitution in the manifest. See `references/session-2026-05-pokemon-anatomy-lock-and-substitution.md`.
- For Mewtwo-like Pokémon/creatures, tail structure is a first-class anatomy constraint from the initial prompt, not something to rely on later edits to fix. Require exactly one long thick smooth purple tail attached at the lower back/base of spine, continuous to the tip, with the purple torso patch visually separate from the tail. Avoid circular tail-orbit compositions when anatomy fidelity is critical; they can invite forked, duplicated, or detached tail fragments. Prefer a side/three-quarter pose with one clean S-curve tail and add hard negatives against second/forked/detached tails or tails connected to shoulder/neck/belly/front. If Nick flags the tail after generation, do not reuse the old image as a corrected result unless a real edit/reference reroll produced a new file. See `references/session-2026-05-mewtwo-tail-structure-edit-failure.md`.
- For CLIProxyAPI `/v1/responses` image-generation batches, avoid forcing `tool_choice` for `image_generation` when it triggers `Tool choice 'image_generation' not found in 'tools' parameter`; the more reliable pattern is a single `image_generation` tool, `partial_images: 1`, no `tool_choice`, and compact prompts. If a request completes with only text messages/no image, do not overwrite completed siblings; compact that one prompt and retry only that index. If explicit IP naming repeatedly returns text-only for a known character, retain visual identity anchors while reducing franchise/name dependence.
- For interrupted named batches with provider 429/cooldown plus partial successes, preserve the manifest as the source of truth and patch retry scripts to skip known `status: done` items. If a retry script accidentally rewrites successful items as failed, restore the known paths from prior logs immediately, then retry only the remaining target index. See `references/onmyoji-generation-continuation-2026-05-09.md` for a concrete four-image continuation.
- When continuing an interrupted CLIProxyAPI image batch, **read the existing manifest before writing any fresh pending manifest to the same path**. Preserve all `status: done` entries and generate only failed/pending indices. A retry script that calls `save_manifest(new_pending_manifest)` first can erase successful paths/status and break later `发布1234` mapping. If that happens, immediately repair the manifest from known output paths before reporting. This applies especially to long IP batches where only the final image hangs: kill the stuck process if necessary, then resume by skipping done paths and retrying only pending entries. If direct known-IP prompts repeatedly return text-only/no image result, retry the item with a generic visual-identity description that keeps the decisive cues while removing the direct character/IP name. See `references/session-2026-05-onmyoji-batch-retry-and-manifest-preservation.md`, `references/onmyoji-popular-generation-2026-05-09.md`, `references/session-2026-05-evangelion-batch-retry-and-ip-prompting.md`, and `references/session-2026-05-eva-human-cast-batch-continuation.md`.
- For known IP batches that mix human pilots/heroes and mecha/robots, keep the reusable style generic and use IP names only for subject fidelity. Emphasize recognizable cues (hair, suit color, props, silhouette, armor palette, weapon, cockpit/medical/hazard motifs) rather than depending on official logos. For youthful human subjects, explicitly require covered, age-appropriate, non-sexualized treatment; for mecha, mark `Non-human mecha` and prioritize silhouette/color/armor/weapon recognizability. Expect small generated text to be unreliable and report it as a caveat instead of treating it as verified copy.
- For live-action TV-series IP batches, when Nick specifies `剧版`, bind prompts to live-action-recognizable costume/face/prop anchors rather than generic comic variants. If a requested character repeatedly returns empty/no-image, retry that character with a shorter prompt first. If Nick only asked for a count such as `生成4张...角色`, substituting another relevant character is acceptable to complete the count, but report the substitution. If Nick named exact characters, do not substitute silently. Never include a failed item as `MEDIA:` and never reuse another image path for a failed index; duplicated paths are suspicious and should be retried or marked unverified.
- For minimal live-action roster requests like `生成2张剧版黑袍纠察队的主要角色图`, default to the two most immediately recognizable core leads unless Nick specifies otherwise. For *The Boys*, prefer **Homelander + Billy Butcher** as the two-image default pair, and keep each prompt anchored to live-action face/costume/prop cues rather than comic-book reinterpretations. See `references/session-2026-05-the-boys-two-main-characters-default.md`.
- For one-off named live-action prompts that follow a previous batch, like `生成深海`, resolve the subject to the live-action character already established in the conversation if the name is an obvious shorthand. For The Boys, `深海` maps to The Deep. A compact fallback prompt may be needed when the full template returns an empty image; prioritize live-action identity anchors, one composition archetype, and a short negative block over full-template verbosity.
- For Gintama rerolls, treat **hair + outfit + prop** as a canonical triad. If the user flags a mismatch, regenerate with a shorter prompt that hard-binds the exact canon anchors before the neon treatment. Current corrected anchors from Nick: **Kamui = orange braid/queue + blue eyes + deep charcoal/dark-gray Chinese mandarin-collar robe/jacket with frog buttons + clearly visible white cape/cloak over shoulders + red umbrella**; Otae = original simple dark straight tied-back/behind-shoulders hairstyle with side-parted bangs + purple kimono + gentle-but-dangerous big-sister demeanor; Elizabeth = plainly non-humanized white mascot silhouette first. Do not let fashion remix override the source silhouette, and keep corrected outfit/hair anchors large and unobscured while pushing sticker density into background, borders, typography, and prop surfaces.
- For Evangelion-style mecha rerolls, avoid letting huge signboards or foreground weapons obscure the head/torso. Unit-00 clarity improves when the prompt foregrounds the yellow/orange prototype head, single cyclopean visor, bulky shoulders, and restraint cables rather than a giant shield/sign. Unit-02 clarity improves when the prompt asks for a readable full red silhouette, visible sharp helmet/green visor/torso, and foreground blades that frame rather than cover the body. Regenerated mecha can remain strongly stylized, but the visual verification should judge whether identity comes from silhouette/color/armor cues, not only from big text labels.
- When Nick says a neon output has good material quality but lacks the neon skill's core, treat this as **insufficient sticker-bomb density**, not a reason to make the subject dirtier or more industrial. The corrected balance is: premium glossy subject material + dense neon sticker-bomb typography/decals around borders, background, cable arcs, prop surfaces, and slight armor-edge overlaps. Protect the head/chest/core silhouette while restoring torn decals, tape strips, barcodes, graffiti tags, halftone print energy, cropped type, chromatic split, and zine-poster chaos. Session notes: `references/session-2026-05-eva-mecha-glossy-stickerbomb-balance.md`.
- For Evangelion neon-skill batches, a proven four-image composition set is: Unit-02 with foreground two-handed Lance of Longinus diagonal; Unit-01 with berserk umbilical-cable circular halo; Unit-00 with side-profile split layout and restraint clamps; Rei Ayanami with diagonal floating blue cockpit halo. Keep Rei explicitly full-plugsuit, covered, age-appropriate, and non-sexualized. For Unit-02 + Lance, require both hands gripping a long crimson shaft and a visible forked/double-pronged spearhead inside the frame; block ordinary swords, dual blades, and off-frame spearheads. Session notes: `references/session-2026-05-evangelion-neon-skill-batch.md`.
- For Dragon Ball villain batches, preserve the exact form-specific silhouette rather than only the franchise label. Frieza, Perfect Cell, and the requested Buu form should each be prompt-locked with explicit negatives for the other forms. If the user asks for `瘦版魔人布欧`, use **Super Buu / lean Majin Buu**, not fat Buu and not Kid Buu: tall lean pink silhouette, long head antenna, black-gold vest, white baggy pants, yellow gloves/boots, candy-magic aura. Use the sidecar manifest to keep completed items while retrying only the pending one with a shorter single-image prompt when batch generation hangs. For Frieza and Perfect Cell rerolls, do not overcorrect stiffness with extreme foreground hands or extreme dive perspective: it caused abnormal fingers, missing/fused limbs, and broken anatomy. Use an anatomy-safe tension balance instead: readable diagonal posture, clear limb count, clear tail/wing/head/torso anchors, energy aura, tail/wing arcs, and slanted typography. See `references/session-2026-05-dragonball-anatomy-tension-balance.md`.
- For repeated Dragon Ball requests like `生成4个其他的龙珠角色` / `生成4个之前未生成过的龙珠角色`, treat non-overlap across the current conversation as part of the request. Track the already-generated roster and choose fresh characters before prompting. A worked rotation from this session: first reroll set = Trunks / Son Gohan / Android 18 / Cell; next non-overlapping set = Goku / Vegeta / Piccolo / Frieza; next = Bulma / Krillin / Android 16 / Beerus; next = Yamcha / Tien Shinhan / Master Roshi / Gotenks. Vary archetypes across each four-image set and save a manifest so later `发布123` maps to the latest generated batch, not an older Dragon Ball batch.
- For repeated Android 18 / 人造人18号 rerolls, keep her identity anchors stable before style density: short blonde bob with side part, blue eyes, denim vest over black shirt, black-and-white striped sleeves, gloves/boots, cool controlled android expression. Rotate away from the previous four skeletons; a useful fresh set is Dutch-angle action, diagonal floating energy burst, foreground chrome scanner/lens depth, and circular spin-kick ring. Keep non-sexualized coverage explicit. If Nick then says `发布23到小红书`, publish indices 2 and 3 from the latest Android 18 manifest as separate Rednote posts, not from an older reroll set.
- Session note added: `references/session-2026-05-gintama-canon-fidelity-and-retry.md`.
- Session note added: `references/session-2026-05-crayon-shinchan-delivery-and-kindergarten-batch.md` for Shin-chan/幼儿园角色 anchors, child-proportion guardrails, successful composition rotation, and missing-attachment resend behavior.
- Session note added: `references/session-2026-05-slam-dunk-basketball-batches.md` for Slam Dunk exact-roster handling, basketball character anchors, composition rotation, manifest discipline, and hand/ball anatomy pitfalls.
- For Demon Slayer rerolls, treat `重新生成炭治郎/祢豆子/伊之助` as a fresh reroll set, not a resend. Rotate the skeleton each time and keep the canon anchors large and readable: Tanjiro = hanafuda earrings + forehead scar + green-black checkered haori + katana; Nezuko = bamboo muzzle + pink kimono + dark hair with orange tips; Inosuke = boar mask + dual serrated swords. Keep all youthful subjects covered, wholesome, age-appropriate, and non-sexualized. If only one slot fails, retry only that slot with a shorter prompt instead of regenerating successful siblings. For Ubuyashiki-household requests, parse phrasing carefully: `发色不同的双胞胎产屋敷日香和产屋敷天音` means one combined twin image, not two separate adult/child portraits; force visibly different hair colors in the first lines (e.g. 日香 black hair vs 天音 white/silver hair), keep both childlike/covered/non-sexualized, and use negatives against same hair color/missing twin/adult bodies. When Nick asks for the twins to be `更加贴近原作形象`, prioritize original-anime fidelity before neon remix: very young shrine-family proportions, simple round child faces, solemn expressions, straight hime-cut hair with blunt bangs/side locks, and fully covered traditional kimono. Put sticker-bomb density in borders/background/talismans/wisteria seals rather than on faces or core silhouettes; explicitly block adultification, fashion-model redesign, same hair color, missing twin, and stickers covering eyes. If Muzan or Ubuyashiki slots return `empty_response`, preserve successful twin/household outputs and retry only the failed slot with a compact anchor-first prompt. See `references/session-2026-05-demonslayer-rerolls.md`, `references/session-2026-05-ubuyashiki-household-rerolls.md`, and `references/session-2026-05-ubuyashiki-twins-canon-hair-fidelity.md`.
- For Demon Slayer Hashira/Twelve Kizuki neon batches, preserve character and prop mechanics before surface style. Mitsuri Kanroji needs an explicitly continuous Love Hashira whip-sword: katana-style handle + visible flower guard + flexible metallic ribbon blade emerging from the front of the guard, with the hilt/guard/blade junction large and unobscured. Push sticker density and `NickZag` away from that junction. If this fails twice, switch to a half-body/three-quarter close-up centered on the readable weapon connection rather than repeatedly rerolling full-body action. When parallel generation has failed slots, do not shift successful sibling outputs into the failed label; record failures separately and save a dedicated batch manifest for the current subject family. See `references/session-2026-05-demonslayer-hashira-kizuki-weapon-and-manifest-lessons.md`.
- For Demon Slayer Hashira rerolls where the user explicitly wants visible hands, do not solve hand instability by hiding the hands or burying them behind props. Use a stable standard grip instead: both hands visible, weapon hilt fully readable, reduced foreshortening, and no typography/stickers crossing the knuckles. Verify with vision before marking the slot acceptable. New support note: `references/session-2026-05-demonslayer-hashira-hand-clarity.md`.
- For Demon Slayer Hashira rerolls where the user repeatedly flags hand problems and does *not* require visible hands, a chest-up / upper-torso crop can still be used as the fallback. Push the sword hilt and grip out of frame, hide any remaining hand behind haori/flame/braids/ribbon/foreground shapes, and add hard negatives against visible hands/fingers/extra hands/fused fingers. See `references/session-2026-05-demonslayer-hand-out-of-frame-retry.md`.
- For repeated minimal Demon Slayer Hashira requests like `生成4张鬼灭之刃的柱角色`, treat non-overlap across the current conversation as part of the implied request. Rotate through the nine Hashira before repeating, unless Nick names specific characters. A good first two four-image rotation is: Giyu / Rengoku / Shinobu / Tengen, then Obanai / Sanemi / Gyomei / Muichiro; the missing ninth is Mitsuri. If Nick then says `重新生成鬼灭之刃9个柱的角色图`, generate the complete nine-member roster in one numbered manifest: Giyu, Rengoku, Shinobu, Tengen, Mitsuri, Muichiro, Obanai, Sanemi, Gyomei. Preserve each Hashira’s canon anchors first, vary composition skeletons across the set, and write a manifest immediately so later `发布123...` maps to the latest all-Hashira batch.
- For repeated two-slot Demon Slayer Hashira rerolls like `重新生成 炎柱 恋柱` / `重新生成恋柱，炎柱`, patch only the affected indices in the existing nine-Hashira manifest rather than creating a new unrelated source of truth: Rengoku = index 2, Mitsuri = index 5. Rotate the action skeleton each time (Rengoku: foreground flame depth / low-angle diagonal / dutch-angle flame slash; Mitsuri: circular motion ring / diagonal airborne spin / dutch-angle twist). If one slot returns `empty_response`, keep the successful sibling and retry only the failed slot with a compact anchor-first prompt. See `references/session-2026-05-demonslayer-hashira-two-slot-reroll-loop.md`.
- For repeated One Piece / Overlord named-character rerolls, treat the request as fresh generation, not a resend, unless Nick explicitly asks to publish/send previous results. Rotate away from the most recent skeleton for that same character and keep a fresh numbered manifest even for two-image batches. Strong anchors from this session: Ainz = skull face + red eye glow + ornate black/gold/white robe + huge collar + staff; Momon = full black armor + closed helmet + twin greatswords + no skull/robe; Nami = orange hair + clima-tact/weather + navigator map/compass; Robin = long black hair + calm archaeologist + book/glyph/flower-hands. For Robin/other stubborn failed slots, preserve successful siblings and retry only the failed item with a shorter anchor-first prompt. For Zoro, Nick repeatedly rejects incorrect katana anatomy, especially wrong hand-grip and mouth-sword blade placement. Use anatomy-safe poses over extreme twists, and prioritize simple readable three-sword construction: exactly one mouth katana with handle at one side of the mouth, guard outside cheek, blade extending horizontally outward; two hand-held katanas with hand gripping wrapped handle only, visible tsuba/guard, and blade extending from the guard on the same axis. Keep sticker/energy effects away from hilt-guard-blade junctions. See `references/session-2026-05-onepiece-and-overlord-repeat-rerolls.md`.
- For One Piece Zoro outputs, sword count/structure is a first-class fidelity check, especially when Nick says to keep the pose but fix the swords. Preserve the accepted pose/composition when possible, then hard-bind exactly three clean katanas: one straight mouth sword plus two hand-held katanas with continuous handle/guard/blade; keep green slash energy visually separate from actual blades; block extra swords, melted blades, floating fragments, missing mouth sword, and face-covering blades. If direct edit fails, regenerate using the previous image as `input_image` with “preserve pose, fix only swords” language.
- If Nick continues flagging Zoro sword anatomy after an image-reference retry, treat it as a composition failure, not only a prompt-detail failure. Reduce pose complexity: use a stable half-body / upper-thigh front or side three-quarter stance, keep hilt/guard/blade junctions large and unobscured, and move sticker-bomb density to borders/background. Describe each weapon separately: mouth sword = handle at one side of mouth, guard just outside cheek, straight blade extending horizontally outward; each hand-held katana = wrapped handle gripped by hand → visible round/oval tsuba/guard → straight blade extending away from hand on the same axis. Add hard negatives against hand gripping blade, missing guard, blade from wrong side of handle, melted/fused handles, extra handles, random sword shards, energy ribbons replacing blades, and stickers covering hilt/guard. See `references/session-2026-05-onepiece-zoro-sword-fix.md` and `references/session-2026-05-onepiece-zoro-sword-anatomy-and-jojo-batch.md`.
- For JoJo character + Stand batches, require **dual readability**: the human character and their Stand/power must both be large, readable, and visibly separate. Put identity anchors for the human first, then Stand anchors, then a hard line like `the Stand is clearly separate from the character, not merged into the body`. Move stickerbomb density to borders/background/power effects/typography so faces, hair, torso, and Stand head remain clear. For non-Stand or ambiguous cases, avoid false canon claims: Jonathan can use a `heroic Hamon spirit avatar`; Bastet can be a magnetic outlet/icon with metal orbit; unclear minor characters can use a `mysterious stand-like visualization`. Add negatives: `no missing separate stand`, `no stand merged into body`, `no stand hiding character face`, `no stand reduced to background smoke`. See `references/session-2026-05-jojo-character-stand-batches.md`.
- Crayon Shin-chan / Action Mask, Gold Saints 1–4, and Valorant core-four prompt anchors from this session are captured in `references/session-2026-05-crayon-valorant-goldsaints-generation.md`. Key reusable correction: for 动感超人, use `CANON-TIGHT ORIGINAL ANIME FIDELITY PRIORITY` and explicitly preserve the blue full-body suit, red gloves/boots/belt, short red scarf/cape accent, white-red mask/helmet, and large round bug-eye lenses before applying stickerbomb density.
- For Crayon Shin-chan / 蜡笔小新 stickerbomb batches, keep all kindergarten-age characters explicitly childlike, covered, wholesome, and non-sexualized. Useful anchors: Shin-chan = round child face, thick eyebrows, small black hair, red shirt, yellow shorts; Shiro/小白 = small white fluffy puppy, black dot eyes, small black nose, soft ears, curled tail, red collar, non-humanized dog silhouette; Action Mask/动感超人 = original-anime tokusatsu fidelity first: smooth blue full-body suit, red gloves/boots/belt or brief accent, short red scarf/cape accent, white-red helmet/mask, large round bug-eye lenses. If Nick says Action Mask should be closer to the source, treat it as a canon-fidelity failure: keep the hero body clean/readable and push stickerbomb density to background/borders/hero-card labels; block generic armored rider, mecha, insect monster, western superhero, and overly realistic movie-suit redesigns.
- For Crayon Shin-chan batches, keep the deliberately simple cartoon canon anchors first and put neon/stickerbomb energy around them, not over them. Children should remain covered, childlike, age-appropriate, and non-sexualized. Useful anchors: Shin-chan = rounded face + thick black eyebrows + small black hair + red shirt + yellow shorts; kindergarten friends = Kazama neat smart-kid cues, Nene bob/bunny prop, Masao rice-ball head/timid expression, Bo-chan sleepy/runny-nose/stone cue; Shiro = small fluffy white puppy + curled tail + red collar. If Nick says Action Mask / 动感超人 should be closer to the original, treat it as a canon-fidelity failure: enforce original-anime Action Mask anchors (smooth blue suit, red gloves/boots/belt, short red scarf/cape accent, white-red mask with large round bug-eye lenses) and block generic armored rider/mecha/insect/western superhero redesigns. Keep sticker density in background/borders/hero cards so the mask/torso/gloves/boots stay clean. See `references/session-2026-05-crayon-shinchan-actionmask-shiro.md`.
- For Captain Tsubasa / 足球小将 batches and rerolls, preserve soccer-action identity before style: readable kit, face, boots, soccer ball, and body mechanics. Useful anchors: 大空翼 = youthful captain/protagonist + short dark hair + white-blue kit + captain armband + drive-shot/dribble energy; 日向小次郎 = fierce dark-spiky-haired striker + dark kit + orange-yellow tiger-shot accents; 若林源三 = serious goalkeeper + green keeper kit + gloves + goal-net stance; 岬太郎 = graceful playmaker + calm expression + blue-white kit + pass/combi motion; 石崎了 = scrappy defender + short dark hair + blue-white kit + muddy slide/face-block energy. Repeated 大空翼/日向小次郎 rerolls should rotate the full skeleton (dribble depth, circular drive-shot ring, Dutch-angle cut, side split, airborne volley, etc.) and explicitly avoid prior layouts. If a slot returns `empty_response`, preserve successful sibling outputs and mark the failed slot in the manifest rather than reusing old paths. See `references/session-2026-05-captain-tsubasa-rerolls.md`.
- For minimal category/franchise requests such as `生成4张闪光宝可梦` or `生成足球小子热门角色图`, choose a recognizable roster and generate immediately unless identity is genuinely ambiguous; do not ask for a roster by default. Save a numbered sidecar manifest even for two-image rerolls. If a count-based roster slot repeatedly returns `empty_response`, substitute another fitting character only when Nick did not name an exact roster, record the failed candidate/substitution, and report it plainly. For fresh rerolls like `重新生成大空翼，日向小次郎`, rotate away from the previous skeleton for those exact characters and save a new reroll manifest. Session note: `references/session-2026-05-shiny-pokemon-and-captain-tsubasa.md`.
- For requests to collect/export historical neon stickerbomb outputs, organize copied images by work/IP name under the requested destination, not by batch/date/character. Scan active-profile `state/`, `cache/`, and manifest JSONs; copy only image files that actually exist; maintain `_index.json`; verify final count/size with filesystem output; and report missing/stale manifest references plainly. See `references/session-2026-05-stickerbomb-library-export.md`.
- For Saint Seiya / 圣斗士五小强 batches and rerolls, preserve bronze-saint identity anchors before style density: Seiya = brown hair + white/silver Pegasus armor + meteor punch; Shiryu = long black hair + emerald Dragon armor + one forearm shield; Shun = green hair + pink Andromeda armor + one continuous chain with visible links/end pieces; Hyoga = blond hair + white/icy-blue Cygnus armor + Diamond Dust / ice/swan cues; Ikki = dark spiky hair + Phoenix armor + orange-red flame aura. Rotate skeletons on every `重新生成`; if Nick requests `一辉的圣衣上半部分改为白色系`, make the upper chest/shoulders/torso armor white/ivory/silver-white while preserving Phoenix identity and dark/flame lower contrast. See `references/session-2026-06-saint-seiya-bronze-saints-rerolls.md`.
- For Saint Seiya Gold Saints / 黄金圣斗士 batches, canon face/armor fidelity comes before neon surface style. Aries Mu / 白羊座穆 needs long pale lavender-purple hair, visible forehead dot, soft calm androgynous face, relaxed narrow eyes, and especially extremely thin/pale/almost invisible eyebrows; block thick/dark/angry/angular eyebrows, harsh masculine/villain face, wrong dark/short hair, stickers covering face/eyes/eyebrows/forehead dot, and over-busy Crystal Wall effects on the face. For multiple Aries variants, side-profile or close-up Crystal Wall compositions may protect face fidelity better than foreground-hand action poses. Cancer Deathmask / 巨蟹座迪斯马斯克 should remain an adult sinister Gold Saint in integrated Cancer Cloth; block giant crab claws, pincer hands, mecha appendages, crab-monster redesign, and oversized crab limbs. For same-character variant sets, save a separate variant manifest rather than silently replacing the main roster manifest; update the roster manifest only after Nick selects a candidate. See `references/session-2026-06-gold-saints-aries-mu-fidelity.md`.
- For Saint Seiya Gold Saints / 黄金圣斗士 batches, preserve the numbered manifest and patch only requested indices on reroll. For Aries Mu / 白羊座穆, if Nick mentions original facial fidelity, make the face and eyebrows first-class anchors before sticker-bomb density: soft refined face, calm narrow eyes, long lavender hair, thin pale/near-invisible lavender eyebrows; explicitly block thick black/angry/bushy brows and keep eyes/eyebrows unobscured. For Cancer Deathmask / 巨蟹座迪斯马斯克, treat visually strong but modern demon-knight/mecha drift as a canon-fidelity failure: require harsh angular adult face, dark blue-purple moderate spiky hair, integrated classic Gold Cancer Cloth, modest crab armor motifs; block giant monster-claw hands, huge feathered cyber hair, soft pretty face, and excessive mecha redesign. See `references/session-2026-06-gold-saints-face-fidelity-rerolls.md`.

- If Nick says an image was not sent/received after generation (for example `皮卡丘没有发我`), do **not** regenerate or over-explain. Resolve the already-generated file from the latest manifest/prior path and resend that exact file as a standalone `MEDIA:/absolute/path` message. Use generation only if Nick explicitly asks for a new image.
- If Nick immediately asks to `重新生成4张2号机` after a prior EVA-02 set, treat it as a new reroll set rather than re-sending prior files. Rotate away from the previous four skeletons and use distinct variants such as cross-lance close assault, shield-breaker side split, beast-sprint cable trail, and halo-drop twin blades. Keep the helmet/green visor/chest readable in every variant, and add explicit negatives against foreground props covering the face/torso. Session notes: `references/session-2026-05-eva02-repeat-reroll-and-verification.md`.
- For Gintama batches, a short-prompt retry often works better than the initial long prompt when a single item hangs or returns text-only. Keep the same manifest index and retry only the failed item. Preserve the smallest identity anchors first and the stickerbomb style second: Gintoki = silver hair + white kimono + wooden sword; Kagura = red hair buns + blue eyes + red Chinese outfit + purple umbrella; Shinpachi = round glasses + blue/white dojo outfit; Sadaharu = giant fluffy white dog + red collar; Katsura = long black hair + purple kimono + Elizabeth; Hijikata = black hair + black-gold Shinsengumi uniform + cigarette + katana + mayo gag; Okita = light brown hair + black-gold Shinsengumi uniform + bazooka + sly deadpan energy. Session notes: `references/session-2026-05-gintama-neon-batch-retry.md`.
- For Gintama canon rerolls after Nick flags `服装不对` / `发型不对`, move the corrected anchor before the neon style and keep it clean/readable: Kamui = orange braid + red umbrella + **dark charcoal-gray Chinese mandarin-collar robe/changshan with frog buttons + white cape/cloak**; Otae = **long dark straight tied-back hair with side-parted bangs/fringe** + purple kimono; Shiroyasha = silver-haired battle-form Gintoki with white/black battle kimono and weapon. 女装/crossdress requests should stay covered and comedic/parody-oriented, preserving Gintoki/Shinpachi identity anchors instead of becoming sexualized pin-up fashion. Session notes: `references/session-2026-05-gintama-canon-fidelity-and-retry.md`.
- For Onmyoji popular-character follow-up batches, a proven non-overlapping set after 酒吞童子/玉藻前/茨木童子/妖刀姬 is: 神乐/Kagura, 大天狗/Ootengu, 阎魔/Enma, 荒/Susabi. Use distinct archetypes: foreground umbrella depth, wing-storm ring, verdict-scroll side split, and diagonal star-orbit floating poster. If Susabi/NickZag text is important, inspect it closely; one successful run rendered the in-scene credit as `NickZano`, which may require a single-image reroll.
- If a 429 `usage_limit_reached` / `model_cooldown` includes `resets_at` or `resets_in_seconds`, calculate the local reset time with `date`/Python before deciding to wait, retry, or change provider/model routing. Switching the dialogue model field (for example `gpt-5.5` to `gpt-5.4`) may route around cooldown while leaving the image tool model as `gpt-image-2`, but it can also trigger transient tool-schema errors; retry only affected indices and protect completed manifest entries.

## Reusable Templates

- `templates/cliproxyapi_neon_batch_with_manifest.py` is a copy-and-fill batch generator template for CLIProxyAPI `/v1/responses` image generation. Use it when generating multi-image neon batches that may later be selectively published by number; it saves a sidecar manifest after every completed image.

## Session References

- `references/session-2026-05-prompt-corrections.md` captures prompt-correction lessons for in-scene `NickZag`, batch composition diversity, prompt-adherence review, and mascot style drift.
- `references/session-2026-05-cliproxyapi-four-character-batch.md` captures a successful four-character neon batch generated through the direct CLIProxyAPI Responses API path after Hermes `image_generate` failed; includes subject/composition selections and operational timing notes.
- `references/session-2026-05-image2skill-neon-publication.md` captures the corrected publication workflow: “发布到 image2skill” means Eagle-backed frontend publication, not Discord channel posting; includes selected-output mapping, Eagle metadata, and rebuild verification.
- `references/eva-mecha-glossy-dense-stickerbomb-balance-2026-05-13.md` captures Nick-corrected EVA/mecha generation balance: preserve canon silhouette and premium glossy material while keeping dense neon sticker-bomb collage as a non-negotiable style core.
- `references/session-2026-05-eva-units-05-08-glossy-dense-stickerbomb.md` captures the EVA Unit-05/Mark.06/Unit-07/Unit-08 generation anchors, QC expectations, and retry pattern for transient CLIProxyAPI `tool_choice` failures.
- `references/session-2026-05-16-evangelion-angels-canon-tight-neon.md` captures a later canon-tight Angel 1–4 run with successful compact retry wording, the direct-IP no-image workaround for Adam, final paths, and shape-fidelity QC outcomes.
- `references/session-2026-05-eva-09-13-08plus02-reroll.md` captures the EVA Unit-09 / Unit-13 / Unit-08+02 anchors, the 8+2 fusion reroll correction, and the CLIProxyAPI `/responses` authorization/payload pitfall found during generation.
- `references/session-2026-05-eva-09-and-08plus02-second-reroll.md` captures the stronger second-reroll anchors for Unit-09 physical scythe/core clarity and Unit-08+02 interwoven hybrid fusion, plus the local CLIProxyAPI payload lesson: avoid `tool_choice`/`partial_images` on this path when it hangs or 502s; retry pending indices with the simpler known-good payload.
- `references/session-2026-05-eva-unit09-color-fidelity-and-publish.md` captures Nick's correction that Unit-09 is not an all-silver/full-chrome body, the corrected ivory/black/orange-gold/red-orange color anchors, and the Eagle API `Connection refused` recovery by opening Eagle before retrying publication.
- `references/session-2026-05-batch-manifest-and-publication-selection.md` captures the stronger batch-generation practice: save a sidecar manifest mapping result numbers to paths, semantic filenames, prompts, and Eagle metadata so later commands like `发布 1 3 4` can publish exactly the selected outputs without reconstructing prompts manually.
- `references/session-2026-05-overlord-batch-anchors.md` captures the Overlord core-four roster and the composition rotation used to keep the batch visually distinct.
- `references/session-2026-05-overlord-repeat-rerolls-and-manifests.md` captures repeated Overlord reroll handling: treat overlapping requests as fresh batches, rotate each character away from its prior skeleton, preserve a new numbered manifest for each set, and retry only failed slots with compact anchor-first prompts after transient provider errors.8plus02-reroll.md` captures the EVA Unit-09 / Unit-13 / Unit-08+02 anchors, the 8+2 fusion reroll correction, and the CLIProxyAPI `/responses` authorization/payload pitfall found during generation.
- `references/session-2026-05-batch-manifest-and-publication-selection.md` captures the stronger batch-generation practice: save a sidecar manifest mapping result numbers to paths, semantic filenames, prompts, and Eagle metadata so later commands like `发布 1 3 4` can publish exactly the selected outputs without reconstructing prompts manually.
- `references/session-2026-05-overlord-batch-anchors.md` captures the Overlord core-four roster and the composition rotation used to keep the batch visually distinct.
- `references/session-2026-05-overlord-repeat-rerolls-and-manifests.md` captures repeated Overlord reroll handling: treat overlapping requests as fresh batches, rotate each character away from its prior skeleton, preserve a new numbered manifest for each set, and retry only failed slots with compact anchor-first prompts after transient provider errors.
- `references/session-2026-05-the-boys-live-action-batches.md` captures live-action The Boys identity anchors, retry/substitution guidance for failed items, and the no-fabrication rule for failed image paths.
- `references/session-2026-05-the-boys-live-action-soldierboy-deep-regeneration.md` captures The Boys live-action reroll anchors plus the partial-success / repeated-empty-response handling pattern for Soldier Boy and The Deep.
- For Evangelion neon-skill batches, a proven four-image composition set is: Unit-02 with foreground two-handed Lance of Longinus diagonal; Unit-01 with berserk umbilical-cable circular halo; Unit-00 with side-profile split layout and restraint clamps; Rei Ayanami with diagonal floating blue cockpit halo.
- `references/session-2026-05-known-female-batch-rotation.md` captures repeated known-female-character batch lessons: avoid immediate character repetition, pre-assign four distinct composition archetypes, maintain latest-manifest mapping, and treat `发布` as Eagle-backed image2skill publication by default.
- `references/session-2026-05-known-female-anime-batch-loop.md` captures the repeated four-image known-female-anime batch loop: direct CLIProxyAPI generation, manifest fields, archetype rotation, latency expectations, and numbered publication mapping.
- `references/session-2026-05-kof-and-known-female-batches.md` captures worked character/composition sets for KOF women and later known-female anime batches, plus covered/non-explicit fighter-prompting and manifest requirements.
- `references/session-2026-05-two-character-batch-and-publication.md` captures the two-character Harley Quinn / Cammy White batch: retrying a timed-out parallel image, preserving a manifest for `发布12`, and verifying shortened image2skill detail-page slugs.
- `references/session-2026-05-honor-of-kings-batch-continuity.md` captures 王者荣耀 batch continuity across interruptions, `继续`, selected-number publishing, and shortened retry prompts after timeouts.
- `references/session-2026-05-honor-of-kings-skin-variants.md` captures named-skin / variant prompting for Honor of Kings characters, especially treating `经典皮肤` as a distinct generation target with variant-specific fidelity cues and non-explicit coverage.
- `references/session-2026-05-honor-of-kings-male-batches-and-retry.md` captures male Honor of Kings batch rotations, numbered publish/Xiaohongshu interpretation, and the compact retry pattern for transient CLIProxyAPI `image_generation` tool-choice failures.
- `references/session-2026-05-honor-of-kings-male-popular-batch.md` captures a worked four-character male 王者荣耀 batch: 李白 / 韩信 / 赵云 / 百里守约, with distinct composition archetypes and manifest-numbering conventions.
- `references/session-2026-05-stickerbomb-manga-panel-duel.md` captures Nick's corrections for stickerbomb + manga分镜 duel images: keep obvious panel gutters/story beats, make the exact center a large frame-breaking special-move clash, enlarge the characters themselves as the visual anchor, and use dense stickerbomb elements without losing readability.
- `references/session-2026-05-evangelion-batch-retry-and-ip-prompting.md` captures a six-image Evangelion batch with human pilots and mecha: IP fidelity cues without style/logo dependence, explicit non-sexualized pilot constraints, mecha-specific composition archetypes, and resume-safe manifest continuation after a hung final image.
- `references/session-2026-05-eva-human-cast-generation-and-rednote.md` captures a seven-item EVA human/mascot batch plus Xiaohongshu/Rednote publication: preserve partial manifests, omit broken `tool_choice`, use `partial_images: 1`, compact prompts when long prompts hang, retry only failed/pending indices, and post one image/copy per Rednote message.
- `references/session-2026-05-eva-human-cast-batch-continuation.md` captures continuation lessons from a seven-image EVA human/mascot cast batch: read the existing manifest first, skip completed paths, use compact prompts with `partial_images: 1`, avoid `tool_choice` on the local CLIProxyAPI image path, and if a known-IP item returns text-only, retry with a generic visual-identity description that preserves the cues.
- `references/session-2026-05-the-boys-image-backend-auth-mismatch.md` captures a non-art failure mode for live-action neon batches: if all generations fail immediately, compare main chat auth vs image_gen auth and treat `401 Unauthorized` or `unknown provider for model gpt-5.5` as backend-configuration triage, not as prompt failure.
- `references/session-2026-05-evangelion-lance-regeneration.md` captures the Unit-02 correction where Nick specifically wanted the Lance of Longinus: require both hands gripping a long crimson shaft, a visible forked/double-pronged spearhead inside frame, and negatives blocking ordinary sword/dual-blade outputs; update the same manifest index after reroll.
- `references/session-2026-05-evangelion-angels-neon-regeneration.md` captures the correction where Evangelion Angels 1–4 had to be regenerated with neon sticker-bomb style rather than publishing a Silver Rebellion batch; includes Angel-specific composition cues and the safer Lilith retry wording after provider empty responses.
- `references/session-2026-05-evangelion-angels-canon-fidelity.md` captures Nick's later correction that the Angels still drifted too far from the TV/anime designs; future rerolls must prioritize canon silhouette/shape fidelity first and push neon sticker-bomb density into background/borders rather than redesigning the Angel body.

## Common Pitfalls

11. **Mitsuri’s whip-sword junction drifts or detaches.**
   - Symptom: the handle, flower-shaped guard, and flexible blade do not read as one continuous weapon; the blade appears to emerge from the wrong end of the handle, floats independently, or is visually replaced by hair/ribbon motion.
   - Fix: make the weapon junction a first-class constraint. Use explicit language like `clear weapon close-up`, `first visible section of the flexible blade attached directly to the front of the guard`, and `do not let stickers or hair cover the weapon junction`. If the full poster prompt fails, retry with a shorter prompt and reduce background density around the sword connection.
   - Reference: `references/session-2026-05-mitsuri-whipsword-continuity.md`.


0. **Generated output only nominally follows the prompt.**
   - Symptom: the requested prop/text exists, but the result becomes a clean official poster, cute merch sheet, gothic text-column, or repeated centered layout instead of the intended glossy neon sticker-bomb poster.
   - Fix: inspect the actual image before declaring success when the user flags mismatch. Rewrite the prompt around the failed mechanism: composition archetype, subject treatment, edge contrast, sticker-bomb density, palette contrast, and `NickZag` integration. Prefer a different archetype over simply adding more adjectives.

0.1. **`NickZag` becomes a watermark or weak dangling charm.**
   - Fix: make `NickZag` part of a real picture-world object with perspective and material: torn tape, prop label, jacket patch, barcode sticker, evidence tag, lens reflection, card corner, blade maintenance sticker. Do not put it as a corner signature, floating overlay, or ordinary watermark.

0.2. **Weapon continuity fails even when the poster looks strong.**
   - Symptom: a whip blade, ribbon sword, chain, fan, cable, or other continuity-sensitive prop looks attractive but does not physically connect to the handle/guard/body anchor, or the connection is hidden by hair, stickers, typography, or effects.
   - Fix: prioritize the attachment point over extra pose drama. Keep the handle/guard/junction and first blade segment unobscured and in a clean readable area; push sticker density to the frame/background until the structural anchor is solved. Do not update a manifest to a visually attractive but structurally wrong version as final.
   - For Mitsuri Kanroji / Love Hashira specifically: the katana-style handle and flower-shaped guard must be visible in both hands, and the thin flexible blade must emerge from the FRONT side of the guard as one continuous metallic whip-blade, not from the handle bottom/behind and not as a detached floating ribbon. See `references/session-2026-05-demonslayer-mitsuri-whip-sword-continuity.md`.

0.3. **Batch images reuse the same body pose and poster skeleton.**
   - Fix: before each image, choose a distinct composition archetype and vary at least four dimensions: camera angle, body orientation, subject placement, prop placement, typography direction, negative space, motion path, and background geometry. If the previous image was centered/low-angle/foreground-prop, rotate to side split, circular motion, floating diagonal, or dutch-angle.

0.3. **User explicitly wants visible hands, but the image keeps hiding them or making the grip unreadable.**
   - Fix: keep the hands in frame with a stable two-hand grip, lower the foreshortening, and move stickers/text away from the knuckles. Do not “solve” the issue by burying the hands behind foreground props or cropping them out when the user asked for readability.
   - See `references/session-2026-05-demonslayer-hashira-dynamic-hand-retry.md`.

1. **Accidentally retaining a third-party style name.**
   - Fix: replace with descriptive mechanics: glossy neon cyber-pop, thick black ink, sticker-bomb typography, etc.

2. **Using the same centered poster layout repeatedly.**
   - Fix: select a different archetype and explicitly prohibit centered title-wall composition. For batches, vary the whole poster skeleton, not just the character name: pose, prop scale, camera, type direction, negative space, motion path, and placement must visibly change.

3. **Claiming a prompt was applied without visual verification.**
   - Fix: if the user says the style or prompt did not apply, inspect the generated image before defending it. Identify exactly which prompt mechanisms applied and which failed, then rewrite the next prompt around corrections rather than minor wording tweaks.

4. **Softening mascot prompts into official/cute merchandise.**
   - Fix: explicitly block official-merchandise, theme-park, scrapbook, stationery, children’s-book, and pastel-only language. Add deep black contrast, chrome scanner/glass/foil props, glitch UI, torn vinyl, warning/evidence labels, barcode fragments, cyan/magenta rim light, and sharper street-poster typography while keeping the mascot wholesome and non-humanized.

5. **Making `NickZag` a watermark.**
   - Fix: attach it to a real scene object: patch, sticker, tape, card, prop label, barcode, evidence tag, lens reflection.

5.1. **User asks to “send the image” or “发我们” after generation.**
   - Fix: deliver the generated file directly in-chat as `MEDIA:/absolute/path/to/file` whenever possible, rather than stopping at a plain filesystem path. If multiple images were generated, send each as its own `MEDIA:` attachment in the same reply.

6. **Over-sexualizing human characters when adapting fashion details.**
   - Fix: use covered streetwear, technical panels, straps, jackets, gloves, patches, and printed motifs instead of exposed-skin emphasis.

7. **Letting IP/world UI overpower the style.**
   - Fix: reduce official UI/dashboard elements; use street-poster labels and abstract role motifs instead.

8. **Decorations becoming random clutter.**
   - Fix: tie icons to subject role, weapon, element, personality, palette, or supplied scene detail.

9. **Text becoming too neat and official.**
   - Fix: use cropped, angled, torn, overprinted typography; make text follow motion paths.

10. **Extreme foreshortening breaks canon anatomy on monsters/mecha/villains.**
   - Symptom: fingers multiply or bend unnaturally, limbs disappear, torsos twist into unreadable knots, or the subject loses its canonical body structure while trying to look dynamic.
   - Fix: keep the pose dynamic but slightly more legible. Prefer readable diagonal motion, clear limb count, visible head/torso anchors, and controlled perspective over heroic distortion. For body-structure-sensitive IPs like Frieza and Cell, do not push the camera so hard that hands, tails, wings, or joints become ambiguous.
   - Reference: `references/anatomy-safe-dynamics.md`.

12. **Hands, limbs, or paired anatomy become inconsistent.**
   - Symptom: left and right hands have different finger counts or species structure; one hand looks human while the other follows canon; arms/legs duplicate, fuse, vanish, or bend through impossible joints; tails/wings attach to the wrong place; a dynamic pose creates accidental extra limbs.
   - Fix: add an `ANATOMY LOCK` before style language and use anatomy-safe composition. State the exact limb/finger/paw/wing/tail count and require matching left/right anatomy. Reduce extreme foreshortening, keep wrists/elbows/knees/attachment points readable, and move sticker-bomb density away from hands, joints, and limb intersections. If the anatomy is still wrong, treat the image as not usable even when style and identity are strong.

## Verification Checklist

Before calling image generation, check:

- [ ] Prompt contains no third-party creator/artist/style names.
- [ ] Prompt does not depend on brand/studio references for the style.
- [ ] Subject identity cues are explicit and concise.
- [ ] A specific composition archetype is chosen.
- [ ] The archetype differs from the previous image when generating a batch.
- [ ] `NickZag` is integrated as an in-scene element, not a watermark.
- [ ] Background is sticker-bomb / torn poster / graffiti / typography driven.
- [ ] Palette includes deep black contrast plus neon cyan/magenta/acid accents.
- [ ] Negative prompt blocks centered layout, watermark behavior, official clean key art, identity loss, wrong hands, mismatched left/right hands, wrong limb count, duplicated/missing/fused limbs, broken joints, and malformed appendages.
- [ ] An `ANATOMY LOCK` is included whenever hands, limbs, tails, wings, paws, or weapon grips are visible.
- [ ] Mascots are not humanized or sexualized; human characters are not unnecessarily sexualized.

After generation, visually inspect if quality matters:

- [ ] Does the image follow the chosen composition archetype?
- [ ] Is the subject recognizable?
- [ ] Is `NickZag` part of the scene rather than an overlay?
- [ ] Is the style edgy/glossy/neon/sticker-bomb rather than clean official art?
- [ ] Is the layout meaningfully different from the previous batch image?
- [ ] Are both hands anatomically consistent with each other and with the subject’s canon/species finger count?
- [ ] Are all visible limbs, paws, feet, tails, wings, horns, joints, and attachment points correct, readable, and not duplicated/missing/fused?
