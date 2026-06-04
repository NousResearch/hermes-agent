# Session note — Dragon Ball villains: canon shape + dynamic pose + Rednote copy hygiene

Date: 2026-05-17

## Context
Nick generated Dragon Ball villain posters in the established glossy neon sticker-bomb style: Frieza, Perfect Cell, Majin Buu, then rerolled Frieza/Cell/Super Buu. The initial outputs preserved identity but Nick rejected the result as too stiff: “姿势过于呆板”. Subsequent Cell rerolls were also requested until the pose became more clearly dynamic.

## Durable generation lesson
For villain/monster known-IP subjects, canon silhouette alone is not enough. The prompt must also specify an explicit action skeleton if the expected image is a dramatic poster.

### Use dynamic action anchors early
Put these before style language:

- `EXTREME MOTION POSE`
- `not standing / no stiff standing pose / no mannequin pose`
- `airborne diagonal lunge`
- `diving pounce from upper left to lower right`
- `torso twisted into an S-curve`
- `one giant hand/claw in extreme foreground`
- `tail spiraling/snapping across the foreground as motion line`
- `wings flared wide`
- `energy orb/beam thrust toward viewer`
- `face and canon head/torso anchors remain readable`

### Keep sticker density outside the identity anchors
For silhouette-driven villains, keep the character body clean/readable and move the neon/sticker-bomb density to:

- background and borders
- huge cropped typography following the motion path
- warning labels and barcode fragments
- torn decals and graffiti tags
- prop/energy/tail arcs

### Character anchors used successfully

**Frieza final form**
- compact sleek white alien tyrant body
- glossy purple head dome and body plates
- long white tail
- red eyes / smug cruel expression
- dynamic airborne diagonal strike
- tail spirals through foreground
- one finger thrusts toward viewer with purple death-ball
- block golden form, bulky monster, missing tail, wrong colors

**Perfect Cell**
- tall green bio-android insect warrior
- black spotted green carapace
- black torso panels
- pale face, purple cheek marks
- crown head crest, black side fins
- wing-like back plates, long spotted tail
- dynamic diving pounce / attack dash
- huge clawed hand in extreme foreground
- wings flared and tail as motion line
- block generic bug monster, robot armor, missing spots/wings, wrong palette

**Super Buu / slim Majin Buu**
- tall lean pink magical villain
- long head antenna
- narrow angry eyes, sinister grin
- black vest with gold trim
- white baggy pants, yellow gloves/boots
- candy beam / smoke ring / elastic twisting kick
- block fat rounded Buu, Kid Buu child body, gray demon, missing antenna, wrong outfit

## Retry pattern
If a batch hangs or only some items finish, preserve the manifest and retry only unfinished or rejected items. For one rejected character, use a single-image compact prompt rather than relaunching the full batch.

## Rednote copy note
When publishing these generated IP posters to Xiaohongshu/Rednote, Nick corrected that captions should not mention the neon/sticker-bomb visual system and should not use `#neonstickerbomb`. Captions should focus on character temperament, story tension, recognizable traits, and role archetype.
