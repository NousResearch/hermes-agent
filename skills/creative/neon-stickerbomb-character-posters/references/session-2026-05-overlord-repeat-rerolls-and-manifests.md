# Overlord repeat rerolls and manifest continuity — 2026-05-26

## Context
Nick repeatedly requested Overlord character batches in the neon sticker-bomb style, including follow-up/reroll sets with overlapping characters:

- First non-core follow-up: Cocytus, Sebas Tian, Aura Bella Fiora, Mare Bello Fiore.
- Named five-character set: Narberal Gamma, Shalltear Bloodfallen, Sebas Tian, Ainz Ooal Gown, Momon.
- Four-character reroll: Narberal Gamma, Sebas Tian, Ainz Ooal Gown, Momon.
- Two-character reroll: Ainz Ooal Gown, Momon.

## Durable workflow lesson
Treat each repeated Overlord request as a fresh generation/reroll set unless Nick explicitly asks to resend old files. Preserve the latest numbered mapping by writing a new manifest for each completed set, even if only two images are generated.

## Composition rotation used successfully
Avoid reusing the immediately prior skeleton for the same character:

- Narberal Gamma: low-angle lightning diagonal -> side-profile split layout.
- Sebas Tian: side-profile split -> foreground fist/card depth -> low-angle fashion diagonal.
- Ainz Ooal Gown: circular necromancy ring -> foreground staff depth -> side-profile split.
- Momon: dutch-angle charge -> circular sword ring -> foreground greatsword depth.

For future rerolls, rotate again rather than falling back to the previous pose. Good next options include:

- Ainz: diagonal floating robe/cosmic spell trail; low-angle staff-command diagonal; dutch-angle dark-magic action poster.
- Momon: side-profile split with crossed swords and negative space; low-angle twin-blade diagonal; diagonal floating armor leap.
- Narberal: foreground spell-card/depth; circular lightning ring; dutch-angle caster action.
- Sebas: side-profile split with calm glove close-up; circular impact ring; dutch-angle martial step.

## Prompt anchor reminders
- Narberal Gamma: long straight black hair, cold severe expression, black-and-white battle maid uniform, white apron, gloves, lightning/caster aura; covered and non-sexualized.
- Sebas Tian: elderly white-haired butler, slicked-back white hair, sharp moustache, stern older face, black formal suit, white gloves, calm martial protector aura.
- Ainz Ooal Gown: skull face, crimson eye glow, ornate black/gold/white robe, huge jeweled collar, bony hands, necromancer staff; non-human skeletal, not living human.
- Momon: full black plate armor, closed helmet, broad armored shoulders, huge twin greatswords, no skull face visible, no robe.
- Shalltear Bloodfallen: pale vampire princess, very long silver hair, red eyes, black/red gothic frilled dress, parasol or crimson lance motif; covered/non-sexualized.

## Provider handling
A couple of full-template parallel generations returned transient 500/internal stream errors. The successful recovery pattern was to preserve successful siblings and retry only the failed slot with a compact anchor-first prompt that keeps: subject anchors, one composition archetype, in-scene NickZag, neon sticker-bomb mechanics, palette, and a short negative block.

Do not record the 500 itself as a durable limitation; the durable lesson is the compact retry pattern plus manifest preservation.
