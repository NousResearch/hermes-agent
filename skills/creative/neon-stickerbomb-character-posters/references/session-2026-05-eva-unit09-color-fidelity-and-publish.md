# Session: EVA Unit-09 color-fidelity correction and selected publish (2026-05-14)

## Trigger

Nick corrected the rerolled EVA Unit-09: `九号机不是全银机体`. The prior prompt over-emphasized silver/white/chrome and produced an all-silver-looking unit, which failed subject fidelity even though glossy material and sticker-bomb density were strong.

## Generation lesson

For EVA Unit-09 / Mark.09 / Adams Vessel, do **not** prompt it as an all-silver or full-chrome mecha. Use:

- mostly white/off-white / warm ivory lacquered armor
- black biomechanical undersuit and dark joints
- orange-gold/yellow helmet and face-plate cue
- red-orange face/chest core glow and red vessel nodes
- only small silver/chrome edge glints as highlights/trim
- slim non-human EVA proportions and control-lock / sealed Adams Vessel context

Prompt negative that worked:

```text
NEGATIVE: no all-silver body, no chrome statue, no generic white Gundam/Transformer, no muted industrial grunge, no sparse clean key art, no subject hidden by giant text, no face hidden, no text-only identification, no watermark/corner signature.
```

Keep the neon-stickerbomb core around the frame/background/weapon/cables, but protect the face, chest, shoulders, orange-gold head cue, and body silhouette.

## QC checklist for future Unit-09 rerolls

Pass only if:

- It is not read as a full silver/chrome robot.
- Body reads as white/off-white/ivory armor over black inner structure.
- Head/face has orange-gold/yellow cue.
- Red-orange face/chest core is visible.
- Mark.09 / Unit-09 / Adams Vessel context is present.
- Sticker-bomb density remains high without hiding identity cues.

Common caveat: high glossy white armor can still create a slight silver highlight illusion. That is acceptable if the base body clearly reads warm white/ivory rather than all-metal silver.

## Publication note

When Nick says `发布1` after the corrected Unit-09, publish only the corrected manifest index 1, not earlier rejected all-silver rerolls. In this session the corrected image was published as Eagle ID `MP4T5Q5M9F8UV`, route `eva-unit-09-9f8uv.html`.

During publication the first attempt failed because Eagle API was not listening on `localhost:41595` (`Connection refused`). Opening Eagle (`open -a Eagle`) brought the API back; retrying the same direct import/rebuild/deploy workflow succeeded.
