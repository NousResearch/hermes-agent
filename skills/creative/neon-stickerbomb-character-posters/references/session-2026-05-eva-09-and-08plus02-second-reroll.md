# Session: EVA Unit-09 and Unit-08+02 second reroll (2026-05-13)

## Context

Nick asked to regenerate EVA Unit-09 and EVA Unit-08+02 after earlier versions passed but still had caveats:

- Unit-09: previous scythe sometimes read as a decorative/control ring rather than a physical weapon; needed stronger red-orange core and clearer Adams Vessel / control-lock presence.
- Unit-08+02: previous result still read as a mostly left-pink/right-red split; needed stronger interwoven fusion while preserving Unit-08 sniper and Unit-02 lance cues.

The governing balance remained: **canon/readable mecha cues + premium glossy material + dense neon sticker-bomb core**. Do not fix identity by reducing sticker density; push stickers to borders, background, weapon/cable arcs, labels, and light armor-edge overlaps while protecting head/chest.

## Unit-09 reroll pattern that worked

Use direct shape/prop requirements rather than only typography:

```text
Vertical 3:4 glossy neon sticker-bomb poster of EVA Unit-09 / 九号机, Adams Vessel.
White/silver Unit-09 body, black biomechanical undersuit, long narrow EVA head, orange-red face slit and single strong chest/core glow, red vessel nodes, slim threatening Adams Vessel silhouette.
Center-forward three-quarter stance.
One hand clearly grips a scythe/halberd handle across lower frame; black-red crescent blade arcs behind head with visible metal edge and tip, not a decorative ring.
Restraint cables and lock clamps spiral behind.
Dense outer sticker wall: EVA-09, 九号机, ADAMS VESSEL, CONTROL LOCK, SEALED, CONTACT PROHIBITED, barcodes, torn red-white tape, cracked halo diagrams, cyan-magenta glitch stamps.
```

QC target:

- Identity should come from white/silver body, black inner frame, long narrow EVA head, and red-orange core — not labels alone.
- The scythe must have a visible handle, edge thickness, and tip. If it only appears as a halo/ring, reroll.
- Control-lock/sealed context should be visual as cables/clamps plus labels.
- It is acceptable if the output is a heavily stylized redesign, but report it instead of claiming official-settei fidelity.

## Unit-08+02 reroll pattern that worked

Use `true interwoven hybrid` and avoid clean vertical split language unless Nick wants half-and-half:

```text
Vertical 3:4 glossy neon sticker-bomb poster of EVA Unit-08+02 / 8+2号机.
True interwoven hybrid, not clean vertical half split: hot pink Unit-08 plates invade the red Unit-02 side, red/orange Unit-02 plates invade the magenta side, split helmet, mismatched shoulders, green eyes/visor, black shared undersuit, crossing sync cables and shared chest core.
Dynamic S-curve center pose, head/chest large and clear.
Unit-08 sniper scope ring with green glass in upper-left foreground; Unit-02 red forked lance cuts lower-right to upper-left, forming X around face without covering it.
Cyan sync lightning stitches both color systems.
Dense labels: EVA-08+02, 8+2号机, HYBRID SYNC, FULL INTERLOCK, TARGET LOCK, LANCE VECTOR, barcodes, magenta target decals, red combat tape, AT-field shards.
```

QC target:

- It may still have a broad left-08/right-02 read; the pass condition is visible interweaving at head/chest/torso/weapon area, not uniform full-body color mixing.
- Unit-08 cue: sniper/target/scope/green glass must be visible. If the scope floats ambiguously, caveat it and next prompt should add a barrel/connection/held rifle.
- Unit-02 cue: forked red lance must be visible. If it becomes a generic blade, next prompt should specify double-pronged/forked spearhead and long shaft.
- Head/chest readability is more important than putting the scope/lance directly over the face.

## CLIProxyAPI payload pitfall

During this reroll, a long script using `tool_choice` and `partial_images` hung for ~10 minutes with no output. A compact retry with `gpt-5.5-mini` returned `HTTP 502 Bad Gateway` for both items. The working retry mirrored the simpler older pattern:

```python
payload = {
  'model': 'gpt-5.5',
  'store': False,
  'instructions': 'Generate exactly one image by using the image_generation tool. Follow the prompt closely.',
  'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}]}],
  'tools': [{'type': 'image_generation', 'model': 'gpt-image-2', 'size': '1024x1536', 'quality': 'medium', 'output_format': 'png', 'background': 'opaque'}]
}
```

Avoid `tool_choice` and `partial_images` for this local CLIProxyAPI image path unless a current working template proves otherwise. If a process is silent for several minutes, kill it, preserve any manifest entries, and retry only pending indices with the simple payload.

## Reporting pattern

For rerolls, report concise Chinese QC with:

- pass/fail
- what improved relative to the prior caveat
- main identity cues present
- remaining caveat
- `MEDIA:/absolute/path` attachment

Do not publish rerolls unless Nick explicitly asks to publish.
