# Session: EVA late units 09 / 13 / 08+02 neon generation (2026-05-13)

## Context

Nick asked to continue the Evangelion neon sticker-bomb mecha sequence with:

- EVA Unit-09 / 九号机 / Adams' Vessel
- EVA Unit-13 / 十三号机
- EVA Unit-08+02 / 8+2号机 hybrid

The prior correction still governed the batch: **premium glossy mecha material is not enough**. The neon skill core requires dense torn decals, labels, barcodes, cropped type, graffiti/sticker chaos, halftone, chromatic split, and zine-poster energy while protecting the mecha silhouette.

## Prompting anchors that worked

Use `SUBJECT FIDELITY FIRST` before the style block, then a composition-specific sticker-bomb layout. Keep the head/chest/core silhouette unobscured and push graphic density to borders, backgrounds, cable/weapon arcs, decals, and slight armor-edge overlaps.

### 1. Unit-09 / Adams' Vessel

Useful cues:

- sleek white/silver EVA-like body
- red/orange face, visor, or core lights
- black undersuit/interior mechanics
- elegant villainous humanoid EVA silhouette
- long narrow head, slim torso, red accents
- `ADAMS VESSEL`, `CONTROL LOCK`, experimental vessel context
- black-red scythe, crescent, halberd, or half-moon weapon cue

Composition that worked:

```text
SIDE-SPLIT VESSEL / SCYTHE ARC: Unit-09 on the right third in polished three-quarter side profile; a huge black-red scythe arc sweeps from bottom-left to top-right as the motion spine. The opposite side is a dense sticker wall with EVA-09 / 九号机 / ADAMS VESSEL / CONTROL LOCK labels, barcode strips, red hazard tape, white lab decals, cracked halo diagrams, cyan-magenta glitch stamps and AT-field fragments.
```

QC target:

- should read as white/silver Adams Vessel, not a generic white robot
- red/orange facial/core details and black inner structure must be visible
- scythe/crescent weapon and control-lock/sealed context help distinguish it

### 2. Unit-13

Useful cues:

- dark purple / indigo EVA body
- long horned head
- four-arm awakened silhouette
- green and red accent lights
- dual spear / double spear motif
- ominous godlike awakening presence

Composition that worked:

```text
FOUR-ARM X-SPEAR AWAKENING: Unit-13 in a low-angle diagonal with head and chest readable. Four arms form an X-shaped frame around the torso; two long crimson spear shafts cross behind/in front as a double-spear symbol, with visible fork/prong tips inside the frame. Dense labels: EVA-13 / 十三号机 / AWAKENING / DOUBLE ENTRY / SPEAR LOCK, torn violet tape, red warning cards, green sync meters, barcode columns, halo shards, cyan-magenta glitch slices and black AT-field geometry.
```

QC target:

- Unit-13 can share purple/green associations with Unit-01, so four arms + dual spear X + `EVA-13` context need to carry the distinction
- do not let weapons hide the face/torso
- if four arms are partially obscured, report it as a caveat, not a silent pass

### 3. Unit-08+02 / 8+2号机

Useful cues:

- asymmetrical half-and-half EVA body fused from Unit-08 and Unit-02
- one side pink/magenta with Unit-08 sniper/targeting cues
- one side red with Unit-02 combat/lance cues
- split helmet, mismatched shoulders, hybrid torso seam
- black undersuit gaps and visible dual identity

Composition that worked:

```text
VERTICAL SPLIT HYBRID COLLISION: hybrid unit in a dramatic front three-quarter pose split by a jagged vertical seam of cyan lightning and torn decals. Left side uses magenta Unit-08 sniper/target stickers; right side uses red Unit-02 combat/lance warning stickers. A rifle barrel/scope and a forked spear fragment cross diagonally in opposite directions, framing the head without covering it. Dense labels: EVA-08+02 / 8+2号机 / HYBRID SYNC / TARGET LOCK / LANCE VECTOR, barcode strips, torn tape, magenta/red halftone bursts, AT-field shards and graffiti arrows.
```

QC target:

- should instantly read as half-pink/half-red hybrid, not a single-color robot
- Unit-08 sniper/targeting elements and Unit-02 red combat/lance elements should both be visible
- a pure left-right split is acceptable; if Nick wants more fusion, strengthen interpenetrating central armor, shared tubes, and sync-energy seams in the next reroll

## Provider/tooling quirk from this session

One attempted direct request to `http://127.0.0.1:8317/v1/responses` without the known local API key failed with:

```text
HTTPError 401: Unauthorized
```

Recovery was to mirror the working local script pattern:

```python
BASE = 'http://127.0.0.1:8317/v1'
KEY = 'sk-hermes-cliproxyapi'
# POST BASE + '/responses'
# headers include Authorization: Bearer <KEY>
```

Use a chat model such as `gpt-5.5` with the `image_generation` tool model set to `gpt-image-2`; include `Authorization` and save the sidecar manifest after every item.

## Manifest practice

Save a numbered sidecar manifest immediately, even for three images, because Nick often follows with compact publish commands such as `发布124`.

Example path:

```text
/Users/nick/.hermes/profiles/jea/state/neon_eva_units_09_13_82_dense_glossy_20260513.json
```

Required fields: index, title, slug, prompt, status, path, elapsed_seconds, provider, size, quality, api_model.

## Reporting pattern

Report each result by number with:

- image attachment path
- concise Chinese QC
- pass/fail
- main identity cues present
- main caveat

Do not overclaim strict canon reproduction; describe them as stylized glossy neon poster redesigns when the silhouette is heavily reinterpreted.
