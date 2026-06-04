# Session: EVA Unit-09 / Unit-13 / Unit-08+02 neon mecha generation and reroll (2026-05-13)

## Context

Nick requested a continuation of the dense glossy EVA neon series:

- `生成九号机 十三号机 8+2号机`
- then `发布2，重新生成1 3`

The governing style remained the corrected EVA balance: **canon/subject cues first + premium glossy mecha material + dense neon sticker-bomb core**.

## Generation anchors that passed QC

### Unit-09 / 九号机 / Adams Vessel

Initial pass already worked, but the reroll improved the concept. Better prompt direction:

- white/silver Adams Vessel body
- black inner frame/undersuit clearly exposed
- red-orange facial/core glow and red control nodes
- control-lock / sealed / contact-prohibited experimental language
- heavy restraint cables and lock clamps
- black-red crescent/scythe arc behind or around the torso
- dense EVA-09 / 九号机 / ADAMS VESSEL / CONTROL LOCK / CONTACT PROHIBITED sticker wall

QC note: if the scythe is only a large arc, it can be read as a seal/AT-field ring. If Nick requires an actual weapon, explicitly require visible handle, blade thickness, and grip.

### Unit-13 / 十三号机

The published image passed with:

- dark purple/indigo armor
- horned long EVA head
- four-arm awakened silhouette
- dual crimson spears forming a visible X
- EVA-13 / 十三号机 / AWAKENING / DOUBLE ENTRY / SPEAR LOCK sticker labels
- dense neon lab-warning collage without hiding head/chest

Pitfall: purple/green can briefly read as Unit-01 if the four arms and dual-spears are weak. Keep four-arm and double-spear X cues explicit; text labels alone are not enough.

### Unit-08+02 / 8+2号机

Initial pass worked but looked too much like a simple left/right split. Reroll direction that passed better:

- “true interlocking fusion, not simple left-right halves”
- pink/magenta Unit-08 and red Unit-02 armor plates alternating like a zipper
- split helmet with green eyes/visor
- shared black undersuit and hybrid torso seam
- Unit-08 sniper/targeting cues: scope ring, target lock, long-range labels
- Unit-02 lance/combat cues: forked red spear, lance vector, red combat warning tape
- cyan lightning seam spiraling around body, not a flat vertical divider

QC note: even the reroll can retain a left-08/right-02 macro split. If Nick wants a stricter fusion, push red armor into left shoulder/left chest/left arm and pink armor into right chest/right arm/weapon interfaces; add shared spine/core/cables.

## Operational pitfall found

A direct CLIProxyAPI script failed with `HTTPError 401: Unauthorized` when it used `/v1/responses` without the authorization header and with the wrong direct image-generation payload shape. The proven working pattern is:

- base: `http://127.0.0.1:8317/v1`
- endpoint: `/responses`
- header: `Authorization: Bearer sk-hermes-cliproxyapi`
- dialogue model: `gpt-5.5`
- image tool model: `gpt-image-2`
- tool choice:

```json
{"type":"allowed_tools","mode":"required","tools":[{"type":"image_generation"}]}
```

Use the batch template or copy the already-working 05–08 script shape rather than inventing a simpler `/v1/responses` image-only request.

## Manifest convention

For mixed commands like `发布2，重新生成1 3`, keep the same manifest as the numbered source of truth:

- publish the selected existing index exactly
- update rerolled indices in the manifest with new `path`, `slug`, and prompt
- do not publish rerolled indices unless Nick asks
- final report should clearly separate `published` from `rerolled`
