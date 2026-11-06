---
name: skintokens-rigging
description: "Auto-rig static meshes into skinned GLBs on HF Spaces."
version: 0.1.0
author: Jonathan Korstad (@Jpalmer95), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [3d, rigging, skinning, game-dev, glb, gradio, huggingface]
    related_skills: [trellis-image-to-3d, game-asset-pipeline]
---

# SkinTokens Auto-Rigging Skill

Predicts a skeleton and skin weights for a static mesh through the
`VAST-AI/SkinTokens` Hugging Face Space, returning a rigged GLB ready for
animation in Godot or Blender. Ships `scripts/skintokens_rig.py` (single and
batch, retry with auto-escalation, resumable manifest) and its own
`scripts/rig_gates.py`, so the skill installs and runs standalone.
It does not animate and does not retarget — that happens in the engine.

## When to Use

- A static GLB/OBJ/FBX needs a skeleton or skin weights ("rig this", "make it
  animatable", "skin this model")
- After image-to-3D generation, when the mesh must deform
- Don't use for: static props (rocks, buildings — rigging adds nothing),
  image-to-3D itself (`trellis-image-to-3d`), or animation retargeting (engine
  work, or the `puppeteer-animation` skill for video-driven animation)

## Prerequisites

- `HF_TOKEN` in the environment Hermes runs with (https://huggingface.co/settings/tokens).
  The Space needs ≥14 GB VRAM, so anonymous queue runs fail often.
- `pip install "gradio_client>=1.0"`.
- Source meshes must be **full-body**. Waist-up or cropped meshes produce a
  valid static model that SkinTokens cannot skeletonize, and no retry rescues
  missing geometry.

## How to Run

Single mesh:

```bash
python3 <skill>/scripts/skintokens_rig.py model.glb -o rigged/model_rigged.glb --preserve-texture-scale
```

Batch (sequential; appends to `<out-dir>/rig_manifest.csv`):

```bash
python3 <skill>/scripts/skintokens_rig.py --batch assets/glb/ -o assets/rigged/ --preserve-texture-scale --resume
```

`<skill>` is the `skill_dir` reported by `skill_view(name='skintokens-rigging')`.
Both run through `terminal(command="…", timeout=…)`; a batch belongs in
`terminal(background=true, notify_on_complete=true)`.

## Quick Reference

| Flag | Effect |
|---|---|
| `--preserve-texture-scale` | keep the input's UVs and world scale — use it for textured image-to-3D meshes |
| `--voxel-postprocess` | extra weight cleanup; the auto-escalation target for "No output files" |
| `--use-existing-skeleton` | input already has a skeleton; predict weights only |
| `--top-k/--top-p/--temperature/--rep-penalty/--beams` | sampler knobs; defaults are the Space's |
| `--resume` | skip meshes whose `_rigged.glb` already exists |
| `--skip-gates` | disable the rig gate |
| `--beams` | integer; the Space rejects a float here |

## Procedure

1. **Screen the inputs.** Full-body, T-pose-ish, limbs clear of the torso.
   Completion: every mesh in the batch is a character or creature that needs
   deformations.
2. **Rig.** Run the batch command. Completion: one `_rigged.glb` per input and a
   `rig_manifest.csv` row for each.
3. **Handle failures deliberately.** "No output files were produced" on a
   translucent, glossy, or rocky surface is a material problem — retry with
   `--voxel-postprocess` (the script escalates on its own after the first
   failure). The same error on a cropped source is a geometry problem and will
   never succeed; regenerate the source image instead.
4. **Verify deformation.** Import a rigged GLB into Blender (Armature modifier)
   or Godot (Skeleton3D), rotate 2–3 bones. Completion: the mesh follows the
   bones without tearing.
5. **Clean up before shipping.** SkinTokens leaves a joint-visualisation
   `Icosphere` mesh in the file; strip it, and budget a manual weight pass for
   hero characters around thin appendages.

## Pitfalls

1. **Auto weights are a starting point.** Antennae, fingers, and held props
   usually need manual weight painting before animation reads well.
2. **Rigging props wastes quota.** Only rig what deforms.
3. **Stray `Icosphere`.** It ships inside the GLB and shows up in engines.
4. **Parallel calls.** ZeroGPU serializes per account; loop sequentially.
5. **Dropping `--preserve-texture-scale`.** TRELLIS-baked textures can be lost
   or rescaled without it.
6. **Pose-dependent joints.** Limbs touching the torso or a weapon crossing the
   body confuse joint prediction; fix the source pose upstream.

## Verification

- [ ] `rig_manifest.csv` exists with one row per input
- [ ] Each accepted GLB parses, has a skin with >5 joints, and carries
      `JOINTS_0`/`WEIGHTS_0`
- [ ] One asset pose-tested in Blender or Godot
- [ ] `Icosphere` removed from anything headed for an engine
