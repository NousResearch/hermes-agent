---
name: skintokens-rigging
description: "Use when rigging static 3D meshes (GLB/OBJ/FBX) with automatic skeleton + skin weights via the VAST-AI/SkinTokens Hugging Face Space (user's HF_TOKEN ZeroGPU quota). Pairs with trellis-image-to-3d — TRELLIS.2 generates static GLBs, SkinTokens rigs them for animation in Godot/Blender. Covers single and batch rigging plus the full concept-to-rigged-asset pipeline."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [3d, rigging, skinning, game-dev, gradio, huggingface, glb, skeleton]
    related_skills: [trellis-image-to-3d, blender-mcp, hermes-3d-mcp, hf-gradio-3d-rig]
---

# SkinTokens Auto-Rigging via Hugging Face Space

## Overview

`VAST-AI/SkinTokens` takes a static 3D mesh (.glb/.obj/.fbx) and predicts a skeleton + skin weights, returning a rigged GLB ready for animation retargeting in Godot 4 or Blender. It needs >=14GB VRAM, so the HF Space with the user's `HF_TOKEN` (ZeroGPU quota) is the practical way to run it. This skill ships `scripts/skintokens_rig.py`, a self-contained gradio_client wrapper.

The Space exposes a single endpoint (verified against `/gradio_api/info`):

`/run_gradio(models, top_k=5, top_p=0.95, temperature=1.0, repetition_penalty=2.0, num_beams=10, use_existing_skeleton=False, preserve_texture_scale=False, voxel_postprocess=False, checkpoint="experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt", hf_path="None") -> (status, rigged_glb)`

`models` accepts a list of files — you can rig a few per call, but sequential single calls are more debuggable and kinder to quota.

## When to Use

- After `trellis-image-to-3d` (or any other source) produced static GLBs that need skeletons/skin weights
- User asks to "rig", "skin", or "make animatable" a 3D model
- Don't use for: static props (rocks, houses — skip rigging), image-to-3D generation (that's trellis-image-to-3d), animation retargeting itself (do that in Godot/Blender)

## Setup

Same as trellis-image-to-3d: `export HF_TOKEN=***` and `pip install "gradio_client>=1.0"`. Hermes loads `.env` from the working directory — putting `HF_TOKEN=...` in the project `.env` (never committed) also works.

## Usage

Single asset:
```bash
python3 scripts/skintokens_rig.py goblin.glb -o rigged/goblin_rigged.glb
```

Batch (sequential; appends status to `<out-dir>/rig_manifest.csv`):
```bash
python3 scripts/skintokens_rig.py --batch assets/ants/glb/ -o assets/ants/rigged/
```

Useful flags (Space defaults shown):
- `--top-k 5 --top-p 0.95 --temperature 1.0 --rep-penalty 2.0 --beams 10` — sampling knobs; leave alone unless output is degenerate, then lower temperature (e.g. 0.7) first.
- `--preserve-texture-scale` — keeps original UV textures and world scale; usually WANT this on for TRELLIS.2 outputs (they're already textured and reasonably scaled).
- `--use-existing-skeleton` — only when input already has a skeleton and you just need skin weights.
- `--voxel-postprocess` — extra cleanup pass for skin weights. **CONFIRMED RULE from live batch: models that fail with "No output files were produced" (3/10 in the ant batch: translucent ice, glossy venom, granite stone textures) succeeded on the FIRST retry with this flag enabled. If a rig fails server-side, retry with `--voxel-postprocess` before anything else.**
- Completion: output GLB exists, opens in Blender with an Armature modifier / in Godot with a Skeleton3D node.

## Full Pipeline (concept -> rigged game asset)

This is the combined trellis-image-to-3d + skintokens-rigging chain — the "10 elemental ant enemies" end-to-end recipe:

1. **Concepts** — `prompts.txt`, one line per asset: `"fire ant warrior, glowing ember carapace, full body T-pose front view, centered, fantasy game asset, dark background"`. T-pose-ish, full-body, uncropped silhouettes matter for BOTH generation quality and rigging quality (SkinTokens finds joints much better on spread limbs).
2. **Images** — generate one PNG per prompt (local ComfyUI preferred, else HF image Space / image_generate). Save `images/NN_slug.png`.
3. **Alpha-mask** — `rembg i in.png out.png` per image, or rely on TRELLIS `--preprocess`. Verify masks before spending GPU.
4. **3D meshes** — `python3 <trellis-skill>/scripts/trellis_gen.py --batch images/ --out-root assets/ants --preprocess --keep-image` (sequential; several min/asset; run with `terminal(background=true, notify_on_complete=true)`).
5. **Rig** — `python3 <this-skill>/scripts/skintokens_rig.py --batch assets/ants/glb/ -o assets/ants/rigged/ --preserve-texture-scale`.
6. **Verify + import** — open a rigged GLB in Blender (check armature bones sit inside the mesh, weights deform plausibly in pose mode) or drag into Godot 4. Completion: N rigged GLBs + manifest.csv + rig_manifest.csv all consistent.
7. **Polish/animate** — retarget animations onto the generated skeleton in Blender (`blender-mcp`) or Godot (`hermes-3d-mcp`). Auto-rigs usually need weight cleanup around thin appendages; budget a manual pass for hero characters.

## Common Pitfalls

1. **No HF_TOKEN** — anonymous runs hit the public queue and frequently fail on this GPU-heavy Space. Export it first.
2. **Rigging props.** Static scenery gains nothing; only rig characters/creatures.
3. **Tangled input poses.** Characters holding weapons across the body or with limbs touching the torso confuse joint prediction. Prefer T-pose source images upstream in TRELLIS.
4. **Cropped/waist-up source images will NOT rig.** Confirmed live: a waist-up wizard portrait produced a valid static GLB but SkinTokens failed with "No output files" even WITH voxel-postprocess — you can't skeletonize legs that don't exist. The retry auto-escalation rescues texture/material failures, NOT missing-geometry failures. Rule: if rigging is the goal, the source image MUST be full-body. (We tried gating this via bounding-box proportions — doesn't work, wide creatures like ants are legitimately squat. No cheap geometric test; enforce it at the image prompt/review stage.)
4. **Expecting production weights.** Auto skinning is a starting point — thin geometry (antennae, fingers, lantern handles) often needs manual weight painting.
5. **Parallel batch calls** — ZeroGPU serializes per account; loop sequentially.
6. **Forgetting --preserve-texture-scale on TRELLIS outputs** — without it you may lose the texture TRELLIS baked.

## Verification Checklist

- [ ] `HF_TOKEN` exported (or in project `.env`)
- [ ] Output GLB contains a skeleton (Blender: Armature; Godot: Skeleton3D)
- [ ] Skin weights deform plausibly (test a few bone rotations)
- [ ] Batch: `rig_manifest.csv` has one row per input with status
- [ ] Textures survived (use `--preserve-texture-scale`)
