---
name: game-asset-pipeline
description: "Concept to rigged game assets in one resumable run."
version: 0.1.0
author: Jonathan Korstad (@Jpalmer95), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [3d, game-dev, pipeline, glb, rigging, asset-pipeline]
    related_skills: [trellis-image-to-3d, skintokens-rigging, comfyui]
---

# Game Asset Pipeline Skill

Drives the image → mask → mesh → rig chain as one run, with a stage ledger so
an edit reruns only what it invalidates. It orchestrates the phase skills
(`trellis-image-to-3d`, `skintokens-rigging`) rather than reimplementing them:
each phase stays independently usable, and the ledger records which stages are
current, stale, or missing per asset.
It does not animate, rig by hand, or do retopology.

## When to Use

- "Make me 10 enemy characters for my game" from text descriptions
- "Turn this folder of concept art into rigged models"
- Resuming a half-finished batch after editing one prompt, image, or mesh
- Don't use for: a single static prop (call `trellis-image-to-3d` directly),
  re-rigging existing meshes (call `skintokens-rigging`), 2D-only work

## Prerequisites

Both phase skills installed, because this skill invokes their scripts and does
not vendor copies:

```bash
hermes skills install official/gaming/trellis-image-to-3d official/gaming/skintokens-rigging
```

Plus each phase's prerequisites: `HF_TOKEN`, `gradio_client>=1.0`, and
`pip install rembg` for local masking. Optional: local ComfyUI for image
generation, Blender (or `blender-mcp`) for import checks.

## How to Run

```bash
python3 <skill>/scripts/asset_stage.py init assets/ants --name ants --target godot
python3 <skill>/scripts/asset_stage.py add assets/ants 01_fire "fire ant warrior, six legs, full body T-pose"
python3 <skill>/scripts/asset_stage.py plan assets/ants --slug 01_fire
python3 <skill>/scripts/asset_stage.py run assets/ants 01_fire --through rig
```

`<skill>` is the `skill_dir` from `skill_view(name='game-asset-pipeline')`.
`plan` prints only the phase commands still needed; `run` executes them and
records each stage as it completes. Run long assets through
`terminal(background=true, notify_on_complete=true)`.

## Quick Reference

| Command | Effect |
|---|---|
| `init <root>` | create the project folder (`project.json` plus canonical dirs) |
| `add <root> <slug> "<prompt>"` | register an asset at the prompt stage |
| `set-prompt <root> <slug> "<prompt>"` | new prompt; marks image onward stale |
| `import <root> <slug> --stage image\|mask\|mesh\|rig --file F` | drop in a file you made elsewhere |
| `adopt <root>` | build `project.json` from an existing `images/` + `glb/` + `rigged/` tree |
| `refresh <root>` | re-hash files; a manual overwrite marks that stage `external-edit` and stales everything after it |
| `status <root> [-v]` | per-asset stage table |
| `plan <root> [--slug X] [--through rig] [--json]` | the remaining commands, nothing else |
| `run <root> <slug> [--through rig] [--dry-run]` | execute and record |
| `--flux-script PATH` | optional image generator (takes `--prompt`/`-o`); without it the image stage is a tool step |

Canonical paths: `images/<slug>.png`, `images_masked/<slug>.png`,
`assets/glb/<slug>.glb`, `assets/rigged/<slug>_rigged.glb`.

## Procedure

1. **Concepts.** Write one line per asset. Characters only for rigging; props
   stop after the mesh stage. Completion: `plan` reports only stages you intend
   to run.
2. **Images.** Generate `images/<slug>.png` with the `image_generate` tool or
   local ComfyUI, then `import --stage image`. Vision-gate each one
   (`vision_analyze`) against a *known-good* reference — full body, limb count,
   plain background — before spending quota. Completion: every image passes the
   gate, or is regenerated.
3. **Masks.** `import --stage mask` after rembg, or let TRELLIS `--preprocess`
   mask server-side. Completion: alpha isolates the subject with props intact.
4. **Meshes and rigs.** `run <root> <slug> --through rig`. The ledger applies
   the engine preset, retries a failed rig once with `--voxel-postprocess`, and
   records `gate_failed` rows instead of silently burning GPU. Completion:
   `status` shows `ok` through the stages you asked for.
5. **Edits rerun downstream only.** Change a prompt: `set-prompt`, then `plan`.
   Replace a PNG or GLB: `import`, or overwrite the canonical file and
   `refresh`. Completion: `plan` lists only the invalidated stages.
6. **Hand off.** Import a rigged GLB into Blender or Godot and pose a few
   bones. Completion: deformations read correctly and no `Icosphere` remains.

## Pitfalls

1. **Prompt/image drift.** A comment line used to shift every prompt by one in
   older runs; `prompts.txt` parsing now skips `#` lines and blanks, and
   `adopt` matches prompts to images by words, not by index.
2. **Skipping `--resume`.** A crash at asset 8 of 10 restarts from 1 otherwise.
   The ledger plus `--resume` on both phase scripts is what makes batches safe.
3. **Rigging props.** Filter `assets/glb/` or split the prompt list; static
   scenery gains nothing from a skeleton.
4. **Gating images on aesthetics.** Gate on riggability instead: symmetrical
   front view, legs clear of the body, no spread wings. A hero render that
   *looks* better fails the mesh and rig phases.
5. **Missing sibling skills.** `plan` prints an install hint and emits no
   command for a phase whose script is absent; that is deliberate, not a bug.
6. **Expecting the ledger to call the network.** It never does — it prints and
   runs phase commands, which is why it is testable offline.

## Verification

- [ ] `status` shows `ok` for every accepted asset, with stale stages explained
- [ ] `manifest.csv` (meshes) and `rig_manifest.csv` (rigs) agree with `status`
- [ ] `plan --slug <x>` after an edit lists only downstream stages
- [ ] One rigged asset opens and poses in the target engine
