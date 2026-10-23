---
name: trellis-image-to-3d
description: Use when generating 3D GLB assets from images via the microsoft/TRELLIS.2 Hugging Face Space (uses the user's HF_TOKEN ZeroGPU quota). Covers single asset generation, preprocessing (alpha-masking/background removal), and batch character pipelines (concept -> image gen -> bg removal -> TRELLIS.2 -> GLB + manifest) for game dev workflows in Godot/Blender.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [3d, game-dev, gradio, huggingface, trellis, glb, asset-pipeline]
    related_skills: [forgedna-comfyui-assets, blender-mcp, hermes-3d-mcp, comfyui]
---

# TRELLIS.2 Image-to-3D via Hugging Face Space

## Overview

`microsoft/TRELLIS.2` converts a 2D image into a textured 3D GLB asset. Running it through the HF Space with the user's `HF_TOKEN` spends THEIR ZeroGPU quota — no local GPU required. This skill ships `scripts/trellis_gen.py`, a self-contained client that handles the full call chain: upload -> preprocess -> generate -> extract GLB -> save locally.

The Space's Gradio API (verified against `/gradio_api/info`):

| Endpoint | Inputs | Outputs |
|---|---|---|
| `/start_session` | — | — |
| `/preprocess_image` | image | alpha-masked RGBA image |
| `/get_seed` | randomize_seed, seed | seed |
| `/image_to_3d` | image, seed, resolution, then 3 stages x (guidance, guidance_rescale, steps, rescale_t) | 3D preview |
| `/extract_glb` | decimation_target, texture_size | GLB path, download file |

Defaults from the Space UI: resolution `512`, stage 1+2 params `7.5 / 0.7 / 12 / 5`, stage 3 params `1 / 0 / 12 / 3`, decimation `300000`, texture size `2048`.

## When to Use

- User asks to turn concept art / generated images into 3D GLB assets
- Batch asset pipelines: "make me 10 elemental ant enemies as 3D models"
- Any game-dev flow targeting Godot / Blender / HermesForge with GLB imports
- Don't use for: rigging/skinning (roadmap: VAST-AI/SkinTokens, see Roadmap section), text-to-3D directly (generate an image first)

## Setup

1. `HF_TOKEN` must be set (read token at https://huggingface.co/settings/tokens). ZeroGPU quota is tied to the user's HF account; without a token the Space runs anonymously with no ZeroGPU priority and may fail.
2. Install the client (once): `pip install "gradio_client>=1.0"` — that's the only dependency.
3. Optional for background removal: `pip install rembg` (+ first run downloads the u2net model ~170MB).

## Single Asset Generation

```bash
python3 scripts/trellis_gen.py input.png -o assets/goblin.glb \
    --name goblin --preprocess --resolution 512 --texture-size 2048
```

Key flags (all optional, Space defaults shown):
- `--preprocess` — call `/preprocess_image` first (alpha-mask the foreground). DO use this for raw images; skip if the image is already RGBA alpha-masked.
- `--seed N` / `--randomize-seed` (default: randomize)
- `--resolution {512,1024,1536}` (512 default; higher = slower, more quota)
- `--decimation 300000` (face target at GLB extraction)
- `--texture-size {1024,2048,4096}`
- `--stage1 "7.5,0.7,12,5"` `--stage2 "7.5,0.7,12,5"` `--stage3 "1,0,12,3"`
- `--keep-image` — also save the (preprocessed) image next to the GLB
- Completion: the `.glb` file exists at `-o` and is non-trivial in size (>50KB for a real asset).

GLB extraction server-side can take 30-60+ seconds and is the most likely step to hit queue timeouts — the script retries with backoff; if it still fails, re-run with the same `--seed` to retry extraction on the cached generation.

## Batch Character Pipeline (the "10 elemental ants" recipe)

End state per run: a working folder containing `images/NN_slug.png`, `glb/NN_slug.glb`, and `manifest.csv` (slug, prompt, image_path, glb_path, seed, resolution, status). Steps:

1. **Concept list.** Write the N prompts to `prompts.txt` (one per line). Be specific and consistent: `"fire ant warrior character, glowing ember carapace, T-pose front view, full body, fantasy game asset, centered, dark background"`. Front/full-body shots with the whole silhouette visible produce drastically better 3D gens.
2. **Image generation.** Generate one image per prompt with an open image Space or local tool — preferred order: local ComfyUI (see `comfyui`/`forgedna-comfyui-assets` skills, free, no quota) > HF image Space via `gradio_client` (e.g. FLUX.1-schnell Spaces) > `image_generate` tool. Save as `images/NN_slug.png`.
3. **Background removal.** TRELLIS.2 wants an alpha-masked foreground. Either pass `--preprocess` (server-side masking, easiest) OR run locally for more control:
   ```bash
   rembg i input.png output_rgba.png
   ```
   Verify the alpha channel actually isolates the character (no floating props cut off, lantern/weapon intact). Bad masks = melted 3D models. Re-generate or hand-fix bad ones before burning ZeroGPU quota.
4. **Batch convert.** Loop `trellis_gen.py` over the images. The script appends each result to `manifest.csv` in the output root. Run sequentially (ZeroGPU is single-queue per account) and expect several minutes per asset; for 10+ assets run it with `terminal(background=true, notify_on_complete=true)`.
   ```bash
   python3 scripts/trellis_gen.py --batch images/ --out-root assets/ants --preprocess --keep-image
   ```
5. **Verify + import.** Open a sample GLB (Blender: File > Import > glTF 2.0; Godot 4: drag into project, auto-imports). Check texture is applied and scale/orientation is sane. Completion: manifest has N rows with `status=ok` and every GLB opens.

## Common Pitfalls

1. **No HF_TOKEN passed.** The script reads `HF_TOKEN` from env; without it you get anonymous quota and frequent queue failures. Export it before batch runs or add it to `~/.hermes/.env` (source it with `set -a; . ~/.hermes/.env; set +a` — new Hermes sessions don't auto-inherit a token exported in another shell).
2. **gradio_client 1.x renamed the auth kwarg to `token=`** (was `hf_token=` in 0.x). Scripts use `token=`; if you see `unexpected keyword argument 'hf_token'` you're reading old examples.
2. **Skipping preprocessing on busy backgrounds.** TRELLIS.2 reconstructs whatever it sees — a background becomes geometry. Always alpha-mask first (flag or rembg).
3. **Cropped silhouettes.** Characters with cut-off feet/weapons generate incomplete meshes. Prompt for "full body, centered, margin around subject".
4. **Parallel batch calls.** ZeroGPU serializes per-account; parallel clients just queue-fight and time out. Sequential loop only.
5. **Expecting rigged output.** TRELLIS.2 GLBs are static meshes. Rigging is the SkinTokens phase (below).
6. **Giving up on extract_glb timeout.** Generation is cached server-side per session; re-running with the same seed often succeeds on retry.

## Roadmap (planned companion skills / phases)

- **Phase 2 — Rigging:** VAST-AI/SkinTokens Space (`https://vast-ai-skintokens.hf.space`, agents.md verified — needs >=14GB VRAM, runs on ZeroGPU with token) converts a static GLB into a skeleton+skin-weight rigged model. Same gradio_client pattern: upload -> `/gradio_api/queue/join` -> stream `/queue/data`. Planned as `scripts/skintokens_rig.py` in this skill or a sibling `skintokens-rigging` skill.
- **Phase 3 — Engine import:** drop rigged GLBs into Blender (`blender-mcp`) for polish, or Godot/HermesForge (`hermes-3d-mcp`) for gameplay wiring.
- **Phase 4 — Unified asset pipeline:** one orchestrating skill (concept -> image -> mask -> 3D -> rig -> engine) that invokes the phase skills only as needed for the user's current goal. Keep each phase independently usable.

## Verification Checklist

- [ ] `HF_TOKEN` exported and valid (script exits early with a clear error otherwise)
- [ ] Input images are RGBA alpha-masked (or `--preprocess` used)
- [ ] Output `.glb` exists, >50KB, opens in Blender/Godot with textures
- [ ] Batch runs: `manifest.csv` row per asset with prompt, seed, paths, status
- [ ] Sequential execution for batches (no parallel ZeroGPU calls)
