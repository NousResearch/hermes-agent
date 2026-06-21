# Example Workflows

These are starter API-format workflows for the most common tasks. They're
ready to run with `scripts/run_workflow.py` once you've installed (or have
cloud access to) the listed models.

| File | Purpose | Required models | Min VRAM |
|------|---------|-----------------|----------|
| `sd15_txt2img.json` | SD 1.5 text-to-image (512×512) | SD1.5 checkpoint, e.g. `v1-5-pruned-emaonly.safetensors` | 4 GB |
| `sdxl_txt2img.json` | SDXL text-to-image (1024×1024) | `sd_xl_base_1.0.safetensors` | 8 GB |
| `flux_dev_txt2img.json` | Flux Dev text-to-image (1024×1024) | `flux1-dev.safetensors`, `t5xxl_fp16.safetensors`, `clip_l.safetensors`, `ae.safetensors` | 24 GB (or use `flux1-dev-fp8`) |
| `sdxl_img2img.json` | SDXL image-to-image | SDXL checkpoint | 8 GB |
| `sdxl_inpaint.json` | SDXL inpainting (image + mask) | SDXL checkpoint | 8 GB |
| `upscale_4x.json` | Standalone 4× ESRGAN upscale | `4x-UltraSharp.pth` (or any upscaler) | 4 GB |
| `animatediff_video.json` | AnimateDiff text-to-video (16 frames) | SD1.5 checkpoint, `mm_sd_v15_v2.ckpt` motion module | 8 GB |
| `wan_video_t2v.json` | Wan 2.x text-to-video (~33 frames) | `wan2.2_t2v_1.3B_fp16.safetensors`, `umt5_xxl_fp16.safetensors`, `wan_2.1_vae.safetensors` | 24 GB |

## Quick start

```bash
# Run a workflow with prompt injection
python3 ../scripts/run_workflow.py \
  --workflow sdxl_txt2img.json \
  --args '{"prompt": "majestic eagle in flight", "seed": 12345, "steps": 35}' \
  --output-dir ./out

# Img2img: upload an input image first via the script's helper
python3 ../scripts/run_workflow.py \
  --workflow sdxl_img2img.json \
  --input-image image=./photo.png \
  --args '{"prompt": "make it watercolor", "denoise": 0.6}' \
  --output-dir ./out

# Cloud (set API key once)
export COMFY_CLOUD_API_KEY="comfyui-..."
python3 ../scripts/run_workflow.py \
  --workflow flux_dev_txt2img.json \
  --args '{"prompt": "a fox in a misty forest"}' \
  --host https://cloud.comfy.org \
  --output-dir ./out

# What can I tweak in this workflow?
python3 ../scripts/extract_schema.py sdxl_txt2img.json --summary-only

# Are all required models / nodes installed?
python3 ../scripts/check_deps.py wan_video_t2v.json
```

## Notes

- **Inpaint masks**: white pixels = "regenerate this region", black = preserve.
  ComfyUI's `LoadImageMask` reads the **red channel** by default; export your
  mask as a single-channel image or as a normal RGB where red==intensity.

- **Denoise strength** in img2img: `0.0` = output identical to input,
  `1.0` = ignore input entirely. Sweet spot is usually 0.4–0.7.

- **Flux Dev** needs ~24 GB VRAM in its base form. The `flux1-dev-fp8.safetensors`
  variant (already on Comfy Cloud) cuts that roughly in half.

- **Video workflows** can take many minutes. The skill auto-detects video
  output nodes and bumps the default timeout to 900s. Override with `--timeout 1800`.

- These JSON files are deliberately **API format** (top-level keys are node IDs
  with `class_type`), not editor format. To open them in ComfyUI's web UI for
  visual editing, use `Workflow → Load (API Format)` or `Workflow → Open` and
  follow the prompt.

## Cloud vs local model names

Comfy Cloud's preinstalled checkpoints sometimes have a `-fp16` suffix
(`v1-5-pruned-emaonly-fp16.safetensors`) while the canonical local download
keeps the original name (`v1-5-pruned-emaonly.safetensors`). The example
workflows use the local-canonical names. When running on cloud, override with:

```bash
python3 ../scripts/run_workflow.py \
  --workflow sd15_txt2img.json \
  --args '{"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors", "prompt": "..."}' \
  --host https://cloud.comfy.org
```

The `ckpt_name`, `vae_name`, `lora_name`, `unet_name`, etc. are all exposed
as controllable parameters by `extract_schema.py` — discover what's installed
with `comfy model list` (local) or `curl /api/experiment/models/checkpoints`
(cloud).
