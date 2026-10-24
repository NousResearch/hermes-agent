# ComfyUI Workflow JSON Format

## Two Formats — Only API Format Is Executable

**API format** is required for `/api/prompt` and every script in this skill.
The web UI also produces an "editor format" used for visual editing, which
**cannot** be submitted directly.

### API Format

Top-level keys are string node IDs. Each node has `class_type` and `inputs`:

```json
{
  "3": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 156680208700286,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": ["4", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    },
    "_meta": {"title": "KSampler"}
  },
  "4": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}
  }
}
```

**Detection:** every top-level value has `class_type`. The skill's
`_common.is_api_format()` does this check.

### Editor Format (not directly executable)

Has `nodes[]` and `links[]` arrays — the visual graph. To convert: open in
ComfyUI's web UI and use **Workflow → Export (API)** (newer UI) or the
"Save (API Format)" button (older UI).

**Detection:** top-level has `"nodes"` and `"links"` keys.

## Inputs: Literals vs Links

```json
"inputs": {
  "text": "a cat",         // literal — modifiable
  "seed": 42,              // literal — modifiable
  "clip": ["4", 1]         // link — wiring; do NOT overwrite
}
```

Links are length-2 arrays of `[upstream_node_id, output_slot]`. The skill's
parameter injector refuses to overwrite a link with a literal (logs a
warning and skips).

## Common Node Types and Their Controllable Parameters

The full catalog lives in `scripts/_common.py` (`PARAM_PATTERNS` and
`MODEL_LOADERS`). Highlights:

### Text Prompts

| Node Class | Key Fields |
|------------|------------|
| `CLIPTextEncode` | `text` |
| `CLIPTextEncodeSDXL` | `text_g`, `text_l`, `width`, `height` |
| `CLIPTextEncodeFlux` | `clip_l`, `t5xxl`, `guidance` |

To distinguish positive from negative the skill traces `KSampler.negative`
back through Reroute / Primitive nodes to the source CLIPTextEncode. Falls
back to `_meta.title` heuristics ("negative", "neg", "anti").

### Sampling

| Node Class | Key Fields |
|------------|------------|
| `KSampler` | `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise` |
| `KSamplerAdvanced` | `noise_seed`, `steps`, `cfg`, `start_at_step`, `end_at_step` |
| `SamplerCustom` | `noise_seed`, `cfg`, `sampler`, `sigmas` |
| `SamplerCustomAdvanced` | `noise_seed` (via RandomNoise input) |
| `RandomNoise` | `noise_seed` |
| `BasicScheduler` | `steps`, `scheduler`, `denoise` |
| `KSamplerSelect` | `sampler_name` |
| `BasicGuider` / `CFGGuider` | `cfg` |
| `ModelSamplingFlux` | `max_shift`, `base_shift`, `width`, `height` |
| `SDTurboScheduler` | `steps`, `denoise` |

### Latent / Dimensions

| Node Class | Key Fields |
|------------|------------|
| `EmptyLatentImage` | `width`, `height`, `batch_size` |
| `EmptySD3LatentImage` | `width`, `height`, `batch_size` |
| `EmptyHunyuanLatentVideo` | `width`, `height`, `length`, `batch_size` |
| `EmptyMochiLatentVideo` | `width`, `height`, `length`, `batch_size` |
| `EmptyLTXVLatentVideo` | `width`, `height`, `length`, `batch_size` |

### Model Loading

| Node Class | Key Fields | Folder |
|------------|------------|--------|
| `CheckpointLoaderSimple` | `ckpt_name` | `checkpoints` |
| `LoraLoader` | `lora_name`, `strength_model`, `strength_clip` | `loras` |
| `LoraLoaderModelOnly` | `lora_name`, `strength_model` | `loras` |
| `VAELoader` | `vae_name` | `vae` |
| `ControlNetLoader` | `control_net_name` | `controlnet` |
| `CLIPLoader` | `clip_name` | `clip` |
| `DualCLIPLoader` | `clip_name1`, `clip_name2` | `clip` |
| `TripleCLIPLoader` | `clip_name1/2/3` | `clip` |
| `UNETLoader` | `unet_name` | `unet` |
| `DiffusionModelLoader` | `model_name` | `diffusion_models` |
| `UpscaleModelLoader` | `model_name` | `upscale_models` |
| `IPAdapterModelLoader` | `ipadapter_file` | `ipadapter` |
| `ADE_AnimateDiffLoaderWithContext` | `model_name`, `motion_scale` | `animatediff_models` |

### Image Input/Output

| Node Class | Key Fields |
|------------|------------|
| `LoadImage` | `image` (server-side filename, after upload) |
| `LoadImageMask` | `image`, `channel` (`red` / `green` / `blue` / `alpha`) |
| `VAEEncode` / `VAEDecode` | (no controllable fields) |
| `VAEEncodeForInpaint` | `grow_mask_by` |
| `SaveImage` | `filename_prefix` |
| `VHS_VideoCombine` | `frame_rate`, `format`, `filename_prefix`, `loop_count`, `pingpong` |

### ControlNet

| Node Class | Key Fields |
|------------|------------|
| `ControlNetApply` | `strength` |
| `ControlNetApplyAdvanced` | `strength`, `start_percent`, `end_percent` |

### IPAdapter (community pack `comfyui_ipadapter_plus`)

| Node Class | Key Fields |
|------------|------------|
| `IPAdapterAdvanced` | `weight`, `start_at`, `end_at` |
| `IPAdapter` | `weight` |

### Embeddings (referenced inside prompt strings)

ComfyUI scans prompt text for `embedding:NAME` syntax. The skill's
`_common.iter_embedding_refs()` extracts these as model dependencies.

```text
"a beautiful cat, embedding:goodvibes:1.2, embedding:art-style"
```

`extract_schema.py` and `check_deps.py` surface these in
`embedding_dependencies` / `missing_embeddings`.

## Parameter Injection Pattern

```python
import json, copy

with open("workflow_api.json") as f:
    workflow = json.load(f)

wf = copy.deepcopy(workflow)
wf["6"]["inputs"]["text"] = "a beautiful sunset"
wf["7"]["inputs"]["text"] = "ugly, blurry"
wf["3"]["inputs"]["seed"] = 42
wf["3"]["inputs"]["steps"] = 30
wf["5"]["inputs"]["width"] = 1024
wf["5"]["inputs"]["height"] = 1024
```

`scripts/extract_schema.py` automates discovering which node IDs/fields
correspond to which user-facing parameters. It returns a `parameters` dict
that `run_workflow.py` reads to inject values from `--args`.

## Identifying Controllable Parameters (Heuristics)

For unknown workflows:

1. **Prompt text** — any `CLIPTextEncode.text`. Use connection tracing back
   from `KSampler.positive` / `.negative` to disambiguate (don't trust
   meta-title alone).
2. **Seed** — `KSampler.seed` / `KSamplerAdvanced.noise_seed` / `RandomNoise.noise_seed`.
3. **Dimensions** — `Empty*LatentImage.width/height` (must be multiples of 8).
4. **Steps / CFG** — `KSampler.steps`, `KSampler.cfg`. Steps 20–50 typical.
   CFG 5–15 typical (Flux uses guidance, not CFG).
5. **Model / checkpoint** — `CheckpointLoaderSimple.ckpt_name`. Filename must
   match an installed file *exactly*.
6. **LoRA** — `LoraLoader.lora_name`, `.strength_model`.
7. **Images for img2img / inpaint** — `LoadImage.image`. Server-side filename
   after upload.
8. **Denoise** — `KSampler.denoise`. 0.0–1.0; 1.0 = ignore input image,
   0.0 = pass through. Sweet spot for img2img: 0.4–0.7.

## Output Nodes

Output is produced by these node types. The skill's `OUTPUT_NODES` set
extends to common community packs.

| Node | Output Key | Content |
|------|-----------|---------|
| `SaveImage` | `images` | List of `{filename, subfolder, type}` |
| `PreviewImage` | `images` | Temporary preview (not saved) |
| `VHS_VideoCombine` | `gifs` (older) or `videos`/`video` (newer cloud) | Video file refs |
| `SaveAudio` | `audio` | Audio file refs |
| `SaveAnimatedWEBP` / `SaveAnimatedPNG` | `images` | Animated images |
| `Save3D` | `3d` | 3D asset refs |

After execution, fetch outputs from `/history/{prompt_id}` (local) or
`/api/jobs/{prompt_id}` (cloud) → `outputs` → `{node_id}` → `{key}`.

## Wrapper Variants

Some saved JSON files wrap the workflow under a `"prompt"` key (matching
the `/api/prompt` payload shape). The skill's `_common.unwrap_workflow()`
handles this — pass any of:

- raw API format: `{"3": {...}, "4": {...}}`
- wrapped: `{"prompt": {"3": {...}}, "client_id": "..."}`

It rejects editor format with a clear error and a re-export instruction.
