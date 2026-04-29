# ComfyUI Workflow JSON Format

## Two Formats

ComfyUI uses two workflow formats. **Only API format works for programmatic execution.**

### API Format (what we use)

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
    "inputs": {
      "ckpt_name": "v1-5-pruned-emaonly.safetensors"
    }
  },
  "5": {
    "class_type": "EmptyLatentImage",
    "inputs": {"width": 512, "height": 512, "batch_size": 1}
  },
  "6": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "a beautiful cat",
      "clip": ["4", 1]
    }
  },
  "7": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "bad quality, ugly",
      "clip": ["4", 1]
    }
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": ["8", 0]
    }
  }
}
```

**How to detect:** Top-level keys are numeric strings, each value has `class_type`.

### Editor Format (not directly executable)

Has `nodes[]` and `links[]` arrays — the visual graph data from the ComfyUI web editor.
This is what "Save" produces. For API use, export with "Save (API Format)" instead.

**How to detect:** Top-level has `"nodes"` and `"links"` keys.

---

## Input Connections

Inputs can be:
- **Literal values**: `"text": "a cat"`, `"seed": 42`, `"width": 512`
- **Links to other nodes**: `["node_id", output_index]` — e.g., `["4", 0]` means
  output slot 0 of node "4"

Only literal values can be modified by parameter injection. Linked inputs are wiring.

---

## Common Node Types and Their Controllable Parameters

### Text Prompts

| Node Class | Key Fields |
|------------|-----------|
| `CLIPTextEncode` | `text` (the prompt string) |
| `CLIPTextEncodeSDXL` | `text_g`, `text_l`, `width`, `height` |

Usually: positive prompt → one CLIPTextEncode, negative prompt → another.
Distinguish by checking the `_meta.title` field or by tracing which feeds into
positive vs negative inputs of the sampler.

### Sampling

| Node Class | Key Fields |
|------------|-----------|
| `KSampler` | `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise` |
| `KSamplerAdvanced` | `noise_seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `start_at_step`, `end_at_step` |
| `SamplerCustom` | `cfg`, `sampler`, `sigmas` |

### Image Dimensions

| Node Class | Key Fields |
|------------|-----------|
| `EmptyLatentImage` | `width`, `height`, `batch_size` |
| `LatentUpscale` | `width`, `height`, `upscale_method` |

### Model Loading

| Node Class | Key Fields | Model Folder |
|------------|-----------|-------------|
| `CheckpointLoaderSimple` | `ckpt_name` | `checkpoints` |
| `LoraLoader` | `lora_name`, `strength_model`, `strength_clip` | `loras` |
| `VAELoader` | `vae_name` | `vae` |
| `ControlNetLoader` | `control_net_name` | `controlnet` |
| `CLIPLoader` | `clip_name` | `clip` |
| `UNETLoader` | `unet_name` | `unet` |
| `DiffusionModelLoader` | `model_name` | `diffusion_models` |
| `UpscaleModelLoader` | `model_name` | `upscale_models` |

### Image Input/Output

| Node Class | Key Fields |
|------------|-----------|
| `LoadImage` | `image` (filename on server, after upload) |
| `LoadImageMask` | `image`, `channel` |
| `SaveImage` | `filename_prefix` |
| `PreviewImage` | (no controllable fields, just previews) |

### ControlNet

| Node Class | Key Fields |
|------------|-----------|
| `ControlNetApply` | `strength` |
| `ControlNetApplyAdvanced` | `strength`, `start_percent`, `end_percent` |

### Video (AnimateDiff)

| Node Class | Key Fields |
|------------|-----------|
| `ADE_AnimateDiffLoaderWithContext` | `model_name`, `motion_scale` |
| `VHS_VideoCombine` | `frame_rate`, `format`, `filename_prefix` |

---

## Parameter Injection Pattern

To modify a workflow programmatically:

```python
import json, copy

with open("workflow_api.json") as f:
    workflow = json.load(f)

# Deep copy to avoid mutating original
wf = copy.deepcopy(workflow)

# Inject parameters by node ID + field name
wf["6"]["inputs"]["text"] = "a beautiful sunset"     # positive prompt
wf["7"]["inputs"]["text"] = "ugly, blurry"           # negative prompt
wf["3"]["inputs"]["seed"] = 42                       # seed
wf["3"]["inputs"]["steps"] = 30                      # steps
wf["5"]["inputs"]["width"] = 1024                    # width
wf["5"]["inputs"]["height"] = 1024                   # height
```

The `scripts/extract_schema.py` in this skill automates discovering which
node IDs and fields correspond to which user-facing parameters.

---

## Identifying Controllable Parameters (Heuristics)

When analyzing an unknown workflow, these patterns identify user-facing params:

1. **Prompt text**: Any `CLIPTextEncode` → `text` field. Title/meta usually
   indicates positive vs negative.

2. **Seed**: Any `KSampler` / `KSamplerAdvanced` → `seed` / `noise_seed`.
   Randomizable — set to different values for variations.

3. **Dimensions**: `EmptyLatentImage` → `width`, `height`. Common: 512, 768,
   1024 (must be multiples of 8).

4. **Steps**: `KSampler` → `steps`. More = higher quality + slower. 20-50 typical.

5. **CFG scale**: `KSampler` → `cfg`. How closely to follow prompt. 5-15 typical.

6. **Model/checkpoint**: `CheckpointLoaderSimple` → `ckpt_name`. Must match an
   installed model filename exactly.

7. **LoRA**: `LoraLoader` → `lora_name`, `strength_model`. Adapter name + weight.

8. **Images for img2img**: `LoadImage` → `image`. Filename on server after upload.

9. **Denoise strength**: `KSampler` → `denoise`. 0.0-1.0. Lower = closer to input
   image. Only relevant for img2img.

---

## Output Nodes

Output is produced by these node types:

| Node | Output Key | Content |
|------|-----------|---------|
| `SaveImage` | `images` | List of `{filename, subfolder, type}` |
| `VHS_VideoCombine` | `gifs` or `videos` | Video file references |
| `SaveAudio` | `audio` | Audio file references |
| `PreviewImage` | `images` | Temporary preview (not saved) |

After execution, fetch outputs from `/history/{prompt_id}` → `outputs` → `{node_id}`.
