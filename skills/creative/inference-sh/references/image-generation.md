# Image Generation Reference

## Model Details

### FLUX (Black Forest Labs)
- **flux/dev-lora** — Quality generation with LoRA style support
- **flux/2-klein-lora** — Smaller, faster variant with LoRA
- **flux/klein-4b** — Ultra-cheap at $0.0001/image

All FLUX models support: prompt, width, height, num_inference_steps, guidance_scale, lora_url.

### Seedream (ByteDance)
- **seedream/4.5** — Latest, 2K-4K cinematic quality, accurate text rendering in images
- **seedream/4.0** — Balanced quality/speed
- **seedream/3.0** — Fastest

Supports: prompt, negative_prompt, width, height, aspect_ratio.

### GPT-Image-2 (OpenAI)
- **gpt-image-2** — Text-to-image, editing, inpainting, multi-image reference, batch generation

Unique features:
- `image` field for edit/inpaint workflows
- `mask` field for inpainting regions
- `reference_images` array for style/subject reference
- `n` field for batch generation (up to 4)

### Reve (Falai)
- **reve/image** — Natural language image editing

Unique features:
- `edit_prompt` for describing changes ("remove the car", "change sky to sunset")
- `source_image` for the base image
- Built-in text rendering and background replacement

### Additional Models
- **gemini/image-3-pro** — Google's highest quality image model
- **grok-imagine** — xAI, fast creative generation, multiple aspect ratios
- **imagineart/1.5-pro** — Ultra-high-fidelity 4K output
- **p-image** — Fastest and cheapest, LoRA support, multiple aspect ratios

## Enhancement Tools

- **topaz/image-upscaler** — Professional upscaling (2x, 4x)
- **real-esrgan** — Open-source upscaling
- **birefnet/bg-remove** — Background removal to transparent PNG

## Common Input Fields

Most image models accept:
```json
{
  "prompt": "description of the image",
  "negative_prompt": "what to avoid (optional)",
  "width": 1024,
  "height": 1024,
  "num_images": 1,
  "seed": -1
}
```

## Tips

- For text in images, use Seedream 4.5 or GPT-Image-2 — they handle typography best.
- For product shots, start with Seedream 4.5 then upscale with Topaz.
- For quick iteration, use p-image or FLUX Klein (cheapest) then switch to higher quality once the prompt is dialed in.
- For editing existing photos, use GPT-Image-2 (inpaint/edit) or Reve (natural language edits).
