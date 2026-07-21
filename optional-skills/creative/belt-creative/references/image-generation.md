# Image Generation Reference

## Models

### FLUX (Black Forest Labs)
- **flux/dev-lora** — Quality generation with LoRA style support
- **flux/2-klein-lora** — Smaller, faster variant with LoRA
- **flux/klein-4b** — Ultra-cheap at $0.0001/image

Supports: prompt, width, height, num_inference_steps, guidance_scale, lora_url.

### Seedream (ByteDance)
- **seedream/4.5** — Latest, 2K-4K cinematic quality, accurate text rendering
- **seedream/4.0** — Balanced quality/speed
- **seedream/3.0** — Fastest

Supports: prompt, negative_prompt, width, height, aspect_ratio.

### GPT-Image-2 (OpenAI)
- **gpt-image-2** — Text-to-image, editing, inpainting, multi-image reference, batch

Unique fields: `image` (edit/inpaint), `mask` (inpainting region), `reference_images` (style/subject), `n` (batch up to 4).

### Reve (Falai)
- **reve/image** — Natural language image editing

Unique fields: `edit_prompt`, `source_image`. Built-in text rendering and background replacement.

### Other Models
- **gemini/image-3-pro** — Google, highest quality
- **grok-imagine** — xAI, fast creative, multiple aspect ratios
- **imagineart/1.5-pro** — Ultra-high-fidelity 4K
- **p-image** — Fastest and cheapest, LoRA support

## Enhancement Tools

- **topaz/image-upscaler** — Professional upscaling (2x, 4x)
- **real-esrgan** — Open-source upscaling
- **birefnet/bg-remove** — Background removal to transparent PNG

## Tips

- For text in images, use Seedream 4.5 or GPT-Image-2 — they handle typography best.
- For quick iteration, use P-Image or FLUX Klein (cheapest) then switch to higher quality.
- For editing existing photos, use GPT-Image-2 (inpaint/edit) or Reve (natural language edits).
- For product shots, generate with Seedream 4.5 then upscale with Topaz.
