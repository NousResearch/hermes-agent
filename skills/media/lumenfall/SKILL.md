---
name: lumenfall
description: Generate images, videos, and edit images via Lumenfall's unified API (200+ models including FLUX, Imagen, Sora, Kling).
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [image-generation, video-generation, image-editing, AI-art, multimodal, lumenfall]
    requires_tools: [lumenfall_list_models]
required_environment_variables:
  - name: LUMENFALL_API_KEY
    prompt: Lumenfall API key
    help: Get one at https://lumenfall.ai
    required_for: full functionality
---

# Lumenfall Image & Video Generation

Generate images, videos, and edit images using Lumenfall's unified API. Lumenfall routes to 200+ models from providers like Black Forest Labs (FLUX), Google (Imagen), OpenAI (GPT Image, Sora), Kuaishou (Kling), and more.

## Available Tools

Call these tools directly — do NOT use terminal or execute_code for Lumenfall operations.

| Tool | Purpose |
|------|---------|
| `lumenfall_list_models` | Discover available models and their capabilities |
| `lumenfall_image_generate` | Generate images from text prompts |
| `lumenfall_video_generate` | Generate videos from text or image prompts |
| `lumenfall_image_edit` | Edit images (inpainting, upscaling, background removal) |

## Workflow

1. **Discover models first**: Always call `lumenfall_list_models` before recommending a specific model. The catalog changes frequently.
2. **Filter by capability**: Use the `capability` parameter to narrow results: `text-to-image`, `text-to-video`, `image-to-video`, `image-edit`.
3. **Generate**: Call the appropriate tool with a model ID from the catalog.

## Model Quick Reference

| Use Case | Recommended Models |
|----------|-------------------|
| Photorealism | `flux.2-max`, `imagen-4.0-ultra-generate-001`, `gpt-image-1.5` |
| Speed/Drafting | `flux.1-schnell`, `imagen-4.0-fast-generate-001` |
| SVG/Design | `recraft-v4-pro-svg`, `recraft-v4-pro` |
| Video | `kling-v3`, `wan-2.6`, `sora-2-pro` |
| Editing | `qwen-image-edit`, `p-image-edit` |

## Generation Timing

Image generation takes 5-30 seconds, video generation 30 seconds to 5 minutes depending on the model. Let the user know before calling so they're not left waiting without context.

## Troubleshooting

- **401 Unauthorized**: Check `LUMENFALL_API_KEY` in your Hermes `.env` file (run `hermes config` to locate it).
- **402 Payment Required**: Account balance is empty. Top up at lumenfall.ai.
- **500 Provider Error**: Try a different model family (e.g., switch from FLUX to Imagen).
- **Video timeout**: Video generation can take 2-5 minutes. The tool polls automatically for up to 10 minutes.
