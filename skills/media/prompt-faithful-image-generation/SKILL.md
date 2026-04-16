---
name: prompt-faithful-image-generation
description: Generate one image from the user's exact prompt and return it natively to the current chat. Use when the user explicitly asks to 生图、画图、生成图片、create an image, or draw something. Key capabilities: exact prompt passthrough, native MEDIA delivery, minimal output.
---

# Prompt-Faithful Image Generation

Use this skill when the user explicitly wants you to generate an image.

## Rules

1. Prefer the `prompt_faithful_image_generate` tool for normal chat-driven image generation.
2. Pass the user's intended image prompt through literally.
3. Do not rewrite, expand, sanitize for style, or embellish the prompt unless the user explicitly asks you to optimize the prompt first.
4. Do not add extra commentary around the generated image.
5. On success, return only the tool-produced `MEDIA:/absolute/path` line, or the exact media tag embedded in the tool result.
6. On failure, reply with one short sentence describing the real failure reason.

## Notes

- The current chat context already decides where the image goes. In a group, the image is posted back to the group. In a DM, it goes back to the DM.
- If the user asks for prompt engineering rather than direct generation, help with the prompt first and only generate after the user confirms.
- If the user asks for multiple variants, ask for that explicitly or make clear you are generating one image unless the tool gains batch support.

## Supporting Script

- `scripts/prompt_faithful_image_generate.py`

Manual usage:

```bash
python scripts/prompt_faithful_image_generate.py "夜晚下雨的重庆街头，35mm 纪实摄影"
```
