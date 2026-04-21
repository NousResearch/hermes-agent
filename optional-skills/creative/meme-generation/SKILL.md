---
name: meme-generation
description: Generate meme images by choosing a template, downloading a blank source from Imgflip when needed, and overlaying short captions with Pillow.
version: 2.1.0
author: adanaleycio
license: MIT
metadata:
  hermes:
    tags: [creative, memes, humor, images]
    related_skills: [ascii-art, generative-widgets]
    category: creative
---

# Meme Generation

Generate real meme images from a topic by selecting an appropriate template, writing short captions, and rendering a .png file.

## When to use

Use this skill when the user wants to:
- make a meme
- caption a reaction image or blank meme template
- search Imgflip for a template/source image
- turn a topic into a meme with text overlay

Do not use this skill for generic photo editing, video memes, or free-form image generation.

## Procedure

1. Identify the joke structure first: chaos, dilemma, comparison, escalation, hot take, irony, and similar patterns.
2. Prefer a curated template when one clearly matches the structure.
3. If the task needs a blank source, use `scripts/imgflip_download.py` to search Imgflip and save the template locally.
4. If the task already has a local image, use `scripts/meme_caption.py` to add captions to that image.
5. Use `scripts/generate_meme.py` when a single entry point is enough or when compatibility with the combined flow is desired.
6. Keep captions short. Shorter text is usually better for readability and timing.
7. Return the resulting file with a `MEDIA:` path.

## Scripts

- `scripts/imgflip_download.py`: search Imgflip and download a blank template to a local file.
- `scripts/meme_caption.py`: add meme captions to an existing local image, with optional padding trim and bars mode.
- `scripts/generate_meme.py`: main renderer and compatibility wrapper for template-based or local-image meme generation.

## References

- Curated template list: `references/curated-templates.md`
- Template selection heuristics and Imgflip source notes: `references/meme-generation.md`
- Usage examples: `references/examples.md`
- Script details and preferred entry points: `references/scripts.md`

## Pitfalls

- Do not force a template just because it matches the topic words; match the joke structure instead.
- Match the number of text fields to the template.
- Keep the final text readable and concise.
- Do not generate hateful, abusive, or personally targeted content.

## Verification

The output is correct if:
- a .png file was created at the output path
- the text is legible on the template
- the chosen template fits the joke
- the file can be delivered via `MEDIA:`
