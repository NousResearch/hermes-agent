---
name: imgflip-meme-generator
description: Create memes with Imgflip's free API.
version: 1.0.0
author: Groot-claw (Groot-claw), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
required_environment_variables:
  - name: IMGFLIP_USERNAME
    prompt: Imgflip username
    help: Use a throwaway Imgflip account. The free creation API uses account credentials, not an API key.
    required_for: meme creation
  - name: IMGFLIP_PASSWORD
    prompt: Imgflip password
    help: Store this in the local Hermes environment. Never paste it into chat.
    required_for: meme creation
metadata:
  hermes:
    tags: [creative, memes, imgflip, image-generation, api]
    related_skills: [meme-generation]
    category: creative
---

# Imgflip Meme Generator Skill

Create hosted meme images through Imgflip's documented free-tier API. This skill is for classic Imgflip template captions, not private/internal memes or local-only rendering.

## When to Use

- The user asks to make a meme on the fly with Imgflip.
- The user wants a classic meme template generated quickly.
- The user asks what the Imgflip free API can do.

## Prerequisites

- `terminal` tool access.
- Network access to `https://api.imgflip.com`.
- For creation, `IMGFLIP_USERNAME` and `IMGFLIP_PASSWORD` must be available in the runtime environment.

Imgflip's free creation endpoint does not use an API key. It uses account username/password in the POST body. Use a throwaway Imgflip account and never ask the user to paste credentials into chat.

Generated Imgflip URLs are publicly accessible to anyone with the exact URL. Do not send private, customer, credential, or internal-sensitive content to Imgflip.

## How to Run

Use the helper script from this skill directory through the `terminal` tool:

```bash
python scripts/imgflip_free_tier.py list --limit 10
```

For creation, run with credentials already present in the environment:

```bash
python scripts/imgflip_free_tier.py make --template-id 181913649 --text0 "WRITING LONG PROMPTS" --text1 "MAKING ONE MEME COMMAND" --output meme.jpg
```

## Quick Reference

Free-tier endpoints verified from `https://imgflip.com/api`:

- `GET https://api.imgflip.com/get_memes`
  - Free.
  - No authentication required.
  - Optional query parameter: `type=image`, `type=gif`, or `type=gif,image`; default is `image`.
  - Returns popular captionable templates with fields such as `id`, `name`, `url`, `width`, `height`, and `box_count`.

- `POST https://api.imgflip.com/caption_image`
  - Free for creating watermarked meme images.
  - Requires `template_id`, `username`, and `password` in the POST body.
  - Simple mode: `text0` and `text1`.
  - Multi-box mode: `boxes[0][text]`, `boxes[1][text]`, etc., up to 20 boxes.
  - Optional parameters include `font` and `max_font_size`.
  - `no_watermark` is Premium-only.

Premium-only endpoints:

- `POST /caption_gif`
- `POST /search_memes`
- `POST /get_meme`
- `POST /automeme`
- `POST /ai_meme`
- `no_watermark` on creation

## Procedure

1. Determine the meme intent and joke structure.
2. Fetch current popular templates if a template ID is needed:
   ```bash
   python scripts/imgflip_free_tier.py list --limit 20
   ```
3. Pick a known template ID or top-template name match.
4. Keep captions short. Classic meme captions should usually be uppercase.
5. Generate the image:
   ```bash
   python scripts/imgflip_free_tier.py make --template-name "Drake" --text0 "MANUAL IMAGE EDITING" --text1 "API MEMES ON DEMAND" --output meme.jpg
   ```
6. For templates with more than two text areas, use custom boxes:
   ```bash
   python scripts/imgflip_free_tier.py make --template-id 87743020 --boxes '[{"text":"OPTION A"},{"text":"OPTION B"},{"text":"ME"}]' --output meme.jpg
   ```
7. Return the generated image URL, or save the file to a platform-appropriate local path and return it as a `MEDIA:` attachment when needed.

## Pitfalls

- There is no official free-tier API key flow in the Imgflip docs; the creation endpoint uses account credentials.
- The full template search endpoint is Premium-only. For free tier, use top templates or known template IDs.
- Generated image URLs are not private.
- When `boxes` is used, `text0` and `text1` are ignored and text is not automatically uppercased.
- Long captions often make bad memes. Shorten before generating.

## Verification

Run the helper list command and verify it returns JSON with `success: true`:

```bash
python scripts/imgflip_free_tier.py list --limit 3
```

For creation, verify `POST /caption_image` returns `success: true`, the returned image URL downloads with an `image/*` content type, and the final response includes either the generated URL or a `MEDIA:` path.
