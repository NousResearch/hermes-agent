---
name: wavespeed
description: "Generate and edit images, video, audio and 3D via WaveSpeed."
version: 1.0.0
author: Zeyi Cheng (chengzeyi), WaveSpeedAI
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [image-generation, video-generation, text-to-image, image-to-video, text-to-video, text-to-speech, upscale, 3d-generation, wavespeed, seedream, seedance, minimax, wan, veo, creative]
    category: creative
    homepage: https://github.com/WaveSpeedAI/agent-skills
    related_skills: [comfyui]
prerequisites:
  commands: [wavespeed]
required_environment_variables:
  - name: WAVESPEED_API_KEY
    prompt: WaveSpeed API key (starts with wsk_)
    help: "Get one at https://wavespeed.ai/accesskey. Not needed if the user has already run `wavespeed login`, which stores the key locally."
    required_for: "running models when no stored login exists"
    optional: true
---

# WaveSpeed Skill

Run hosted generative-media models on the WaveSpeed platform through the open-source `wavespeed` CLI: text-to-image, image editing, text-to-video, image-to-video, video editing, upscaling, face swap, text-to-speech, and 3D. Every model is one `wavespeed run <model-id>` call with explicit inputs; this skill covers finding the right model, reading its schema, quoting the price, and running it. It does not run models locally and does not cover ComfyUI workflows (see `comfyui`).

## When to Use

- The user wants an image, video, audio clip, or 3D asset created or edited: hero images, product photo edits, short clips from a still, voiceovers, upscales, watermark cleanup on their own media.
- The user names a hosted model (Seedream, Seedance, MiniMax H3, Wan, Veo, Kling, Nano Banana, MiniMax Speech) or asks which model to use for a media task.
- The user asks what a generation would cost before spending credits.

Do not use it for local diffusion pipelines or for media the user has no right to edit (see Pitfalls).

## Prerequisites

- Node.js 18+ and the CLI: `npm install -g @wavespeed/cli`. Run `scripts/check_setup.py` via `terminal` to confirm the binary is on PATH and the account is signed in.
- Auth, one of: `wavespeed login` (opens https://wavespeed.ai/accesskey and stores the key), or `WAVESPEED_API_KEY` in the environment. Never ask the user to paste a key into the chat; if `wavespeed status` reports signed out, ask them to run `wavespeed login`.
- Generations spend account credits; `wavespeed balance` shows the remaining amount.

## How to Run

All commands go through `terminal`. Always pass `--json` so the result is machine-readable.

```bash
# 1. Find a model on the live catalog
wavespeed models "seedream"
wavespeed models --type image-to-video --popular

# 2. Read its input schema (fields, enums, defaults)
wavespeed run bytedance/seedream-v5.0-pro -h

# 3. Quote, then run
wavespeed price bytedance/seedream-v5.0-pro -p "a cyberpunk skyline at golden hour" -i resolution=2k
wavespeed run bytedance/seedream-v5.0-pro \
  -p "a cyberpunk skyline at golden hour" \
  -i aspect_ratio="16:9" -i resolution="2k" --json
```

`run --json` prints `{ id, model, prompt, outputs: [url, ...], saved: [path, ...], elapsed_ms, raw }`. Report `outputs[0]` to the user; add `--download "./out/{index}.{ext}"` when they want the file on disk.

Local files: prefix the path with `@` and the CLI uploads it and substitutes the hosted URL.

```bash
wavespeed run bytedance/seedream-v5.0-pro/edit \
  -p "replace the background with a sunlit kitchen" \
  -i images='["@./input.jpg"]' --json

wavespeed run wavespeed-ai/minimax-h3/image-to-video \
  -p "subtle parallax, gentle wind" -i image=@./hero.jpg -i duration=5 --json
```

## Quick Reference

| Task | Start with | Notes |
|---|---|---|
| Text to image | `bytedance/seedream-v5.0-pro` | `aspect_ratio`, `resolution` 1k/2k/4k |
| Image edit | `bytedance/seedream-v5.0-pro/edit` | `images` is a JSON array of URLs or `@paths` |
| Text/image to video | `wavespeed-ai/minimax-h3/text-to-video`, `.../image-to-video` | open-weights, cheap, native audio; `duration` 3-15 |
| Highest-quality video | `bytedance/seedance-2.5/*` | same field names, up to 4k |
| Video edit / extend | `wavespeed-ai/minimax-h3/video-edit`, `.../video-extend` | `video` is required |
| Text to speech | `minimax/speech-2.6-turbo` | `text`, `voice_id` |
| Upscale | `wavespeed-ai/image-upscaler`, `wavespeed-ai/ultimate-video-upscaler` | target resolution field |

Other useful commands: `wavespeed show <id>` (recover a run by id), `wavespeed upload <file>` (hosted URL only), `wavespeed usage` and `wavespeed billings` (spend), `wavespeed models --type text-to-3d`.

Per-model detail (all parameters, pricing tiers, prompt tips) for Seedream, Seedance, Veo 3.1, Wan, Nano Banana, MiniMax Speech and the upscalers lives in `references/models.md`.

## Procedure

1. Confirm setup with `scripts/check_setup.py`; stop and ask for `wavespeed login` if it reports signed out.
2. Pick the model. Search the catalog with `wavespeed models`; never invent a model id. Use the Quick Reference defaults unless the user names a model.
3. Read the schema with `wavespeed run <model> -h` and map the user's request onto the documented fields only.
4. Quote with `wavespeed price` when the run is a video, a 4k image, or the user asked about cost; state the number before running.
5. Run with `--json`, keep the returned `id`, and give the user `outputs[0]` (or the `saved` path when `--download` was used).
6. If the run fails, read the `raw` error in the JSON; schema mistakes and unsupported values are the usual causes. Fix the inputs and retry once.

## Pitfalls

- Bare local paths are passed through untouched and rejected by the model. Only `@`-prefixed values upload.
- Field names are per model; `-h` is the source of truth. A field that works on one model may not exist on another.
- Video runs take 30 seconds to several minutes; the CLI waits, so do not spawn a second run for the same request.
- Face swap and watermark removal act on other people's likeness or marks. Only proceed when the user owns the media or has consent and rights, never for impersonation or stripping third-party attribution.
- The CLI never rewrites prompts. If the user wants prompt help, write the prompt yourself before running.

## Verification

- `wavespeed status` shows the signed-in account and `wavespeed balance` a non-zero credit.
- A successful run returns JSON with a non-empty `outputs` array and an `id` that `wavespeed show <id>` can reload.
- Open the output URL (or the downloaded file) to confirm the asset matches the request before reporting done.
