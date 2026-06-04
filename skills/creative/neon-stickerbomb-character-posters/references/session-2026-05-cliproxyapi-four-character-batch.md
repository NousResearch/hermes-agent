# Session note: four-character neon batch via CLIProxyAPI direct path (2026-05-05)

## Trigger

Nick asked to generate 4 female anime characters after Hermes `image_generate` had recently failed with mixed `openai-codex` 401s and `cliproxyapi-gptimage2` 404s.

## What worked

Use `codex-ops` direct CLIProxyAPI Responses API workaround rather than the failing `image_generate` tool path:

- Endpoint: `POST http://127.0.0.1:8317/v1/responses`
- Auth header: local CLIProxyAPI bearer key (do not print it)
- Model: `gpt-5.5`
- Required tool: `image_generation`
- Image model: `gpt-image-2`
- Size: `1024x1536`
- Quality: `medium`
- Output: PNG, opaque background
- One request per image

A four-image sequential batch succeeded with these subjects and composition archetypes:

1. Yoruichi / 夜一 — low-angle lightning dash + cat-shadow diagonal
2. Erza / 艾露莎 — foreground chrome sword + armor split layout
3. Homura / 焰 — foreground time-shield lens + broken clock circular ring
4. Android 18 / 人造人18号 — dutch-angle cyborg kick + capsule gadget foreground

Saved paths:

```text
/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260505_145826_yoruichi_lightning_cat_dash.png
/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260505_150127_erza_armored_sword_split.png
/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260505_150300_homura_time_glass_circle.png
/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260505_150612_android18_capsule_cyber_kick.png
```

## Prompt pattern used

For this class of batch, each prompt should still follow the skill contract:

- vertical 3:4 glossy neon cyber-pop sticker-bomb character poster
- explicit character identity cues
- adult/covered/non-explicit or age-appropriate/non-sexualized boundary
- one distinct composition archetype per image
- in-scene `NickZag` as a physical object label/tape/sticker/patch, not watermark
- style mechanics block: thick black manga outlines, sharp cel shading, deep black/violet/crimson shadows, wet glossy highlights, cyan/magenta rim light, acid accents, halftone, spray paint grain, offset print, torn vinyl decals
- negative block against clean official key art, centered pose, watermark, sexuality, identity loss, hand/prop deformation

## Operational lesson

If direct CLIProxyAPI generation is needed for neon batches, put the script in a temp file and run it as a background process. Use `print(..., flush=True)` or `python3 -u`; otherwise no progress may appear while requests are in flight. Wait in chunks because medium portrait generations can take 80-190 seconds each.
