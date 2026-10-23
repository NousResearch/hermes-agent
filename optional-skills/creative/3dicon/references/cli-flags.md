# iconloop CLI flags (1:1 with `scripts/iconloop/cli.py` at upstream 9539d3a)

Invocation: `cd "${HERMES_SKILL_DIR}/scripts" && .venv/bin/python -m iconloop [--out DIR] <stage> [flags]`.
`--out` (default `out`, relative to the cwd — pass an absolute path) is a
global flag and goes **before** the stage name. Every stage first runs
`config.load_dotenv()` (nearest `.env` / `.env.local` walking up from cwd;
never overrides real env vars) and `config.require_tool("ffmpeg", ...)`.

## `still` and `run`

| flag | default | meaning |
|---|---|---|
| `--prompt` | required | what the object is |
| `--backend` | `$ICONLOOP_IMAGE_BACKEND` or `openai` | `openai` \| `gemini` \| `openrouter` |
| `--variants N` | `1` | generate N to choose from (`still_1.png` …); keep at 1 unless asked |
| `--raw` | off | send the prompt verbatim (skip the baked-in style rules) |

Unless `--raw`, the prompt is appended with the STILL_RULES sentence: stylised
3D icon (not a photograph), simplified friendly forms, soft matte surfaces,
rounded edges, clean flat colour with gentle shading, centred on a fully
transparent background, soft even upper-left lighting, no text/watermark/
border/ground plane/cast shadow. Output: `<out>/still.png`.
`run` = `still` + a printed reminder to look before spending on motion.

## `animate`

| flag | default | meaning |
|---|---|---|
| `--motion "…"` | none | extra motion detail, in physical language |
| `--strategy` | none | `event` \| `native` \| `part` \| `surface` |
| `--energy` | follows strategy | `still` \| `calm` \| `lively` \| `playful` |
| `--emit` | off | let the object briefly produce sparks, fragments, droplets |
| `--preset` | none | archetype shortcut: `battery` `bell` `camera` `clock` `cloud` `droplet` `flame` `gear` `heart` `leaf` `star` |
| `--feel` | none | `organic` \| `settle` \| `smooth` \| `snappy` |
| `--still PATH` | `<out>/still.png` | override the still to animate (transparent PNG) |
| `--duration N` | `5` | clip seconds |
| `--via` | `$ICONLOOP_VIDEO_BACKEND` or `openrouter` | `openrouter` \| `replicate` (fallback, stops at Kling 2.5) |
| `--model ID` | `$ICONLOOP_VIDEO_MODEL` / `bytedance/seedance-2.0` (openrouter); `$ICONLOOP_KLING_MODEL` / `kwaivgi/kling-v2.5-turbo-pro` (replicate) | video model id for the chosen backend |
| `--dry-run` | off | print the composed motion prompt and stop, spending nothing |

Known upstream bug: `--preset camera` raises `NameError: name 'ACCENT' is not
defined` (motion.py references an undefined constant); every other preset works.

`--feel` meanings (motion.QUALITY): `settle` — movements overshoot slightly and
settle rather than stopping dead; `snappy` — each movement is fast and
decisive, with a held beat between; `smooth` — even and unhurried throughout;
`organic` — no two cycles identical, rhythm varies slightly.

Writes `<out>/sent_to_kling.png` (still flattened onto backing RGB 158,158,158,
padded to 1024 px) and `<out>/render.mp4`. The same image is sent as first and
last frame on both backends.

## `matte`

| flag | default | meaning |
|---|---|---|
| `--video PATH` | `<out>/render.mp4` | override the clip to matte |
| `--model` | `isnet-general-use` | rembg model |
| `--extract N` | `640` | matting resolution |
| `--master N` | `512` | master frame size |

Writes `<out>/src/*.png` (extracted frames) and `<out>/master/*.png` (RGBA).

## `encode`

| flag | default | meaning |
|---|---|---|
| `--source-fps` | `24.0` | frame rate of the render |
| `--fps` | source rate | output rate; lower drops frames and causes judder |
| `--size N` | `384` | output pixel size (never above the 512 master) |
| `--quality N` | `60` | WebP quality; barely visible, ~1/255 colour for +50 % bytes |
| `--name` | `icon` | output basename |
| `--sweep` | off | print size vs fps table and stop |
| `--webm` | off | also write VP9 + alpha |
| `--mp4` | off | also write flattened H.264 |
| `--mp4-bg` | `0xFFFFFF` | flatten colour for `--mp4` |

## `verify`

No flags. Reports motion inside the object and loop-seam quality from
`<out>/master/`, writes `<out>/contact_sheet.png`, then reports every
`<out>/*.webp`. Exits non-zero if any check fails — believe it.

## Environment variables (`scripts/env.example`)

| variable | default | used by |
|---|---|---|
| `ICONLOOP_IMAGE_BACKEND` | `openai` | still backend |
| `OPENAI_API_KEY` | — | `--backend openai` (newest `gpt-image-*` resolved at runtime) |
| `GOOGLE_API_KEY` | — | `--backend gemini` (Gemini image models) |
| `OPENROUTER_API_KEY` | — | `--backend openrouter` and `--via openrouter` (default video path) |
| `ICONLOOP_IMAGE_MODEL` | auto-resolve newest | pin an image model |
| `ICONLOOP_VIDEO_BACKEND` | `openrouter` | `--via` default |
| `ICONLOOP_VIDEO_MODEL` | `bytedance/seedance-2.0` | OpenRouter video model; alternatives `bytedance/seedance-2.5`, `google/veo-3.1`, `minimax/hailuo-3`, `alibaba/wan-2.7` |
| `REPLICATE_API_TOKEN` | — | `--via replicate` only |
| `ICONLOOP_KLING_MODEL` | `kwaivgi/kling-v2.5-turbo-pro` | Replicate model |
