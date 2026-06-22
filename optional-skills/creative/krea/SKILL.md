---
name: krea
description: Use when generating or transforming media through Krea's hosted API — text-to-image, image editing, text/image-to-video, upscaling/enhance, and LoRA styles across 40+ models (Flux, Imagen, Nano Banana, Krea 2, Veo, Kling, Seedance, Topaz). Covers both the Krea MCP tools and the krea CLI.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [krea, image-generation, video-generation, upscaling, mcp, media]
    related_skills: [hermes-media-generation, native-mcp]
---

# Krea

Generate and transform media through Krea's hosted API: 40+ image/video models
behind one credential, plus Topaz upscaling and custom LoRA styles. Krea exposes
two surfaces Hermes can drive — a hosted **MCP server** (first-class tools) and a
**CLI** (`krea`, headless/agent-friendly). Prefer whichever is already set up.

## When to Use

- Text-to-image, image editing (image-to-image), text/image-to-video
- Upscaling / enhancement (Topaz standard, bloom, generative — up to 22K)
- Generating with a specific Krea-hosted model not available via Hermes' own
  `image_generate` / `video_generate` (e.g. Nano Banana Pro, Veo 3.1, Kling 3.0)
- Training or using a custom LoRA style; running a Krea node app

Don't use for: a quick one-off image/video where Hermes' built-in
`image_generate` / `video_generate` already covers it — those need no extra
credential or balance. Reach for Krea when you specifically want its model
lineup, upscaling, or styles.

## Surface Check (do this first)

Use whichever is present. Don't fall back to raw HTTP unless neither exists.

**MCP (preferred when available):** look for `mcp_krea_*` tools in your tool list
(`list_models`, `get_model_schema`, `generate_image`, `generate_video`,
`enhance_image`, `upload_asset`, `get_job`, `cancel_job`, node-app tools). Install
from the catalog if absent:

```bash
hermes mcp install krea     # prompts for the API token, writes the auth header
# start a new session so the mcp_krea_* tools load
```

**CLI:** check it's installed and authenticated:

```bash
command -v krea && krea doctor      # all checks should be ok
```

Install + auth if needed (the token billing draws the workspace API balance):

```bash
npm install -g @krea-ai/cli
krea auth login --api-key "$KREA_API_KEY"   # stored in the OS keyring
```

## Billing — read before generating

- Krea's **API balance is SEPARATE** from any Krea web-app subscription /
  compute units. API calls draw a **prepaid USD balance**; an empty balance
  returns HTTP 402 with a "top up your API balance" message even though auth
  succeeds. Top up at krea.ai/app/api (workspace owners only).
- **Reads are free** (`list_models`, `get_model_schema`, `get_job`). Completed
  generations are billed per model; **failed/cancelled jobs are not charged**.
- There is **no API endpoint to check balance** — monitor it in-app.
- A token can only be created by a workspace **owner/admin** at
  krea.ai/settings/api-tokens.

Confirm with the user before firing paid generations. Rough prices: Flux
$0.04/img, Nano Banana Pro 4K $0.30/img, Veo 3.1 $0.20/sec, Topaz $0.10/img.

## Generate (MCP)

1. `mcp_krea_list_models` → pick a model `id` (matches API paths, e.g.
   `google/nano-banana-pro`, `krea/krea-2/medium`, `google/veo-3.1`).
2. `mcp_krea_get_model_schema` for that id → use ONLY the fields it accepts.
   Param shapes differ per model (aspect_ratio/resolution vs width/height;
   creativity sliders for Krea 2; duration/mode for video).
3. `mcp_krea_generate_image` / `generate_video` / `enhance_image` with the
   model id + input. Submit **async** (default) — the response carries a
   `job_id`. The MCP Apps widget auto-polls; only call `mcp_krea_get_job`
   yourself if there's no widget or the user asks for a text status.
4. Pull the result URL(s) into chat when complete.

For image/video inputs (editing, i2v, enhance): pass an external URL, a base64
data URI, or `mcp_krea_upload_asset` a local file (≤75MB) and use the returned
URL.

## Generate (CLI)

Discover live shapes from the CLI itself — don't trust remembered flags:

```bash
krea models list --json | jq '.[] | select(.category=="image") | .id'
krea generate image --help
krea generate image -p "a cyberpunk cat" -m bfl/flux-1.1-pro --wait -o ./out.png
krea generate video -m google/veo-3.1 -p "..."     # async; then: krea jobs wait <id>
```

`--wait` blocks and prints the URL; `-o <path>` implies `--wait` and downloads.

## Common Pitfalls

1. **Empty API balance looks like an auth failure but isn't.** A 402 / "top up"
   message means auth worked and the balance is $0. Don't re-check the key —
   top up at krea.ai/app/api. Web-app compute units do NOT cover the API.
2. **Stale `KREA_API_KEY` env var shadows a good credential.** The CLI reads env
   before keyring; an exported placeholder/wrong value yields a 401 with
   `source=env`. `unset KREA_API_KEY` and re-check `krea auth status`.
3. **Per-model param schemas differ.** Always `get_model_schema` (MCP) or
   `generate ... --help` (CLI) first. Passing Krea-2 fields to Seedream, or
   aspect_ratio to a width/height model, fails validation.
4. **Submit async, don't block.** Default async + widget auto-poll is cheapest
   and cleanest. Only poll `get_job` manually when there's no widget.
5. **Generation is OAuth-or-token, billed differently.** MCP OAuth bills compute
   units; API-token (the catalog install) bills the prepaid API balance. The
   catalog entry uses the token path.

## Verification Checklist

- [ ] Surface confirmed: `mcp_krea_*` tools present OR `krea doctor` all-ok
- [ ] Model id taken from a live `list_models`, not memory
- [ ] Request fields validated against `get_model_schema` / `--help`
- [ ] User confirmed spend for paid generations; balance is non-zero
- [ ] Job submitted async; result URL retrieved before reporting done
