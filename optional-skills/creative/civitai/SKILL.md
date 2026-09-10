---
name: civitai
description: Search Civitai for AI models (LoRA, Checkpoint, Flux, SDXL, Pony,
  Illustrious, Wan Video, Hunyuan Video, Mochi, LTXV, CogVideoX), browse top
  example media (images and videos) with generation prompts, identify
  .safetensors files by hash, and produce ready-to-run download commands for
  ComfyUI.
version: 1.1.0
author: Lulu Xiao (inspired by timoncool/civitai-mcp-ultimate, MIT)
license: MIT
metadata:
  hermes:
    tags:
      - aigc
      - image-generation
      - stable-diffusion
      - lora
      - safetensor
      - flux
      - sd3
      - wan-video
      - hunyuan-video
      - creative
      - generative-ai
      - civitai
    config:
      - key: civitai.comfyui_path
        description: Default ComfyUI models directory for download commands
        default: ""
        prompt: "ComfyUI models path"
---

# Civitai

Search & inspect Civitai AI image/video models — listings, metadata,
example media with prompts, hash lookups, ComfyUI download commands.

> **Windows:** use `python` instead of `python3` in every command below.

> **"images" commands return videos too.** The `images` / `top-images` /
> `prompts` subcommands surface whichever media type the model produced.
> For video models (Wan Video, Hunyuan Video, Mochi, LTXV, LTXV2,
> CogVideoX), video URLs are the **correct, expected** result — don't
> refuse with "no images found". Use `--content-type {image,video}` to
> filter.

## When to Use

Triggers: **Civitai**, LoRA, Checkpoint, Flux, SDXL, Pony, Illustrious,
Wan Video, Hunyuan Video, Mochi, LTXV, CogVideoX, Stable Diffusion model,
AI video model, ComfyUI download, generation prompts, `.safetensors`, or
asks to identify a model file, mine prompts, or compare AI-art / AI-video models.

## API key (optional)

`CIVITAI_API_KEY` is **not required**. SFW search, model details, hash
lookups, prompt mining, and download-command generation work without it.
Set it to lift rate limits, access NSFW, or generate runnable download
commands.

Setup: <https://civitai.com> → **Account settings** → **API Keys** →
**Add API key** (token shown once). Add `CIVITAI_API_KEY=<token>` to the
Hermes env file (`hermes config env-path`, usually `~/.hermes/.env`,
`chmod 600`). Verify:
`python3 ${HERMES_SKILL_DIR}/scripts/search.py top-models --type LORA --limit 1`
— no `auth failed` = live.

**Never echo the token, pass it as a CLI argument, or commit `.env`.**
Scripts read from `os.environ` and never print the value.
`error: auth failed — check CIVITAI_API_KEY` = key invalid.

## NSFW filtering

**`--nsfw` works on all four search subcommands** (`models`, `top-models`,
`images`, `top-images`):

```bash
python3 ${HERMES_SKILL_DIR}/scripts/search.py top-models --nsfw --type LORA --base-model "Pony"
python3 ${HERMES_SKILL_DIR}/scripts/search.py images    --nsfw --model-id <id>
```

For granular per-level control on **image commands only**, use
`--browsing-level` (`PG`=1, `PG-13`=2, `R`=4, `X`=8, `XXX`=16; OR-combine
with commas, e.g. `X,XXX`; `all`=31). It's rejected by model commands.
If both flags are passed to an image command, `--browsing-level` wins.
Without an API key, NSFW silently downgrades to SFW (with a stderr
warning on image commands).

## Quick Reference

All commands prefix with `python3 ${HERMES_SKILL_DIR}/scripts/`.

| Intent | Command |
|---|---|
| Top models for a base model | `search.py top-models --type LORA --base-model "..."` |
| Search models by name | `search.py models --query "..."` |
| **Batch fetch many models in one call** | `search.py models --ids "1,2,3,..." --limit 100 --json` |
| Top media (images + videos) this week | `search.py top-images --period Week --has-meta` |
| Media for a model (images **or** videos) | `search.py images --model-id <id> --has-meta` |
| Full model details + versions | `show.py model <id>` |
| Version details + file hashes | `show.py version <vid>` |
| Working prompts (image **or** video) | `show.py prompts <id>` |
| Identify .safetensors by hash | `show.py hash <SHA256_or_AutoV2>` |
| Download command for ComfyUI | `download.py <id> --comfyui-path "..."` |

Every script supports `--json` and `--help`. IDs in output use `#` prefix
(greppable as `#\d+`). **For many models at once, prefer batch fetch
(`--ids`) over looping** — see Pitfalls #6.

## Procedure

### Recipe 1 — Top LoRAs for a base model

```bash
# SFW
python3 ${HERMES_SKILL_DIR}/scripts/search.py top-models --type LORA --base-model "Flux.1 D" --period Month --limit 10
# Include NSFW
python3 ${HERMES_SKILL_DIR}/scripts/search.py top-models --type LORA --base-model "Pony" --period Month --limit 10 --nsfw
```

### Recipe 2 — Search models by name

```bash
python3 ${HERMES_SKILL_DIR}/scripts/search.py models --query "hands fix" --type LORA --limit 5
```

Text queries route through Meilisearch; combining `--query` with `--type`
/ `--base-model` / `--tag` / `--username` works. REST is used only for
`--ids` / `--favorites`.

### Recipe 3 — Mine working prompts (image or video)

```bash
python3 ${HERMES_SKILL_DIR}/scripts/show.py prompts 257749 --limit 5
```

Auto-resolves latest version. Returns prompts, negatives, sampler/steps/
CFG/seed, and stacked LoRAs. Works identically for video models.

### Recipe 4 — Identify a local .safetensors

```bash
python3 ${HERMES_SKILL_DIR}/scripts/show.py hash E837144C55
```

Accepts SHA256, AutoV2 (10-char prefix), CRC32, or BLAKE3 (auto-uppercased).

### Recipe 5 — ComfyUI download command

```bash
python3 ${HERMES_SKILL_DIR}/scripts/download.py 257749 --comfyui-path "D:/ComfyUI/models"
```

**Agent: forward `civitai.comfyui_path` from the skill activation message
as `--comfyui-path` when set.** The script doesn't read `config.yaml`. If
unset, output uses a `<COMFYUI_PATH>` placeholder. Emits `curl`, `wget`,
and PowerShell variants — references `$CIVITAI_API_KEY` by name, never
prints the value. `--version-id` overrides "latest".

### Recipe 6 — Media for a model (images or videos)

```bash
# Image or video model — same command, returns whatever the model produced
python3 ${HERMES_SKILL_DIR}/scripts/search.py images --model-id <id> --has-meta --limit 5
# Include NSFW
python3 ${HERMES_SKILL_DIR}/scripts/search.py images --model-id <id> --has-meta --nsfw
# Force one media type
python3 ${HERMES_SKILL_DIR}/scripts/search.py images --model-id <id> --content-type video --has-meta
```

`--model-id` auto-resolves to the latest version — see Pitfalls #2.

### Workflow A — Find best LoRA → mine prompts → download

```bash
python3 ${HERMES_SKILL_DIR}/scripts/search.py top-models --type LORA --base-model "Flux.1 D" --period Week --limit 10
python3 ${HERMES_SKILL_DIR}/scripts/show.py model <ID>
python3 ${HERMES_SKILL_DIR}/scripts/show.py prompts <ID> --limit 3
python3 ${HERMES_SKILL_DIR}/scripts/download.py <ID> --comfyui-path "<configured-path-or-omit>"
```

### Workflow B — Compare popularity across base models

```bash
python3 ${HERMES_SKILL_DIR}/scripts/search.py top-models --type Checkpoint --base-model "Flux.1 D" --period Month --limit 20
python3 ${HERMES_SKILL_DIR}/scripts/search.py top-models --type Checkpoint --base-model "SDXL 1.0" --period Month --limit 20
```

### Delivering media

After fetching image or video URLs, append `[[as_document]]` to the final
response so Hermes delivers media as file attachments — quality matters
for AI media, and inline previews compress lossily (or, for video, may
not play at all).

## Enum cheat sheet

### Model Types (`--type`)

`Checkpoint`, `LORA`, `LoCon`, `DoRA`, `TextualInversion`, `Hypernetwork`,
`Controlnet`, `Poses`, `AestheticGradient`, `Wildcards`, `MotionModule`,
`VAE`, `Upscaler`, `Workflows`, `Detection`, `Other`.

### Base Models (`--base-model`)

| Family | Values |
|---|---|
| SD 1.x | `SD 1.4`, `SD 1.5`, `SD 1.5 LCM`, `SD 1.5 Hyper` |
| SD 2.x | `SD 2.0`, `SD 2.1` |
| SDXL | `SDXL 1.0`, `SDXL Lightning`, `SDXL Hyper` |
| Flux | `Flux.1 S`, `Flux.1 D`, `Flux.1 Krea`, `Flux.1 Kontext`, `Flux.2 D`, `Flux.2 Klein 9B/4B` |
| Anime | `Pony`, `Pony V7`, `Illustrious`, `NoobAI`, `Anima` |
| Z-Image | `ZImageBase`, `ZImageTurbo` |
| Other | `Qwen`, `Chroma`, `HiDream`, `AuraFlow`, `Hunyuan 1`, `Kolors`, `PixArt a/E`, `Lumina` |
| Video | `CogVideoX`, `Hunyuan Video`, `LTXV`, `LTXV2`, `Mochi`, `Wan Video 1.3B/14B/2.2/2.5` |

### Sort (`--sort`)

- **Models (Meilisearch):** `Most Downloaded`, `Highest Rated`, `Most Collected`, `Most Comments`, `Most Tipped`, `Newest`, `Oldest`
- **Models (REST fallback):** only `Highest Rated`, `Most Downloaded`, `Newest` (others silently downgraded)
- **Images:** `Most Reactions`, `Most Comments`, `Most Collected`, `Newest`, `Oldest`

### Period (`--period`)

`AllTime`, `Year`, `Month`, `Week`, `Day`.

### Content Type (`--content-type`)

`image`, `video` (image commands only).

### ComfyUI folder map (`download.py`)

`Checkpoint`→`checkpoints/`, `LORA`/`LoCon`/`DoRA`→`loras/`,
`TextualInversion`→`embeddings/`, `Controlnet`→`controlnet/`,
`VAE`→`vae/`, `Upscaler`→`upscale_models/`,
`Hypernetwork`→`hypernetworks/`, `Poses`→`poses/`, others→`other/`.

## Pitfalls

1. **REST text-search is broken (since May 2025).** `GET /models?query=...`
   returns empty; `search.py models --query ...` auto-routes to Meilisearch.
   REST is only used for `--ids` / `--favorites`. If a `--query` search
   comes back empty, try `--username` — most reliable filter.

2. **`/images?modelId=...` is silently ignored — only `modelVersionId`
   works.** `search.py images --model-id <ID>` and `show.py prompts <ID>`
   both auto-resolve to the latest version and emit
   `# resolved model <ID> → version <VID>` on stderr. For an older version,
   call `show.py model <ID>` first to list versions, then pass
   `--model-version-id`.

3. **`--browsing-level` is a bitmask.** `X,XXX` → 24; `all` → 31. Levels
   not in the OR are excluded.

4. **`get_creators` / `get_tags` aren't exposed** — those endpoints are
   slow (30s+) and frequently 500. Use `search.py models --username "name"`
   for creators; `search.py top-images --tag "anime"` for tag discovery.

5. **Meilisearch key is Civitai's public frontend key.** The hardcoded
   `MEILI_KEY` in `_common.py` is the search-only key shipped in
   civitai.com's JS bundle — every visitor's browser receives it. Not a
   secret, not tied to any account; that's why model search works without
   `$CIVITAI_API_KEY`. Civitai rotates it occasionally.

   **If a Meili call fails with `meilisearch http 401/403`, the key has
   rotated.** Recovery: open <https://civitai.com>, DevTools → Network,
   filter `multi-search`, trigger a search, click the POST to
   `search-new.civitai.com/multi-search`, and copy the value after
   `Authorization: Bearer` (64-char hex). Update `MEILI_KEY` in
   `scripts/_common.py`, **or** set `MEILISEARCH_KEY=<new_key>` in the
   Hermes env file (env wins). For remote sandboxes, add
   `MEILISEARCH_KEY` to `terminal.env_passthrough` in `config.yaml`.

   Sanity-check:

   ```bash
   curl -X POST https://search-new.civitai.com/multi-search \
     -H "Authorization: Bearer <new_key>" -H "Content-Type: application/json" \
     -d '{"queries":[{"q":"test","indexUid":"models_v9","limit":1}]}'
   ```

   `results` in response = live; another 401/403 = wrong string copied.

6. **Don't fan out parallel API calls — Civitai rate-limits aggressively.**
   20+ parallel `show.py model <id>` invocations (concurrent.futures,
   repeated `execute_code` shells, etc.) reliably hit 429/502. Scripts
   retry with backoff (3 attempts, ~8s) but parallel callers collide on
   backoff windows and most still fail. Better patterns:

   1. **Batch fetch with `--ids`** — `/models` accepts up to 100 IDs per
      call and returns full `modelVersions` per model. Use this for any
      list of IDs from a search. **Pass `--nsfw` if any IDs may be NSFW** —
      flags don't carry between calls; without it Civitai silently drops
      NSFW models and the script warns which IDs are missing.

      ```bash
      python3 ${HERMES_SKILL_DIR}/scripts/search.py models --ids "1,2,3,...,27" --limit 100 --json --nsfw
      ```

   2. **Serial loop** — if batch isn't an option (e.g. `show.py prompts`
      per version), one shell at a time with a small `sleep`:

      ```bash
      for id in 1 2 3; do
        python3 ${HERMES_SKILL_DIR}/scripts/show.py model $id --json
        sleep 1
      done
      ```

   Rule of thumb: **one Civitai shell process at a time**.

7. **`show.py … --json` and `search.py models --ids --json` are slimmed.**
   Raw `/models/{id}` payloads can exceed 100 KB (HTML descriptions, image
   arrays, six hash formats); a 16-ID batch easily hits 600 KB+. Hermes's
   terminal cap is ~20-50 KB; oversized JSON gets truncated mid-stream and
   breaks `json.loads` — this caused the historical "5/22 parse" bug.

   Since v1.1.0, `show.py {model,version,prompts} --json` and
   `search.py models --ids --json` route through `slim_*` helpers that
   strip HTML descriptions, per-version `images` arrays, all hashes except
   SHA256/AutoV2, non-LoRA `resources` (prompts), and versions beyond #20
   (a `_versions_truncated: true` flag surfaces when clipped). Preserved:
   id/name/type/stats/creator/tags, per-version id/name/baseModel/
   trainedWords, full file info (name, sizeKB, primary, scan results,
   hashes, downloadUrl). For the raw payload, `web_fetch
   https://civitai.com/api/v1/models/<id>` to a file — don't capture
   through `terminal` stdout.

## Safety

- **Store `CIVITAI_API_KEY` in `.env` only.** Scripts read from
  environment, never echo the value.
- **`download.py` shell-escapes untrusted strings.** Filenames, URLs,
  target paths wrapped with `shlex.quote()` (bash) / PowerShell
  single-quote literals. Only `$CIVITAI_API_KEY` is unquoted, so the
  shell expands it at run time.
- **Append `[[as_document]]`** for image/video previews so Hermes delivers
  them as file attachments rather than lossy inline previews.
- **Treat unfamiliar models with caution.** `show.py version <vid>` shows
  `pickleScanResult` and `virusScanResult`. If either is not `Success`,
  warn the user before generating a download command.

## Verification

```bash
# Quick check
python3 ${HERMES_SKILL_DIR}/scripts/search.py top-models --type LORA --base-model "Flux.1 D" --limit 1
# Comprehensive
python3 ${HERMES_SKILL_DIR}/scripts/health_check.py
```

Quick check expects one LoRA card (`#ID`, name, creator, base model,
downloads, tags). Errors: `auth failed` = key invalid;
`meilisearch http 401` = Meili key rotated (Pitfalls #5);
`network error after retries` = host unreachable. SFW results work
without env vars. `health_check.py` runs static + network + auth +
functional checks; exits 0 on success.
