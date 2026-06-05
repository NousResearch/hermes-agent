---
version: 0.3.0
name: higgsfield-generate
description: |
  Generate images/videos via Higgsfield AI. Default: GPT
  Image 2 for images/design/text, Seedance 2.0 for video,
  Nano Banana 2/Pro for character/reference image work,
  Marketing Studio for ads with avatars/products/hooks,
  settings, plus Soul V2/Cinema/Cast/Location and Kling
  3.0. Use when: "generate an image", "make a video",
  "animate this photo", "image-to-video",
  "edit/stylize/remix this image", "produce a clip",
  "create an ad", "make a UGC video", "product demo",
  "unboxing", "brand video", "presenter video",
  "import product from URL", "create avatar for ad",
  or "analyze video virality". Supports image-to-image,
  image-to-video, references, job/upload IDs, and
  Marketing Studio. Chain with higgsfield-soul-id for
  face/identity consistency. Virality Predictor
  (`brain_activity`) analyzes video virality: hook strength,
  attention, retention, distraction risk, and creative
  score. NOT for: Soul Character training (use
  higgsfield-soul-id), product photoshoots, marketplace
  listing cards, text/chat/TTS tasks.
argument-hint: "[prompt-or-analysis-request] [--model <name>] [--image|--video <path-or-id>]"
allowed-tools: Bash
---

# Higgsfield Generate

Submit jobs to any Higgsfield model. Wraps the `higgsfield` CLI. Covers generic image/video gen, Marketing Studio (branded ads, avatars, products, hooks, settings), and, secondarily, Virality Predictor video scoring.

## Step 0 — Bootstrap

Before any other command:

1. If `higgsfield` is not on `$PATH`, install it:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/higgsfield-ai/cli/main/install.sh | sh
   ```
2. If `higgsfield account status` fails with `Session expired` / `Not authenticated`:
   a. First check `ls -la ~/.config/higgsfield/credentials.json.lock` — if the file exists and is 0 bytes, auth was never completed.
   b. If the user is available, ask them to run `higgsfield auth login` (interactive browser OAuth).
   c. If the user is NOT available or the session can't be fixed immediately, **fall back to MiniMax CLI (`mmx`)** for image and music generation. Install with:
      ```bash
      npm install -g mmx-cli
      # Or via npx:
      npx mmx-cli auth login --api-key $MINIMAX_API_KEY
      ```
      The MiniMax API key is stored in the agent secret manager as `MINIMAX_API_KEY`; expose it to the runtime environment before authenticating the CLI.
      **Do not** crash or stall because Higgsfield is unavailable — MiniMax can generate images (`mmx image generate --out <path> --aspect-ratio 9:16`) and music (`mmx music generate --instrumental --out <path>`) that match or exceed the quality needed for social reels.


## UX Rules

1. Be concise. No raw IDs, no JSON dumps in chat. Print the media URL for generated assets, or the text summary for Virality Predictor.
2. No internal jargon. Don't narrate "calling higgsfield cost", "polling job".
3. Detect the user's language from the first message and reply in it. Technical args (`--aspect_ratio 16:9`) stay English.
4. Don't batch-ask. Pick a sane default model and ask one thing at a time only if genuinely missing.
5. Don't pre-estimate cost or optimize for cheaper models unless the user asks. Prefer the quality default first.
6. Pass `--wait` to `generate create` so the command blocks until done and prints the result URL itself. Avoid the two-step `create` → `wait` pattern.

## Discovery guardrail

When looking for a Higgsfield feature/model, do not rely only on semantic search or CLI `--help`. First run an unfiltered model list, then inspect likely `job_set_type` names. If the user says a model exists but search returns no results, trust that signal and verify with the full model list before answering.

Virality Predictor is exposed as:

- Customer-facing name: Virality Predictor
- Technical `job_set_type`: `brain_activity`
- Category/output: text report. This is video-in/text-out analysis, not a text/chat generation model.
- Input: uploaded video
- Purpose: finished-video hook, attention, retention, and virality analysis

If the user says "analyze this video", "score this ad", "evaluate the hook", or similar, route to `brain_activity` even though it appears under text/analysis models. Classify by task intent and required input, not by output category alone.

## Workflow — generic generation

1. **Pick a model.** Start with the core defaults unless the brief clearly needs a specialist:

   - **GPT Image 2** → default image model for high-fidelity general generation, graphic design, UI, banners, typography, and on-image text.
   - **Seedance 2.0** → default video model for serious motion, cinematic clips, multi-shot work, image-to-video, and 4–15s production-quality output. 12s is valid.
   - **Nano Banana 2/Pro** → default for character, cartoon, stylized, and reference-driven image work; use Pro for harder briefs.
   - **Marketing Studio** → default for ads, UGC, product demos, unboxing, TV spots, presenter videos, and brand/product workflows.

   **Image:**
   - Brand product visual (Pinterest pin, lifestyle, hero banner, ad pack, virtual try-on) → use `higgsfield-product-photoshoot` instead. NOT this skill.
   - Generated product concept / packaging / can / bottle with brand name or label text → GPT Image 2.
   - Branded ad image with avatar + product (Marketing Studio shape) → Marketing Studio Image (see Marketing Studio below)
   - Aesthetic UGC / fashion editorial / lifestyle character → Soul 2.0
   - Cinematic still frame → Soul Cinema
   - Highly characterful creative persona (text-only, distinctive) → Soul Cast
   - Locations / environments / no-people scenes → Soul Location (best in class)
   - Vector illustrations OR face edit + complex scene swap → Seedream 4.5
   - Soul Character (reference id from `higgsfield-soul-id`) → Soul 2.0 for stills, Soul Cinema for cinematic
   - Character or cartoon-style work → Nano Banana 2; step up to Nano Banana Pro on hard cases
   - Fast and cheap iteration → Z Image
   - **Default for everything else → GPT Image 2.** Graphic design, UI, banners, typography, and high-fidelity general generation.

   **Video:**
   - All advertising / commercial / branded ad video → Marketing Studio (see Marketing Studio below)
   - **Default all-purpose serious video (multi-shot, consistent identity, motion-heavy, image-to-video, 4–15s requests) → Seedance 2.0.** SOTA. Do not downgrade to Seedance 1.5 just because its duration enum is easier to read; validate Seedance 2.0 first.
   - Single-plane scene without strong dynamics, cheaper than Seedance 2.0 → Kling 3.0
   - Cheap clean shot without cuts, only when the user asks for cheaper/budget output → Seedance 1.5 Pro
   - Cinema-grade highest fidelity → Cinema Studio Video 3.0
   - Cheap with strong physics, no audio needed → Minimax Hailuo
   - Fast batch / volume → Veo 3.1 Lite

   **Video analysis:**
   - Rate a finished video's hook, virality potential, attention, retention, or distraction risk → Virality Predictor (`brain_activity`). This is a video analysis model that returns a text score/report, not a generated media asset.

   For the actual `--model` ID to pass to `higgsfield generate create`, run `higgsfield model list --json | jq` to map display names to IDs. See `references/model-catalog.md` for the full table.

2. **Pass media inputs straight to flags.** Media flags accept a local file path **or** a UUID. CLI auto-uploads paths and auto-detects job vs upload for UUIDs. No need to pre-upload. Each model declares accepted roles (`image`, `start_image`, `end_image`, `video`, `audio`) — see `references/media-inputs.md`.
3. **Validate quickly.** If unsure of params, run `higgsfield model get <jst> --json` once and pass only what's needed. Validate the preferred model before falling back to an older one. Use schema defaults otherwise. The server returns `adjustments` for non-fatal coercions (e.g. `aspect_ratio=99:99` → closest match) and a structured error for invalid declared-param values.
4. **Submit and wait in one shot.** `higgsfield generate create <jst> [--prompt "..."] [media flags] [param flags] --wait`. Blocks until terminal status and prints the result on stdout. Tunables: `--wait-timeout 20m` (default 10m), `--wait-interval 5s` (default 3s). Virality Predictor does not need a prompt; pass `--video`.
   - If your agent runner has a foreground timeout cap (common in tool wrappers), start the command in background and poll the process until exit; then return the final URL from stdout.
5. **Deliver.** For generated media, send the URL plus a one-line summary (model, duration if video). For Virality Predictor, deliver the scores, business interpretation, and the Open report link. Do not surface `.glb`, `.bin`, or region-table internals in normal chat output.

To inspect or rerun later, `higgsfield generate list --json` and `higgsfield generate get <id> --json` work for retrospection. `higgsfield generate wait <id>` is still available if you ever need to rejoin a job started without `--wait`.

## Media flags

| Flag | Purpose | Models that accept it |
|---|---|---|
| `--image <path-or-id>` | reference image | most image models, `seedance_2_0`, `veo3`, `marketing_studio_video` |
| `--start-image <path-or-id>` | first frame for image-to-video transitions | `kling3_0`, `kling2_6`, `veo3_1`, `seedance_2_0`, `marketing_studio_video` |
| `--end-image <path-or-id>` | last frame for transitions | `kling3_0`, `seedance_2_0`, `marketing_studio_video` |
| `--video <path-or-id>` | reference or analyzed video | `seedance_2_0`, `brain_activity` |
| `--audio <path-or-id>` | reference audio (lipsync, soundtrack match) | `seedance_2_0` (use this, NOT `--generate-audio`) |

Each flag accepts either a local file path (auto-uploaded) or a UUID (upload id from `higgsfield upload create`, or a previous job id). Each model declares its own role set via `MEDIA_ROLES`. See `references/media-inputs.md` for the full table.

## Common params

Flags pass through to model schema. Use `higgsfield model get <jst>` to discover.

```bash
higgsfield generate create gpt_image_2 --prompt "neon city at dusk" --aspect_ratio 16:9 --resolution 2k --wait
higgsfield generate create nano_banana_2 --prompt "anime character concept, expressive pose" --image ./ref.png --wait
higgsfield generate create seedance_2_0 --prompt "camera dollies in" --start-image ./first.png --duration 12 --wait
higgsfield generate create text2image_soul_v2 --prompt "..." --soul-id <soul_ref_id> --quality 2k --wait
higgsfield generate create brain_activity --video ./ad.mp4 --wait
```

For machine-readable output (chained pipelines, agent context), add `--json`. With `--wait --json` you get the final job object array. Without `--wait`, you get the job IDs. Virality Predictor stores raw analysis and render artifacts in the job params, but the default text output should stay to scores plus Open report.

Stdin prompt: `echo "..." | higgsfield generate create z_image --wait`.

Soul image quality: for `text2image_soul_v2` and `soul_cinematic`, pass `--quality 1.5k` or `--quality 2k`. These are UI-facing tiers; the backend maps them to `720p`/`1080p` and model-specific dimensions from the selected `--aspect_ratio`. `soul_location` has no quality selector; it uses fixed dimensions per aspect ratio.

## Marketing Studio

Branded image/video gen: avatars + products + optional setup hooks/settings + ad-style modes. Use models `marketing_studio_video` and `marketing_studio_image`.

### Concepts

- **Avatar** — presenter face. Curated `preset` (browse `higgsfield marketing-studio avatars list`) or `custom` (uploaded photos via `higgsfield marketing-studio avatars create`). For UGC modes, an avatar is optional if the brief clearly mentions a person; the backend can create a Soul Character automatically. Pass an avatar when the user wants a specific presenter.
- **Product** — brand item with title + reference images. Imported from URL (`higgsfield marketing-studio products fetch --url ...`) or created from uploaded images (`higgsfield marketing-studio products create`).
- **Webproduct** — App Store / web page version. Auto-routes when fetching App Store URLs.
- **Hook** — reusable opening angle / ad hook. Browse with `higgsfield marketing-studio hooks list`. Hook text is prepended to the user's prompt; it does not replace `--prompt`.
- **Setting** — reusable environment / scene context. Browse with `higgsfield marketing-studio settings list`.
- **Ad reference** — reusable inspiration video that can be bound to an avatar and/or product. Created from an uploaded video (`--video-input <upload_id>`) or a previous generation job (`--job <job_id>`). Browse with `higgsfield marketing-studio ad-references list`. See `references/marketing-ad-references.md`.
- **Brand kit** — captures a brand's identity (name, logo, hero images, colours, fonts, tone) for reuse across image generations. Created by handing in a website URL (`higgsfield marketing-studio brand-kits fetch --url https://… --wait`). See `references/marketing-brand-kits.md`.
- **Ad format** — presets that drives the visual structure of a generated image (`headline`, `bullet-points`, etc.). Read-only, browse with `higgsfield marketing-studio ad-formats list`. Required input for `dtc-ads generate`.

### Discovery commands

Use these exact list commands when the user asks what already exists:

```bash
higgsfield marketing-studio avatars list --json
higgsfield marketing-studio products list --json
higgsfield marketing-studio hooks list --json
higgsfield marketing-studio settings list --json
higgsfield marketing-studio ad-references list --json
higgsfield marketing-studio brand-kits list --json
higgsfield marketing-studio ad-formats list --json
```

`--hook_id` and `--setting_id` are supported by `marketing_studio_video` only; do not pass them to `marketing_studio_image`.

### UX rules (additional)

- One question per phase. Don't ask product+avatar+mode upfront.
- **Two ad approaches are mutually exclusive.** Either the user gives an ad reference video (reference-driven) **or** picks hook/setting blocks (composed-from-blocks) — never both. If the user has an ad reference selected, do not offer hook/setting; if hook/setting are picked, do not offer to attach an ad reference.
- **Ad reference source.** The only valid inputs are a local video file (uploaded via `higgsfield upload create ... --video`) or a prior video job. If the user provides anything else, ask for a local file.
- **`dtc-ads` ad format is mandatory.** Always ask the user to pick from `ad-formats list`. There is no auto-default — both the CLI and server reject calls without `--format-id`.
- **`dtc-ads` optional inputs.** Suggest avatars, products, and reference media when the brief calls for them; only attach what the user picks.

### Workflow — quick ad video

1. **Get product.**
   - Existing product → `higgsfield marketing-studio products list --json`
   - URL → `higgsfield marketing-studio products fetch --url <url> --wait` (polls until import done)
   - Local images → `higgsfield upload create <photo>...` then `higgsfield marketing-studio products create --title "..." --image <id>...`
   Capture product id. When using `--hook_id`, strongly prefer passing `--product_ids`; hooks are designed to pivot into a product and work poorly without product context.
2. **Pick avatar if needed.**
   - Default: `higgsfield marketing-studio avatars list` and pick a preset matching the brand voice.
   - Custom: `higgsfield marketing-studio avatars create --name "..." --image <upload_id>`.
   For UGC modes, you may omit `--avatars` when no specific presenter is required and the brief mentions a person; the backend can synthesize a Soul Character.
3. **Optionally pick setup items.**
   - Hook: `higgsfield marketing-studio hooks list --json`
   - Setting: `higgsfield marketing-studio settings list --json`
   Pass selected IDs as `--hook_id <hook_id>` and `--setting_id <setting_id>` for `marketing_studio_video` only. Do not copy the hook's prompt into `--prompt` unless the user explicitly wants to reinforce the same wording.
4. **Pick mode if needed.** Default is `ugc`; `--mode` is not required just because `--hook_id` is present. Other current slugs: `ugc_how_to`, `ugc_unboxing`, `product_showcase`, `product_review`, `tv_spot`, `wild_card`, `ugc_virtual_try_on`, `virtual_try_on`. **Hook/setting are valid only for `ugc`, `ugc_how_to`, `ugc_unboxing`, `product_review`, `ugc_virtual_try_on`** — do not pass `--hook_id` / `--setting_id` with the other modes. See `references/marketing-modes.md`.
5. **Generate (one-shot).**
   ```bash
   PRODUCT_IDS_JSON=$(mktemp)
   AVATARS_JSON=$(mktemp)
   printf '["<product_id>"]' > "$PRODUCT_IDS_JSON"
   printf '[{"id":"<avatar_id>","type":"preset"}]' > "$AVATARS_JSON"

   higgsfield generate create marketing_studio_video \
     --prompt "..." \
     --avatars @"$AVATARS_JSON" \
     --product_ids @"$PRODUCT_IDS_JSON" \
     --mode ugc \
     --duration 15 \
     --resolution 720p \
     --aspect_ratio 9:16 \
     --wait
   ```
   Add `--hook_id <hook_id>` and/or `--setting_id <setting_id>` when a setup hook/setting was selected.
   `product_ids` and `avatars` are JSON arrays; pass them via `@/path/to/file.json`. Do not pass a bare UUID to `--product_ids`.
   Resolution is `480p` or `720p`. Aspect ratio is one of `auto`/`21:9`/`16:9`/`4:3`/`1:1`/`3:4`/`9:16`. `--generate-audio true` is supported here (unlike `seedance_2_0`). `--wait` blocks until done; bump `--wait-timeout 30m` for longer ad runs.
6. **Deliver.** URL + one-line summary (mode, duration).

### Click-to-Ad shortcut (URL-driven)

When the user gives a product URL and wants a marketing video in one go:

```bash
# 1. Trigger fetch (returns the product id, import runs in the background)
higgsfield marketing-studio products fetch --url https://shop.example.com/sneakers --wait

# 2. Generate the marketing video against the same URL — backend reuses the entity
higgsfield generate create marketing_studio_video \
  --url https://shop.example.com/sneakers \
  --mode ugc \
  --duration 15 \
  --aspect_ratio 9:16 \
  --wait
```

Backend dedupes by URL, so repeated runs reuse the existing entity instead of re-fetching.

### Workflow — marketing image

Same as above but use `marketing_studio_image` model:

```bash
higgsfield generate create marketing_studio_image \
  --prompt "..." \
  --aspect_ratio 1:1 \
  --resolution 2k \
  --wait
```

## Quality gate for premium social videos (new)

When the user asks for **"quality" / "cinema" / "studio-level"** output, do **not** deliver slideshow-style fallback videos as final creative.

Operational rule:
1. Start with **Cinema Studio** (`cinematic_studio_3_0` or current cinema variant) or **Marketing Studio Video** for ad-focused reels.
2. Run **Virality Predictor** (`brain_activity`) on the generated clip before presenting it as final.
3. If the hook is weak, iterate prompt/model before publishing. Practical thresholds for first-pass acceptance:
   - `hook_score >= 40` preferred
   - `viral_potential >= 55` preferred
   - `sustain` can be high but does **not** compensate for weak hook.
4. In user-facing output, report only: URL + concise score interpretation + next iteration decision.

## Pitfall to avoid:
- A clip can look "pretty" but still fail the growth goal if early-hook metrics are weak. Always separate **aesthetic quality** from **distribution performance**.

## CRITICAL: Preview before publish (user-enforced rule)

When producing **original content** (not reposts of existing viral material):

1. **NEVER publish without showing the user a preview first.** The agent's taste is not the user's taste.
2. Send the video file to Telegram via `MEDIA:<path>` or upload it, then **wait for explicit approval** before touching `INSTAGRAM_POST_IG_USER_MEDIA`.
3. If the user rejects it, ask what to change (image style, text, music, duration) and rebuild.
4. Only after `✅ Está bien` or similar explicit approval → publish.

Reposts of existing viral reels (where the user provided the URL) are exempt — the user already approved the source.

**Failure mode observed:** generating a 9s typography reel on a dark background with no image when the reference had a cinematic photograph with text overlay → user called it "basura" and deleted it. Always match the production style of the reference, not just the text content.

## Virality Predictor video scoring

Use Virality Predictor (`brain_activity`) when the user wants to evaluate a finished video as a business creative: hook strength, virality potential, attention, retention, or how well the content/product holds focus and minimizes distraction. Treat "Virality Predictor" as the customer-facing feature name; `brain_activity` is only the CLI/job_set_type.

```bash
higgsfield generate create brain_activity --video ./creative.mp4 --wait
```

The result is text, not a generated image/video. Report the overall score, peak hook second, sustain score, strongest/weakest regions, and report URL if present. Interpret it as an objective attention proxy for creative testing: higher Visual/Auditory/Language/Attention scores suggest stronger stimulus and focus; lower Default Mode is better because it suggests less mind-wandering.

The CLI prints an Open report URL like `https://<app-domain>/apps/virality-predictor?resultJobId=<job_id>`. Send that URL for the visual report. Raw artifact URLs such as `brain_example_url`, `vertexMapBinaryUrl`, and `vertexMapUrl` are implementation details; mention them only when the user asks for raw data or implementation details.

Good final shape:

```text
Overall score: 44/100
Peak hook: 49% at 1s
Sustain: 89%
Strongest region: Visual Cortex
Risk: Default Mode is high, which can indicate mind-wandering.

Open report: <report_url>
```

## Extended workflow — 30–45s social promo with voiceover (assembly pattern)

Use this when the user asks for a full promotional video longer than a single model run (e.g., 30–45s Reel/TikTok):

1. Generate 3 short vertical clips (typically `seedance_2_0`, `--duration 10` or `12`, `--aspect_ratio 9:16`) with scene-specific prompts (intro / value / CTA).
2. Download clips and concatenate with FFmpeg.
3. Generate spoken narration from a scripted CTA voiceover (TTS).
4. If narration is longer than video, extend the last frame with FFmpeg `tpad` and mux narration as final audio.
5. Verify final duration with `ffprobe` and deliver the final MP4.

Reference FFmpeg pattern:

```bash
# concat
printf "file 's1.mp4'\nfile 's2.mp4'\nfile 's3.mp4'\n" > list.txt
ffmpeg -y -f concat -safe 0 -i list.txt -c copy base_concat.mp4

# extend tail + mux voiceover
ffmpeg -y -i base_concat.mp4 -i voiceover.ogg \
  -filter:v "tpad=stop_mode=clone:stop_duration=4.3" \
  -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 20 -c:a aac -b:a 192k -shortest final.mp4

# verify
ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 final.mp4
```

## Resilience pattern (provider instability)

- If a generation fails with transient upstream errors (e.g., HTTP 5xx), do one immediate retry.
- If repeated identical failure persists, change strategy (simplify prompt, switch model/shot, or proceed with alternate clip and finish assembly) instead of looping the same command.

## Errors

- `Missing required params: prompt` → user gave no prompt; ask for it.
- `Missing required params: medias` on `brain_activity` / Virality Predictor → pass exactly one video via `--video <path-or-id>`.
- `Invalid values: aspect_ratio=99:99 (allowed: ...)` → bad enum; pick from allowed.
- `Unknown params: foo` → schema doesn't accept that flag; check `higgsfield model get <jst>`. If this happens for `hook_id` or `setting_id`, the selected model/job_set_type does not support Marketing Studio setup items.
- `Session expired` → `higgsfield auth login`.

See `references/troubleshooting.md` for more.

## Reference docs

Load on demand:

- `references/model-catalog.md` — picking the right model for the task
- `references/prompt-engineering.md` — writing prompts that work
- `references/media-inputs.md` — image/video/audio reference flows and Virality Predictor video analysis
- `references/troubleshooting.md` — common errors and fixes
- `references/marketing-avatars.md` — preset vs custom avatars
- `references/marketing-products.md` — URL fetch vs manual product create
- `references/marketing-setup-items.md` — hooks/settings discovery and usage
- `references/marketing-ad-references.md` — ad reference videos (create/list/get)
- `references/marketing-brand-kits.md` — brand kits (fetch from URL, list, get)
- `references/marketing-dtc-ads.md` — DTC Ads Engine (`dtc-ads generate`)
- `references/marketing-modes.md` — every Marketing Studio mode
- `references/minimax-fallback.md` — when Higgsfield auth is unavailable, use MiniMax CLI for image + music gen
