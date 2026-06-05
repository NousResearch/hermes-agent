---
name: higgsfield-cli
description: "Generate AI images and video on Higgsfield's hosted backend (Nano Banana Pro, FLUX.2, Soul V2, Veo 3.1, Kling v3.0, etc.) via CLI. No local GPU needed. Use when the user asks for image/video generation, brand/product photoshoot, marketplace product cards, or character/SoulID training."
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [image, video, generation, marketing, higgsfield, mcp]
    related_skills: [ffmpeg-media, hyperframes-render, vimax-video]
---

# Higgsfield CLI

Installed at `higgsfield` (aliases: `higgs`, `hf`). Calls the
Higgsfield cloud backend — no local GPU required.

## First-time auth (browser OAuth)
```bash
higgsfield auth login          # opens browser, ~5 sec OAuth
higgsfield account status      # confirm: <email> — <plan>, <N> credits
```
Token is short-lived; re-run `auth login` if you see `401 / Session expired`.

## Quick recipes

```bash
# Cheap image smoke test
higgsfield generate create z_image --prompt "test" --wait

# Photoshoot from a product photo (hosted prompt enhancement)
higgsfield upload --file ./product.jpg                          # → upload_id
higgsfield product-photoshoot create --image <upload_id> \
  --mode product --prompt "white studio, soft shadow, 50mm"

# Marketplace product card (e-commerce style assets)
higgsfield marketplace-cards create --image <upload_id> --variants 4

# Train a Soul ID (character / brand mascot from 5–20 references)
higgsfield soul-id train --name "vonash-mascot" --refs ./refs/

# Video (Kling / Veo / Soul V2 — model list with `model list --video`)
higgsfield generate create kling_v3 \
  --prompt "slow drone over a Caribbean beach at golden hour" \
  --image <upload_id> --duration 5s --wait
```

## When to invoke
- "Generate an image of …" / "make a 5-second video clip of …" → here first.
- Marketing studio assets (carousels, hero shots, A/B variants).
- Building a brand mascot / consistent character (SoulID).
- Product photoshoots from a single phone photo.

## Pricing
Credit-based per render — model + resolution + duration determine cost.
Plan credits show with `higgsfield account status`. The CLI itself is free,
the renders are not. **Always confirm with the user before kicking off batch
jobs** (`marketplace-cards --variants 8`, video > 5s, etc.).

## Companion skills already installed
The `higgsfield-ai/skills` repo is cloned into Paimon's skills dir, exposing
4 task-specific skills the agent can invoke directly:
- `higgsfield-generate` — base image/video generation flow
- `higgsfield-product-photoshoot` — e-commerce product photography
- `higgsfield-marketplace-cards` — variant grids for marketplaces
- `higgsfield-soul-id` — character/brand reference training

Same skills are installed for `claude-code` and `codex` CLIs at
`~/.claude/skills/higgsfield/` and `~/.codex/plugins/higgsfield/`.

## Learning captured from `@higgsfield_ai` playbook (operational)

When producing social assets (posts/reels/video ads), prefer this execution order:

1. **Start with distribution goal** (DTC ad, organic reel, carousel, teaser).
2. **Feed real brand context** into generation (website URL copy + product image/reference).
3. **Select one output format first** (don't batch all formats blindly).
4. **Generate a single ready-to-run creative**, then iterate style variants.
5. **Scale only after winner signals** (hooks/readability/message clarity).

Practical implications for future agents:
- Prioritize **actionable workflow outputs** over vague creative prompts.
- Favor copy-safe renders (legible text overlays, concise hook-first messaging).
- Run small-batch tests first; escalate variants only with explicit user approval (credit spend).
- Keep a reusable structure: `hook -> value prop -> CTA -> format adaptation`.

Suggested prompt skeleton for ad creatives:

```text
Goal: <campaign objective>
Audience: <ICP>
Offer: <product/value>
Hook: <1-line thumb-stop opener>
Proof: <social proof / concrete claim>
CTA: <single action>
Format: <reel|story|feed|carousel|short video>
Constraints: legible on mobile, no broken text, brand-consistent colors
```

## UGC pipeline (stage-gated) for this operator

When the user asks to "test the full pipeline" and explicitly wants to review stages together, execute in checkpoints and report each stage before publishing:

1. Story concept (target, narrator style, location, <=45s script)
2. Generate two avatar/location stills from different providers (e.g., FAL + Higgs), then select winner with explicit QA rationale.
3. Generate voiceover (ElevenLabs) matched to narrator persona.
4. Before lip-sync step, probe model capabilities with:
   - `higgsfield model list --video --json`
   - `higgsfield model get <job_set_type> --json`
   Confirm whether media inputs (`--image`, `--audio`) are accepted.
5. If lip-sync input support exists, run it. If not, immediately switch to fallback path (Ken Burns/talking-photo style + captions) and surface status clearly as `fallback_used=true`.
6. Add captions/titles optimized for mobile readability.
7. Produce final Reel copy and publish only after final verification (`permalink`).

Rule: do not silently skip blocked stages. Mark exact stage status (`ok`, `blocked`, `fallback`) and continue with best available production path.

Field note (2026-05, this host CLI build): `generate cost` can accept `--audio` for some video models (`kling3_0`, `wan2_6`, `wan2_7`) but `create` may still fail (`kling3_0` HTTP 500, `wan2_6` job-level failed). Also, some models reject mixed media despite docs/blog naming overlap (e.g., `kling2_6` returned "Model accepts a single image input."). So capability checks must include an actual `generate create ... --wait` smoke test, not cost-only validation.

Flow-presets note (Lipsync Studio): UI preset names (e.g., Kling Avatars 2.0, Sync Lipsync 2 pro, Infinite Talk) map to internal flow IDs and may not appear as `hf model list` entries. For long-form avatar speech, prefer Kling Avatars 2.0 path with explicit media linking order: upload audio first, then bind audio+image in the avatar flow before generation.

Reference: `references/ugc-stage-gated-production.md`.
See also: `references/lipsync-studio-flow-mapping.md`.
See also: `references/longform-no-lipsync-mini-doc.md` (45–60s documentary-style reel pipeline without avatar/lipsync).

Field note (2026-05, long-form production): `higgsfield generate create ... --json` may return a JSON **array** (list of jobs) instead of a single object. Parsers must handle both shapes and extract `result_url` from the first item when list-shaped output is returned.

Field note (2026-05, async submit behavior): `higgsfield generate create <video_model> --json` **without** `--wait` can return an array of bare job-id strings (e.g. `["<uuid>"]`) instead of an object payload. Robust runners should:
1) accept string-array submit responses,
2) map those IDs back to scene indexes,
3) finalize each with `higgsfield generate wait <job_id> --json` before download.

Field note (2026-05, Seedance params): on this CLI/API surface, `seedance_2_0` rejects `generate_audio` as an explicit param (`Unknown params: generate_audio`). Treat audio as model-default behavior and derive accepted flags from `higgsfield model get <job_set_type> --json` before submission.

Field note (2026-05, moderation resilience): some shots can return `status=nsfw` with empty `result_url`; keep spare prompts and automatic shot-replacement logic so a 45–60s edit can still complete without manual restart.

Field note (2026-05, family/emotion creatives): prompts mentioning children (e.g., "father with daughter") can trigger false-positive `nsfw` moderation on `z_image`. Preferred fallback sequence:
1) retry with adult-safe wording (`adult son with mother`, `adult family reunion`),
2) remove child-specific terms while preserving emotional intent,
3) keep a generic safe fallback prompt to avoid pipeline abort.
Always handle `status!=completed` by retrying with alternate prompt before failing the batch.

## Notes
- No `mcp` subcommand in this CLI build — agent integration is **via skills**,
  not via an MCP server. The Higgsfield site advertises MCP for Claude Code /
  Cursor / Codex; current `0.1.40` of the CLI only exposes the
  `auth/account/generate/marketing-studio/marketplace-cards/model/upload/
  soul-id/product-photoshoot` subcommands.
- All renders run in Higgsfield's cloud — no GPU load on this RTX 2080.

## Reel production QA addendum (important)
- For vertical outputs, use `--aspect_ratio 9:16` (underscore parameter name). Do not use `--aspect-ratio`.
- `generate create --json` returns an **array** payload; parse `result_url` from the first item (`[0]`) after status=`completed`.
- If user requests "no subtitles", ensure subtitle filters are removed from final export.
- Always verify final audio is not cut: probe voice duration with `ffprobe`, and if video is shorter, extend last frame with `tpad` before final mux.
- Detailed checklist: `references/reel-postproduction-qa.md`.

## Lipsync-family model mapping pitfall (UI vs CLI/API)

When a user says "SyncLipsync2", "Kling Lipsync", "Infinite Talk", "Kling Avatar 2", "Wan 2.5 Speak", or "Higgsfield Speak 2", **do not assume those exact names are valid `job_set_type` values in this CLI**.

Required verification sequence before claiming support or failure:
1. Run `higgsfield model list --json` and `higgsfield model list --video --json`.
2. Probe candidate IDs with `higgsfield model get <job_set_type> --json`.
3. Validate accepted input params (`input_image`, `medias`, `audio`, etc.) from model schema.
4. Only then declare which path is available in this environment.

Reason: Higgsfield UI/marketing names can differ from CLI/API `job_set_type` identifiers per account/build.

For this operator, keep a session note in `references/lipsync-ui-vs-cli-mapping.md` and refresh it whenever model exposure changes.
