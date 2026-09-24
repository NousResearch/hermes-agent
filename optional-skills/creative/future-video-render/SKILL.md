---
name: future-video-render
description: Create hosted Future Video Studio renders.
version: 1.0.0
author: Future Video Studio
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [creative, video, ai-video, mcp, rendering, future-video-studio]
    category: creative
    related_skills: [kanban-video-orchestrator, hyperframes, blender-mcp, concept-diagrams, songwriting-and-ai-music, youtube-content, mcporter, native-mcp]
    homepage: https://future.video
    mcp_server: https://mcp.future.video/mcp
---

# Future Video Render

Create Future Video Studio renders through the hosted FVS MCP server.

Use this skill for multi-shot scenes, music videos, product teasers, reference-guided custom productions, account-wallet renders, Link pay-as-you-go quotes, status polling, cancellation, and final signed video URLs.

The MCP endpoint is `https://mcp.future.video/mcp`.

For Hermes MCP setup details, load `references/hermes-mcp-config.md`.
For request fields and response shapes, load `references/api.md`.

## Prerequisites

- Prefer the hosted MCP server at `https://mcp.future.video/mcp`.
- Account mode uses `FVS_AGENT_API_KEY` or the MCP secret header `X-FVS-Agent-Key`.
- Treat account keys as wallet-backed credentials. Show a concise render summary and get explicit approval before spending wallet credits.
- Pay-as-you-go mode omits the API key and uses `fvs_create_paid_render_quote`.
- Never ask for raw card details. Link payment happens through the returned `payment_url`.

## When to Use

Use this skill when the user asks to:

- generate a polished AI video from a written prompt, screenplay, or shot list
- create a multi-shot video with continuity across shots
- make a product launch teaser, brand video, narrative short, or music video
- use public HTTPS image, audio, PDF, or style-guide URLs as references
- get a pay-as-you-go quote before rendering
- submit an account-wallet render with an FVS Agent API key
- poll, cancel, or retrieve a Future Video Studio render

Do not use this skill for local-only HTML or code-rendered videos that should be built directly in the workspace. Use `hyperframes`, `blender-mcp`, `concept-diagrams`, or direct FFmpeg workflows when those are a better fit.

## Hermes Setup

Before submitting or quoting a render, check whether the FVS MCP server is configured in Hermes.

Recommended Hermes config:

```yaml
mcp_servers:
  future_video_studio:
    url: "https://mcp.future.video/mcp"
```

For account mode, add the API-key header:

```yaml
mcp_servers:
  future_video_studio:
    url: "https://mcp.future.video/mcp"
    headers:
      X-FVS-Agent-Key: "${FVS_AGENT_API_KEY}"
```

If the MCP tools are unavailable, tell the user to add the config above to `~/.hermes/config.yaml` and restart Hermes. Do not ask the user to paste API keys into chat.

## Tool Names in Hermes

Hermes prefixes MCP tools with the configured server name. If the server is named `future_video_studio`, the important tools are typically:

- `mcp_future_video_studio_fvs_submit_render`
- `mcp_future_video_studio_fvs_create_paid_render_quote`
- `mcp_future_video_studio_fvs_get_render_status`
- `mcp_future_video_studio_fvs_get_paid_render_status`
- `mcp_future_video_studio_fvs_cancel_render`
- `mcp_future_video_studio_fvs_download_final_video`
- `mcp_future_video_studio_fvs_example_render_request`

If the server uses another name, adjust the prefix accordingly. In normal reasoning, prefer the tool descriptions over hardcoding a prefixed name.

## How to Run

1. Convert the user's brief into an FVS render request.
2. Ask for clarification only when missing details affect cost, legality, or feasibility.
3. Choose the billing path:
   - account mode if `FVS_AGENT_API_KEY` is configured and the user approves spending wallet credits
   - pay-as-you-go mode if no key is configured or the user wants a visible price first
4. For account mode, show a concise render summary and get explicit approval before calling `fvs_submit_render`.
5. For pay-as-you-go mode, call `fvs_create_paid_render_quote`, return the `payment_url`, and explain that payment happens through Link.
6. Poll with the matching status tool until the job completes, fails, halts for review, or the user asks to stop.
7. Return the final signed `final_video_url` exactly as returned by FVS.

## Request Shape

Use `project_mode`:

- `scene` for screenplay or shot-list videos without an uploaded soundtrack
- `music` for soundtrack-led music videos
- `custom` for brand, product, or reference-heavy productions

Useful fields:

- `name`
- `project_mode`
- `screenplay`
- `instructions`
- `additional_lore`
- `visual_style_preset`
- `visual_style_custom_instructions`
- `scene_target_duration_seconds`
- `shot_count`
- `ingredients_mode_enabled`
- `image_model`, `image_resolution`
- `video_model`, `video_resolution`
- `music_model`

Hosted MCP clients must use public HTTPS `upload_urls` for assets. Local file paths are not visible to the hosted MCP. For trusted local-file account workflows only, use the bundled helper script in `scripts/future_video_render.py`.

## Verification

Poll with the matching status tool until the render completes, fails, halts for review, or the user asks to stop.

Inspect:

- `status`
- `current_stage`
- `is_running`
- `final_video_url`
- `last_error`
- `payment_url`, `quote_id`, and `claim_token` for paid quotes

Treat the job as terminal when `status` is `completed` or `failed`, `current_stage` is `halted_for_review`, or the job is not running and has no active queued or running state.

Return `final_video_url` promptly when present. If the signed URL expires, poll status again for a fresh result URL.

## Pay-As-You-Go Flow

Use pay-as-you-go when the user does not have an FVS API key or wants to approve a one-off payment.

1. Call `fvs_create_paid_render_quote` with the render request and any public HTTPS `upload_urls`.
2. Return the quote amount and `payment_url`.
3. Tell the user to pay through Link or the approved payment surface.
4. After payment, poll with `fvs_get_paid_render_status` using either the returned `status_url` or `quote_id` plus `claim_token`.
5. Return `final_video_url` when the render completes.

Never ask for raw card details.

## Account/API-Key Flow

Use account mode only when `FVS_AGENT_API_KEY` is configured.

Before submitting:

- summarize the video request
- include duration, shot count, resolution, and selected models when known
- state that the render will spend FVS wallet credits
- ask for explicit approval

Then call `fvs_submit_render`. Poll with `fvs_get_render_status`.

## Safety And Cost Rules

- Never submit an account render without explicit user approval.
- Never create a second paid render after a failed or halted render unless the user approves the extra spend.
- Never invent or rewrite final video URLs.
- Report exact FVS error messages if a render fails or halts.
- Use status polling instead of long blocking waits.
- Keep `scene_target_duration_seconds` between 4 and 600.
- Keep `shot_count` between 1 and 64.

## Local Helper Fallback

The bundled Python helper is for account-mode direct Agent API calls when a trusted local-file workflow needs multipart uploads:

```bash
python ${HERMES_SKILL_DIR}/scripts/future_video_render.py submit --request-file request.json --file reference.png --poll
```

It uses only the Python standard library and reads `FVS_AGENT_API_KEY` from the environment. It does not accept API keys on the command line. Prefer the hosted MCP tools whenever they are available.
