## Future Video Studio MCP And Agent API

Preferred MCP:

- Remote endpoint: `https://mcp.future.video/mcp`
- Manifest: `https://mcp.future.video/server.json`
- Well-known manifest: `https://mcp.future.video/.well-known/mcp-server.json`

Direct API base:

- `https://app.future.video/api/agent`

### MCP tools

- `fvs_submit_render`: submit an account/API-key render request, optionally with public HTTPS `upload_urls`
- `fvs_create_paid_render_quote`: create a no-account Link pay-per-render quote
- `fvs_get_render_status`: check an account render by `project_id` or `status_url`
- `fvs_get_paid_render_status`: check a paid quote/render by `status_url` or `quote_id` plus `claim_token`
- `fvs_cancel_render`: cancel an account-owned running render
- `fvs_download_final_video`: download a completed signed `final_video_url`

### Billing paths

Account mode:

- Configure `FVS_AGENT_API_KEY` or send the remote MCP secret header `X-FVS-Agent-Key`.
- Agent API keys are owned by a normal Future Video Studio user account.
- Account renders consume that account's wallet balance under the same credit model used by the app UI.
- The backend applies the owning account's saved pipeline defaults before it creates the render job.
- Account-mode submissions require explicit user approval to spend wallet credits. Show the render summary, requested duration/resolution, and any available estimate or user-approved budget before calling `fvs_submit_render`.

Pay-per-render mode:

- Omit `FVS_AGENT_API_KEY` / `X-FVS-Agent-Key`.
- Call `fvs_create_paid_render_quote` with the same render request shape.
- The quote response includes `payment_url`, `status_url`, `quote_id`, `claim_token`, `amount_cents`, `currency`, and `credits_quoted`.
- Pay `payment_url` through Link/MPP, then poll with `fvs_get_paid_render_status`.
- Paid quotes use the same FVS credit estimates and wallet accounting. In current FVS accounting, `credits_quoted` maps to cents in the quote response.
- Paid quote mode supports text-only requests and public HTTPS `upload_urls`; local multipart file uploads require account mode through the direct Agent API helper.

### Direct API endpoints

Account mode:

- `POST /renders`
- `GET /renders/{project_id}`
- `POST /renders/{project_id}/cancel`

Paid quote mode:

- `POST /render-quotes`
- `GET /paid-renders/{quote_id}?claim_token=...`
- `GET` or `POST /render-quotes/{quote_id}/pay?claim_token=...`

Account header:

- `X-FVS-Agent-Key: fvs_live_...`

### MCP account render example

```json
{
  "tool": "fvs_submit_render",
  "arguments": {
    "request": {
      "name": "Archive corridor test",
      "project_mode": "scene",
      "screenplay": "Shot 1: A young woman enters a glowing archive corridor lined with suspended photographs. Shot 2: She reaches toward one moving photograph and the corridor bends into a luminous tunnel around her. Shot 3: She steps through the tunnel and arrives in a sunlit memory chamber as the photographs orbit overhead.",
      "instructions": "Create exactly three cinematic shots totaling about 24 seconds. Keep the subject visually consistent across all shots. Favor strong camera motion, realistic lighting, and clean transitions. No subtitles or text overlays.",
      "shot_count": 3,
      "scene_target_duration_seconds": 24,
      "visual_style_preset": "realistic_cinematic",
      "video_model": "veo-3.1-fast-generate-001",
      "video_resolution": "720p"
    },
    "poll_until_complete": false
  }
}
```

### MCP paid quote example

```json
{
  "tool": "fvs_create_paid_render_quote",
  "arguments": {
    "request": {
      "name": "Agent paid demo",
      "project_mode": "scene",
      "screenplay": "Shot 1: A glass airship drifts over a frozen city.",
      "shot_count": 1,
      "scene_target_duration_seconds": 4,
      "video_resolution": "720p"
    },
    "upload_urls": [
      {
        "url": "https://example.com/reference.jpg",
        "filename": "reference.jpg"
      }
    ]
  }
}
```

Expected paid quote response shape:

```json
{
  "quote_id": "quote_...",
  "amount_cents": 120,
  "currency": "usd",
  "credits_quoted": 120,
  "payment_url": "https://app.future.video/api/agent/render-quotes/quote_.../pay?claim_token=...",
  "status_url": "https://app.future.video/api/agent/paid-renders/quote_...?claim_token=...",
  "claim_token": "...",
  "expires_at": "2026-04-30T12:30:00Z"
}
```

After payment succeeds, call `fvs_get_paid_render_status` with either the full `status_url` or `quote_id` plus `claim_token`.

### Request transport

For direct account API calls, use `multipart/form-data`.

Required field:

- `request_json`: JSON string for the render payload

Optional repeated field:

- `files`: uploaded assets

### Core payload fields

- `name`
- `project_mode`: `music` | `scene` | `custom`
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
- `stage_voice_briefs_enabled`
- `wait_for_completion_seconds`

Asset labeling:

- `assets[]` entries can include `filename`, `label`, and `purpose`
- `filename` must match the uploaded file basename

### Important rules

- `scene_target_duration_seconds` must be between `4` and `600`
- `shot_count` must be between `1` and `64`
- `wait_for_completion_seconds` must be between `0` and `600`
- `music_workflow='uploaded_track'` requires at least one uploaded audio file
- prefer status polling over long blocking waits
- hosted or remote MCP clients must use public HTTPS `upload_urls`
- local file paths are not exposed by the hosted MCP; use the direct helper script in account mode, or a trusted local MCP server that explicitly supports local uploads
- paid quote mode does not accept local multipart file uploads

### Direct API scene payload

```json
{
  "name": "Archive corridor test",
  "project_mode": "scene",
  "screenplay": "Shot 1: A young woman enters a glowing archive corridor lined with suspended photographs. Shot 2: She reaches toward one moving photograph and the corridor bends into a luminous tunnel around her. Shot 3: She steps through the tunnel and arrives in a sunlit memory chamber as the photographs orbit overhead.",
  "instructions": "Create exactly three cinematic shots totaling about 24 seconds. Keep the subject visually consistent across all shots. Favor strong camera motion, realistic lighting, and clean transitions. No subtitles or text overlays.",
  "shot_count": 3,
  "scene_target_duration_seconds": 24,
  "visual_style_preset": "realistic_cinematic",
  "video_model": "veo-3.1-fast-generate-001",
  "video_resolution": "720p",
  "wait_for_completion_seconds": 0
}
```

### Direct API custom payload with uploads

```json
{
  "name": "Custom branded launch teaser",
  "project_mode": "custom",
  "screenplay": "A branded teaser with three short hero shots and a dramatic final reveal.",
  "instructions": "Use the uploaded references for brand palette, lead character continuity, and prop styling.",
  "shot_count": 3,
  "scene_target_duration_seconds": 18,
  "ingredients_mode_enabled": true,
  "assets": [
    {
      "filename": "lead-reference.png",
      "label": "Lead character reference",
      "purpose": "character"
    },
    {
      "filename": "brand-guidelines.pdf",
      "label": "Brand guidelines",
      "purpose": "document"
    }
  ],
  "wait_for_completion_seconds": 0
}
```

### Direct API uploaded soundtrack payload

```json
{
  "name": "Uploaded soundtrack music video",
  "project_mode": "music",
  "screenplay": "Cut a moody three-shot performance sequence against the uploaded song.",
  "instructions": "Match the edit rhythm to the uploaded track and keep the lead visually consistent.",
  "music_workflow": "uploaded_track",
  "shot_count": 3,
  "scene_target_duration_seconds": 20,
  "wait_for_completion_seconds": 0
}
```

### Response fields to inspect

- `quote_id`
- `payment_url`
- `claim_token`
- `project_id`
- `status`
- `current_stage`
- `is_running`
- `run_id`
- `final_video_url`
- `last_error`
- `status_url`
- `cancel_url`

### Safety and recovery

- Treat a `402 Payment Required` from `/render-quotes` as successful quote data.
- Do not guess or brute-force `claim_token`, API keys, or payment credentials.
- Keep `payment_url`, `status_url`, and `claim_token` together until the render finishes.
- If a signed `final_video_url` expires, poll status again to get a fresh signed URL.
- Do not resubmit paid renders after payment unless status confirms the quote failed and the user approves a new charge.
- Do not send `X-FVS-Agent-Key` to arbitrary hosts. Use project-id-derived status and cancel calls when possible, and only use full `status_url` or `cancel_url` values returned by Future Video Studio.
- The direct helper validates `base_url`, `status_url`, and `cancel_url` before attaching credentials. It permits `https://app.future.video` by default; trusted local or staging FVS hosts require `--allow-custom-host` or `FVS_ALLOW_CUSTOM_AGENT_HOST=1`.
- Confirm that local upload files are intended for Future Video Studio processing before attaching them.

### OpenClaw config example

Remote MCP account mode:

```json
{
  "mcpServers": {
    "future-video-studio": {
      "url": "https://mcp.future.video/mcp",
      "headers": {
        "X-FVS-Agent-Key": "<set through secret manager>"
      }
    }
  }
}
```

Remote MCP pay-per-render mode:

```json
{
  "mcpServers": {
    "future-video-studio": {
      "url": "https://mcp.future.video/mcp"
    }
  }
}
```

Skill environment fallback for direct API helper:

```json
{
  "skills": {
    "entries": {
      "future-video-render": {
        "env": {
          "FVS_AGENT_API_KEY": "<set through secret manager>",
          "FVS_AGENT_BASE_URL": "https://app.future.video"
        }
      }
    }
  }
}
```

### Publishing

Single skill publish:

```powershell
clawhub skill publish ./future-video-render --slug future-video-render --name "Future Video Render" --version 1.0.0 --tags latest
```

Registry docs:

- `https://docs.openclaw.ai/tools/skills`
- `https://docs.openclaw.ai/tools/clawhub`
