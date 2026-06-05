# Higgsfield Lipsync Studio: UI names vs CLI/API IDs

## Why this exists
Higgsfield product/UI labels (e.g., in Lipsync Studio/blog posts) may not match the `job_set_type` names exposed by the local CLI/account at runtime.

## Verified workflow (authoritative)
1. Enumerate models:
   - `higgsfield model list --json`
   - `higgsfield model list --video --json`
2. Inspect candidate model schema:
   - `higgsfield model get <job_set_type> --json`
3. Confirm media input contract before generating:
   - Look for fields like `input_image`, `medias`, and whether audio/video/image are accepted.
4. For CLI generation, media flags accepted by tool wrapper are:
   - `--image`, `--start-image`, `--end-image`, `--video`, `--audio`
   - Each can be UUID or local file path (auto-uploaded by CLI).

## Session snapshot (May 2026)
User-provided functional family names (UI side):
- SyncLipsync2
- Kling lipsync
- Infinite talk
- Kling avatar 2
- Google veo 3
- Wan 2.5 speak
- Higgsfield speak 2

Local CLI-discovered video IDs included:
- `veo3`, `veo3_1`, `veo3_1_lite`
- `kling2_6`, `kling3_0`
- `wan2_6`, `wan2_7`
- `soul_cast`
- others (cinematic/marketing/seedance/minimax)

Observed mismatch:
- Direct `model get` probes for `synclipsync2`, `lipsync2`, `infinite_talk`, `kling_avatar_2`, `wan2_5_speak`, `higgsfield_speak_2` were not exposed in this CLI surface.
- This does **not** prove feature absence in Higgsfield globally; it only proves mapping mismatch for this environment/account/tool surface.

## API baseline (official docs)
- Base: `https://platform.higgsfield.ai`
- Submit: `POST /{model_id}`
- Status: `GET /requests/{request_id}/status`
- Cancel: `POST /requests/{request_id}/cancel`
- Auth: `Authorization: Key {API_KEY}:{API_SECRET}`

## Operational rule
Never promise "model X unsupported" until you:
- map UI label -> actual `job_set_type`/`model_id`, and
- verify schema accepts required media inputs for the requested workflow.