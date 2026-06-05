# Lipsync Studio flow mapping (session note, 2026-05)

## Why this exists
Lipsync Studio UI model names do not always match `higgsfield model list` names in the CLI. For API/automation work, capture the observed UI-to-flow mapping and validate with a real create smoke test.

## Confirmed UI preset labels and internal values (from live page select options)
Source page pattern:
- `https://higgsfield.ai/flow/avatar/preset/kling?flowId=<flow-id>`

Observed mappings:
- `Kling Avatars 2.0` -> `kling-speak`
- `Higgsfield Speak 2.0` -> `speak-hf-v2`
- `Wan 2.5 Speak` -> `speak-wan-2.5`
- `Wan 2.5 Speak Fast` -> `speak-wan-2.5-fast`
- `Google Veo 3` -> `speak-veo-preview`
- `Google Veo 3 Fast` -> `speak-veo-fast`
- `Infinite Talk` -> `infinite_talk`
- `Sync Lipsync 2 pro` -> `sync-so`
- `Kling Lipsync` -> `kling-lipsync`
- `Kling 2.6 Lipsync` -> `kling-2-6-lipsync`

## Workflow correction from user (important)
For long audio avatar output, use **Kling Avatars 2.0** with this order:
1. Upload audio first.
2. Link uploaded audio to avatar flow.
3. Link image.
4. Generate.

Do not assume prompt-only or image-first path is equivalent for long-form speech.

## Practical verification rule
Always run:
1. capability probe (`model get` / flow option inspection), then
2. real `create --wait` smoke test.

Cost checks alone are insufficient for media-input compatibility.