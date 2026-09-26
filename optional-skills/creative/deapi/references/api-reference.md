# deAPI v2 REST — condensed reference

Full docs: https://docs.deapi.ai (OpenAPI: https://docs.deapi.ai/openapi-v2.json)

## Auth & bases

- REST v2 base: `https://api.deapi.ai/api/v2`
- OpenAI-compatible gateway: `https://oai.deapi.ai/v1` (images, speech,
  transcriptions, embeddings, videos; **no** `/chat/completions` — deAPI hosts
  no LLMs)
- Headers: `Authorization: Bearer <DEAPI_API_KEY>` + `Accept: application/json`
- Keys start with `dpn-sk-`. Rate-limit headers: `X-RateLimit-Limit`,
  `X-RateLimit-Remaining`, `X-RateLimit-Type` (`minute`|`daily`), `Retry-After`.

## Async job pattern (all generation endpoints)

1. `POST /api/v2/<endpoint>` → `{"data":{"request_id":"<uuid>"}}`
2. Poll `GET /api/v2/jobs/{request_id}` →
   `{"data":{"status":"pending|processing|done|error","progress":0-100,"result_url":...,"result":...,"results_alt_formats":{...}}}`
3. `result` carries text output (transcription/OCR); `result_url` carries file
   output. **Result URLs expire after ~24 h — download immediately.**

Alternatives to polling: `webhook_url` in the request body (HMAC signature in
`X-DeAPI-Signature`), WebSockets (Pusher protocol at `soketi.deapi.ai:443`,
channel `private-client.{client_id}`). OCR/transcription also accept
`return_result_in_response`.

Every POST endpoint has a price estimator sibling: `POST <endpoint>/price`.

## Endpoints

| Task | Endpoint | Body | Notes |
|---|---|---|---|
| List models | `GET /api/v2/models` | query: `per_page`, `page`, `filter[inference_types]=txt2img,...` | Returns `slug`, `inference_types`, `info.limits`, `info.features`, `info.defaults`, `languages` (TTS voices), `loras` |
| Image gen | `POST /api/v2/images/generations` | JSON: `prompt, model, width, height, guidance, steps, seed` (+`negative_prompt`, `loras`) | |
| Image edit | `POST /api/v2/images/edits` | multipart: `prompt, model`, `image` (or `images[]`, ≤3 for Klein) | |
| OCR | `POST /api/v2/images/ocr` | multipart: `image` (≤10 MB), `model`, `language?`, `format: text\|json` | |
| Background removal | `POST /api/v2/images/background-removals` | multipart: `image, model` | transparent PNG out |
| Upscale | `POST /api/v2/images/upscales` | multipart: `image, model` | scale is per-model (x2/x4) |
| TTS | `POST /api/v2/audio/speech` | multipart: `text, model, lang, speed, format, sample_rate` (+`mode`: `custom_voice`\|`voice_clone`\|`voice_design`) | voices per model in `languages` |
| Transcription | `POST /api/v2/audio/transcriptions` | multipart: `model, include_ts` + exactly one of `source_url` \| `source_file` | URL: YouTube, X, Twitch, Kick, TikTok. Files: audio ≤20 MB, video ≤50 MB |
| Music | `POST /api/v2/audio/music` | multipart: `caption, model, lyrics, duration (10-600s), inference_steps, guidance_scale, seed, format` (+`bpm, keyscale, timesignature, reference_audio`) | `lyrics: "[Instrumental]"` for no vocals |
| Text-to-video | `POST /api/v2/videos/generations` | JSON: `prompt, model, width, height, guidance, steps, seed, frames` (+`fps, negative_prompt`) | |
| Image-to-video | `POST /api/v2/videos/animations` | multipart: `prompt, first_frame_image, model, width, height, guidance, steps, seed, frames, fps` (+`last_frame_image, negative_prompt`) | `fps` is required by the live API despite being marked optional in the spec |
| Embeddings | `POST /api/v2/embeddings` | JSON: `input` (string or array), `model` | |
| Prompt booster | `POST /api/v2/prompts/enhancements` | multipart: `prompt, type` (v2 dot notation, e.g. `images.generations`, `videos.animations`) + `model_slug?, negative_prompt?, image?` (image required for `images.edits`/`videos.animations`) | returns enhanced `prompt` (+`negative_prompt`) |
| Balance | `GET /api/v2/account/balance` | — | `{"data":{"balance": 19.72}}` |

## Errors

- `401/403` — bad key or no credit; check https://app.deapi.ai
- `422` — parameter outside the model's limits; body explains which
- `429` — rate limited; honor `Retry-After`
