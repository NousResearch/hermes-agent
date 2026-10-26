# Model discovery — never hardcode slugs

Model availability changes over time. Always resolve the current list from the
live API instead of hardcoding slugs:

```bash
python3 scripts/deapi.py models                  # everything
python3 scripts/deapi.py models --type image     # by task
python3 scripts/deapi.py models --type tts --json  # full objects (limits, voices)
```

Raw call:

```bash
curl "https://api.deapi.ai/api/v2/models?per_page=100" \
  -H "Authorization: Bearer $DEAPI_API_KEY" -H "Accept: application/json"
```

## Model object fields that matter

- `slug` — the string to pass as `model` in requests
- `inference_types` — array of task types the model serves (both short
  `txt2img` and kebab `text-to-image` spellings occur across docs; the helper
  accepts either)
- `info.limits` — min/max width/height/steps, `resolution_step`, frames, fps,
  duration, text length (varies by type)
- `info.features` — `supports_guidance`, `supports_negative_prompt`,
  `supports_steps`, `supports_last_frame`, `supports_voice_clone`, …
- `info.defaults` — the model's recommended parameter values (the helper uses
  these when you omit `--steps` / `--guidance` / `--frames`)
- `languages` — TTS only: language → voices (`name`, `slug`, `gender`)
- `loras` — optional LoRA list for image models

## Example slugs (illustrative only — verify against the live list)

These appeared in official docs at the time of writing. Treat them as
*examples of what slugs look like*, not as guaranteed-available models:

- Image: `Flux_2_Klein_4B_BF16` (FLUX.2 Klein — fixed 4 steps, no guidance,
  256–1536 px), `Flux1schnell`, `ZImageTurbo_INT8`
- Image edit: `Flux_2_Klein_4B_BF16` (≤3 input images), `QwenImageEdit_Plus_NF4`
- TTS: `Kokoro` (54+ voices, e.g. `af_bella`)
- Transcription: `WhisperLargeV3`
- OCR: `Nanonets_Ocr_S_F16`
- Background removal: `Ben2`
- Upscale: `RealESRGAN_x2`, `RealESRGAN_x4`
- Music: `ACE-Step-v1.5-turbo`
- Video: `Ltx2_19B_Dist_FP8`
- Embeddings: `Bge_M3_FP16` (1024 dims)
