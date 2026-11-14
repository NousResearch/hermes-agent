# OpenRouter Image API (gpt-image family)

Verified 2026-08. For OpenAI-compatible image generation when you have an OpenRouter key but no direct OpenAI image key.

## Discovery
- List image-capable models: `POST https://openrouter.ai/api/v1/images/models`, or filter the models list:
  `curl -s https://openrouter.ai/api/v1/models -H "Authorization: Bearer $KEY" | jq -r '.data[].id' | grep -i image`
- OpenAI image models observed on OpenRouter:
  - `openai/gpt-5.4-image-2` — the "GPT Image 2" model on OpenRouter (quality-first)
  - `openai/gpt-5-image` — higher quality / pricier
  - `openai/gpt-5-image-mini` — cheaper/faster
  - (also `openai/gpt-image-1` / `openai/gpt-image-2` in docs examples, and Google Gemini-image family)
- There is NO model literally named `gpt-image-2` on the /models list; the "Image 2" naming maps to `gpt-5.4-image-2`. Confirm with the user before committing to a specific slug.

## Endpoint & request
`POST https://openrouter.ai/api/v1/images`
```json
{
  "model": "openai/gpt-5.4-image-2",
  "prompt": "...",
  "n": 1,
  "quality": "high",            // auto|low|medium|high
  "aspect_ratio": "2:3",        // or size:"2048x2048" / resolution:"2K"
  "input_references": [         // image-to-image: guide generation
    {"type": "image_url", "image_url": {"url": "<data:image/jpeg;base64,...> OR https URL>"}},
    {"type": "image_url", "image_url": {"url": "<2nd reference>"}}
  ]
}
```
- Reference images via `input_references` accept HTTP(S) URLs or base64 data URLs. For local files use data URLs.
- `n` up to 10 but many providers cap at 1; default 1.
- `output_format` png/jpeg/webp; `background` auto/transparent/opaque.
- `provider.only` / `provider.order` / `provider.options.<slug>` for routing.

## Response
```json
{
  "created": 1748372400,
  "data": [{"b64_json": "<base64>", "media_type": "image/png"}],
  "usage": {"total_tokens": 0, "cost": 0.13}
}
```
Decode `data[0].b64_json` to bytes and write to file. `usage.cost` gives per-call cost.

## Billing (important)
- **All-or-nothing**: a generation is either completed and billed in full, or fails (502) and is NOT billed. No partial charges.
- Observed cost: `gpt-5.4-image-2`, `quality=high`, `aspect_ratio=2:3`, 1 ref image ≈ **$0.19/image**. Cost reported per-image in `usage.cost`.
- Failed/cancelled runs billed as $0.

## Batch scripting recommendations
- Send `input_references` as base64 data URLs built from local files.
- Generate into clearly named outputs: `ghost_<COLOR>.png` then `indossato_<POSA>_<COLOR>.png`.
- Guard each step on existence of its OUTPUT file (idempotency), with retries (e.g. 3 attempts, backoff) and per-call cost accumulation.
- Use the images endpoint timings generously (each gen ~60–150s at high quality).
