# API Reference

## Primary Endpoint: Pollinations.ai (Flux)

**Cost:** Completely free. No API key required. No sign-up. No credit limits.

### Endpoint
```
https://image.pollinations.ai/prompt/{prompt}?width=1024&height=1024&model=flux&nologo=true
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string (URL-encoded) | required | The combined Style + Content prompt |
| `width` | int | 1024 | Image width in pixels |
| `height` | int | 1024 | Image height in pixels |
| `model` | string | `flux` | Model to use (`flux` recommended for best quality) |
| `nologo` | bool | `true` | Removes Pollinations watermark |
| `seed` | int | random | Set for reproducibility across generations |
| `enhance` | bool | `false` | Model-side prompt enhancement (may fight your carefully crafted style) |

### Recommendations
- **Always use `model=flux`** — superior texture, lighting, and prompt adherence.
- **Always use `nologo=true`** — clean output.
- **Use `seed` for series** — ensures structural consistency across a batch; the Style Prompt handles aesthetic consistency.
- **Avoid `enhance=true`** — the model's auto-enhancement may override your Style Prompt's specific directions.
- **URL-encode the prompt** — spaces become `%20`, special characters must be encoded.
- **Prompt length** — Flux handles long prompts well (8000+ chars effective). Concatenate Style + Content freely.

### Example Call (Python)
```python
import requests
from urllib.parse import quote

style_prompt = "Perspective & Composition: Extreme macro, flat-on subject isolation..."
content_prompt = "A raven perched on a craggy outcrop, wings half-spread, facing left."
full_prompt = f"{style_prompt} {content_prompt}"

url = f"https://image.pollinations.ai/prompt/{quote(full_prompt)}?width=1024&height=1024&model=flux&nologo=true&seed=42"
response = requests.get(url, timeout=120)
with open("output.jpg", "wb") as f:
    f.write(response.content)
```

### Timeout
Pollinations.ai can take 10–60 seconds per image. Set timeout to at least 120 seconds.

### Rate Limiting
No documented rate limit, but rapid sequential requests may queue. Allow ~2 seconds between calls for reliability.

---

## Alternative Endpoints (If Pollinations.ai Is Down)

| Service | Free Tier | Model | Notes |
|---------|-----------|-------|-------|
| Hugging Face Inference API | Generous free tier | Stable Diffusion XL, Flux | Can have cold starts; requires free API key |
| Together AI | $5 free credits (one-time) | FLUX.1-schnell | Credit-based, eventually runs out |
| Replicate | Some free models | Various | Limited free predictions |

These are **not** recommended as primary — Pollinations.ai is the only truly zero-cost, zero-auth option.