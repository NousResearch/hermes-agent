# Vision Tool Pitfalls

Claude Haiku 4.5 is the recommended aux vision model for Instagram
image analysis. It exists and supports `image_input: true`.

## Dual-section config trap

`config.yaml` has TWO `vision:` sections:

- `auxiliary.vision:` (line ~164) — the ACTUAL config used by the vision tool
- `vision:` (bottom of file) — a duplicate, NOT used

`hermes config set vision.model XYZ` targets the WRONG section (the unused
bottom one). Always use:

```bash
hermes config set auxiliary.vision.model claude-haiku-4-5-20251001
hermes config set auxiliary.vision.provider anthropic
```

Verify with: `grep -A3 'auxiliary.vision:' ~/.hermes/config.yaml`

## Session caching

The vision tool loads its model configuration at session start. Config
changes (even via `hermes config set`) do NOT take effect in the running
session. A `/reset` or new session is required after changing
`auxiliary.vision.*`.

## Model availability

| Model | Status | Image support |
|-------|--------|---------------|
| `claude-haiku-4-5-20251001` | ✅ Active | ✅ |
| `claude-sonnet-4-20250514` | ❌ 404 (retired) | — |
| `claude-3-5-haiku-20241022` | ❌ 404 (retired) | — |

Verify a model exists before configuring:
```bash
curl -s https://api.anthropic.com/v1/models/MODEL_ID \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"
```

## DeepSeek V4 incompatibility

DeepSeek V4 supports multimodal only through the Anthropic-format API
(`https://api.deepseek.com/anthropic`). The OpenAI-compatible endpoint
(`https://api.deepseek.com/v1`) rejects `image_url` content blocks.
Hermes `vision_analyze` sends images in OpenAI format — DeepSeek
cannot be used as `vision.provider`.

## When alt text is sufficient (skip vision)

Instagram carousel images include full slide text in the `<img alt>` attribute
when extracted via Playwright. Use vision_analyze only as a fallback when
alt text is empty.

## Direct Anthropic API (when vision tool is broken)

If the vision tool can't be reconfigured mid-session, call Anthropic directly:

```python
import base64, json
from urllib.request import Request, urlopen

with open('image.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = json.dumps({
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
        {"type": "text", "text": "Extract all text from this image."}
    ]}]
}).encode()

req = Request('https://api.anthropic.com/v1/messages', data=payload,
    headers={'x-api-key': api_key, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'})
resp = json.loads(urlopen(req, timeout=60).read())
print(resp['content'][0]['text'])
```
