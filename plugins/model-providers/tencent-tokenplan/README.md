# Tencent TokenPlan

Routes Hermes through Tencent's [LKEAP](https://cloud.tencent.com/product/lkeap)
TokenPlan gateway to the **Hy3** (Hunyuan) model over an Anthropic
Messages-compatible endpoint.

## Setup

Add your API key to `~/.hermes/.env`:

```bash
TOKENPLAN_API_KEY=your-api-key
```

Then run:

```bash
hermes chat --provider tencent-tokenplan --model hy3
```

Aliases: `tokenplan`, `tencent-lkeap`.

## Configuration

| Env var | Purpose |
|---|---|
| `TOKENPLAN_API_KEY` | API key for the LKEAP TokenPlan gateway |
| `TOKENPLAN_BASE_URL` | Override the base URL (default: `https://api.lkeap.cloud.tencent.com/plan/anthropic`) |

The base URL ends with `/anthropic`; the Anthropic SDK appends `/v1/messages`
automatically, so do **not** include that suffix in overrides.
