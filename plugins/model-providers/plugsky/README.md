# Plugsky provider

OpenAI-compatible API at `https://api.plugsky.com/v1`.

- Signup and keys: https://plugsky.com/
- Docs: https://plugsky.com/docs
- Free tier: `plugsky-micro` and `plugsky-lite` (100 requests/day, 50K tokens/day, 1 concurrent, 32K input / 2K output; no card).
- Agent loops: Hobby at $8/month gives 1,000 requests/day, 2M tokens/day, 4 concurrent, 120 RPM, 64K input / 8K output and the full 39-model catalogue.

Models are discovered live from `/v1/models`; the bundled fallbacks are the two free IDs.
