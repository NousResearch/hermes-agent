# Smart Model Router Skill

A Hermes Agent skill that routes tasks to the optimal model based on task intensity, maximizing free tier usage and minimizing paid token burn.

## Quick Start

1. Install the skill:
   ```bash
   hermes skills install smart-model-router
   ```

2. Verify detection:
   ```bash
   python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py --check
   ```

3. Sync routing table:
   ```bash
   python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py --sync
   ```

## Prerequisites

- Python 3.11+
- `pyyaml` package (`pip install pyyaml`)

## Security Notice

**NEVER add secrets (API keys, tokens, passwords) to `config.yaml`.** Store all secrets exclusively in `~/.hermes/.env`.

The skill's detection scripts intentionally do NOT read `.env` or check API key presence. Provider detection is driven entirely by `config.yaml` (primary provider, fallback providers, custom providers). If a provider is listed in config but credentials are missing, Hermes's built-in failover handles the 401 automatically — no need for the scripts to check.

This design keeps secret values out of script memory and avoids the security risk of parsing `.env` files.

## Auto-Detection

When you add a new provider to `config.yaml` (in `model.provider`, `fallback_providers`, or `custom_providers`), run:

```bash
python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py --sync
```

The new provider is automatically detected and added to the routing table with the right tier assignments.

## Routing Tiers

| Tier | Task Types | Cost Target |
|------|-----------|-------------|
| **Light** | Lookups, confirmations, formatting, monitoring | FREE |
| **Medium** | Code writing, refactoring, docs, research | Free tier or cheapest paid |
| **Heavy** | Large features, architecture, complex debugging | Best model (Claude Opus) |

## Provider Support

16+ providers in the detection catalog: Anthropic, OpenRouter, OpenCode Zen/Go, LM Studio (local), Groq, Nous Portal, OpenAI Codex, Google Gemini, DeepSeek, xAI Grok, Kimi/Moonshot, MiniMax, AWS Bedrock, NVIDIA NIM, and custom providers.

## Scripts

- `scripts/model-router.py` — Quick CLI model picker (`--task light|medium|heavy`)
- `scripts/detect-providers.py` — Provider auto-detection and routing table sync (`--check`, `--sync`, `--watch`, `--patch`)

## References

- `references/config-recommendations.md` — Copy-paste config.yaml snippets
- `references/openrouter-free-models.md` — Current OpenRouter free models list

## License

MIT
