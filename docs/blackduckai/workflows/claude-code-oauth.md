# Claude Sonnet via Claude Code OAuth

BlackDuckAi's preferred Claude lane in Hermes uses Claude Code OAuth / Claude Pro-Max credential storage instead of an Anthropic API key.

## Current default-profile config

```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6
```

Credential behavior:

- Hermes resolves Anthropic credentials from Claude Code's credential store, e.g. `~/.claude/.credentials.json`.
- The active Hermes default profile clears `ANTHROPIC_TOKEN` and `ANTHROPIC_API_KEY` so stale API-key values do not shadow Claude Code OAuth.
- `ANTHROPIC_API_KEY` is legacy/pay-per-token fallback only.

## Setup / repair commands

Install or update Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

Log in through Claude Code:

```bash
claude
# or, if a setup-token flow is needed:
claude setup-token
```

Set Hermes to use Anthropic + Sonnet:

```bash
# TOP local helper: sets model.provider/default and clears stale ANTHROPIC_* env entries.
python C:/Users/black/AI_Project/scripts/setup/set-claude-code-oauth.py

# Manual equivalent:
hermes config set model.provider anthropic
hermes config set model.default claude-sonnet-4-6
```

If the profile previously used an API key, clear stale Anthropic env values instead of storing secrets in docs:

```bash
# Preferred via Hermes internals / model flow:
hermes model
# choose Anthropic → existing Claude Code credentials / OAuth → claude-sonnet-4-6
```

## Verification

```bash
hermes auth list
hermes chat --provider anthropic --model claude-sonnet-4-6 -q 'Reply exactly: CLAUDE_OAUTH_OK' --toolsets safe --quiet
```

Expected:

```text
anthropic: claude_code oauth
CLAUDE_OAUTH_OK
```

## Model/auth matrix for BlackDuckAi

- GPT-5 / orchestration: `openai-codex` via Codex/ChatGPT OAuth; no OpenAI API key required.
- GPT-Image-2 / ad images: `image_gen.provider: openai-codex`; no OpenAI API key required.
- Claude Sonnet / executive language, copywriting, QA: `anthropic` via Claude Code OAuth; no Anthropic API key required.
- Gemini / research and Veo: API-key backed unless Hermes adds OAuth support for that lane.
- Grok/xAI: API-key backed.

## Safety

Never commit or document credential values: no API keys, OAuth access tokens, refresh tokens, setup tokens, cookies, or page tokens.
