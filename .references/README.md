# Local Reference Snapshots

This directory is for local, ignored source-code and documentation snapshots that help agents work with fast-moving APIs, SDKs, and upstream projects.

Examples:

```text
.references/openai-agents-sdk/
.references/anthropic-docs/
.references/supabase-mcp/
.references/telegram-bot-api/
.references/hermes-upstream/
```

Rules:

- Do not commit downloaded reference trees.
- Do not store credentials, `.env` files, tokens, auth stores, or logs here.
- Keep only this README and `.gitkeep` tracked.
- Search a specific reference folder when current source/docs are needed; do not dump entire reference trees into prompts.

Distinction from skill references:

- `skills/<category>/<skill>/references/` contains committed support docs for that skill.
- `.references/` contains uncommitted local snapshots of upstream source/docs.
