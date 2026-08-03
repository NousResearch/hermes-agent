# Adding a New Language to Hermes Desktop

## Quick Start

1. Copy the English source file as your template:
   ```bash
   cp src/locales/en.json src/locales/{your-code}.json
   ```

2. Translate the values in your new file. Group by section prefix:
   - `appearance.*` — Settings > Appearance
   - `settings.*` — Settings sidebar
   - `chat.*` — Chat interface
   - `commandCenter.*` — Command center
   - `composer.*` — Message composer
   - `gateway.*` — Gateway settings
   - `model.*` — Model settings & auxiliary tasks
   - `sessions.*` — Sessions list
   - `onboarding.*` — First-run setup
   - `config.*` — Config panel labels & descriptions
   - `mcp.*` / `keys.*` — MCP & API Keys settings
   - `common.*` — Shared labels (Save, Cancel, etc.)
   - `nav.*` — Navigation
   - `tool.*` — Tool output
   - `errors.*` / `updates.*` / `boot.*` — System messages

3. Add your language's self-name to the `language.{code}` entry

4. Open a PR with just your new `{code}.json` file

## Auto-Discovery

The i18n system uses `import.meta.glob` — any `.json` file in `src/locales/` is automatically detected. The language selector in Settings > Appearance will pick it up without any code changes.

## Translation Guidelines

- Keep `{variable}` placeholders intact (e.g., `{count}`, `{name}`, `{time}`)
- Do NOT translate language self-name keys (`language.en`, `language.ja`, etc.)
- Do NOT translate provider names in `keys.*` descriptions
- Use natural, conversational tone
- Test by building and running: `npm run build && npm run pack`

## Current Status

| Language | Code | Keys Translated | Coverage |
|----------|------|----------------|----------|
| English (source) | en | 857/857 | 100% |
| Simplified Chinese | zh-CN | 843/857 | 98.4% |
| Japanese | ja | 79/857 | 9.2% |
| Korean | ko | 79/857 | 9.2% |
| German | de | 58/857 | 6.8% |
| Spanish | es | 58/857 | 6.8% |
| French | fr | 58/857 | 6.8% |
