---
name: mealie
description: "Use when working with a self-hosted Mealie instance: recipe search, recipe imports, duplicate checks, meal planning, shopping lists, categories, tags, Mini Mealie browser-extension workflows, Mealie API actions, or Docker/self-hosting troubleshooting."
version: 1.0.0
author: Gerrit Jessen + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [mealie, recipes, meal-planning, shopping-list, self-hosted, api, xprivat]
    related_skills: [hermes-agent]
---

# Mealie

## Overview

Use this skill to help with Mealie, a self-hosted recipe manager and meal-planning system. It supports both guidance and direct API actions: searching recipes, importing URLs, testing scrapes, checking duplicates, inspecting user/API setup, and troubleshooting the Mini Mealie browser-extension workflow.

The bundled API helper is intentionally small and uses only Python standard-library modules so it can run in normal Hermes sessions without extra dependencies.

## When to Use

- The user asks to find, inspect, import, create, update, or troubleshoot Mealie recipes.
- The user asks about meal plans, shopping lists, categories, tags, households, or Mealie admin tasks.
- The user mentions a Mealie domain, API token, Mini Mealie, recipe scraping, failed imports, duplicate recipes, CORS, Docker, reverse proxies, or Hermes API actions for Mealie.
- The user wants both instructions and actual API execution.

Don't use this skill for generic cooking advice unless Mealie storage, import, planning, or API automation is involved.

## Configuration

Resolve Mealie connection settings in this order:

1. Explicit values provided by the user or command flags.
2. Environment variables: `MEALIE_BASE_URL` or `WXT_MEALIE_SERVER`; `MEALIE_API_TOKEN` or `WXT_MEALIE_API_TOKEN`.
3. Env files: a supplied `--env-file`, then `mini-mealie/.env.local`, then `.env.local`, then `~/.config/mealie.env`.

Use a base URL without a trailing slash, for example `https://mealie.example.com`.

Never print API tokens. Prefer environment variables or env files over passing tokens on the command line.

## API Helper

Run the helper from the skill directory:

```bash
python3 ~/.hermes/skills/xprivat/mealie/scripts/mealie_api.py whoami
python3 ~/.hermes/skills/xprivat/mealie/scripts/mealie_api.py search "pasta"
python3 ~/.hermes/skills/xprivat/mealie/scripts/mealie_api.py recent --limit 10
python3 ~/.hermes/skills/xprivat/mealie/scripts/mealie_api.py test-scrape "https://example.com/recipe"
python3 ~/.hermes/skills/xprivat/mealie/scripts/mealie_api.py import-url "https://example.com/recipe"
```

For less common endpoints:

```bash
python3 ~/.hermes/skills/xprivat/mealie/scripts/mealie_api.py raw GET /api/recipes
python3 ~/.hermes/skills/xprivat/mealie/scripts/mealie_api.py raw POST /api/households/mealplans --json '{"date":"2026-06-14"}'
```

Read `references/api.md` before raw calls, meal-plan mutations, shopping-list operations, or endpoints not covered by the helper.

## Recipe Workflows

For URL imports:

1. Use `test-scrape` first when the user wants a confidence check.
2. Use `import-url` for normal server-side imports.
3. If import fails on a paywalled or JavaScript-heavy page, explain that Mini Mealie HTML mode may work better because it sends rendered page HTML to `/api/recipes/create/html-or-json`.
4. Return the recipe slug and direct URL when available.

For search and duplicate checks:

1. Use `search` for user-supplied text queries.
2. Use `recent --limit 100` for duplicate checks by source URL.
3. Compare `orgURL` values after normalizing fragments, common tracking parameters, `www`, and trailing slashes.
4. Keep results compact: recipe name, slug, id, and `orgURL` when relevant.

## Mini Mealie Notes

The Mini Mealie browser extension uses these Mealie endpoints:

- `GET /api/users/self`
- `POST /api/recipes/create/url`
- `POST /api/recipes/create/html-or-json`
- `POST /api/recipes/test-scrape-url`
- `GET /api/recipes?search=...&perPage=...`

When debugging Mini Mealie, separate browser-only problems from API problems. A successful `whoami` from this helper proves the Mealie API and token work from the terminal, but it does not prove browser CORS is configured correctly.

## Common Pitfalls

1. **Missing token:** Ask the user to set `MEALIE_API_TOKEN` or `WXT_MEALIE_API_TOKEN`; do not request that they paste a token into chat unless unavoidable.
2. **Wrong base URL:** Check reverse-proxy subpaths and trailing slashes. The helper trims trailing slashes but does not guess path prefixes.
3. **Repeated imports:** Do not retry mutating import calls blindly after ambiguous failures; first search recent recipes for a created slug or matching `orgURL`.
4. **CORS confusion:** CLI success does not guarantee browser-extension success. CORS is enforced by browsers, not by Python CLI calls.
5. **Large raw responses:** Summarize recipe lists instead of dumping full payloads unless the user asks for raw JSON.

## Verification Checklist

- [ ] `whoami` succeeds before mutating calls.
- [ ] Base URL and token came from env/config, not printed output.
- [ ] Mutating operations match the user's explicit intent.
- [ ] Import results include slug and URL when available.
- [ ] Failures include HTTP status and concise response details.
