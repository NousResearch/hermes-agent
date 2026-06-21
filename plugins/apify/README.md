# Apify Actor Tools

Bundled plugin that brings [Apify](https://apify.com/) Actors into Hermes. Apify
hosts 20,000+ ready-made Actors for web automation and data extraction
(Instagram, YouTube, Google Maps, LinkedIn, Amazon, and more). The agent can
discover the right Actor, inspect its input schema, run it, and collect
structured results.

## Tools

| Tool | What it does |
|------|--------------|
| `apify_discover` | Search the Apify Store by keyword, or fetch a specific Actor's input schema + README by `actor_id`. |
| `apify_start` | Fire-and-forget batch Actor starts (up to 10 per call). Returns run refs immediately so the agent keeps reasoning while Actors run. |
| `apify_collect` | Poll run statuses and return completed dataset results, wrapped in `EXTERNAL_UNTRUSTED_CONTENT` markers. Supports `limit` / `may_have_more` pagination. |

## Setup

1. Create an Apify account and get a token at
   <https://apify.com/account/integrations>.
2. Run `hermes tools`, open **Apify Actors**, enable the toolset, and paste your
   token. (The token is stored in `~/.hermes/.env` as `APIFY_API_TOKEN` and is
   never sent to the model.)

The `apify` toolset is **off by default**. The three tools register on startup
but stay invisible to the model until `APIFY_API_TOKEN` is set (runtime
`check_fn` gate).

## Architecture

This is a bundled `kind: backend` plugin (auto-loads, no opt-in), modeled on the
Spotify plugin. Tool registration goes through the plugin API
(`ctx.register_tool()`), so the Apify tools never enter `_HERMES_CORE_TOOLS`. The
`apify-client` SDK is installed on demand via `tools.lazy_deps` (`search.apify`)
the first time an Actor runs.

| File | Purpose |
|------|---------|
| `__init__.py` | `register(ctx)` — wires the three tools via `ctx.register_tool()`. |
| `tools.py` | Handlers + JSON schemas. |
| `client.py` | Lazy `apify-client` import, token validation, module-level cache. |
