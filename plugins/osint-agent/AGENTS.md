# osint-agent plugin rules

Unified OSINT brief stack: SitDeck + World Monitor Free + government RSS + MHLW + Computer Use + multilayer web search.

## Computer Use + multilayer (Phase D)

1. Enable: `hermes osint-agent stack enable` (includes toolsets `computer_use`, `web`, `search`).
2. Plan: call `osint_agent_computer_use_plan`, then execute via `computer_use` on https://worldmonitor.app/ and SitDeck.
3. Layers: call `osint_agent_multilayer_collect`, run L5 queries with `web_search`, merge into `osint_agent_brief`.
4. Prefer Free JSON / Playwright SitDeck crawl for bulk/cron; use CU for live UI / map verification only.
5. Never invent sources, echo SitDeck passwords, or proceed to paid checkout unless the user explicitly asks.
6. Fail soft per layer; keep open-source / non-classified only.

See `README.md` for operator commands and L1–L6 table.
