# Hermes Workspace Integration

## Current Architecture (post-2026-05-15)
Instead of six persistent tmux workers, the system now runs a **signal monitor daemon** alongside a reduced tmux swarm. The daemon handles continuous competitive monitoring (pricing diffs, RSS feeds, page checks) without LLM token burn. Hermes workers query its SQLite database when they run. Cloudflare Browser Rendering is used selectively for JS-heavy pages, with 24-hour caching. A source maintenance job auto-heals dead monitor URLs weekly.

## Pricing Change Detection
The daemon uses `PricingExtractor` to strip HTML noise and hash only pricing-relevant text (dollar amounts, plan names, per-seat pricing, contact-sales signals). Full-page SHA hashing produced constant false positives.

## Key Rules
- Treat Built LMS prompt files as role instructions until they are represented by Workspace roster/profile/wrapper/tmux runtime state
- GitHub is implementation evidence
- Linear is product intent
- Obsidian wiki is durable synthesis