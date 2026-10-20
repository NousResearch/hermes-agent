# Signal Monitor Architecture

## Architecture (post-2026-05-15)
- **Signal monitor daemon**: Continuous competitive monitoring (pricing diffs, RSS feeds, page checks) without LLM token burn
- **Reduced tmux swarm**: Hermes workers query daemon's SQLite database when they run
- **Cloudflare Browser Rendering**: Used selectively for JS-heavy pages, 24-hour caching
- **Source maintenance job**: Auto-heals dead monitor URLs weekly

## Agent Naming Convention
Use descriptive names in human-facing artifacts:
- swarm13 = Built Chief of Staff
- swarm14 = Built Product Ops
- swarm15 = Built Market Intel
- swarm16 = Built GTM Content / Claim Safety
- swarm17 = Built Discovery / Outreach
- swarm18 = Built Knowledge Base / Scribe

## Pricing Change Detection
PricingExtractor strips HTML noise, hashes only pricing-relevant text (dollar amounts, plan names, per-seat pricing, contact-sales signals). Full-page SHA hashing produced constant false positives.

## Reset Procedure
See references/pricing-change-detection.md for edge cases and reset procedure.