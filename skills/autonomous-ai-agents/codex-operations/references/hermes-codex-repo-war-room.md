# Hermes/Codex Repository Adaptation War Room

Date: 2026-07-04  
Target repo: `hermes-agent`  
External clone workspace: `/tmp/hermes-repo-warroom-mNN06w`

## Decision summary

| Repository | Decision | Hermes/Codex seam | Rationale | Risk |
|---|---|---|---|---|
| `microsoft/playwright-mcp` | **Implement now** | `optional-mcps/playwright/manifest.yaml` | Official MCP server; direct fit for Hermes MCP catalog; useful for browser automation loops that need persistent page state and accessibility snapshots. | Browser automation can act on live accounts; default must avoid persistent profile state. |
| `yamadashy/repomix` | **Implement now** | `repomix-context-packaging` skill + Codex Ops workflow | Strong fit for Codex context packaging, token budgeting, repo handoffs, and broad code review. | Can accidentally package sensitive/noisy files if used carelessly. |
| `nizos/tdd-guard` | **Adapt principles / spike later** | `test-driven-development` + Codex Ops guidance | Valuable enforcement model, but implementation is Claude Code hook-specific and now points new work to Probity. Hermes should not vendor it directly. | Hook/runtime mismatch; possible API-key billing/security pitfalls. |
| `obra/superpowers` | **Research/adapt selectively** | Existing Hermes skills already adapted some patterns | Good methodology source; direct wholesale import would duplicate current skills and impose foreign harness assumptions. | Skill drift and duplicated doctrine. |
| `wshobson/agents` | **Watchlist/selective import only** | Potential future optional skill/agent catalog | Large catalog of agent personas; useful as inspiration, not a runtime integration. | Quality variance, prompt bloat, overlapping Hermes subagent model. |
| `affaan-m/ECC` | **Watchlist/research only** | Cross-harness workflow ideas | Broad harness OS with Hermes/Codex claims and security guide material; too overlapping and too large for direct import. | High maintenance burden; verify claims before trusting; avoid wholesale adoption. |
| `letta-ai/claude-subconscious` | **Spike later** | Memory/session reflection ideas | Interesting memory loop, but Hermes already has persistent memory, session search, skills, and cron jobs. | Claude-specific hook model; memory auto-write risk. |
| `smtg-ai/claude-squad` | **Reject for now** | None; duplicates delegation/worktrees | Multi-agent TUI overlaps Hermes delegation/Codex Ops; AGPL raises redistribution constraints. | License/architecture mismatch. |
| `hesreallyhim/awesome-claude-code` | **Reject as dependency; use as reading list only** | Manual research only | Useful index, but CC BY-NC-ND prevents modified integration into Hermes. | Noncommercial/no-derivatives license. |
| `multica-ai/andrej-karpathy-skills` | **Reject/watchlist** | Possible manual prompt inspiration | Small Claude/Cursor skill set; no visible license in clone summary, narrow direct value. | License uncertainty; low unique value. |

## Implemented slice

1. Added Playwright MCP to the curated optional MCP catalog with safe defaults:
   - pinned package: `@playwright/mcp@0.0.77`
   - launch: `npx -y @playwright/mcp@0.0.77 --isolated --headless`
   - no auth prompts
   - post-install warning about live-account browser automation

2. Added package metadata coverage and a regression test for MCP catalog manifests:
   - all `optional-mcps/*/manifest.yaml` files must parse
   - every optional MCP manifest must be declared under `[tool.setuptools.data-files]`
   - this also fixed the pre-existing packaged-data omission for `optional-mcps/unreal-engine`

3. Added `repomix-context-packaging` skill:
   - safe commands for temporary context bundles
   - narrow include examples
   - Codex handoff pattern
   - verification checklist and pitfalls

4. Updated `codex-operations` skill:
   - references `repomix-context-packaging` and `test-driven-development`
   - adds war-room guidance to package context before broad Codex launches and require TDD gates for non-spike work

## Deferred implementation tracks

- **Harness-agnostic TDD guard:** evaluate Probity/TDD Guard concepts against Hermes plugin hooks and Codex Ops telemetry. Do not import Claude Code hook scripts directly.
- **Skill/agent catalog curation:** evaluate `superpowers`, `wshobson/agents`, and ECC skill material one capability at a time. Import only specific, licensed, tested skills.
- **Memory-loop design:** compare `claude-subconscious` against Hermes memory/session-search/skill-curation rules. Do not enable automatic memory writes without explicit policy gates.

## Operating notes

- Prefer manifest/skill adapters over vendoring external runtimes.
- Keep browser automation isolated by default.
- Keep generated context bundles out of repos unless intentionally committed.
- Licenses matter: indexes or prompts under noncommercial/no-derivatives terms are research inputs, not integration sources.
