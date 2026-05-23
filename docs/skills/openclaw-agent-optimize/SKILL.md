# Agent Optimization Skill

Tune agent workspaces for cost-aware routing, parallel-first delegation, and lean context.

## Default Posture

This skill is **advisory first**. It produces:

- audit -> options -> recommended plan -> exact patch -> rollback -> verification
- No persistent mutations without explicit approval.

## Quick Start

### 1\) Full audit \(safe, no changes\)

> Audit my agent setup for cost, reliability, and context bloat. Output a prioritized plan with rollback notes. Do NOT apply changes.

### 2\) Context bloat / transcript noise

> My agent context is bloating. Identify the top offenders and propose the smallest reversible fixes first.

### 3\) Routine/delegation posture

> Review my routines for delegation efficiency. Which could be consolidated, simplified, or run less frequently?

## What Good Output Looks Like

- Executive summary
- Top drivers \(cost, context, reliability, operator friction\)
- Options A/B/C with tradeoffs
- Recommended plan \(smallest safe change first\)
- Exact proposals + rollback + verify

## Safety Contract

- Do not mutate persistent settings without explicit approval.
- Do not create/update/remove scheduled actions without explicit approval.
- If optimization reduces monitoring coverage, present options and require choice.
- Before any approved change, show:
  1. Exact change
  2. Expected impact
  3. Rollback plan
  4. Post-change verification

## High-ROI Optimization Levers

### 1\) Output discipline for automation

Make maintenance routines truly silent on success. Only surface errors and state changes. Use `[@suppress-notification]` for routine completions that don't need user attention.

### 2\) Separate work from notification

Do the work quietly; notify out-of-band with a short human receipt. Keep interactive context lean.

### 3\) Skill discipline

Keep skills concise and load-bearing. Move long runbooks into companion reference files. SKILL.md should be the entry point; detail goes in reference files read on demand.

### 4\) Ambient skill surface reduction

A common hidden tax is too many always-visible skills in the catalog.

- Prefer on-demand activation via `use_skill` over ambient injection
- Audit whether all installed skills are actively used
- Trim unused skills before tuning routine tool lists

### 5\) Memory hygiene

- Audit global vs routine-specific memories for duplication
- Remove stale memories that reference deprecated behavior
- Keep memories actionable and specific, not vague preferences

### 6\) Routine consolidation

- Look for routines that could share triggers or be merged
- Identify routines running more frequently than needed
- Check for redundant tool grants across routines

### 7\) Measure optimizations authoritatively

For Town: check routine run frequency, token usage via `get_analytics`, and skill catalog size.
For OpenClaw: prefer fresh-session `/context json` or equivalent receipts over "feels better."

### 8\) Verification-first ops hygiene

After any approved optimization, verify:

- Core functionality still works
- Recall/behavior did not degrade
- New session actually picks up the change
- Rollback path is proven, not theoretical

## Audit Workflow

### Town Environment

1. Audit memories: `get_memories()` \+ per-routine memories for active routines
2. Audit skills: `town_ls skills://` \- identify unused or oversized skills
3. Audit routines: `list_routines()` \- check for redundancy, frequency, tool bloat
4. Audit analytics: `get_analytics(period_days=7)` \- identify high-cost routines
5. Recommend the smallest viable change first

### OpenClaw/Hermes Environment

1. Audit rules + memory: keep restart-critical facts only
2. Audit skill surface: trim ambient specialists before touching tool surface
3. Audit transcripts/noise: silence cron and heartbeat success paths
4. Audit model routing and delegation posture
5. Recommend the smallest viable change first
6. Verify on a new session when skill/bootstrap snapshotting exists

## Notes

- Some runtimes snapshot skills/config per session - start new session after changes
- Prefer short SKILL.md + companion reference files for long runbooks
- If context bloat is the main complaint, audit ambient skills first



## External Landscape (May 2026)

### OpenClaw Security Posture
- Texas A&M SUCCESS Lab (arXiv:2603.27517, April 2026): Systematic taxonomy of 470 advisories against OpenClaw, organized by architectural layer (7 layers) and attack type (5 techniques)
- Three independently Moderate/High-severity advisories compose into a complete unauthenticated RCE path from LLM tool call to host process
- Exec allowlist bypass via line continuation, busybox multiplexing, and GNU long-option abbreviation
- Malicious skill distributed through plugin channel executed two-stage dropper within LLM context, bypassing exec pipeline
- Dominant pattern: per-layer, per-call-site trust enforcement rather than unified policy boundaries

### Hermes Agent (Competitive Frame)
- Nous Research, launched February 2026, MIT license, $70M funded ($1B valuation)
- 153K GitHub stars (as of May 16, 2026 - single dated data point)
- Core differentiator: self-learning memory system with persistent Skill Documents (Markdown files created after 5+ tool calls)
- Self-evolving skill loop is governance-incompatible with CCFT unless every skill mutation is versioned, reviewed, and routed through candidate-note / isolation-test / approval path
- Recommendation: Monitor, do not adopt. hermes_claw_migrate command indicates Nous targeting OpenClaw installed base.

### Governance Artifacts Landed (PR #286, merged May 16, 2026)
- governance/AGENT_ROUTING_POLICY.md - Tier 0-4 routing policy (300 lines)
- governance/STATUS.md - Enforcement gap made explicit
- governance/HASH_ROTATIONS.md - Empty rotation log with required fields
- Compliance memo finalized: "Why the DEM 27-Agent Fleet Is Insulated from Model-Output-as-Control-Signal Failures"

### Cost and Performance Context
- Local LLM breakeven vs frontier APIs: 3-6 months on DGX Spark at moderate utilization
- Against cheap open-weight cloud APIs ($0.30/1M tokens): local never breaks even within hardware life
- Qwen 2.5 Coder 32B (Q4) scored 9.3/10 on agent tool calling, beating Claude Sonnet 4.5 API (8.6/10) - but requires 24GB+ VRAM
- DEM Tier 6 decision (local LLMs deferred) validated by benchmark data on 16GB hardware
