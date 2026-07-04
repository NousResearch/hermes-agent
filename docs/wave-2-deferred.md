# Wave 2 Deferred Items

These items were considered during Wave 1 token-leak mitigation
(`ralplan-hermes-token-leaks` v3, 2026-05-01) and intentionally deferred.

## Phase 4.3 — tool description compaction

Multiline Markdown stuffed into tool `description` fields can often be
60% shorter without losing semantic meaning. Top-5 fattest schemas:
`delegate_task` (6.4KB), `terminal` (4.7KB), `skill_manage` (3.0KB).

**Gate**: ship only after a tool-selection regression suite reports
≤2% selection-accuracy delta on a 50-prompt eval set.

**Eval set**: 50 prompts × known-correct-tool labels, run through both
pre- and post-compaction tool definitions, compare LLM-selected tool
match rate.

**Owner**: Wave 2 contributor.

## Phase 5 — SKILL.md content cache (provider-cache integration)

Provider-side cache via `cache_control` breakpoints on the SKILL.md
block of the cron prompt would eliminate ~2.2M tokens/day of re-injection
for bluenode-style profiles with stable skill lists. Requires Anthropic
backend; bluenode currently runs `google/gemma-4-31b-it` on Nous Portal.

**Path forward**: revisit when bluenode migrates to Claude or when Nous
Portal ships gemma cache breakpoints.

## Phase 6 — precheck_command schema for cron

Top-level `precheck_command` field on cron jobs that runs BEFORE LLM
invoke. Returns JSON `{"shouldInvoke": bool, "reason": str}`. Skip LLM
entirely when shouldInvoke=false.

This eliminates the cron LLM-mandatory architecture (the largest leak
vector). Requires upstream RFC + cross-team approval.

**Target**: 2026-05-15 RFC filing.

## Vector 11 — conversation history compression on cron

Long-running cron sessions accumulate full message history per
iteration. Existing `context_compressor` only fires when threshold hit;
short cron runs (≤10 iterations) miss compression. A streaming-aware
compression for cron-mode would address this.

**Target**: Wave 3.
