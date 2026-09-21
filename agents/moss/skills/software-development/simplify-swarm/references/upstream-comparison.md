# Upstream Simplify-Code Comparison

Research conducted 2026-06-08 comparing 11 code simplification implementations
and the upstream Hermes Agent `simplify-code` skill (merged #41691, 2026-06-07).

## Sources Analyzed

| # | Implementation | Source | Key Contribution |
|---|---|---|---|
| 1 | Anthropic Official | claude-plugins-official | 5-principle model (preserve, standards, clarity, balance, scope) |
| 2 | nicknisi/essentials | nicknisi/claude-plugins | Dead code removal, AI slop detection, communication protocol |
| 3 | Navigator | alekspetrov/navigator | ROI scoring gate, config-driven, auto vs interactive matrix |
| 4 | Addy Osmani | addyosmani/agent-skills | Chesterton's Fence, language-specific guidance, Rule of 500 |
| 5 | OMC code-simplifier | Yeachan-Heo/oh-my-claudecode | LSP diagnostics, Failure Modes, structured output |
| 6 | OMC ai-slop-cleaner | Yeachan-Heo/oh-my-claudecode | 4-pass cleanup, writer/reviewer separation, UI checklist |
| 7 | GitHub Copilot gem | github/awesome-copilot | Structured JSON output, knowledge sources, reverse-dep order |
| 8 | ECC refactor-cleaner | affaan-m/everything-claude-code | knip/depcheck/ts-prune dead code detection, risk categorization |
| 9 | Meta OpenEnv | meta-pytorch/OpenEnv | TDD Red-Green-Refactor integration |
| 10 | WalletConnect | WalletConnect/skills | Early returns, constant extraction, single responsibility |
| 11 | Hermes upstream | NousResearch/hermes-agent | Parallel 3-agent batch dispatch, focus modifiers, dry-run |

## Upstream vs Simplify-Swarm — Key Deltas

### What Upstream Has That We Don't

1. **Focus modifiers** — "simplify focus on efficiency" runs only that reviewer.
   Good for targeted passes.
2. **TOCTOU detection** — Efficiency reviewer checks existence pre-checks.
3. **Dry-run as user modifier** — Cleaner UX than config toggle for one-off passes.
4. **Brevity** — 175 lines, zero ceremony. Our comprehensive version is heavyweight.

### What We Have That Upstream Doesn't

1. **Risk-tiered application** — SAFE auto-apply, CAREFUL verified, RISKY flagged.
2. **Detailed agent prompts** — 6-12k word reference files with exact detection patterns.
3. **Language-specific detectors** — Concrete TS/JS, Python, Go patterns.
4. **AI slop detection** — `as any` casts, defensive checks, inconsistent patterns.
5. **Stale state detection** — Duplicate stores, abandoned state machines.
6. **Silent failure detection** — Empty catch, ignored errors, propagation gaps.
7. **Structured JSON output** — Typed schema per agent vs freeform.
8. **Chesterton's Fence** — Mandatory git blame before removal.
9. **Config file** — Project-level tuning (.kensei/simplify.yaml).
10. **Integration matrix** — How it plugs into requesting-code-review, subagent-driven-development, etc.

## Focused Enhancement PR (Submitted to upstream #41691)

Five enhancements added to upstream `simplify-code`:
1. Risk-tiered application (SAFE/CAREFUL/RISKY)
2. Chesterton's Fence (git blame before removal)
3. AI slop detection in Quality reviewer
4. Silent failure detection in Efficiency reviewer
5. Structured output with confidence+risk tags
6. 3 new pitfalls (over-trusting tools, public contracts, error handling)

+45/-8 lines. Keeps 212-line compact spirit.

## Community Repo

https://github.com/Sahil-SS9/hermes-simplify-swarm

Standalone installable skill. MIT licensed. 6 files (~41k chars).
