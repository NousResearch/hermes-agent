# Hermes Agent Contribution Rubric

This reference preserves the detailed contribution intent that used to live in `AGENTS.md`.

## What this document is for

Use this when reviewing or making Hermes Agent changes that may affect the core agent, model tool schema, contributor policy, or PR triage decisions.

## Core stance

Hermes ships a broad product surface, but the **core agent + model tool schema** is the narrow waist. Product reach can expand aggressively at the edges; core prompt/tool footprint stays conservative.

## What we want

- **Fix real bugs, well.** Reproduce on current `main`, identify the exact line where the symptom manifests, and fix the whole bug class, not just one call path.
- **Expand reach at the edges.** Platform adapters, providers, models, desktop/TUI/dashboard features, and integrations are welcome when wired into existing setup/config UX.
- **Refactor god-files into clean modules.** Declared extraction refactors can be large when they reduce concentrated complexity.
- **Keep the core narrow.** New model tools are expensive because every tool ships on every API call.
- **Extend, don't duplicate.** Reuse existing infrastructure before adding managers/hooks/modules.
- **Behavior contracts over snapshots.** Tests should assert invariants, not freeze model lists, config versions, or enumeration counts.
- **E2E validation.** For config propagation, security boundaries, remote backends, file/network I/O, and resolution chains, exercise the real path with temp `HERMES_HOME`.
- **Cache-, alternation-, and invariant-safe.** Preserve prompt caching, message role alternation, and byte-stable system prompts.
- **Contributor credit preserved.** Prefer cherry-picking/rebase-merging external work when possible.

## What we do not want

- Speculative infrastructure with no concrete consumer.
- New `HERMES_*` env vars for non-secret behavior. Use `config.yaml`; `.env` is for secrets.
- New core tools when terminal + file, CLI + skill, plugin, or MCP can do the job.
- Lazy-reading pagination for instructional tools that agents must read fully.
- Security fixes that destroy the feature they are meant to secure.
- Outbound telemetry/usage attribution without opt-in gating.
- Change-detector tests.
- Cache-breaking mid-conversation behavior.
- Dead code wired in without E2E proof.
- Plugins that modify core files.
- Third-party product integrations shipped into the core tree; publish standalone plugins instead.

## Verify the premise before calling something a bug

Common failure modes:

- A limitation may be intentional design, not a gap.
- A PR may be based on an incorrect mental model of runtime behavior.
- An omission may be load-bearing.
- A fix may overreach or revive an approach maintainers already rejected.

Rule: verify both the **claim** and the **intent** against the codebase before writing or merging a fix.

## The Footprint Ladder

Choose the highest, least-footprint rung that solves the problem:

1. **Extend existing code** — no new surface.
2. **CLI command + skill** — default for config/state/infra workflows; zero model-tool footprint.
3. **Service-gated tool (`check_fn`)** — appears only when prerequisites are configured.
4. **Plugin** — third-party, niche, user-specific capability; lives outside core.
5. **MCP server in the catalog** — structured tool capability without growing core tool schema.
6. **New core tool** — last resort; only for fundamental, broadly useful capabilities unreachable via terminal/file or MCP.

When 3+ PRs integrate the same category, design an ABC + orchestrator, keep the built-in as the first provider, and turn competing implementations into plugins.
