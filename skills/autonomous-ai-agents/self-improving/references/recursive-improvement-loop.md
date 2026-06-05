# Recursive improvement loop (Hermes)

Read-only bootstrap and audit — no automated writes to `.learnings/` or artifacts.

## Session start

1. `fleet_context_snapshot(summary=True)` or `town_brief()` — fleet + governance context
2. `self_improvement_snapshot(summary=True)` — memory tier health and proposals
3. `learnings_read(file="memory.md")` — HOT memory content
4. `knowledge_read(artifact="latest_state")` — operational truth
5. `knowledge_read(artifact="held_spec_ledger")` — active holds

## During work

- Log explicit corrections to `.learnings/corrections.md` (append-only).
- After significant tasks, use the self-reflection block in SKILL.md.

## Before changing HOT memory

1. Run `self_improvement_snapshot()` (full) or `python -m tools.self_improvement_audit --json`.
2. Apply only **operator-approved** edits to `memory.md`.
3. Knowledge layer wins for operational state; HOT is a bounded bootstrap (≤100 lines).

## Governance

Town-Hermes automated memory sync is **FROZEN**. MCP tools never write memory or knowledge files.
