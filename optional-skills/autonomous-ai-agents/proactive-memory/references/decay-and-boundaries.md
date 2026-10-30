# Behavioral state decay, and why the reminder rides a boundary

Background for the proactive-memory skill: what the paper proposes, what Hermes can and cannot
adopt from it, and where the reminder is allowed to appear.

## The failure the paper names

arXiv:2607.08716 (*Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents*, Wu et
al., Jul 2026) argues that long-horizon agents fail not because facts are missing from context but
because decision-relevant state stops influencing the next action as the trajectory grows —
"behavioral state decay". Their remedy is a lightweight memory agent watching the action agent: it
maintains a structured bank (status / knowledge / procedural) and injects a memory-grounded
reminder only when it is likely to change the next action. Reported: Terminal-Bench 2.0 37.6% →
45.9%, τ²-Bench 55.0% → 61.8% for Sonnet 4.5.

Two ideas are portable and two are not.

## Portable → this skill

| Paper idea | How the skill does it |
|---|---|
| Structured bank of status / knowledge / procedural entries | `scripts/memory_bank.py`, mirroring `todo_list` (status), `memory` (knowledge) and skill Pitfalls (procedural) |
| Selective intervention, not always-inject | `check --action` returns only entries whose trigger is present in the proposed action; an unrelated action returns none |
| Reminders grounded in real trajectory entries | every entry requires a `--trigger`; a triggerless (generic) entry is rejected |
| Keep the working set small | the bank is capped per kind, newest wins — an unbounded bank has already decayed |

## Not portable → deliberately declined

| Paper mechanism | Why Hermes will not do it in core |
|---|---|
| Inject the reminder into the next action-agent **prompt**, mid-trajectory | Hermes keeps the system prompt byte-stable for the life of a conversation and forbids a synthetic user message injected mid-loop (`agent/AGENTS.md`). Both break per-conversation prompt caching and strict role alternation — the project's first invariant. |
| A separate always-on memory agent watching every step via a second model call each turn | That is heavy, always-on **core** surface, paid on every turn. The Footprint Ladder puts new capability at the edges first; a per-boundary skill call is the honest cost. |

The issue (#99138) is labelled `needs-decision` precisely because the core-module version trades
against caching and alternation. That trade is a maintainer's call, not a skill's.

## Where a reminder is allowed to appear

Hermes already has legal places to surface memory-grounded content without touching a live prompt.
The skill uses these boundaries, each of which the agent reaches under its own control:

- **Before a committing action** — the agent calls `check` and reads the result as ordinary tool
  output, then decides. No context mutation.
- **At the start of a scheduled run** — each cron run is a fresh session (a fresh cache); loading
  the skill in the job rebuilds the bank and checks against it at turn one.
- **At session resume / a new session** — a new conversation is a new cached prefix by definition,
  so seeding the bank from `session_search` there costs nothing in cache terms.
- **At a delegated subtask boundary** — a subagent starts with its own context; the parent can pass
  the relevant reminders into the delegation prompt.

The throughline: the reminder is *pulled* by the agent at a boundary, never *pushed* into a
running turn. That is the difference between fighting decay and breaking the cache.

## The acceptance criteria, mapped

| #99138 criterion | Where it lands |
|---|---|
| Enable/disable per profile | The skill is optional (installed per profile) and loaded per job; nothing runs until it is. No core `config.yaml` key is added — that would be the `needs-decision` core module. |
| Reminders grounded in real entries, not generic advice | Enforced: `record` rejects a triggerless entry. |
| Selective beats always-inject | `check` returns a subset keyed to the action; an unrelated action returns none, where always-inject returns everything. |
| No regression on short-horizon tasks when disabled | An optional skill adds nothing to the prompt until installed, and even then only its ≤60-char description joins the skills index. |
