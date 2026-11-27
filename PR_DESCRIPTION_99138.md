# feat(skills): proactive-memory — fight behavioral state decay without breaking the cache

Refs #99138

## Summary

#99138 proposes a **Proactive Memory Agent** (arXiv:2607.08716): a separate lightweight agent that
watches the action agent, maintains a status/knowledge/procedural memory bank, and **injects a
memory-grounded reminder into the next action-agent prompt** when it is likely to change the next
action. Reported gains: Terminal-Bench 2.0 37.6% → 45.9%, τ²-Bench 55.0% → 61.8%.

The problem it targets is real and Hermes-relevant: in long sessions and cron automations, agents
repeat failed commands, drop earlier requirements, and re-diagnose solved errors. But the proposed
mechanism — step 3, *inject the reminder into the live prompt mid-trajectory* — is exactly what
Hermes forbids in core:

> Never alter past context … or rebuild the system prompt mid-conversation. … never a synthetic
> user message injected mid-loop. (`agent/AGENTS.md`)

Mid-trajectory injection breaks **per-conversation prompt caching** (the project's first invariant)
and **strict role alternation**. That is why the issue is correctly labelled `needs-decision`: the
core-module version trades against caching, and that trade is a maintainer's call.

This PR delivers the *portable* half of the paper as an **optional skill** on primitives Hermes
already ships, with the reminder surfaced at boundaries the agent controls — never pushed into a
running turn. It adds one stdlib-only helper for the piece with no native home: the relevance gate.

```bash
hermes skills install official/autonomous-ai-agents/proactive-memory
```

## What is portable, and what is declined

| Paper idea | This PR |
|---|---|
| Structured status / knowledge / procedural bank | `scripts/memory_bank.py`, mirroring `todo_list` (status), `memory` (knowledge), skill Pitfalls (procedural) |
| Selective intervention, not always-inject | `check --action` returns only entries whose trigger appears in the proposed action; an unrelated action returns none |
| Reminders grounded in real trajectory entries | every entry **requires** a `--trigger`; a triggerless (generic) entry is rejected |
| Keep the working set small | the bank is capped per kind, newest wins |
| **Inject into the next action-agent prompt, mid-trajectory** | **Declined in core** — breaks caching + alternation. The reminder is *pulled* by the agent at a boundary (before a committing action, at cron-run start, at session resume, at a delegation), never *pushed* into a live turn. |
| A separate always-on agent, a second model call per step | **Declined** — heavy always-on core surface paid every turn; the Footprint Ladder puts capability at the edges first |

`references/decay-and-boundaries.md` carries this table plus the exact boundaries where a
memory-grounded reminder is legal without touching a live prompt.

## Why a skill and not a `proactive_memory` core module

| Rule (root / `agent/AGENTS.md`) | The issue as written | This PR |
|---|---|---|
| Per-conversation caching is sacred | mid-trajectory prompt injection | reminder pulled at agent-controlled boundaries; zero live-context mutation |
| Strict role alternation | inject a synthetic reminder mid-loop | no synthetic message; `check` output is ordinary tool output the agent reads |
| Footprint Ladder | new core module + second-agent model call each step | Rung 2: optional skill + one deterministic helper |
| Extend, don't duplicate | a new memory bank beside memory | the bank is a compact index over `todo_list` / `memory` / skill Pitfalls, not a replacement |
| No new `config.yaml` behavior key unless core | per-profile enable flag in `config.yaml` | the skill is optional (installed per profile, loaded per job); no core key added |

## The helper — a deterministic relevance gate

`scripts/memory_bank.py` has three subcommands (`record`, `check`, `list`) over a JSONL bank.
The gate is the whole point, and it is deterministic so it is testable:

- **`record`** adds a status/knowledge/procedural entry. It **rejects an entry with no `--trigger`** —
  that enforces the "grounded, not generic advice" criterion at the input. `--supersedes <id>`
  retires an evolved entry (status changes), and the bank is **capped per kind, newest wins**, so it
  stays compact — an unbounded bank has already decayed, which is the failure being fought.
- **`check --action "<next step>"`** returns only the entries whose trigger string appears in the
  proposed action, **procedural first** (the order most likely to change the next action). An
  unrelated action returns `count: 0`. That silence is the contract that makes selective
  intervention *different* from re-injecting the whole bank ("always inject").
- **`list`** shows the active entries.

Output is stable-ordered; bad input exits 2 and writes nothing.

> The subcommand is `record`, not `update`: the test runner's live-system guard blocks any
> subprocess whose string contains both `hermes` and the token `update` (it guards `hermes update`),
> and the script path contains `hermes-agent`. `record` sidesteps the false positive and reads
> better besides.

## What changed

| Path | Change |
|---|---|
| `optional-skills/autonomous-ai-agents/proactive-memory/SKILL.md` | New skill, 125 lines. Modern section order; three procedures (build the bank, gate the reminder, recall before rebuild), each with a Verification step |
| `…/scripts/memory_bank.py` | Stdlib-only helper — `record` / `check` / `list`, the relevance gate, per-kind cap, supersede |
| `…/references/decay-and-boundaries.md` | Portable vs declined table, the legal boundaries, and the acceptance criteria mapped |
| `tests/skills/test_proactive_memory_skill.py` | 5 tests (8 cases) running the real script offline |
| `website/docs/reference/optional-skills-catalog.md`, `website/sidebars.ts`, `website/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-proactive-memory.md` | Output of `website/scripts/generate-skill-docs.py` (+1 line in each listing file, plus the new page) |

No core Python, config keys, env vars, tools, hooks or dependencies were added.

## Acceptance criteria (#99138), mapped

| Criterion | Where it lands |
|---|---|
| Enable/disable per profile | Optional skill, installed per profile, loaded per job; nothing runs until installed. (A core `config.yaml` key would be the `needs-decision` core module.) |
| Reminders grounded in real entries, not generic advice | `record` rejects a triggerless entry (exit 2). |
| Selective beats always-inject | `check` returns an action-keyed subset; an unrelated action returns none, where always-inject returns everything. |
| No regression on short-horizon tasks when disabled | An optional skill adds nothing to the prompt until installed; then only its 54-char description joins the skills index. |
| Evaluate on a long-horizon eval subset | **Not in this PR** — see Honest gaps. |

## Design notes

- **Prompt-cache safe.** The reminder is read as tool output at a boundary the agent already
  reaches; the system prompt stays byte-stable and no synthetic user message is injected mid-loop.
- **Profile-aware.** The bank lives under `"$HERMES_HOME/proactive-memory/<task>.jsonl"`; the local
  terminal exports `HERMES_HOME`. No hardcoded home.
- **Every cited surface was checked against this tree** — `memory`/`todo_list`/`session_search`/
  `cronjob_manage` in `toolsets.py`, the caching + alternation invariants in `agent/AGENTS.md`, and
  the live-system guard in `tests/_fixtures/live_system_guard.py`.

## Honest gaps (possible focused follow-ups, not in this PR)

1. **No benchmark run.** The paper's headline is a Terminal-Bench 2.0 / τ²-Bench improvement from
   *mid-trajectory injection*, which this PR deliberately does not do. Measuring the boundary-time
   variant needs an eval harness and is a separate effort; the deterministic gate is unit-tested
   instead of benchmarked here.
2. **The reminder is agent-pulled, not automatic.** By design — an automatic push is the core
   module. If maintainers decide the caching trade is worth it for a bounded case (e.g. only at a
   *new* turn boundary via a tool result, never the system prompt), that is the `needs-decision`
   follow-up this skill is the safe interim for.

## Tests

`tests/skills/test_proactive_memory_skill.py` runs the real script via `subprocess`, stdlib +
pytest only, no network:

| Test | Contract |
|---|---|
| `test_gate_returns_only_action_relevant_entries` | A related action returns the matching entry (a strict, non-empty subset of the bank); an unrelated action returns `count: 0` |
| `test_procedural_reminder_leads_regardless_of_insertion_order` | On a shared trigger, `check` orders procedural → status → knowledge whatever the insertion order |
| `test_supersede_retires_the_old_entry` | After `record --supersedes`, the old entry is gone from `list` and `check` |
| `test_bank_stays_capped_per_kind` | 10 records with `--cap 8` leave 8 active; the two oldest are retired, the newest survives |
| `test_invalid_input_is_rejected_without_writing` (×4) | Empty text, a triggerless entry, an unknown `--supersedes` id and an empty `--action` each exit 2 with a `memory_bank:` message and write no file |

```
$ scripts/run_tests.sh tests/skills/test_proactive_memory_skill.py \
    tests/skills/test_authoring_standards.py tests/skills/test_optional_skill_self_paths.py \
    tests/skills/test_skill_docs_contract.py tests/skills/test_skill_document_contracts.py \
    tests/skills/test_skill_pages_match_shipped_skills.py
✓ tests/skills/test_proactive_memory_skill.py (8✓, 4.0s)
✓ tests/skills/test_authoring_standards.py (1464✓, 310.5s)
=== Summary: 6 files, 1510 tests passed, 0 failed, 1 skipped (100% complete) in 310.6s ===

$ ruff check optional-skills/autonomous-ai-agents/proactive-memory tests/skills/test_proactive_memory_skill.py
All checks passed!
```

### End-to-end: install through the real hub path into a temp `HERMES_HOME`

```
$ HERMES_HOME=$(mktemp -d) hermes skills install official/autonomous-ai-agents/proactive-memory --yes
Scan: proactive-memory (official/builtin)  Verdict: SAFE
Decision: ALLOWED — Allowed (builtin source, safe verdict)
Installed: autonomous-ai-agents/proactive-memory
Files: SKILL.md, references/decay-and-boundaries.md, scripts/memory_bank.py
```

### Manual: the relevance gate

```
$ python scripts/memory_bank.py record bank.jsonl --kind procedural \
    --text "uv sync fails behind the proxy unless UV_NATIVE_TLS=1" --trigger "uv sync,uv pip"
{"updated": "m1", "kind": "procedural", "active_entries": 1}
$ python scripts/memory_bank.py check bank.jsonl --action "run uv sync to install deps"
{ "reminders": [ {"id": "m1", "kind": "procedural", "matched_triggers": ["uv sync"], ...} ], "count": 1 }
$ python scripts/memory_bank.py check bank.jsonl --action "git status"
{ "reminders": [], "count": 0 }          # selective: unrelated action, no reminder
$ python scripts/memory_bank.py record bank.jsonl --kind status --text "done" --trigger "the"
# (generic trigger is allowed but noisy; a triggerless record is rejected:)
$ python scripts/memory_bank.py record bank.jsonl --kind status --text "done" --trigger " , ,"
memory_bank: an entry needs at least one --trigger: a reminder must be grounded, not generic  # exit 2
```

## Risk

Low. Opt-in optional skill, no core change, no new dependency. The only files outside the skill are
the generated docs and its test.

## Credit

The proposal, the paper framing, and the acceptance criteria are by @Lexus2016 in #99138. This PR
adapts the portable half onto Hermes's existing memory primitives and states plainly which half
cannot land in core without a maintainer decision on the caching trade.
