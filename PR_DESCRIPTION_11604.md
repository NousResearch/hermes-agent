# feat(skills): self-improvement-loops — the "Living Subsystem" loops on existing primitives

Refs #11604

## Summary

#11604 proposes a **Living Subsystem Framework**: 12+ autonomous subsystems (Governance,
ScienceLoop, ReflectiveEvolution, FitnessBuilder, Knowledge, Reasoning, ...) in a new core
`subsystems/` package. Each would persist JSON under `~/.hermes/` and expose `.run()` / `.status()`
for cron and slash commands.

The loops it describes are worth having. Nearly every piece already ships as a Hermes primitive,
though, and building it as core modules would conflict with the project's design rules (below).
This PR delivers the loops as an **optional skill** that routes each one to the surface that
already exists. It adds one small helper script for the only piece with no native home: weighted
fitness scoring.

```bash
hermes skills install official/autonomous-ai-agents/self-improvement-loops
```

## Why a skill and not `subsystems/`

| Rule (root `AGENTS.md`) | Proposal as written | This PR |
|---|---|---|
| Footprint Ladder: extend existing code → CLI + skill → … → new core surface | 12 new core modules, new slash commands | Rung 2: an optional skill driving existing `hermes` commands and tools |
| "Extend, don't duplicate" | `governance_audit.json`, `goals.json`, `learnings.json` sit alongside approvals, kanban, `/goal`, memory and skills | Uses those stores, so future sessions and the curator actually read the state |
| No speculative infrastructure | ABC + registry with no in-tree consumer | No new hooks, ABCs or registries |
| Never hardcode `~/.hermes` | `self.home = Path.home() / ".hermes"` | Everything goes through `$HERMES_HOME`, so it works with profiles |
| No new core tool when terminal + skill suffice | Agent-facing `Governance().block_action(...)` | Existing `hooks.pre_tool_call` + `approvals.deny` |

## Subsystem → primitive map

The same table ships in `references/subsystem-map.md`.

| Proposed | Existing surface the skill uses |
|---|---|
| **Governance**: pre-flight block | `approvals.mode` / `approvals.deny` / `command_allowlist`; `hooks.pre_tool_call` shell hooks (exit 2 or `{"action":"block"}`, `fail_closed`); `hermes approvals test -- <cmd>`; the built-in dangerous-command detector and hardline floor |
| **Governance**: audit trail | The session DB stores every tool call; `hermes sessions export --session-id ID --format jsonl`; `hermes logs --component tools`; optional `hooks.post_tool_call` |
| **Governance**: dangerous code | `hermes plugins enable security-guidance` |
| **ScienceLoop** | Kanban: one task per hypothesis, `kanban_comment` per experiment, `kanban_complete(metadata={"verdict": ...})`, `kanban_link` for revisions, grouped by `tenant`. For a single measurable hypothesis: `/goal` with a `verification:` contract + `/goal gate add` |
| **ReflectiveEvolution** | `session_search` to recall past failures; `/refine`, `/learn`, `skill_manage` (Pitfalls), `memory` for cross-task facts |
| **FitnessBuilder** | **Gap**: new `scripts/fitness.py` + a `cronjob_manage` schedule that loads the skill |
| Knowledge / Reasoning | `memory`, skills + `related_skills`, `hermes journey`; decisions recorded as kanban comments or goal contracts |
| Evolution / Reflection / Metacognitive / Self-model / Identity | `/curator`, `/refine`, `SOUL.md`, `USER.md` |
| Memory tiering | Built-in memory + memory providers (`hermes memory setup`) |
| Orchestrator | `delegate_task`, kanban swarm, `dynamic-workflow` skill |
| Quota display | `hermes insights`, `/usage` |
| `.run()` / `.status()` | `cronjob_manage` / `hermes cron` with `skills=["self-improvement-loops"]`; `hermes status`, `hermes cron status`, `hermes kanban stats`, `hermes hooks doctor` |

## What changed

| Path | Change |
|---|---|
| `optional-skills/autonomous-ai-agents/self-improvement-loops/SKILL.md` | New skill. Modern section order; four procedures (hypothesis, lessons, guardrails + audit, fitness), each with a Verification step |
| `…/scripts/fitness.py` | Stdlib-only helper. Validates the spec (positive weights summing to 1, unique dimensions) and the scores (exactly the spec's dimensions, each in [0, 1]). Prints the weighted score and per-dimension breakdown. With `--log`, appends to a JSONL history and reports `previous` / `delta` for the same spec. Bad input exits 2 without writing |
| `…/references/subsystem-map.md` | The full proposal → primitive table |
| `tests/skills/test_self_improvement_loops_skill.py` | Two tests that run the real script offline (below) |
| `website/docs/reference/optional-skills-catalog.md`, `website/sidebars.ts`, `website/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-self-improvement-loops.md` | Output of `website/scripts/generate-skill-docs.py` (+1 line in each listing file, plus the new page) |

No core Python, config keys, env vars, tools or hooks were added.

### Why the fitness math is a script

`OPENAI_MODEL_EXECUTION_GUIDANCE` already tells several model families to use tools for
arithmetic, because traces showed financial math done in prose going wrong. A weighted score
that's compared across weeks must be reproducible, so the skill tells the model *never* to
compute it in prose. Skill authoring standard #6 says to ship a helper for logic like this rather
than have the model write it inline on every call.

## Design notes

- **Prompt-cache safe.** It's an optional skill: nothing enters the system prompt until the
  user installs it, and even then only the ≤60-char description is added to the skills index.
- **Profile-aware.** Specs and history live under `"$HERMES_HOME/fitness/"`. The local terminal
  exports `HERMES_HOME` (`tools/environments/local.py`). The skill never names a hardcoded home.
- **Changing guardrails needs the user's consent.** The skill has the agent show the exact
  `config.yaml` YAML and ask before changing `approvals.*` or `hooks.*`, and test with
  `hermes approvals test` / `hermes hooks test` before calling it done.
- **Passes the skills-guard scan.** An early draft of the map listed the covered commands
  literally (`rm -rf /`, `mkfs`, `curl | sh`), and the hub scan rated it DANGEROUS on install.
  The map now describes those command classes in words, and the scan verdict is **SAFE**.
- **Every cited command was checked against code**, not recalled. For example, the cron tool is
  `cronjob_manage`, kanban has no tags field (so `tenant` groups the ledger), and there is no
  `hermes goal` CLI (only `/goal`).

## Honest gaps (possible focused follow-ups, not in this PR)

1. **No cross-session tool-call audit listing.** `hermes sessions export` works per session, and
   the tool calls are in `state.db`, but there's no `hermes sessions tool-calls --since 7d`.
   That would be a small CLI addition on the existing DB, and worth doing only if someone asks.
2. **No kanban labels.** `tenant` works for grouping but isn't a real tag.

## Tests

`tests/skills/test_self_improvement_loops_skill.py` runs the real script with `subprocess`,
using stdlib + pytest only and no network:

| Test | Contract |
|---|---|
| `test_history_log_reports_weighted_score_and_delta` | Each run's `score` equals Σ weight × score from the spec. The first run has `previous`/`delta` = `null`; the second run's `previous` equals the first score and its `delta` is the difference. The log gains one line per run, and its parent directory is created |
| `test_invalid_input_is_rejected_without_logging` (×4) | Weights not summing to 1, a missing dimension, an out-of-range score and an extra dimension each exit 2 with a `fitness:` stderr message, and no history file is written |

```
$ scripts/run_tests.sh tests/skills/test_self_improvement_loops_skill.py \
    tests/skills/test_authoring_standards.py tests/skills/test_optional_skill_self_paths.py \
    tests/skills/test_skill_docs_contract.py tests/skills/test_skill_document_contracts.py \
    tests/skills/test_skill_pages_match_shipped_skills.py
=== Summary: 6 files, all passed ===

$ ruff check optional-skills/autonomous-ai-agents/self-improvement-loops tests/skills/test_self_improvement_loops_skill.py
All checks passed!
```

`test_skill_docs_contract::test_catalog_lists_every_shipped_skill[optional-skills]` failed before
the docs were regenerated, and passes after.

### End-to-end: install through the real hub path into a temp `HERMES_HOME`

```
$ HERMES_HOME=$(mktemp -d) hermes skills install official/autonomous-ai-agents/self-improvement-loops --yes
Scan: self-improvement-loops (official/builtin)  Verdict: SAFE
Installed: autonomous-ai-agents/self-improvement-loops
Files: SKILL.md, references/subsystem-map.md, scripts/fitness.py
```

### Manual: the helper script

```
$ python scripts/fitness.py latency.json '{"speed":0.5,"accuracy":0.5}' --log h.jsonl
{ "name": "latency", "score": 0.5, "breakdown": {"speed": 0.2, "accuracy": 0.3}, "previous": null, "delta": null }
$ python scripts/fitness.py latency.json '{"speed":1.0,"accuracy":0.25}' --log h.jsonl
{ "name": "latency", "score": 0.55, "breakdown": {"speed": 0.4, "accuracy": 0.15}, "previous": 0.5, "delta": 0.05 }
$ python scripts/fitness.py latency.json '{"speed":1}'
fitness: scores must cover exactly the spec's dimensions (missing=['accuracy'], extra=[])   # exit 2
```

## Risk

Low. It's an opt-in optional skill with no core change. The only files outside the skill are
generated docs.

## Credit

The design proposal is by @haoqimeng1992 in #11604. This PR adapts its loops to the existing
Hermes surfaces.
