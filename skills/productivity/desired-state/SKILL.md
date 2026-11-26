---
name: desired-state
description: Track durable goals and current-vs-target gaps.
version: 1.0.0
author: Benlloyd Goldstein (benlloydg), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: ["python3"]
metadata:
  hermes:
    tags: [goals, planning, personal-assistant, tracking, desired-state, productivity]
    related_skills: [plan]
    category: productivity
---

# Desired-State Skill

Capture durable desired-state artifacts for goals the user wants to track across
sessions. Each artifact can carry a target, current value, baseline, dates,
links, and milestones, while the helper scripts compute repeatable gap and pace
summaries instead of relying on model estimates. This is not the session `todo`
tool or `/goal` loop; it is cross-session goal state that lives under
`HERMES_HOME`.

## When to Use

- The user states a target or aspiration: "I want to hit a 15% savings rate,"
  "get my resting HR under 60," "ship the podcast pilot this quarter."
- The user asks how they're doing against a goal, or "what should I focus on?"
- A weekly/periodic check-in: record fresh current values and surface what's
  behind pace.
- The user wants a goal to persist and be referenced across future sessions.

**Don't use for:**

- In-session task decomposition or a working checklist for the current task —
  that is the `todo` tool (session-scoped scratchpad), not a durable goal.
- Driving one session to completion turn-by-turn — that is the `/goal` loop
  (`hermes_cli/goals.py`), a different altitude. Desired-state is cross-session
  and cross-domain; it does not run the agent in a loop.
- One-off facts about the user ("prefers dark mode") — that is memory.

## Prerequisites

- Python 3.11+ with stdlib only; no PyYAML or runtime third-party dependencies.
- Run commands through the `terminal` tool from the skill's `scripts/` directory.
- `HERMES_HOME` is optional; when unset the scripts use the Hermes default store.

## How to Run

Use the `terminal` tool to run `python3 ds.py <command>` from
`skills/productivity/desired-state/scripts/`. Put `--json` before the command
for machine-parseable output.

## Quick Reference

### Store Layout

```
~/.hermes/state/desired/<domain>/<goal-slug>.md   # one artifact per goal
```

`<domain>` and `<goal-slug>` are slugified (lowercase, hyphenated). Resolve the
root with the store, never hardcode `~/.hermes` — it honors `HERMES_HOME`.

### CLI Commands

| Command | Purpose |
|---|---|
| `python3 ds.py define <domain> "<goal>" [field flags]` | Create a goal artifact |
| `python3 ds.py track <domain> <slug> <value>` | Record a new measured current value |
| `python3 ds.py edit <domain> <slug> [field flags]` | Change frontmatter fields |
| `python3 ds.py show <domain> <slug>` | Print one artifact (raw, or `--json`) |
| `python3 ds.py list [--domain D] [--status S]` | List goals |
| `python3 ds.py gap [<domain> <slug>]` | Gap for one goal, or all **active** goals |
| `python3 ds.py report [--domain D]` | Per-domain rollup + what's behind pace |
| `python3 ds.py milestone <domain> <slug> [--add TEXT \| --check N \| --uncheck N]` | List / add / check the plan's milestones |
| `python3 ds.py archive <domain> <slug> [--status …]` | Soft-close (never deletes) |

**Field flags** (for `define` / `edit`): `--horizon {short,medium,long}`,
`--status {active,paused,achieved,dropped}`, `--direction {increase,decrease,maintain}`,
`--target`, `--current`, `--baseline`, `--unit`, `--target-date`, `--start-date`,
`--source`, `--project` (repeatable), `--person` (repeatable), `--todo`
(repeatable), `--tag` (repeatable), `--body` / `--body-file`.

## Procedure

### Define a Good Goal

Set the fields that make the gap meaningful; omit what you don't have.

1. **`--direction`** decides how progress reads. `increase` (savings rate up),
   `decrease` (resting HR down), or `maintain` (stay in a band). If omitted it
   is inferred from whether the target is above or below the current value.
2. **`--baseline`** is the starting value. With it, progress is
   `(current − baseline) / (target − baseline)` — the honest "how far from where
   I started." Without it, progress falls back to `current / target`.
3. **`--target-date` + `--start-date`** unlock pace: the report compares how far
   along the goal is against how much of the window has elapsed, and labels it
   ahead / on track / behind.
4. **Milestones** go in the body as `- [ ]` / `- [x]` checkboxes. For a goal with
   no numeric target, checkbox completion becomes the progress signal.

Completion criterion: after `define`, run `python3 ds.py gap <domain> <slug>`
and confirm the summary reads the way the user means it (right direction, sane
percentage). If it doesn't, the direction or baseline is wrong — fix it before
moving on.

### Core Recipes

**Define from a stated target**
```
python3 ds.py define finance "Hit 15% savings rate" \
  --direction increase --baseline 9 --current 9 --target 15 --unit "%" \
  --start-date 2026-01-01 --target-date 2026-12-31 --source "ynab export"
```

**Weekly check-in** — record fresh numbers, then surface the picture:
```
python3 ds.py track finance hit-15-savings-rate 12.4
python3 ds.py report
```

**"What should I focus on?"** — `report` lists every active goal with its pace
and flags the ones `behind`. Lead the user with those, not the whole list.

**Close the loop** — when a goal is met, `gap` reports `pace: met` (it does not
auto-flip status; that stays a decision). Confirm with the user, then
`python3 ds.py archive <domain> <slug> --status achieved`.

### Propose the Path

Tracking the gap is half the job — the point is *closing* it. When a goal is
`behind`, or the user asks "how do I hit this?", don't stop at the number:
propose the concrete path.

1. **Ground the proposal in the artifact.** Read the body (context, constraints,
   open `- [ ]` milestones) and the gap (how far behind, days left, direction).
   The plan must respect the goal's stated constraints — a savings goal that
   says "no lifestyle cuts that hurt output" rules out the obvious cut.
2. **Propose the 1–3 highest-leverage next actions**, not a generic list. Each
   should measurably move `current` toward `target` or unblock a milestone.
3. **Persist the plan so it's a living artifact, not a one-off message.** Add each
   durable next step as a milestone —
   `python3 ds.py milestone finance hit-15-savings-rate --add "Renegotiate rent by March"` —
   and mark them done as they land (`--check N`). `gap` and `report` then show
   milestone progress (`· 2/3 milestones`) alongside the numeric gap.
4. **Be honest when the pace says so.** If a goal is `behind` with little time
   left, say the target or date likely needs to move
   (`edit --target-date`/`--target`) rather than proposing a path that can't
   close the gap.

Completion criterion: the user has concrete, constraint-aware next actions, and
any durable ones are written to the artifact so the plan survives the session.

### Explain Gap and Pace

So you can explain it correctly:

- **progress** is 0–100% toward target, direction-aware (see baseline rule
  above). It can exceed 100% when the target is passed.
- **pace** compares progress against elapsed time: `ahead` (progress leads
  elapsed by >10%), `behind` (trails by >10%), `on_track` (within the band),
  `met` (target reached), `unknown` (no dates or no numeric target).
- **milestones** (`done/total`) are reported alongside, and drive progress when
  there is no numeric target.

### Integration Notes

- **Links are soft references.** `--project`, `--person`, `--todo` store IDs/slugs
  for context and future cross-linking; this skill does not require those stores
  to exist and does not write to them.
- **Not memory.** Desired-state artifacts are structured, long-lived goal state.
  Do not duplicate them into `MEMORY.md` (char-capped; it would evict them). A
  one-line pointer in memory ("tracks savings-rate goal via desired-state") is
  fine.
- **Files are truth.** Edits the user makes to an artifact by hand are picked up
  on the next read. Never cache goal state in the conversation.

## Pitfalls

1. **Using it as a todo list.** A desired state is a target with a gap, not a
   task. If it has no notion of "closer/further," it belongs in the `todo` tool.
2. **Omitting `--direction` on a decrease goal.** Inference usually catches it,
   but state it explicitly for weight/HR/cost goals so progress never inverts.
3. **No dates, then expecting pace.** Without `--start-date`/`--target-date`,
   pace is `unknown` by design — don't report "behind"; report the raw progress.
4. **Auto-marking achieved.** `pace: met` is a prompt to confirm with the user,
   not a signal to silently `archive`.
5. **Hardcoding `~/.hermes`.** Always let the scripts resolve the root so
   `HERMES_HOME` and profiles work.
6. **Deleting goals.** There is no delete — `archive` sets a terminal status and
   keeps the file. Dropped goals are history, not garbage.

## Verification

- [ ] `python3 ds.py define …` created a file under `~/.hermes/state/desired/`
- [ ] `python3 ds.py gap <domain> <slug>` summary reads correctly (direction + %)
- [ ] For time-boxed goals, `--start-date` and `--target-date` are set so pace works
- [ ] `python3 ds.py report` groups by domain and flags what's behind
- [ ] For a behind goal, proposed constraint-aware next steps and persisted the
      durable ones as milestones on the artifact
- [ ] The user confirmed before any `archive --status achieved`
- [ ] No goal state cached in the conversation; artifacts remain the source of truth
