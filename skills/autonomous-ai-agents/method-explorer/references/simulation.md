# Simulation Mode — scenario spec, journal schema, worked examples

This reference backs the **Simulation Mode** and **The Journal** sections of
`SKILL.md`. It is loaded on demand; the SKILL.md body stays lean.

Simulation Mode replays the recursive question loop **deterministically and with
no real side effects** so you can watch it converge or correctly dead-end. The
planner behaves identically to a real run — only the *source* of each node's
outcome changes from "execute the action" to "read the scenario".

## Activation

Active when either holds:
- env var `HERMES_SIM_SCENARIO=<path-to-scenario.json>` is set, or
- the prompt contains `scenario: <path>`.

When active, the **ATTEMPT** step of `solve(node)` reads the node's declared
outcome instead of executing. When inactive, the skill runs normally (real
execution); these files are inert.

## Scenario spec

A scenario is JSON. It is a declarative guide the planner *follows* (not code) —
keep it unambiguous.

```json
{
  "intent": "optional restatement of the goal (for the reviewer)",
  "default": "passthrough | progress | tombstone",
  "notes": "free text describing what this scenario exercises",
  "rules": [
    {
      "id": "r-primary",
      "stage": "S1",
      "match": { "action_contains": "primary", "tool": "terminal" },
      "on_occurrence": 1,
      "after_tombstones": 0,
      "outcome": "tombstone | no-progress | progress | success",
      "reason": "human-readable why",
      "opens": ["verify"]
    }
  ]
}
```

Field semantics:
- **`default`** — outcome for any node no rule matches.
  - `passthrough` → run the real action (mixed sim/real; only safe in a sandbox).
  - `progress` → treat unmatched nodes as progress (advance).
  - `tombstone` → treat unmatched nodes as no-progress. Safest for a fully
    side-effect-free run; pair with explicit `progress` rules for the path you
    want to succeed.
- **Rule matching** — a rule fires when its conditions hold for the current node:
  - `stage` — match the planner's stage id (e.g. `S1`).
  - `match.action_contains` — substring of the node's action text (semantic; the
    planner maps its own stage wording to this). `""` matches everything.
  - `match.tool` — the Hermes tool the node would call.
  - `on_occurrence` — fire only on the Nth matching attempt (default: every time).
  - `after_tombstones` — fire only once ≥N tombstones are already journaled
    (situational, journal-state-dependent).
  First matching rule wins; order rules most-specific first.
- **`outcome`** — `tombstone`/`no-progress` are equivalent (branch is dead);
  `progress`/`success` advance. `success` additionally signals the intent's
  success criteria are met.
- **`opens`** — branch ids this progress unlocks (added to the Frontier).
- **`reason`** — copied into the journal `reason` field.

## Journal schema — `journal.jsonl`

Append-only, one object per loop cycle, at
`${HERMES_HOME}/plans/<task-slug>/journal.jsonl`.

```json
{"node":"S1","q":"fetch valid JSON with key ok from the primary?","chosen":"alfa","expected":"HTTP 200 + JSON with key ok","verdict":"fail","evidence":"sim: r-primary → tombstone (503)","next":"try->S1b"}
```

| field | meaning |
|-------|---------|
| `node` | stage/branch id being attempted |
| `q` | the key question this cycle answers |
| `chosen` | the method attempted |
| `expected` | the postcondition predicted **before** acting |
| `verdict` | `success` / `progress` / `fail` |
| `evidence` | the re-checked receipt (real mode); in **sim** it is the scenario rule's declared `reason` echoed back — **not** a real receipt |
| `next` | the single regenerate move (`advance`, `try->X`, `backtrack->Y`, `relax->C`, `jump->P`, `EXHAUSTION`) — exactly ONE action |

This is the **lean** schema (matches SKILL.md "The Journal"). The old
`ts/depth/questions/attempt/matched_rule/progress/branches_open` fields were dropped in the
lean redesign. **Record each fact once** — do NOT roll the journal up into a prose plan-tree
Decision-log (that duplication was removed; the plan-tree's marker map already reflects the
final state).

**Locality labels in sim:** classify LOCAL-transient vs STRUCTURAL from the declared
`reason`'s *semantics* ("source is down" → LOCAL-transient; "permission denied" →
STRUCTURAL) — never from the fact that a sim retry re-fails. Scenario rules re-declare
the same outcome by design, so retry behavior carries zero information about
standing-ness in sim (SKILL.md Key Question #3).

## Worked example 1 — recursive backtrack, siblings exhausted (`backtrack-demo.json`)

Intent: produce a local data file containing valid JSON with key `"ok"`.
Soft constraint (declare in the task): *prefer a fresh network source over a
local cache*.

Expected loop:
1. `S1` fetch **primary** A → rule `r-primary` → **tombstone** (503). Journal;
   REGENERATE → try sibling.
2. `S1b` fetch **mirror** B → rule `r-mirror` → **tombstone** (timeout). Both network
   siblings are now dead and **no untried network sibling remains**.
3. **Siblings exhausted → ordinary back-up (NOT a K-jump).** The "use a network source"
   parent has an empty local frontier, so back up to the parent and relax the soft
   constraint *prefer network* → opens the local path. (The K=5 upstream-jump heuristic is
   a *different* move — it fires only when K siblings tombstone while MORE remain — and
   does not apply here.)
4. `S2'` read **local cache** → rule `r-cache` → **progress** (opens `verify`).
5. `verify` → rule `r-verify` → **progress** → success criteria met → **SUCCESS**.

What to check in `journal.jsonl`: two tombstones, a `next:"relax->prefer-network"`
(or `backtrack->`) cycle, then progress on the local path, then success — and
that **no real fetch/write ever ran**.

## Worked example 2 — genuine exhaustion (`exhaustion-demo.json`)

Intent: obtain data available only from a single hard-constrained, unreachable
source. The task declares **no soft constraints**.

Expected loop: every attempt matches `default: tombstone` → no progress rule
exists → the Frontier empties with nothing to relax → **EXHAUSTION-STOP** with the
structured no-viable-path report (blocking hard constraint + unblock condition).
The planner must **not** fabricate output and must distinguish this from a
guard-halt. `journal.jsonl` ends with a `verdict:"tombstone"` /
`next:"EXHAUSTION"` cycle and no `progress:true` line.
