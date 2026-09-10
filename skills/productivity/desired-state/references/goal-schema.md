# Goal Artifact Schema

Full frontmatter reference for a desired-state artifact
(`~/.hermes/state/desired/<domain>/<slug>.md`). `domain` and `goal` are
required; everything else is optional. Unknown keys are ignored on read.

| Field | Type | Default | Notes |
|---|---|---|---|
| `domain` | string | — | **Required.** Life area: `finance`, `health`, `career`, `projects`, `learning`, … Slugified into the path. |
| `goal` | string | — | **Required.** Short human title. Slugified into the filename. |
| `horizon` | `short` \| `medium` \| `long` | `medium` | Rough time scale. |
| `status` | `active` \| `paused` \| `achieved` \| `dropped` | `active` | Terminal statuses (achieved/paused/dropped) skip pace math and report themselves. |
| `direction` | `increase` \| `decrease` \| `maintain` | inferred | How progress reads. Inferred from target vs current if omitted; state it for decrease goals. |
| `target_value` | number \| string | — | The desired value. Numeric enables gap math. |
| `current_value` | number \| string | — | Latest measured value. Updated by `track`. |
| `baseline_value` | number | — | Starting value. Enables honest `(current−baseline)/(target−baseline)` progress. |
| `unit` | string | — | e.g. `%`, `bpm`, `kg`, `$`, `hrs`. Display only. |
| `target_date` | ISO date | — | With `start_date`, enables pace (ahead/on_track/behind). |
| `start_date` | ISO date | `created_at` | Window start for pace. |
| `measurement_source` | string | — | Where the current value comes from (e.g. "ynab export"). |
| `linked_projects` | list[string] | `[]` | Soft references (project slugs). Not required to exist. |
| `linked_people` | list[string] | `[]` | Soft references (people). |
| `linked_todos` | list[string] | `[]` | Soft references to todo IDs. |
| `tags` | list[string] | `[]` | Free-form tags. |
| `created_at` | ISO datetime | set on create | Stamped by the store; don't edit. |
| `updated_at` | ISO datetime | set on write | Stamped by the store; don't edit. |

## Body

Markdown after the closing `---`. Freeform, but two conventions matter:

- **Milestones** as GitHub checkboxes — `- [ ]` (open) / `- [x]` (done). Their
  completion ratio is reported alongside the numeric gap, and *is* the progress
  signal for goals with no numeric target.
- **Context** — why the goal matters, constraints, and notes. This is what the
  agent reads to give grounded, specific help.

## Progress semantics

- `increase`: `(current − baseline) / (target − baseline)` (baseline defaults to 0)
- `decrease`: `(baseline − current) / (baseline − target)` (falls back to `target / current`)
- `maintain`: in-band → 1.0, else scaled by distance from target

Pace compares progress to elapsed-time fraction with a ±10% band.
