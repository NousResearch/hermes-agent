---
name: commander-api
description: "Commander sprint API — read /api/agent-guide docs first."
version: 1.0.0
author: zealchaiwut, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [commander, sprints, tickets, github, agents, devops, api]
    category: software-development
    related_skills: [rest-graphql-debug]
    requires_toolsets: [terminal]
    config:
      - key: commander_api.host
        description: Host Commander's dashboard API is reachable on
        default: "localhost"
        prompt: "Commander host (usually localhost)"
      - key: commander_api.port
        description: Dashboard port — 8000 is PRD, 8001 is UAT
        default: 8000
        prompt: "Commander API port (8000=prd, 8001=uat)"
      - key: commander_api.token
        description: COMMANDER_API_TOKEN bearer token. Leave blank if calling from localhost with no token configured server-side — write calls from 127.0.0.1 are exempt from auth.
        default: ""
        prompt: "Commander API bearer token (blank if none configured)"
      - key: commander_api.default_project
        description: Default project identifier for status checks when the user doesn't name one (repo name, e.g. "commander")
        default: ""
        prompt: "Default project (bare repo name, optional)"
---

# Commander API

Bridges Hermes to a locally running **Commander** dashboard — a FastAPI app
that runs a BA → Coder → Tester → UAT agent pipeline against GitHub Issues,
with sprints, tickets, milestones, and a Kanban board. ~271 routes, no
GraphQL/gRPC, a handful of SSE streams. Commander's own docs already name
Hermes as an intended headless caller (`Authorization: Bearer <token>`), so
this skill is read **and** write — but every mutating call is confirm-gated,
and the highest-blast-radius ones (dispatching a sprint, deleting data,
merging/deploying code) require the user's *explicit, specific* approval
before the script will run them at all.

## When to Use

- User asks about sprint/board/ticket status ("what's running", "what's on
  the board for commander")
- User wants to dispatch, monitor, or finish a sprint
- User wants to create, triage, or approve tickets/issues
- User wants advisor suggestions or mis-sizing flags for a project
- Anything else touching Commander's ~271 routes — reachable via the
  generic client, catalogued in `references/endpoints.md`, or Commander's
  own docs (see Canonical Docs below)

## Prerequisites

- Commander running locally, reachable at `http://<host>:<port>`
- `python3` (stdlib only — no pip installs)
- Skill config (set via `hermes skills config`, or read from the
  `[Skill config]` block injected when this skill loads):
  - `commander_api.host` — default `localhost`
  - `commander_api.port` — `8000` (PRD) or `8001` (UAT)
  - `commander_api.token` — bearer token, blank is fine for localhost calls
  - `commander_api.default_project` — optional, saves asking every time

## Canonical Docs — read these, never grep Commander's source

For anything beyond this file's own live-verified notes, Commander's own
docs are the authority. Read via `call GET <path>` (see How to Run), in
this order:

| Priority | Call | Why |
|---|---|---|
| 1 | `call GET /api/agent-guide` | Canonical operate recipes — run sprint, rerun, complete, sign-off chains. Response is `{content, version}`; the `version` field is a fingerprint — cache the content and only re-fetch when it changes. |
| 2 | `call GET /api/projects/commander/docs/docs/features/api.md` | Full API reference — all ~271 routes, auth model, body fields (`callback_url`, `migrate_from`, `ticket_numbers`). |
| 3 | `call GET /api/projects/commander/docs/docs/workflow.md` | The Bulk Create → Run Sprint → Finish/Rerun operating loop. |
| 4 | `call GET /api/projects/commander/docs/docs/features/sprint-manager.md` | Sprint manager semantics — dispatch levels, gates, fix-loop, rerun lineage. |
| 5 | `spec` (`GET /openapi.json`) | Machine schema when exact request/response models are needed. |

For anything not covered by the above: `call GET /api/projects/commander/docs`
lists all ~90 doc paths — fetch the one you need by path. The same pattern
works for other tracked projects: `/api/projects/perf-coach/docs`,
`/api/projects/asset-studio/docs`, etc.

**Never search the Commander source tree on disk for how something works —
read it from the running server via these calls instead**; the source can
be a different worktree/branch than what's actually serving the API (see
How to Run), and these doc routes are the maintained, agent-facing
source of truth.

## Critical: project identifier quirk

Verified against a live instance — **Commander uses three different project
identifier formats** across its own routes, and the wrong one either 404s
or (worse) silently returns an empty/wrong result with a 200:

- Query-string routes (`?project=...` — `board`, `running`) want the
  internal **id**: `owner-repo` with a dash, e.g. `zealchaiwut-commander`.
- Path-param routes (`/api/projects/{project}/...` — `running_sprint`,
  `advisor_suggestions`, `todos`, sprint-scoped `?project=` query params
  like `sprint_state`/`preflight`/`mis_sizing_flags`) want the **bare repo
  name**, e.g. `commander`.
- `sprint_columns` (`/api/sprint-nav-status?repo=...`) and `rerun`/`rerun_preview`
  (`.../rerun`, `.../rerun-preview`, `?project=...`) both want the **full
  `owner/repo` string**, e.g. `zealchaiwut/commander` — despite `rerun`'s
  query param being named `project` just like the bare-repo-name group
  above, it is NOT in that group; passing the bare name 502s with
  `expected the "[HOST/]OWNER/REPO" format`. `sprint_columns` fails quieter:
  the bare name or dashed id both return `{"has_sprint": false}` with a
  plain `200`, which looks like "no sprint" instead of "wrong identifier".
  The `status` subcommand already gets `sprint_columns` right by reading `repo`
  straight off `GET /api/projects`; the `sprint_columns`/`rerun_preview`/`rerun`
  shortcuts in this script all require `--repo`/`--project` in full
  `owner/repo` form for exactly this reason — don't strip the owner off out
  of habit from the other two groups.

Same route family, same-looking `project`/`repo` query param name, three
different expected value formats depending on the specific route — always
check this section (or `call GET /api/projects` + trial) before assuming a
new route follows the same pattern as one you've already used.

## How to Run

All calls go through one client script, via `terminal`. **Never write ad hoc
`execute_code`/`curl`/`requests`/`subprocess` calls against this API, and
never search the Commander source tree for how to do something — this
script is the only sanctioned path, including for routes it has no named
shortcut for (use `call`, see below).**
`$HERMES_HOME/skills/software-development/commander-api/scripts/` is where
this skill lives once installed:

```bash
python3 $HERMES_HOME/skills/software-development/commander-api/scripts/commander_api.py \
  --port <commander_api.port> --token <commander_api.token> <subcommand> [args]
```

Every subcommand prints JSON with a `status` field, and list-type responses
are capped at 15 items so a single call can't blow the context budget.

## Quick Reference

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `status` | `GET /api/home` | **Start here for "what's running/pending"** — one call, every tracked project's live `status` (idle/uat-pending/running), UAT count, real backlog count |
| `health` | `GET /api/health` | Overall health snapshot |
| `home` | `GET /api/home` | Same data `status` is built from, if you want the full raw payload (aggregate stats too) |
| `sprint_columns --repo <owner/repo>` | `GET /api/sprint-nav-status` | Ticket-column breakdown for one project's latest GitHub-tracked sprint — **not a live-running signal**, see Pitfalls |
| `board --project <id>` | `GET /api/board` | Kanban board for a project |
| `running --project <id>` | `GET /api/running` | Currently-running agents/jobs for a project |
| `sprints` | `GET /api/sprints` | List **every** sprint label ever created for the default project (not "pending" — see Pitfalls) |
| `sprint_state <label> --project <repo>` | `GET /api/sprints/{label}/state` | Sprint state snapshot |
| `sprint_progress` | `GET /api/sprint-progress` | Sprint progress bar data |
| `issues` | `GET /api/issues` | List issues |
| `running_sprint <repo>` | `GET /api/projects/{project}/running-sprint` | Is a sprint running right now (dispatch guard) |
| `todos <repo>` | `GET /api/projects/{project}/todos` | Project todos |
| `advisor_suggestions <repo>` | `GET /api/projects/{project}/advisor/suggestions` | Pre-computed advisor suggestions |
| `mis_sizing_flags <label> --project <repo>` | `GET .../mis-sizing-flags` | Pre-computed ticket mis-sizing flags |
| `preflight <label> --project <repo>` | `GET .../preflight` | Full preflight report before dispatch |
| `rerun_preview <label> --project <owner/repo>` | `GET .../rerun-preview` | Preview what re-running a sprint would do (SAFE) — needs full `owner/repo`, not bare name |
| `milestones <repo>` | `GET /api/projects/{slug}/milestones` | List milestones + `active` one — check before `plan-next` |
| `spec [--path <substr>]` | `GET /openapi.json` | Live schema — the source of truth if this doc drifts |
| `stream <path> [--max-seconds N]` | any SSE route | Capped read of a live stream (default 20s) |
| `call <METHOD> <path> [--json '<body>'] [--confirm]` | any of the ~271 routes | Escape hatch — see `references/endpoints.md` or Canonical Docs |

## Procedure

### Status Check — one or more projects ("what's running/pending on commander")

This is the main use case: a remote status check standing in for opening
the dashboard UI. Get it right in one shot.

1. Run **`status`** — one call, no project resolution needed, covers every
   tracked project. Do not call `board`, `running_sprint`, `sprint_columns`, or
   loop over guessed sprint labels for this ask; `status` already did that
   correctly. If the user named one specific project, still run `status`
   (it's one call) and just report that project's entry.
2. Read the per-project `status` field — this is Commander's own live
   computation, the same one its dashboard UI shows, not something this
   skill derives:
   - `"running"` — an agent is actively working right now (`running_sprint_label`
     / `running_elapsed_sec` are present). Nothing else can be dispatched
     for that project until it finishes.
   - `"uat-pending"` — no agent is currently running; the most recent
     sprint's tickets are sitting in UAT awaiting sign-off (`uat_count`).
     This is idle from a dispatch standpoint, but flag the pending UAT
     count since that's usually why the user is checking.
   - `"idle"` — no agent running, nothing awaiting UAT. Fully idle, room to
     plan or dispatch the next sprint.
   Commander does not expose a literal "pending sprints" list — `sprints`
   returns *every* label ever created (finished ones included), so
   `sprints minus the running one` is **wrong** and lists old, already-
   merged sprints as if they were queued. Don't compute pending that way;
   `status`'s `status` field is the real signal.
3. `backlog_open` is the project's real untriaged-issue count (no sprint
   assigned yet) — mention it when nonzero, especially for an idle project,
   since that's the natural next question ("room to plan the next sprint
   from N backlog issues").
4. Reply in a **compact list, one line per project** — repo name, status,
   uat_count if nonzero, backlog_open if nonzero. Example shape (not
   literal wording, adapt to what `status` actually returned):
   `commander: uat-pending, 5 in UAT, 26 backlog`
   `perf-coach: running sprint-110.1 (6m elapsed)`
   `asset-studio: idle, 17 backlog`
   No headers, no per-project subsections, no restating the question, no
   paragraph of caveats — this is a status ping, not a report.

**Never do these, on this or any status ask** (this is what broke last
time — see Pitfalls for why each one specifically matters):
- Don't show the `terminal`/`python3 .../commander_api.py ...` command or
  its raw JSON in the reply — run it, then speak the answer in plain
  language. Only show the command if the user explicitly asks "what
  command does that" or similar.
- Don't invent "illustrative" or "assuming the API returns..." example
  data. Every number in the reply must come from an actual call you just
  made. If a call fails, say it failed — don't paper over it with a
  plausible-looking guess.
- Don't explain how the API/endpoints work unless asked. The user wants
  the sprint status, not a tutorial on `running` vs `sprints` vs
  `sprint-nav-status`.
- If the user asked about "each project"/"all projects", answer for all
  of them in this one reply. Don't stop after one and ask "which project
  would you like next" — that's only appropriate if they asked about one
  project and you're offering to check others.

### Sprint Dispatch — HIGH-RISK, confirm first

Dispatching spawns real, paid Coder/Tester agent runs. Never call this
without the user explicitly saying to go ahead on *this specific sprint*.

1. `running_sprint <repo>` — confirm nothing is already running for this
   project (a 200 with a label means something's already in flight; don't
   double-dispatch).
2. `preflight <label> --project <repo>` — surface blocking issues
   (unestimated tickets, stale estimates, missing acceptance criteria,
   dependency cycles) to the user before proposing dispatch.
3. Summarize the sprint's ticket list and preflight status, then ask: "Dispatch
   `<label>` now? This spawns paid Coder/Tester agent runs." Wait for a yes.
4. Only after that explicit yes: `call POST /api/sprints/run --json '{"sprint_label": "<label>", "project": "<repo>"}' --confirm`.
5. Report the response and point to `sprint_columns`/`stream` for progress —
   don't poll in a loop unattended.

**Re-running** an already-finished/failed sprint is a different call, not
`/api/sprints/run` again: `rerun_preview <label> --project <owner/repo>`
first (SAFE, shows what would happen — ticket list and the suggested
versioned label like `sprint-103.1`), summarize it and get the same
explicit per-action confirmation, then
`call POST /api/sprints/{label}/rerun --json '{"project": "<owner/repo>"}' --confirm`.
Note the full `owner/repo` form here — see the identifier quirk section.

### Sprint Monitoring ("how's sprint X doing")

1. `status` first for the live signal (is it actually running right now,
   and for how long — see Status Check above). `sprint_columns --repo <owner/repo>`
   adds ticket-column breakdown (backlog/in-progress/sit/uat/done/needs-rework)
   for the project's latest GitHub-tracked sprint, but its own `state` field
   means "has a Sprint N Executive Summary issue been posted" — not "is
   this actively running" — don't quote `sprint_columns.state` as the answer to
   "is it running", only `status`'s `status` field answers that.
2. If the user wants a live tail: `stream /api/sprints/<label>/live/stream --max-seconds 20`
   — one capped read, not an open-ended watch.
3. Relay fields plainly; don't re-interpret `state`/`dag`/warnings into your
   own verdict, and don't conflate `sprint_columns`'s GitHub-derived `state`
   with live process state (see Pitfalls).

### Backlog → Sprint (project is idle, add its backlog/follow-up tickets)

The trigger is usually "project X is idle, clean up/queue its backlog." Two
real tools exist, and they are not equally safe across multiple projects —
read the caveat before picking one.

1. First confirm idle: `status` — if that project's `status` field is
   `"running"`, stop here and say so; Commander runs one sprint per project
   at a time, nothing can be added until it finishes. (`"uat-pending"` and
   `"idle"` both mean no agent is currently running, so planning is fine —
   don't use `sprint_columns` for this check, its `state` field answers a
   different question, see Pitfalls.)
2. **Auto-fill the next sprint from the backlog (preferred, multi-project
   safe):** `plan_next` isn't a named shortcut (it's a write) — check
   `milestones <repo>` first (bare repo name) and read the top-level
   `active` field. If `active` is `null`, say so plainly and stop or ask
   whether the user wants to set a milestone on GitHub first — this is a
   real precondition in Commander's own planner (issue #861: "no active
   milestone → nothing to plan"), not a skill limitation. If there is an
   active milestone, confirm with the user, then
   `call POST /api/sprints/plan-next --json '{"project": "<owner/repo>", "replace": false}' --confirm`.
   Relay the returned `status` verbatim (`ok` / `no_milestone` / `empty` /
   `conflict`) — each is a real, documented outcome, not an error to work
   around. `conflict` means a pending-sign-off draft already exists; only
   retry with `"replace": true` after the user explicitly says to discard
   that existing draft.
3. **Manually add one specific issue to a sprint — CAVEAT:** the only routes
   for this (`POST /api/sprint-planning/assign`,
   `POST /api/issues/{issue_id}/sprint-label`) take **no project parameter
   at all** (verified by reading Commander's own source, not just the
   OpenAPI spec) — they act on whatever project Commander's server
   currently considers active, which this API gives no way to read or set.
   With more than one tracked project this can silently label an issue in
   the wrong repo. Only use these when you're already certain which
   project is server-side-active (e.g. it's the only one with a UI session
   open), and say that assumption out loud to the user. When unsure, use
   `plan_next` instead (it takes an explicit project) or tell the user this
   one needs the Commander UI directly.

### Ticket Creation / Backlog Triage

1. Draft first, never post directly: `call POST /api/tickets/draft --json '{"description": "..."}' --confirm`
   (WRITE-tier — drafting doesn't touch GitHub yet).
2. Show the draft to the user. Only on approval:
   `call POST /api/tickets/create --json '<edited draft>' --confirm`.
3. For backlog triage, always run the `cleanup-preview` / preview-mode call
   first (SAFE) and show what would change before running the applying
   `triage` call (WRITE, confirm).

### Advisor / Estimate Narration

Commander pre-computes these judgments — relay them, don't re-derive:

1. `advisor_suggestions <repo>` and `mis_sizing_flags <label> --project <repo>`
   are read-only and already scored. State the suggestion/flag as Commander
   reports it.
2. Only call the `advisor/run` or `mis-sizing/rebuild` routes (HIGH-RISK —
   they cost LLM calls) if the user explicitly asks to refresh.

### Everything Else

For any of the remaining routes: look up the method/path/risk tier in
`references/endpoints.md`, or check the Canonical Docs above (`agent-guide`,
`docs/features/api.md`), then `call <METHOD> <path> [--json ...] [--confirm]`.
If the reference looks stale, `spec --path <substring>` pulls the live
schema straight from Commander.

## Pitfalls

- Don't guess the project identifier — the three-format split above is real
  and unverified assumptions will silently 404 or (for `sprint_columns`)
  silently return a wrong-but-valid-looking `has_sprint: false`. Run
  `call GET /api/projects` when unsure, or just use `status`, which already
  gets this right.
- `sprints` is a full historical label list, not a "pending" list — don't
  compute "pending" as `sprints` minus the running one, and don't guess a
  next label by incrementing the highest known number. Neither is reliable;
  see Status Check above for the actual read of `status`.
- **`sprint_columns`'s `state` field is not a live-running signal — confirmed
  wrong in practice, not just theoretically.** It's derived from GitHub
  issue labels ("has a Sprint N Executive Summary issue been posted"),
  cached 30s, and can read `"running"` for a project that finished its
  actual agent work hours ago and just hasn't been through the finish
  step yet — this produced a real bad reply (`status: "running"` for a
  project the user could see was idle in the dashboard UI). The `status`
  command's `status` field (`idle`/`uat-pending`/`running`, sourced from
  `/api/home`, the same computation Commander's own dashboard uses) is the
  only field to answer "is it running right now". Likewise `sprint_columns`'s
  `columns.backlog` is the tiny in-sprint kanban backlog column (0-2
  tickets), not the project's real untriaged backlog — that's `status`'s
  `backlog_open` (sourced from `/api/home`'s `backlog_count`), which was
  also wrong in the same bad reply for the same underlying mix-up.
- `pending-signoff` timed out during testing against a project with a lot
  of history — prefer `status`/`home` for awaiting-UAT counts instead of
  calling it per project.
- `sprint-planning/assign` and `issues/{id}/sprint-label` have no project
  parameter — verified in Commander's own source, not just the spec. Don't
  treat them as safe for an arbitrary project; see Backlog → Sprint above.
- `plan-next` requires an active GitHub milestone on the project (a real
  precondition, not a bug) — check `milestones <repo>` first rather than
  calling it blind and being confused by a `no_milestone` result.
- Don't show the raw `terminal` command or JSON response in a status reply,
  don't fabricate illustrative/example data, don't explain API mechanics
  unprompted, and don't leave part of a multi-project question unanswered
  to ask which project to check next — see the "Never do these" list under
  Status Check, all four came from a real bad reply this skill produced.
- Never pass `--confirm` on a mutating call without the user having approved
  *that specific action* in chat — a standing "sure, go ahead" earlier in
  the conversation doesn't cover a different sprint/branch/deploy later.
- Don't loop `stream` or poll a running sprint's state in a tight unattended
  loop — one capped read per ask, per the perf-coach skill's precedent for
  keeping cron/unattended runs to a bounded number of tool calls.
- `/api/fs/list` and `.../environments/{env}/env-vars` can return local
  paths and secrets — never echo env-var values verbatim into chat.
- Don't re-derive advisor/mis-sizing judgments from raw numbers; always
  relay Commander's own pre-computed verdict.

## Verification

- `python3 scripts/commander_api.py health` returns `status: 200` with no
  connection error when Commander is running.
- `python3 scripts/commander_api.py status` returns a `projects` array
  covering every project from `/api/home`, each with a real `status`
  (idle/uat-pending/running) that matches what `running_sprint <repo>`
  and the dashboard UI itself report — cross-check the two if in doubt,
  don't trust `sprint_columns.state` as a substitute.
- `call POST ...` without `--confirm` always exits non-zero and refuses —
  confirming the safety gate is structural, not just documented.
- A HIGH-RISK path (e.g. `/api/sprints/run`, any `DELETE`) prints the loud
  warning banner even with `--confirm` passed.
- A multi-project status reply is a compact list (one short line per
  project) with zero fabricated numbers, zero shown commands/JSON, and zero
  unrequested API explanation — and answers every project the user asked
  about in that one reply, not a subset with a follow-up question.
