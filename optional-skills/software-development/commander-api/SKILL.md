---
name: commander-api
description: "Commander sprint dashboard API: board, sprints, tickets, issues."
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
with sprints, tickets, milestones, and a Kanban board. ~155 routes, no
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
- Anything else touching Commander's ~155 routes — reachable via the
  generic client, catalogued in `references/endpoints.md`

## Prerequisites

- Commander running locally, reachable at `http://<host>:<port>`
- `python3` (stdlib only — no pip installs)
- Skill config (set via `hermes skills config`, or read from the
  `[Skill config]` block injected when this skill loads):
  - `commander_api.host` — default `localhost`
  - `commander_api.port` — `8000` (PRD) or `8001` (UAT)
  - `commander_api.token` — bearer token, blank is fine for localhost calls
  - `commander_api.default_project` — optional, saves asking every time

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
- `nav_status` (`/api/sprint-nav-status?repo=...`) and `rerun`/`rerun_preview`
  (`.../rerun`, `.../rerun-preview`, `?project=...`) both want the **full
  `owner/repo` string**, e.g. `zealchaiwut/commander` — despite `rerun`'s
  query param being named `project` just like the bare-repo-name group
  above, it is NOT in that group; passing the bare name 502s with
  `expected the "[HOST/]OWNER/REPO" format`. `nav_status` fails quieter:
  the bare name or dashed id both return `{"has_sprint": false}` with a
  plain `200`, which looks like "no sprint" instead of "wrong identifier".
  The `status` subcommand already gets `nav_status` right by reading `repo`
  straight off `GET /api/projects`; the `nav_status`/`rerun_preview`/`rerun`
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
| `status` | `/api/projects` + `/api/sprint-nav-status` × N | **Start here for "what's running/pending"** — one call, every tracked project's current sprint + state + done/total/UAT counts |
| `health` | `GET /api/health` | Overall health snapshot |
| `home` | `GET /api/home` | Cross-project rollup (running count, awaiting-UAT, backlog totals) |
| `nav_status --repo <owner/repo>` | `GET /api/sprint-nav-status` | Current sprint + ticket-column breakdown for one project |
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
| `spec [--path <substr>]` | `GET /openapi.json` | Live schema — the source of truth if this doc drifts |
| `stream <path> [--max-seconds N]` | any SSE route | Capped read of a live stream (default 20s) |
| `call <METHOD> <path> [--json '<body>'] [--confirm]` | any of the ~155 routes | Escape hatch — see `references/endpoints.md` |

## Procedure

### Status Check — one or more projects ("what's running/pending on commander")

This is the main use case: a remote status check standing in for opening
the dashboard UI. Get it right in one shot.

1. Run **`status`** — one call, no project resolution needed, covers every
   tracked project. Do not call `board`, `running_sprint`, `sprint_state`,
   or loop over guessed sprint labels for this ask; `status` already did
   that correctly. If the user named one specific project, still run
   `status` (it's one call) and just report that project's entry.
2. Commander does not expose a literal "pending sprints" list — `sprints`
   returns *every* label ever created (finished ones included), so
   `all sprints minus the running one` is **wrong** and will list old,
   already-merged sprints as if they were queued up. The correct read of
   `status`'s per-project `state` field:
   - `"running"` — a sprint is actively being worked; nothing else can be
     dispatched for that project right now.
   - `"finished"` (or no entry / `has_sprint: false`) — the project is idle;
     there's room to plan or dispatch the next sprint. This does *not* mean
     a specific next sprint is already queued — say "idle, room to run
     something" rather than inventing a next label. If the user wants to
     know the next label to use, that's a separate ask (see Sprint Dispatch).
3. Reply in a **compact list, one line per project** — repo name, sprint
   number, state, `done/total` done, `uat` count if nonzero. Example shape
   (not literal wording, adapt to what `status` actually returned):
   `commander: sprint-103 running (0/6 done, 5 in UAT)`
   `asset-studio: sprint-8 finished (9/9 done)`
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
5. Report the response and point to `nav_status`/`stream` for progress —
   don't poll in a loop unattended.

**Re-running** an already-finished/failed sprint is a different call, not
`/api/sprints/run` again: `rerun_preview <label> --project <owner/repo>`
first (SAFE, shows what would happen — ticket list and the suggested
versioned label like `sprint-103.1`), summarize it and get the same
explicit per-action confirmation, then
`call POST /api/sprints/{label}/rerun --json '{"project": "<owner/repo>"}' --confirm`.
Note the full `owner/repo` form here — see the identifier quirk section.

### Sprint Monitoring ("how's sprint X doing")

1. `nav_status --repo <owner/repo>` for a snapshot with ticket-column
   breakdown (or `status` if checking multiple projects at once — see
   Status Check above). Use `sprint_state <label> --project <repo>` only if
   you need fields `nav_status` doesn't return.
2. If the user wants a live tail: `stream /api/sprints/<label>/live/stream --max-seconds 20`
   — one capped read, not an open-ended watch.
3. Relay state fields plainly; don't re-interpret `state`/`dag`/warnings
   into your own verdict.

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

For any of the remaining ~140 routes: look up the method/path/risk tier in
`references/endpoints.md`, then `call <METHOD> <path> [--json ...] [--confirm]`.
If the reference looks stale, `spec --path <substring>` pulls the live
schema straight from Commander.

## Pitfalls

- Don't guess the project identifier — the three-format split above is real
  and unverified assumptions will silently 404 or (for `nav_status`)
  silently return a wrong-but-valid-looking `has_sprint: false`. Run
  `call GET /api/projects` when unsure, or just use `status`, which already
  gets this right.
- `sprints` is a full historical label list, not a "pending" list — don't
  compute "pending" as `sprints` minus the running one, and don't guess a
  next label by incrementing the highest known number. Neither is reliable;
  see Status Check above for the actual read of `state`.
- `pending-signoff` timed out during testing against a project with a lot
  of history — prefer `status`/`home` for awaiting-UAT counts instead of
  calling it per project.
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
  covering every project from `/api/projects`, each with a real `state`.
- `call POST ...` without `--confirm` always exits non-zero and refuses —
  confirming the safety gate is structural, not just documented.
- A HIGH-RISK path (e.g. `/api/sprints/run`, any `DELETE`) prints the loud
  warning banner even with `--confirm` passed.
- A multi-project status reply is a compact list (one short line per
  project) with zero fabricated numbers, zero shown commands/JSON, and zero
  unrequested API explanation — and answers every project the user asked
  about in that one reply, not a subset with a follow-up question.
