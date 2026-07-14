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

Verified against a live instance — **Commander uses two different project
identifiers** and the wrong one 404s or resolves to nothing:

- Query-string routes (`?project=...` — `board`, `running`) want the
  internal **id**: `owner-repo` with a dash, e.g. `zealchaiwut-commander`.
- Path-param routes (`/api/projects/{project}/...` — `running_sprint`,
  `advisor_suggestions`, `todos`, sprint-scoped `?project=` query params
  like `sprint_state`/`preflight`/`mis_sizing_flags`) want the **bare repo
  name**, e.g. `commander`.

When unsure, run `call GET /api/projects` first and read `id` vs `repo` off
the real project list — don't guess.

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
| `health` | `GET /api/health` | Overall health snapshot |
| `board --project <id>` | `GET /api/board` | Kanban board for a project |
| `running --project <id>` | `GET /api/running` | Currently-running agents/jobs for a project |
| `sprints` | `GET /api/sprints` | List sprint labels |
| `sprint_state <label> --project <repo>` | `GET /api/sprints/{label}/state` | Sprint state snapshot |
| `sprint_progress` | `GET /api/sprint-progress` | Sprint progress bar data |
| `issues` | `GET /api/issues` | List issues |
| `running_sprint <repo>` | `GET /api/projects/{project}/running-sprint` | Is a sprint running right now (dispatch guard) |
| `todos <repo>` | `GET /api/projects/{project}/todos` | Project todos |
| `advisor_suggestions <repo>` | `GET /api/projects/{project}/advisor/suggestions` | Pre-computed advisor suggestions |
| `mis_sizing_flags <label> --project <repo>` | `GET .../mis-sizing-flags` | Pre-computed ticket mis-sizing flags |
| `preflight <label> --project <repo>` | `GET .../preflight` | Full preflight report before dispatch |
| `spec [--path <substr>]` | `GET /openapi.json` | Live schema — the source of truth if this doc drifts |
| `stream <path> [--max-seconds N]` | any SSE route | Capped read of a live stream (default 20s) |
| `call <METHOD> <path> [--json '<body>'] [--confirm]` | any of the ~155 routes | Escape hatch — see `references/endpoints.md` |

## Procedure

### Status Check ("what's going on with the board / sprint")

1. Resolve the project: use what the user named, or `commander_api.default_project`.
2. Call `board --project <id>` and, if relevant, `running --project <id>`.
   Two calls, no more.
3. Narrate: current sprint, ticket counts per column (running / needs_rework
   / ready_to_merge / draft / backlog), and which agents are active on what.
4. If the user asks for percent-complete specifically, add `sprint_progress`.

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
5. Report the response and point to `sprint_state`/`stream` for progress —
   don't poll in a loop unattended.

### Sprint Monitoring ("how's sprint X doing")

1. `sprint_state <label> --project <repo>` for a snapshot.
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

- Don't guess the project identifier — the id-vs-repo-name split above is
  real and unverified assumptions will silently 404 or hit the wrong
  project. Run `call GET /api/projects` when unsure.
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
- `call POST ...` without `--confirm` always exits non-zero and refuses —
  confirming the safety gate is structural, not just documented.
- A HIGH-RISK path (e.g. `/api/sprints/run`, any `DELETE`) prints the loud
  warning banner even with `--confirm` passed.
- Status narration after `board`/`running` never invents ticket counts or
  agent states not present in the response body.
