# Commander API — Full Endpoint Reference

~155 routes, no global prefix (a few routers set one — noted per group). This
file is the on-demand catalogue; `SKILL.md` only carries the ~12 routes an
agent reaches for daily. For anything not listed here, or to check whether a
route's shape has changed, fetch the live schema:

```bash
python3 scripts/commander_api.py spec --path <substring>
```

## Risk tiers

- **SAFE** — GET, read-only, call freely.
- **WRITE** — mutates state but is low blast-radius (a todo, a dismissed
  suggestion, a draft). Requires `--confirm`; mention to the user what
  you're about to do first, but a quick heads-up is enough.
- **HIGH-RISK** — spawns paid agent runs, deletes data, merges/deploys code,
  or otherwise can't be casually undone. Requires `--confirm`; get the
  user's *explicit, specific* approval first ("dispatch sprint-119?", not a
  standing "sure go ahead").

## Project identifier quirk (verified against a live instance — this is not
## documented anywhere in Commander itself, so don't assume it, use it)

Commander uses **two different project identifiers** depending on which
router you're hitting, and passing the wrong one either 404s or silently
resolves to nothing:

- Query-string style (`?project=...`) — e.g. `/api/board`, `/api/running` —
  wants the dashboard's internal **id**: `owner-repo` with a dash
  (`zealchaiwut-commander`).
- Path-param style (`/api/projects/{project}/...`) — e.g. `running-sprint`,
  `advisor/suggestions`, `todos` — wants the **bare repo name**
  (`commander`, not `zealchaiwut/commander` and not `zealchaiwut-commander`).

When in doubt, call `GET /api/projects` first and use `id` for query-string
routes, `repo` (stripped of the `owner/` prefix) for path-param routes.

---

## System / health / diagnostics

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/health` | Health snapshot (shortcut: `health`) |
| SAFE | GET | `/api/version` | Build/version info |
| SAFE | GET | `/api/environment` | prd/uat env tag |
| SAFE | GET | `/api/repo/config` | Repo config |
| SAFE | GET | `/api/github/labels` | List GH labels |
| WRITE | POST | `/api/github/labels` | Create/update GH labels |
| SAFE | GET | `/api/gh-auth-status` | GitHub CLI auth status |
| WRITE | POST | `/api/gh-auth/login/start` \| `/input` \| `/cancel` \| `/token` | GH login flow |
| SAFE | GET | `/api/doctor` | Preflight host doctor report |
| SAFE | GET | `/diagnostics` | Diagnostics page/JSON |
| SAFE | GET | `/api/debug/api-volume` | Request-volume debug counters |

## Logs / events / activity

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/agents` | List active agents |
| SAFE | GET | `/api/events` | List recent events |
| SAFE | GET | `/api/projects/{slug}/events` | Project event feed |
| WRITE | POST | `/api/agent-event` | Ingest agent tool-call/event (used by hooks) |
| WRITE | POST | `/api/token-usage` | Report token usage |
| WRITE | DELETE | `/api/events/test` | Clear test events |
| SAFE (SSE) | GET | `/events` | Live event stream — use `stream` subcommand |
| SAFE | GET | `/api/logs/search` | Full-text log search (prefix `/api/logs`) |
| SAFE | GET | `/api/logs/runs/{sprint_label}/ticket-stats` | Per-ticket run stats |
| SAFE | GET | `/api/logs/runs/{sprint_label}/ica-cost` | ICA cost for sprint |
| SAFE | GET | `/runs`, `/run-browser` | Forensic run browser (HTML) |
| SAFE | GET | `/runs/{sprint}/{issue}/{agent}/log` | Paginated log content |
| SAFE | GET | `/logs/tail`, `/runs/{sprint}/{issue}/{agent}/log/tail` | Tail of a log file |

## Projects

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/projects` | List tracked projects — **use this to resolve id/repo** |
| WRITE | POST | `/api/projects` | Add project (201) |
| HIGH-RISK | DELETE | `/api/projects/{owner}/{repo_name}` | Remove project tracking |
| SAFE | GET | `/api/projects/{project}/running-sprint` | Running sprint (shortcut: `running_sprint`) |
| WRITE | POST | `/api/projects/{owner}/{repo_name}/approve-batch` | Batch-approve tickets |
| WRITE | POST | `/api/projects/init` | Initialize new project scaffolding |
| SAFE | GET | `/api/project-details` | Project detail lookup |
| HIGH-RISK | POST | `/api/projects/{owner}/{repo_name}/sprint-branch-merge` | Merge sprint branch — merges code |
| SAFE | GET | `/api/projects/{owner}/{repo}/branches/stale` | List stale merged branches |
| HIGH-RISK | DELETE | `/api/projects/{owner}/{repo}/branches/{branch}` | Delete a branch |

## Milestones / roadmap

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/projects/{slug}/milestones`, `/api/milestones` | List milestones |
| WRITE | POST/PATCH | `/api/projects/{slug}/milestones` \| `/{number}` | Create/update milestone |
| WRITE | DELETE | `/api/projects/{slug}/milestones/{number}` | Delete milestone |
| SAFE | GET | `/api/projects/{slug}/issues` | Issues for milestone view |
| SAFE | GET | `/api/roadmap` (prefix `/api/roadmap`) | Roadmap data |
| WRITE | POST/PATCH | `/api/roadmap/milestones` \| `/{number}` | Create/update roadmap milestone |
| WRITE | POST | `/api/roadmap/milestones/{number}/close` \| `/reopen` | Close/reopen milestone |
| WRITE | PUT | `/api/roadmap/settings` | Update roadmap settings |
| SAFE | GET | `/api/home/milestone` (prefix `/api/home`) | Home-page milestone summary |

## Issues / backlog / triage

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/issues` | List issues (shortcut: `issues`) |
| WRITE | POST | `/api/issues/{issue_id}/approve` \| `/reject` \| `/close` | Issue lifecycle actions |
| SAFE | GET | `/api/issues/{issue_id}/test-report` | Tester report for issue |
| WRITE | POST | `/api/issues/{issue_id}/sprint-label` | Set sprint label on issue — **no project param** (acts on Commander's ambient active project, verified via source; unsafe to assume for a specific project) |
| SAFE | POST | `/api/projects/{owner}/{repo}/backlog/cleanup-preview` \| `/triage-apply` (preview mode) | Preview backlog cleanup/triage |
| WRITE | POST | `/api/projects/{owner}/{repo}/backlog/triage` | Run backlog triage (applies) |
| WRITE | POST | `/api/tickets/{issue_id}/approve` | Approve single ticket |
| WRITE | POST/DELETE | `/api/tickets/bulk/{job_id}/stop` \| (DELETE) | Stop/delete bulk-ticket job |

## Bulk ticket creation

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/tickets/bulk/{job_id}` | Job status |
| SAFE (SSE) | GET | `/api/tickets/bulk/{job_id}/stream` | Job progress stream |
| WRITE | POST | `/api/tickets/draft` | Draft a ticket via LLM (not posted yet) |
| WRITE | POST | `/api/tickets/create` | Create ticket (201) — posts to GitHub |
| WRITE | POST | `/api/tickets/bulk` | Start bulk-create job (202) |
| WRITE | POST | `/api/tickets/bulk/{job_id}/estimate-draft` \| `/skip` \| `/retry` \| `/redraft` \| `/retry-with-body` \| `/retry-with-image` \| `/retry-all` | Bulk-job item actions |
| WRITE | POST | `/api/tickets/bulk/{job_id}/post-selected` | Post drafted tickets to GitHub |
| WRITE | POST | `/api/tickets/bulk/{job_id}/size-remedy-comment` \| `/size-remedy-images` | Size-remedy actions |

## Todos (prefix `/api/projects`)

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/projects/{project}/todos` | List todos (shortcut: `todos`) |
| WRITE | POST | `/api/projects/{project}/todos` | Create todo |
| WRITE | DELETE | `/api/projects/{project}/todos/done` \| `/{todo_id}` | Clear/delete todo(s) |
| WRITE | PATCH | `/api/projects/{project}/todos/{todo_id}` | Update todo |
| WRITE | POST/GET/DELETE | `/api/projects/{project}/todos/{todo_id}/attachments/...` | Attachments |
| SAFE | GET | `/api/todos` | Batch todos across projects |

## Sprints — CRUD / planning

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/sprints` | List sprints (shortcut: `sprints`) |
| SAFE | GET | `/api/sprints/goal` \| `/order` \| `/pending-signoff` \| `/running-all` | Reads |
| WRITE | POST | `/api/sprints/goal` \| `/order` | Set goal/order |
| WRITE | POST | `/api/sprints/plan-next` | Draft next sprint from the active milestone's backlog, capacity-aware — properly project-scoped via `project` body field (full `owner/repo`); requires an active GitHub milestone or returns `no_milestone` (check `milestones <repo>` first) |
| SAFE | GET | `/api/sprints/{label}/dispatch-log` \| `/preview-dag` \| `/dag-order-preview` | Preview reads |
| WRITE | POST | `/api/sprints/create` | Create sprint |
| WRITE | POST | `/api/sprints/{label}/rename` \| `/tickets/reorder` \| `/plan` | Sprint edits |
| WRITE | POST | `/api/sprints/delete-empty` \| `/cleanup-empty` | Cleanup empty sprints |
| HIGH-RISK | DELETE | `/api/sprints/{label}` | Delete sprint |

## Scheduler

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/scheduler/config` \| `/sprints` | Reads |
| WRITE | PUT | `/api/scheduler/config` \| `/sprints` | Updates |
| HIGH-RISK | POST | `/api/scheduler/tick` | Trigger a scheduler tick — can dispatch sprints |

## Sprint dispatch / run / live

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/board` | Kanban board data (shortcut: `board`) |
| SAFE | GET | `/api/running` | Currently-running agents/jobs (shortcut: `running`) |
| SAFE | GET | `/api/sprint-management/issues` | Issues for sprint management view |
| HIGH-RISK | POST | `/api/sprint-run`, `/api/sprints/run` | **Dispatch a sprint — spawns paid Coder/Tester agent runs.** Confirm explicitly. |
| HIGH-RISK | DELETE | `/api/sprints/run/{label}` | Cancel a running sprint |
| HIGH-RISK | POST | `/api/sprints/{label}/rerun` | Rerun sprint |
| SAFE | GET | `/api/sprints/{label}/branch-status` \| `/rerun/preview` \| `/rerun-preview` | Reads |
| SAFE | GET | `/api/sprints/{label}/state` | Sprint state (shortcut: `sprint_state`) |
| SAFE | GET | `/api/logs/runs` | Live runs list |
| WRITE | POST | `/api/logs/sync-github` | Force GitHub sync |
| SAFE | GET | `/api/sprints/{label}/state-full` \| `/issue/{n}/log` \| `/state-timing` | Detail reads |
| SAFE (SSE) | GET | `/api/sprints/{label}/live/stream` | Live sprint stream |
| SAFE | GET | `/api/sprints/{label}/live` | Live sprint snapshot |
| SAFE | GET | `/api/sprint-nav-status` \| `/sprint-progress` \| `/sprint-nav-summary` | Nav reads (shortcuts: `sprint_columns` for nav-status, `sprint_progress` for progress) — `sprint-nav-status`'s `state` is GitHub-label-derived, not live-running; use `status`/`/api/home` for that |
| SAFE | GET | `/api/sprint-status` \| `/sprint-summary` \| `/home` \| `/sprint-history` \| `/sprint-history-content` \| `/sprints/timeline` \| `/sprints/summaries` | Summary reads |
| WRITE | POST | `/api/sprint-status` | Post status |

## Sprint finish / signoff / preflight / labels / history

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/sprints/{label}/finish-card` \| `/finish-preview` \| `/bulk-complete-preview` \| `/conflict-status` | Previews |
| HIGH-RISK (SSE) | POST/GET | `/api/projects/{owner}/{repo}/sprints/{label}/finish-bg` \| `/finish-stream` | Background finish + progress stream |
| HIGH-RISK | POST | `/api/projects/{owner}/{repo}/sprints/{label}/finish` | Finish sprint — merges/closes |
| HIGH-RISK | POST | `/api/projects/{owner}/{repo}/sprints/{label}/bulk-complete` | Bulk-complete sprint |
| WRITE | POST | `/api/projects/{owner}/{repo}/sprints/{label}/complete-step` | Complete one step |
| WRITE | POST | `/api/sprints/{label}/approve` \| `/reject` | Sign-off |
| WRITE | POST | `/api/sprints/{label}/preflight-fix` | Auto-fix preflight issues |
| SAFE | GET | `/api/sprints/{label}/cycle-check` \| `/conflicts` \| `/dep-order` \| `/preflight` | Preflight reads (shortcut: `preflight`) |
| SAFE | GET | `/api/open-issues` | Open issues not yet in a sprint |
| WRITE | POST | `/api/sprints/batch-labels` | Batch label update |
| SAFE | GET | `/api/debug/sprint-collisions` \| `/api/sprints/history` | Debug/history reads |
| SAFE | GET | `/api/sprints/{label}/reconcile-preview` | Dry-run reconcile |
| WRITE | POST | `/api/sprints/{label}/reconcile` | Apply reconcile |
| WRITE | POST | `/api/sprints/{label}/split-xl/{issue}/preview` \| `/apply` | XL-ticket split |
| SAFE | GET | `/scan-stale-branches` | Scan stale branches |
| WRITE | POST | `/cleanup-stale-branches` | Cleanup stale branches |
| SAFE | GET | `/api/sprints/{label}/run-stats` \| `/timeline` | Reads |
| WRITE | POST | `/api/sprints/{label}/clear-stale-labels` | Clear stale status labels |

## Estimates / mis-sizing / calibration

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/sprints/{label}/estimate-summary` \| `/estimate` \| `/outcome` \| `/estimate-vs-actual` | Reads |
| SAFE | GET | `/api/estimates/batch` | Batch estimates |
| SAFE | GET | `/api/sprints/{label}/mis-sizing-flags` | (shortcut: `mis_sizing_flags`) |
| WRITE | POST | `/api/sprints/{label}/mis-sizing-flags/generate` \| `/{issue_id}/action` | Generate/act on flags |
| SAFE | GET | `/api/mis-sizing/history` \| `/config` | Reads |
| WRITE | POST | `/api/mis-sizing/rebuild` \| `/config` | Rebuild/set config |
| SAFE | GET | `/api/calibration` \| `/api/projects/{slug}/analytics/calibration` | Reads |
| SAFE | GET | `/api/estimate-jobs/{job_id}` | Job status |
| WRITE | POST | `/api/sprints/{label}/estimate` | Trigger estimate job (202) |
| WRITE | POST | `/api/system-misc /api/issues/{issue_id}/estimate` | Trigger single-issue estimate |
| WRITE | POST | `/api/sprints/{label}/xl-suggestions/{issue}/dismiss` | Dismiss XL-split suggestion |

## Advisor / suggestions / sprint planning

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/projects/{project}/advisor/suggestions` | (shortcut: `advisor_suggestions`) — **pre-computed, relay verbatim, don't re-derive** |
| HIGH-RISK | POST | `/api/projects/{project}/advisor/run` | Run advisor — costs LLM calls |
| SAFE | GET | `/api/projects/{project}/advisor/look-ahead` | Look-ahead analysis |
| HIGH-RISK | POST | `/api/advisor/tick` | Advisor tick — costs LLM calls |
| WRITE | POST | `/api/projects/{project}/advisor/suggestions/{id}/dismiss` \| `/accept` | Accept/dismiss suggestion |
| SAFE | GET | `/api/projects/{project}/advisor/dismissed` | Dismissed suggestions |
| SAFE | GET | `/api/sprint-planning/issues` | Candidate issues for planning — **no project param**, returns whichever project is Commander's ambient active one |
| WRITE | POST | `/api/sprint-planning/assign` | Assign issue to sprint label — **no project param** (same ambient-project caveat as `/api/issues/{id}/sprint-label`; prefer `plan-next` for a specific project) |

## Brief / summary / changelog / docs

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/projects/{slug}/brief` \| `/api/brief` | Project/home brief |
| SAFE | GET | `/api/projects/{slug}/brief/summary` \| `/api/brief/summary` | LLM summary (pre-generated) |
| HIGH-RISK | POST | `.../brief/summary/regenerate`, `.../brief/daily/regenerate` | Regenerate — costs LLM calls |
| SAFE | GET | `/api/projects/{slug}/brief/daily` \| `/api/brief/daily` | Daily brief artifact |
| SAFE | GET | `/api/projects/{slug}/changelog` | Project changelog |
| SAFE | GET | `/api/projects/{slug}/docs` \| `/docs/{path}` | List/fetch docs |
| SAFE | GET | `/api/agent-guide` | Agent onboarding guide content |

## Analytics / metrics / status

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/metrics/sprints` \| `/api/projects/{slug}/analytics/metrics` \| `/cost` | Reads |
| WRITE | POST | `/api/maintenance/calibration/rebuild` | Rebuild calibration data |
| SAFE | GET | `/api/status` \| `/status/sprints` \| `/status/health` \| `/status/queue` | Status reads |

## Settings / config / secrets / scaffolding

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/settings` \| `/api/projects/{slug}/settings` \| `/deploy-config` | Reads |
| WRITE | PUT | `/api/settings` \| `.../settings` \| `.../deploy-config` | Updates |
| WRITE | DELETE | `/api/projects/{slug}/settings` | Delete project settings |
| WRITE | POST | `.../environments/{env}/deploy-config/validate` | Validate deploy config |
| SAFE | GET | `/api/fs/list` | Local filesystem browse — **can read arbitrary local paths, treat as sensitive** |
| SAFE | GET | `.../environments/{env}/env-vars` | Get env vars — **may contain secrets, never print verbatim to chat** |
| WRITE | PUT | `.../environments/{env}/env-vars` | Set env vars |
| SAFE | GET | `.../docs/scaffold/check` | Check doc scaffold status |
| WRITE | POST | `.../docs/scaffold/apply` | Apply doc scaffold |
| SAFE | GET | `/api/projects/notes` | Get project notes |
| WRITE | POST | `/api/projects/notes` | Save project notes |
| SAFE | GET | `/api/settings/provider` | LLM provider setting |
| WRITE | POST | `/api/settings/provider` | Set LLM provider |
| SAFE | GET | `/api/settings/sync/status` | Settings sync status |
| WRITE | POST | `/api/settings/sync/diff` \| `/commit` | Diff/commit settings sync |
| SAFE | GET | `/api/backup/status` | Backup status |

## Environments / deploy

| Risk | Method | Path | Purpose |
|---|---|---|---|
| HIGH-RISK | POST | `/api/projects/{slug}/environments/{env}/deploy` | Deploy |
| HIGH-RISK | POST | `.../restart` \| `/stop` \| `/start` | Environment lifecycle |
| SAFE | GET | `.../run-state` \| `/deploy-status` | Reads |
| SAFE | GET | `/api/projects/{slug}/environments` | List environments |
| WRITE | PUT | `/api/projects/{slug}/environments` | Update environments config |
| HIGH-RISK | POST | `/api/deploy/promote` | Promote deploy |
| SAFE | GET | `/api/deploy/overview` | Deploy overview across projects |

## Conflict resolution / maintenance

| Risk | Method | Path | Purpose |
|---|---|---|---|
| HIGH-RISK | POST | `/api/projects/{owner}/{repo}/resolve-branch-conflict` | Auto-resolves merge conflicts — can clobber code |
| SAFE (SSE) | GET | `.../resolve-conflict-stream/{job_key}` | Progress stream |
| WRITE | POST | `/api/maintenance/tests/cleanup` | Cleanup test artifacts |
| WRITE | POST | `/api/maintenance/sprints/cleanup` | Sprint cleanup maintenance |
| SAFE | GET | `/api/maintenance/sprints/export` | Export sprint data |
| HIGH-RISK | POST | `/api/maintenance/sprints/import` | Import sprint data — overwrites |

## Alerts / docs-freshness

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/alerts` \| `/api/docs-freshness/warnings` | Reads |
| WRITE | POST | `/api/alerts` \| `/api/docs-freshness/check` | Create alert / trigger check |
| WRITE | DELETE | `/api/docs-freshness/warnings/{id}` | Dismiss warning |
| SAFE | GET | `/api/estimator/health` | Estimator subsystem health |
| WRITE | POST | `/api/issues/{issue_id}/estimate` | Trigger single-issue estimate |
