# Control Centre phase 2 audit

What exists before this phase, so nothing here rebuilds it. Checked against the source;
file:line references included so the claims are verifiable.

The short version: **the backend is almost complete and the frontend is the gap.** Seven
agent write routes exist, are gated, audited and tested, and none of them has a form. One
read route is genuinely missing, and one runtime capability is not exposed at all.

---

## 1. Existing frontend routes

Hash routes, one per `NAV` entry (`components/shell.tsx:28`), plus one member route.

| Route | Screen | Writes today |
|---|---|---|
| `#/overview` | Overview | — |
| `#/agents` | Agent list | — |
| `#/agents/<id>` | Agent profile — tabs: Work, **Soul**, Knowledge, Channels, Permissions, Usage | **Soul only** |
| `#/objectives` | Objectives | submit |
| `#/work` | Kanban board | decide |
| `#/approvals` | Blocked work | decide |
| `#/activity` | Audit decisions | — |
| `#/knowledge` | Corpora | — |
| `#/channels` | Declared + all 22 available | apply |
| `#/automations` | Schedules | pause / resume / delete / declare |
| `#/policies` | Compiled policy | — |
| `#/usage` | Budget observations | — |

There is **no** schedule *detail* route, and no route for creating an agent.

## 2. Existing backend routes

**Reads** (`nova/control/api.py:263-296`): `/health`, `/identity`, `/agents`, `/tasks`,
`/tasks/<id>`, `/policy`, `/policy/simulate`, `/decisions`, `/budget`, `/knowledge`,
`/objectives`, `/automations`, `/channels`, `/agents/<id>/soul`,
`/agents/<id>/automations`.

**Writes** (`nova/control/auth.py`, `WRITE_ROUTES`): `/work/decide`, `/objectives/submit`,
`/channels/apply`, `/automations/decide`, `/automations/create`, `/agents/create`,
`/agents/update`, `/agents/soul`, `/agents/duplicate`, `/agents/archive`,
`/agents/restore`, `/agents/delete`.

All twelve are `admin` except the two viewer-readable member leaves. RBAC is enforced
twice — at the transport (`server.py`) and again in `ControlAPI.write` — and every write
emits `intent` → `committed`/`failed` with the authenticated human's name.

## 3. Agent operations already available

`nova/agents/manage.py`, each one an edit to the bundle through
`nova.spec.writer.edit` (validated against the whole bundle, atomic, traversal-proof):

| Operation | Function | Route |
|---|---|---|
| Create | `create_agent` | `POST /agents` |
| Update declared fields | `update_agent` | `POST /agents/<id>/update` |
| Read persona | `instructions_of` | `GET /agents/<id>/soul` |
| Rewrite persona | `set_instructions` | `POST /agents/<id>/soul` |
| Duplicate | `duplicate_agent` | `POST /agents/<id>/duplicate` |
| Archive / restore | `archive_agent` | `POST /agents/<id>/archive` `…/restore` |
| Delete | `delete_agent` | `POST /agents/<id>/delete` |

Settable fields are allowlisted (`AGENT_FIELDS`): name, role, description, enabled, model,
tools, knowledge, permissions, approval, limits, delegation.

## 4. Schedule operations already available

`nova/runtime/hermes/automations.py` over `cron.jobs`, which stays the only scheduler:

| Operation | Adapter | Route |
|---|---|---|
| List | `list_automations` | `GET /automations` |
| List for one agent + executions | — | `GET /agents/<id>/automations` |
| Declare | `create` (via the compiler) | `POST /automations` |
| Pause / resume | `set_enabled` | `POST /automations/<id>/pause`, `…/resume` |
| Edit name/schedule | `update` → `cron.jobs.update_job` | `POST /automations/<id>/update` |
| Delete | `delete` | `POST /automations/<id>/delete` |
| Executions | `executions` → `cron.executions.list_executions` | included above |
| Scheduler liveness | `scheduler_health` | included in `/automations` |

`update` is narrow by design: **name and schedule only.** The objective passed the compiler
on the way in; a route that could rewrite it afterwards would make the compiler a gate you
walk through once and then step around.

## 5. UI components to reuse

| Component | File | Use |
|---|---|---|
| `GlassPanel`, `GlassCard`, `SectionHeader`, `StatusPill`, `InfoDot` | `components/glass.tsx` | every surface |
| `PanelBody`, `MetricCard`, `ErrorState` | `components/panel.tsx` | loading / empty / error states |
| `usePanel(path, refreshMs, nonce)` | `lib/hooks.ts` | all reads; `nonce` re-reads after a write |
| `load` / `post` | `lib/api.ts` | `post` throws with the server's own message |
| `SoulEditor` | `screens/soul.tsx` | **the save-behaviour pattern this phase copies** |

The Soul editor already implements every requirement in §5 of the brief: a null draft means
"showing the server's copy" so a background poll cannot clobber an unsaved edit; Save is
disabled while clean; saving, error and success states are distinct; and *saved* and
*applied* are reported separately rather than as one tick.

## 6. Missing frontend functionality

| # | Gap | Backend ready? |
|---|---|---|
| 1 | Create Agent form | yes |
| 2 | Edit identity (name, role, description) | yes |
| 3 | Edit model | yes |
| 4 | Edit tools / permissions | yes |
| 5 | Duplicate / archive / restore / delete controls | yes |
| 6 | Agent status beyond idle/working | partly — see below |
| 7 | Schedule create form | yes |
| 8 | Schedule edit form | yes |
| 9 | Schedule detail with executions | yes |
| 10 | Agent ↔ schedule navigation both ways | yes |
| 11 | Activity / logs surface | **no** |

### The two genuine backend gaps

**a. No route returns an agent's editable declaration.** `GET /agents` returns a
presentation view — it carries `model`, `limits` and `knowledge_sources` but **not `tools`
or `permissions`**, and permissions are only reachable through `/policy`. An edit form needs
exactly the shape `update_agent` accepts. That is one new read route
(`GET /agents/<id>/config`), not a second data model: it is the read counterpart of
`AGENT_FIELDS`.

**b. Toolsets are not exposed.** `toolsets.TOOLSETS` (60 entries) is the runtime's own
registry. NOVA cannot import it outside the adapter package, so it needs an
`AgentRuntime.toolsets()` method the way channel discovery got one.

### Choices for the form's option lists — all from existing reads

| Field | Source |
|---|---|
| Permissions | `/policy` → `permissions` keys (declared in `policy.yaml`) |
| Toolsets | new `toolsets()` adapter method over `toolsets.TOOLSETS` |
| Knowledge | `/knowledge` → `sources` |
| Channels | `/channels` → `catalogue` (22, from plugin manifests) |
| Model | free text plus `deployment.yaml` defaults |

### Agent status — what can honestly be shown

Backed by real data: **Disabled** (`enabled: false`), **Working** (running/ready tasks),
**Needs a human** (blocked tasks), **Scheduled** (has an enabled automation), **Drifted**
(`in_sync: false`), **Idle** (materialised, nothing running).

**Not available:** a live process heartbeat per agent. There is no per-agent liveness signal
— Hermes runs agents as spawned workers, and the control plane sees their work rows, not
their processes. So "Running" in the sense of "a process is alive right now" is **not**
shown. An agent with no work and no schedule reads *Idle*, and where the runtime cannot be
reached at all the screen says *Status unavailable* rather than guessing.

## 7. Implementation plan

1. `GET /agents/<id>/config` — the editable declaration, admin-only (it carries permissions).
2. `AgentRuntime.toolsets()` + Hermes implementation, surfaced on the config read.
3. A shared `useEditor` hook extracted from the Soul editor's proven save behaviour, so
   every new form gets dirty-tracking, disabled-when-clean, saving/error/success and the
   saved-versus-applied split without reimplementing them.
4. Agent profile: Overview (editable identity), Model, Tools, Channels, Schedules tabs.
5. Lifecycle controls: duplicate (with id + name), archive, restore, delete-with-confirm.
6. Create Agent: a stepped form — Identity, Instructions, Model, Capabilities, Channels,
   Review — posting once at the end.
7. Schedules: create and edit forms with simple and advanced modes, validated by the
   runtime's own parser before saving; schedule detail with real executions; links both ways
   between an agent and its schedules.
8. Activity: structure only, wired to the reads that exist (`/decisions`, task runs and
   events). No log tailing — that route does not exist, and inventing one is out of scope.

## 8. Testing plan

Unit and API tests beside the existing ones, then Playwright against a real control plane
for: create → appears in list → open profile → edit description → save → refresh →
persisted; the existing Soul flow unchanged; duplicate → independent; archive → restore;
schedule create → edit → refresh → disable. Plus one failure case per operation — invalid
configuration, unknown agent, invalid schedule, permission denied.


---

## 9. Result

Added after implementation, so this document is not a plan nobody checked against its
outcome.

**Complete and verified in a browser** (35 checks, `tests/browser/control_centre_e2e.py`):
create an agent through the stepped form; edit identity, model, tools, permissions and
knowledge; edit the Soul; duplicate, archive, restore and delete with typed confirmation;
list, edit, pause and delete schedules; navigate agent → schedules and schedule → agent.
Every check asserts against the files the runtime ended up with, not against the UI's own
claims. 798 platform tests and the protected-identifier check pass.

**Not done, and why:**

- **Channel association from the agent form.** A grant is declared on the connection
  (`channels.yaml`), which Phase 9 made the single place it lives. The Channels step shows
  what would reach the agent and where to change it; offering a second place to set it would
  guarantee the two disagree.
- **Live "Running" status.** There is no per-agent process heartbeat. Disabled, Not yet
  applied, Awaiting a human, Working and Idle are all backed by real data; a sixth state
  claiming a process is alive would not be.

---

## 10. Phase 3: runtime logs, execution monitoring, and credential entry

Both were listed above as not done — logs because the route did not exist, credential entry
because NOVA deliberately could not write a secret. The administrator asked for both, so
both now exist. The second changes a security property of the platform, so it is worth
stating what changed and what did not.

### What was added

| Surface | Route | Role |
|---|---|---|
| Which logs an agent has | `GET /agents/<id>/logs` | admin |
| A bounded tail of one log | `GET /agents/<id>/logs?stream=&lines=` | admin |
| Work, executions, decisions, log inventory | `GET /agents/<id>/activity` | admin |
| Which credentials are needed, and which are set | `GET /agents/<id>/credentials` | admin |
| Set or clear credentials | `POST /agents/<id>/credentials` | admin |

### Credential entry: what changed, precisely

**NOVA can now write a secret.** It could not before, and that was a deliberate property.
What changed is narrower than "the restriction was lifted":

* **`.env` is still on `materialize.NEVER_WRITE`.** `nova apply` still cannot touch it, so a
  configuration push cannot overwrite a credential. Credential entry is a *separate* path,
  not a relaxation of the existing one. A test applies the bundle after a write and asserts
  the value survives.
* **The name is allowlisted, and this is the load-bearing control.** `<profile>/.env` is
  loaded into the environment of the process that runs the agent. A write path accepting any
  name could set `LD_PRELOAD`, `PYTHONPATH`, `PATH` or `BASH_ENV` and turn "set a Slack
  token" into "run my code inside the agent". `nova/credentials.py` derives the writable set
  from the tenant's own declaration — the channels that grant this agent, its model, the
  deployment default — and everything else is refused by name. Revoking a channel grant also
  revokes the ability to write that channel's credentials, because the set is derived per
  request rather than cached.
* **Values go one way only.** There is no read that returns one. The status of a credential
  is "set" or "not set". The audit log records which names changed and who changed them, and
  a test greps it for the value to prove it is absent.
* **Written 0600, merging.** The temporary file is created with that mode rather than
  chmod-ed afterwards, so the value is never briefly world-readable. Comments and variables
  NOVA does not model survive the write: the file is the operator's.

### Logs: what is and is not done

* Tailed from the end, capped in **bytes before lines**, so a caller cannot ask for a
  megabyte by asking for a lot of lines.
* The stream name is looked up in an explicit map, never resolved against a directory —
  which is how traversal happens even when each individual check looks fine.
* **Admin-only, and no sanitisation is attempted.** A log line can carry anything the
  runtime wrote — a prompt, a tool argument, part of a document NOVA never saw. A sanitiser
  that misses one pattern is worse than a clear statement of who may read, so the gate is
  the role.
* **Not done: live tailing.** Each read is a snapshot with a Refresh button. Streaming would
  need a long-lived connection through the control plane, and the value of watching a log
  update in real time did not justify that surface today.

### Verified

19 unit tests (`tests/platform/test_credentials_and_logs.py`) and 20 browser checks
(`tests/browser/credentials_and_logs_e2e.py`), which type a distinctive secret through the
real form and then grep every read route and the audit log for it from inside the browser's
own session. 817 platform tests pass.
