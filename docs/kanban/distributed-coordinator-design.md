# Distributed Kanban: coordinator / supervisor split

Status: implemented baseline; this document records the design constraints and
remaining follow-up work.
Scope: run Hermes agents across private-network machines (for example, Linux
and macOS workers over Tailscale).
sharing one Kanban board, with tasks, comments, and artifacts crossing machines.

---

## 1. The finding that drives the design

`release_stale_claims()` (`hermes_cli/kanban_db.py:3600`) reclaims any task where
`claim_expires < now`. The **only** escape is this predicate:

```python
if (
    host_local                      # claim_lock.startswith(f"{socket.gethostname()}:")
    and row["worker_pid"]
    and _pid_alive(row["worker_pid"])
    and not heartbeat_stale
):
    # extend the lease instead of reclaiming
```

A remote worker is never `host_local`, so it can **never be rescued**. It is
reclaimed at `DEFAULT_CLAIM_TTL_SECONDS` (900s) even when perfectly healthy.
The task returns to `ready`, the dispatcher spawns it again — while the original
worker is still running. **Duplicate execution.**

This is not a theoretical race. Lease renewal today is *model-driven*:
`heartbeat_claim()` (the only writer of `claim_expires`) is reached solely via the
`kanban_heartbeat` tool, which the LLM chooses to call. `_touch_activity` bridges
API traffic into `last_heartbeat_at` but **not** into `claim_expires`. So healthy
workers routinely blow past the 15-minute lease and survive *only* because of the
PID rescue above. Single-host that is a sound design. Cross-host the safety net
is simply absent.

Note what this means: `last_heartbeat_at` being fresh cannot save a remote claim.
`heartbeat_stale` only *removes* the rescue; it never grants one.

### Corollary

You cannot fix this by teaching `release_stale_claims` to check a remote PID.
"Is the owning process alive?" is a question **only the owning machine can answer**.

---

## 2. The seam already exists inside `dispatch_once`

Draw the line at *"does this operation need a local process table?"*

Grep for `os.kill`, `/proc`, `Popen`, `_pid_alive`, `reap`. Everything that hits
them belongs on the machine that owns the process. Everything else is pure state
transition and belongs in one central place.

| `dispatch_once` step | Today | Belongs to |
|---|---|---|
| promote `todo`→`ready` when parents done | SQL | **Coordinator** |
| expire leases (`claim_expires < now`) | `release_stale_claims` | **Coordinator** (time only) |
| PID-alive lease extension | `release_stale_claims` | **Supervisor** (becomes renewal) |
| crashed-worker detection | `detect_crashed_workers` | **Supervisor** |
| stale-heartbeat detection | `detect_stale_running` | **Coordinator** (decide) |
| kill the stale worker | `detect_stale_running` → `os.kill` | **Supervisor** (execute) |
| `enforce_max_runtime` | mixed | Coordinator decides, Supervisor kills |
| zombie reaping | `reap_worker_zombies` | **Supervisor** |
| circuit breaker / `consecutive_failures` | `_record_task_failure` | **Coordinator** |
| spawn the worker | `_default_spawn` → `Popen` | **Supervisor** |
| dispatch tick lock | `_dispatch_tick_lock(db_path)` — a *local file lock* | **Coordinator** (becomes real, single-process) |

**The rule: the coordinator decides, the supervisor executes.**

Any reclaim with a kill side-effect becomes two-phase. The coordinator cannot
`os.kill` across a network, so it sets `cancel_requested` on the task; the owning
supervisor observes it and terminates its child. This state does not exist today.

### The interface is already half-parameterized

`dispatch_once` already accepts `spawn_fn=` and `signal_fn=` as injection points
(used by tests). `_pid_alive` and `reap_worker_zombies` are the two remaining
local-only calls. That is the entire supervisor interface:

```python
class Supervisor(Protocol):
    def spawn(self, task, workspace_path, board) -> int | None: ...   # -> worker_pid
    def signal(self, pid: int, sig: int) -> None: ...
    def is_alive(self, pid: int) -> bool: ...
    def reap(self) -> None: ...
```

Three of four already exist as parameters. Build 0 is mostly *naming* this.

---

## 3. Lease redesign

Replace the model-driven heartbeat + PID rescue with a **machine-driven lease**.

- Each machine runs one **agent supervisor** process.
- The supervisor claims a task, spawns the child, and then renews the lease on a
  timer for as long as its child is alive (this is where `_pid_alive` legitimately
  lives — it is probing *its own* child).
- `release_stale_claims` becomes purely time-based: no PID, no host prefix. An
  expired lease now genuinely means *"the owning machine stopped vouching for
  this task."*

Consequences:

- **The lease gets much shorter.** 900s was chosen because renewal was unreliable
  and PID-rescue was the real mechanism. With automatic renewal, a 60–90s lease is
  fine. Renew at `TTL/3`. Shorter lease ⇒ faster recovery when the MacBook dies.
- **`kanban_heartbeat` stops being load-bearing** for ownership. It stays as a
  progress signal (feeding `last_heartbeat_at`, the wedged-in-a-logic-loop
  backstop), which is what it should always have been.
- **The coordinator must stamp `claim_expires` itself.** Never accept a
  client-supplied expiry. Two machines, two clocks — a skewed MacBook clock would
  otherwise grant itself an arbitrary lease. The API verb is `renew`, not
  `set_expiry(T)`.

### Fencing (the partition case)

MacBook alive, tailnet drops. The supervisor cannot renew; the coordinator expires
the lease. Nobody else can claim it (capability filter — there is no second macOS
machine), so it sits `ready`. Then the tailnet comes back and the MacBook's child
is *still running*.

The fence token already exists: `heartbeat_claim()` returns `bool` — "True if we
still own it" — because its `UPDATE` is guarded by `AND claim_lock = ?`. Today
nothing acts on that return value. Cross-machine it becomes load-bearing:

> **A failed renewal is not a retry. It is a kill signal.**
> The supervisor must immediately terminate the orphaned child.

Otherwise a reconnecting machine resumes a task the coordinator already gave away.

---

## 4. Routing

Confirmed absent from the schema: there is no `target_machine`, no `machine_id`,
no capability column. `tasks` has `assignee` (a profile name) and nothing else,
and `dispatch_once` claims any `ready` task that has one. Point two machines at
one board today and the Linux box will happily claim the Xcode task.

Capability filtering is therefore a **precondition for a shared board existing**,
not a later refinement.

### Two routing keys, not one

- **`assignee` → profile.** The supervisor can *enumerate profiles from disk*.
  This is a **fact**. If the Mac has no `ios-specialist` profile, `_default_spawn`
  fails, `consecutive_failures` climbs, and the breaker trips after 2.
- **`required_capabilities` → declared.** `macos`, `xcode`, `bluetooth`. This is a
  **promise**. Unverifiable, but expresses intent.

Use both:

```
claimable ⟺ task.assignee ∈ machine.profiles
          ∧ task.required_capabilities ⊆ machine.capabilities
          ∧ (task.target_machine IS NULL ∨ task.target_machine = machine.id)
```

Schema:

```sql
CREATE TABLE machines (
  id            TEXT PRIMARY KEY,   -- stable UUID, NOT hostname
  hostname      TEXT,
  last_seen_at  INTEGER
);
CREATE TABLE machine_profiles     (machine_id TEXT, profile    TEXT);
CREATE TABLE machine_capabilities (machine_id TEXT, capability TEXT);
CREATE TABLE task_capabilities    (task_id    TEXT, capability TEXT);

ALTER TABLE tasks ADD COLUMN target_machine TEXT;   -- optional hard pin
ALTER TABLE tasks ADD COLUMN machine_id     TEXT;   -- owner of worker_pid
ALTER TABLE tasks ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0;
```

Subset check as clean SQL, inside the existing `write_txn` before the CAS:

```sql
NOT EXISTS (
  SELECT 1 FROM task_capabilities tc
   WHERE tc.task_id = t.id
     AND tc.capability NOT IN (SELECT capability
                                 FROM machine_capabilities
                                WHERE machine_id = :me)
)
```

**Use a stable machine UUID, not `socket.gethostname()`.** `claim_lock` already
encodes host (`host:pid` from `_claimer_id()`), which gives a migration path — but
hostnames collide (`MacBook-Pro.local`), change, and are trivially spoofed.

### Empty-capabilities default

A task with no required capabilities is claimable anywhere — which is exactly the
Xcode-task-on-Linux failure. Do not fix this at the call site. Let **profiles
declare their own requirements** (`requires: [macos, xcode]` in the profile's
`config.yaml`) and have `create_task` copy them onto the task. `kanban_create`'s
signature never changes, and the existing single-machine board keeps working
because its profiles declare nothing.

---

## 5. Transport and deployment

A Tailscale address in `100.64.0.0/10` is already a flat, authenticated,
encrypted network with no inbound ports exposed, which is most of what a TLS +
public-coordinator design buys.

- Coordinator binds to the tailnet interface. No certs, no public URL.
- Per-machine bearer token anyway, as defense-in-depth against a compromised
  tailnet node.
- **Store: keep SQLite**, held by the coordinator process. One writer, one
  machine, no network filesystem. This is the store you already have and it is
  correct for two machines. Postgres when you add a third or actually contend.
- Artifacts: coordinator local disk. S3 later, behind the same API.

Every function in `kanban_db.py` is already `(conn, ...) -> result`. That is a
service layer in a trenchcoat. The coordinator holds the `conn`; `RemoteBackend`
mirrors the existing signatures minus that first argument. No Protocol needs to be
invented from scratch.

### Forbidden

`HERMES_KANBAN_DB` already exists as a path override. Pointing it at NFS,
Syncthing, or Dropbox will appear to work and will corrupt the board. Say so
explicitly in the config docs — it is the path of least resistance.

---

## 6. Consumers beyond the nine tools

A `KanbanBackend` covering the agent tools leaves these behind — they open their
own connections:

- `gateway/kanban_watchers.py` — tails `task_events` to push completions to chat
- `hermes_cli/kanban_swarm.py`, `kanban_decompose.py`, `kanban_specify.py`,
  `kanban_diagnostics.py`
- `plugins/kanban/dashboard/plugin_api.py` — the only writer of attachments

Also: there are **no agent-facing attachment tools today**. Attachments exist only
through the dashboard's HTTP API, which stores a resolved absolute local path
(`stored_path=str(dest_path.resolve())`). `kanban_artifact_*` is net-new surface,
not an adaptation.

The nine registered tools are: `kanban_show`, `kanban_list`, `kanban_complete`,
`kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`,
`kanban_unblock`, `kanban_link`. There is no `kanban_claim` — claiming is the
dispatcher's job, and workers never claim. Any design where the worker claims is
reinventing `dispatch_once`.

---

## 7. Workspaces

`_default_spawn` passes `HERMES_KANBAN_WORKSPACE`, `HERMES_KANBAN_BRANCH`, and
`TERMINAL_CWD`; `tasks` carries `workspace_path`. All local paths.

The baseline remote worker runs scratch tasks in a machine-local workspace.
Cross-machine project and worktree tasks still need a portable workspace
specification that the worker's own machine resolves — clone, fetch, checkout —
never a path handed over the wire. That remains a follow-up before routing a
shared repository task to another machine.

---

## 8. Increments

**Build 0 — no network at all.**
Split `dispatch_once` into `decide()` and `execute()` along the process-table line.
Name the `Supervisor` protocol; `spawn_fn`/`signal_fn` already exist as parameters.
Add the routing schema and the claim filter. Replace PID-rescue with
supervisor-driven lease renewal. Make failed renewal kill the child.
Existing tests (`tests/hermes_cli/test_kanban_db.py`,
`tests/plugins/test_kanban_worker_runs.py`) are the guard.
*Exit criterion: on this one box, a task tagged `macos` is never claimed locally,
and a healthy long-running worker is never reclaimed.*

**Build 0.5 — coordinator as a thin wrapper.**
HTTP server over the existing `kanban_db` functions, holding the `conn`, bound to
the tailnet. `RemoteBackend` mirrors the same signatures. Point the *MacBook's
supervisor* at it — not its tools yet.
*Skip a `remote_task_create` tool. It would be deleted in Build 1.*

**Build 1 — the tools.** Route the nine tools through the backend, plus the
watchers, swarm, decompose, diagnostics.

**Build 2 — workspace specs, then artifacts.**

**Build 3 — live events, inbox, mentions.** Deferring this is correct.

---

## 9. Corrections to the original sketch

- `delegate_task` does **not** block the parent. `run_agent.py:5705` reads
  `background=(not _is_subagent)`; top-level delegations return a handle
  immediately. Background children are also **detached** from the parent's
  `_active_children` and are *not* cancelled on parent interrupt
  (`delegate_tool.py:2831-2857`). Two of the three stated reasons for rejecting
  `delegate_task` are false. The third — children are `AIAgent` objects on a
  `DaemonThreadPoolExecutor`, in-process threads — is true, sufficient, and the
  only one that matters: **a thread cannot run on a MacBook.**
- `resolve_secret()` does not exist. The helper is `get_secret(name, default=None)`
  in `agent/secret_scope.py:123`. `cfg_get` does exist as written.
- `hermes worker start` does not exist. A worker is
  `hermes -p <profile> --cli chat -q "work kanban task <id>"` with
  `HERMES_KANBAN_TASK` in env. The sketch's `worker_loop()` — which claims — is
  the wrong shape; see §6.
- **`detect_crashed_workers` is already host-safe.** It filters on the `claim_lock`
  host prefix and its docstring states PIDs from other hosts are meaningless. The
  cross-host PID hazard is in `release_stale_claims`' *rescue* path (§1), not here.
