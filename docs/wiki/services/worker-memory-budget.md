# Worker Memory Budget (t_8e8ec4de)

> Wiki page for the worker-scope memory overcommit fix.
>
> Status: shipped on branch `canonical-deploy-t_7161d9a9`.
>
> The per-worker cgroup `MemoryMax` and the dispatcher's concurrency cap are
> ONE budget. They are computed from a single shared module
> (`hermes_cli/memory_budget.py`) so they can never disagree:
>
> ```
> effective_max_workers x per_worker_memory_max <= 0.7 x MemTotal
> ```
>
> The language server is bounded too — `lsp.max_server_heap_mb` caps the V8
> heap and `lsp.max_clients` caps the number of language-server PROCESSES one
> Hermes process may run.

## Overview

The host's memory pressure was **not** a host-starvation problem and **not** an
`rpc.gssd` leak. It was a **worker-scope overcommit**.

Each Kanban worker got a cgroup `MemoryMax` of half the host's RAM (3971 MiB
on a 7942 MiB host), but the dispatcher would run up to 8 workers concurrently.
The scopes could therefore collectively claim **4x more memory than the host
had**, and the kernel OOM-killed whichever scope crossed the line — taking a
healthy worker with it.

On 2026-10-28 13:01:11 UTC the kernel OOM-killed **two** processes in
`hermes-worker-kanban-t_3ae7ccc7-run-36574.scope`: `hermes` (pid 1433203) and
`node-MainThread` (pid 1434287, RSS 1.4 GB — the pyright LSP server the worker
spawned). The worker was healthy; it was killed by a memory limit it never
should have been exposed to.

### The arithmetic (verified 2026-10-28)

| Quantity | Before | After |
|---|---|---|
| Per-worker scope limit | `min(slice, MemTotal/2, 4 GiB)` = 3971 MiB | 1400 MiB |
| Concurrency cap | `clamp(MemTotal/512, 2, 8)` = 8 | `floor(0.7 x MemTotal / limit)` = 3 |
| Sum of scope limits | 31,768 MiB (**4.0x over**) | 4,200 MiB (within 5,559 MiB) |

At the board's own `max_spawn` of 3 the host was already oversubscribed by 50%.
Nothing enforced that the *sum* of the scopes' limits fit in RAM.

### Second finding — the pyright LSP is the dominant consumer, and it was unbounded

`~/.hermes/config.yaml` and `profiles/ultron/config.yaml` both enabled the LSP
with no memory cap, no idle timeout, and no max-one-per-host setting:

| pid | profile | RSS |
|---|---|---|
| 1493079 | librarian | 1,786 MB |
| 1588219 | librarian | 1,485 MB |
| 1776321 | ultron | 432 MB |

Two librarian workers each spawned their own pyright. The node process is the
OOM victim class (`task=node-MainThread` in every kernel OOM record). One
pyright at 1.7 GB is 21% of the host's RAM.

The mechanism: a Kanban worker has **no operator anchor**
(`operator_workspace_roots()` returns empty when `HERMES_KANBAN_TASK` is set),
so every checkout it touches is UNTRUSTED — and an untrusted root gets its own
client key, so pyright's multi-root sharing never engages. A worker that
wrote into two repos therefore ran TWO pyright processes.

## Architecture

### One shared module, two consumers

`hermes_cli/memory_budget.py` is the single source of truth. It is
deliberately dependency-free (stdlib only) so `tools` can import it without a
circular import back into `hermes_cli.kanban_db_dispatch`.

| Function | Purpose |
|---|---|
| `host_memory_total_bytes()` | Physical RAM, or `None` when unknown. `None` means "unknown" and every caller fails open. |
| `per_worker_memory_max_bytes(host, max_workers, slice_limit)` | The per-worker cgroup `MemoryMax`. |
| `effective_max_workers(host, per_worker_bytes)` | The concurrency cap. |
| `budget_is_consistent(max_workers, per_worker_bytes, host)` | The invariant check. |

### The derivation order — limit PRIMARY, cap derived from it

The two quantities cannot be chosen independently. A worker's cgroup limit has
to be large enough that a real worker is not OOM-killed by its own scope (the
exact failure `t_4d12c8eb` fixed when a mis-inherited 1 GiB limit killed workers
mid-implementation). The concurrency cap then has to be small enough that the
sum of those limits fits the host.

So:

1. **per-worker limit** = the worker's real footprint (measured, with margin),
   bounded by the host's per-worker ceiling and by the slice's own limit;
2. **concurrency cap** = `floor(budget / per_worker_limit)`.

Deriving the cap from the limit (rather than the limit from the cap) guarantees
the invariant holds AND that the limit is never too tight to run a worker.

`MEMORY_GUARD_MB_PER_WORKER = 1400` is the MEASURED peak of a real worker
scope on the homelab host (2026-10-28, `memory.peak` = 1350995968 /
1408446464 / 386093056 bytes across three live scopes): the Hermes runtime plus
a pyright language server. The old value was 512 MiB — a 2.7x under-estimate
that made the derived cap of 8 workers unfundable on a 7942 MiB host.

### The per-worker limit is the MINIMUM of

- `mb_per_worker` (1400 MiB) — the worker's real footprint, so a worker is never
  OOM-killed by its own scope (the `t_4d12c8eb` failure mode);
- `0.7 x host / max_workers` — the host-wide budget; keeps the SUM of
  concurrent scopes inside the host's budget;
- `0.7 x host / 2` — even ONE worker may not claim more than half the budget;
- `4 GiB` — no single worker may claim the host;
- the slice's `memory.max` — the real parent cgroup. The gateway unit's own
  `MemoryMax` is deliberately NOT used (see `t_4d12c8eb`).

A slice limit BELOW the 64 MiB floor is not a usable bound and is ignored —
clamping it up to the floor would silently turn "no usable bound" into "a 64 MiB
limit", which kills every worker.

### The two consumers

**`hermes_cli/kanban_db_dispatch.py::derive_default_max_in_progress()`** —
replaces `clamp(MemTotal/512, 2, 8)` with
`clamp(floor(0.7 x MemTotal / per_worker_limit), 2, 8)`.

**`tools/process_registry.py::_worker_memory_max_bytes()`** — the scope's
`MemoryMax`, now derived from the SAME cap the dispatcher uses
(`_effective_max_workers_for_scope()` → `resolve_max_in_progress(
configured_max_in_progress())`).

An explicitly configured `kanban.max_in_progress` overrides the derived cap; the
operator's number then wins and the limit shrinks to keep the host-wide
invariant. The per-BOARD cap (`kanban.max_spawn`) is deliberately NOT used:
scopes from several boards share one host, and the host-level cap is what
bounds how many scopes can exist at once across all of them.

### The scope budget is an admission check over EVERY scope (2026-10-29)

The invariant above is stated over the **kanban** concurrency cap. That is not
the whole population of scopes that claim memory.

Cron executions are dispatched through the SAME scope mechanism
(`cron/scheduler.py` → `restart_safe_gateway_child_argv` →
`_build_systemd_scope_argv`), so every cron execution holds a scope with the
SAME per-worker `MemoryMax`. Cron concurrency is bounded by
`cron.max_parallel_jobs`, which is **unset by default and therefore
unbounded**.

Measured on the homelab host (2026-10-29), a 3-minute sample of the live
cgroup tree:

```
peak 24 concurrent hermes-worker-*.scope  ->  33600 MiB claimed
budget (0.7 x 7942 MiB)                   ->   5559 MiB
kanban-only accounting said                ->   3 x 1400 = 4200 MiB  "consistent"
```

The invariant was **true** and the host was **6x overcommitted**, because the
accounting did not cover the scopes that were really running.

The fix is an **admission check**, not a resize:

`tools/process_registry.py::worker_scope_budget_exhausted()` counts the live
`hermes-worker-*` scopes (kanban AND cron), plus one for the scope about to be
created, and compares that against `budget // per_worker_limit`. When the host
cannot afford another scope, `restart_safe_gateway_child_argv()` raises
`WorkerScopeBudgetExhausted` and **no scope is created**.

Why admission and not resizing:

- The `MemoryMax` is read at **scope-creation** time. An existing scope keeps
  the limit it was created with, so shrinking a NEW scope's limit does not
  reduce the collective claim — existing scopes still hold theirs.
- Shrinking the new scope's limit WOULD starve the new worker (at 24 live
  scopes the limit would fall to 231 MiB, far below the measured 1400 MiB
  footprint). That re-creates the `t_4d12c8eb` failure: a healthy worker
  OOM-killed by its own scope mid-task.
- A scope that is never created costs nothing. A scope created without room
  costs a healthy worker.

Callers handle the refusal as an **infrastructure** condition, not a failure:

- **Kanban** (`hermes_cli/kanban_db_dispatch.py`) —
  `_is_infrastructure_spawn_failure()` classifies it alongside
  `RestartSafeScopeUnavailable`, so the card is re-queued and retried on a
  later tick WITHOUT spending the card's retry budget (#114720).
- **Cron** (`cron/scheduler.py`) — `_defer_job_for_scope_budget()` parks the
  job the way `cron.quota_hold` parks a job behind a closed provider window:
  the fire is coalesced to a later instant, the claim is released, and NO
  failure is recorded (no cron incident, no burned repeat).

### The LSP bounds

Two new `lsp` config keys, both with safe defaults in
`hermes_cli/config_defaults.py`:

| Key | Default | Effect |
|---|---|---|
| `max_server_heap_mb` | `768` | V8/Node heap ceiling for Node-based servers (pyright, typescript). Implemented by adding `--max-old-space-size` to `NODE_OPTIONS`. Idempotent: an existing `--max-old-space-size` is replaced, other `NODE_OPTIONS` flags preserved. `0` = no ceiling. |
| `max_clients` | `2` | Hard cap on how many language-server PROCESSES one Hermes process may run at once. When the cap is reached the least-recently-used client is shut down and respawned on demand. |

The client cap is a **RESOURCE bound, not a trust relaxation**: an untrusted
workspace keeps its own language-server process (the trust boundary is
unchanged), but a worker that touches many checkouts cannot accumulate one
process per checkout.

Eviction semantics (`LSPService._evict_for_capacity`): the cap counts LIVE
language-server processes. When the key the caller wants is already live,
nothing new is spawned and the service only has to come down TO the cap. When
it is NOT live, a new process is about to start — so if the service is sitting
exactly ON the cap, one extra client is evicted to make room. A service already
OVER the cap is brought back down to the cap either way.

## Operation

### Verify the invariant on the live host

```bash
cd /home/serveradmin/.hermes/hermes-agent
./venv/bin/python - <<'PY'
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import memory_budget as mb
import tools.process_registry as pr
host = mb.host_memory_total_bytes()
cap = kbd.derive_default_max_in_progress()
limit = pr._worker_memory_max_bytes()
print(f"host {host//2**20} MiB  cap={cap}  limit={limit//2**20} MiB")
print(f"sum {cap*limit//2**20} MiB  budget {int(host*mb.MEMORY_BUDGET_FRACTION)//2**20} MiB")
print("invariant holds:", mb.budget_is_consistent(cap, limit, host))
PY
```

Expected on the homelab host (7942 MiB): `cap=3`, `limit=1400`, `sum=4200 MiB`,
`budget=5559 MiB`, invariant `True`.

### Verify the live scope limits

```bash
SL=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice
SUM=0
for d in $SL/hermes-worker-kanban-*.scope; do
  [ -d "$d" ] || continue
  m=$(cat "$d/memory.max"); SUM=$((SUM+m))
  echo "$(basename "$d" .scope): max=$((m/1048576))MiB cur=$(( $(cat "$d/memory.current") /1048576 ))MiB"
done
echo "SUM of scope limits: $((SUM/1048576)) MiB"
```

The limit is read at SCOPE-CREATION time. Scopes created before the change
keep the old value until the worker is re-spawned.

### Verify the LIVE scope count (the budget covers every scope, not just kanban)

```bash
SL=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice
ls "$SL" | grep -c '^hermes-worker-'
```

The invariant above is expressed over the KANBAN cap. The scopes that actually
claim memory include CRON executions — they use the same scope mechanism and
the same per-worker `MemoryMax`. See "The scope budget is an admission check"
under Architecture.

### Check for OOM kills

```bash
journalctl --since "-24h" --no-pager -k | grep "oom-kill" | \
  sed 's/.*oom_memcg=//' | cut -c1-140
```

A cgroup-limit OOM names the scope that hit its own `MemoryMax`:

```
oom-kill:constraint=CONSTRAINT_MEMCG,...,oom_memcg=/user.slice/.../hermes-worker-kanban-<task>-run-<n>.scope
```

### Run the regression tests

```bash
cd /home/serveradmin/.hermes/hermes-agent
./venv/bin/python -m pytest \
  tests/hermes_cli/test_worker_memory_budget.py \
  tests/agent/lsp/test_client_cap.py \
  tests/agent/lsp/test_heap_cap.py \
  tests/tools/test_process_registry.py -q
```

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `TERMINAL_LOCAL_MEMORY_MAX_MB` | unset | Operator override of the per-worker limit, in MiB. Values below the 64 MiB floor are ignored with a warning. When it TIGHTENS the bound it wins. |
| `kanban.max_in_progress` | unset (derived) | Explicit host-level concurrency cap. Overrides the derived value; the per-worker limit then shrinks to keep the invariant. |
| `kanban.max_spawn` | per-board | Per-board concurrency. NOT used for the scope limit — boards share the host. |
| `cron.max_parallel_jobs` | unset (**unbounded**) | Max concurrent cron executions per profile home. Unset means no limit, so cron scopes can exceed the host budget on their own. Set it (e.g. `2`) to bound the cron side of the scope budget. |
| `lsp.max_server_heap_mb` | `768` | V8 heap ceiling for Node language servers. `0` disables. |
| `lsp.max_clients` | `2` | Language-server processes per Hermes process. Clamped to >= 1. |

The memory-pressure watchdog's remediation text (`~/.hermes/scripts/
memory_pressure_watchdog.py`) was corrected in the same change: it used to
tell the reader to "raise the scope's limit", which is the OPPOSITE of the fix
for this root cause. It now states the invariant explicitly and names both
likely causes — a runaway language server (check `lsp.max_server_heap_mb`) or
too many scopes running at once.

## Troubleshooting

### Symptom: a worker was OOM-killed with `constraint=CONSTRAINT_MEMCG`

The scope hit its own `MemoryMax`. Two possible causes:

1. **One worker used more than its share** — a runaway language server. Check
   `lsp.max_server_heap_mb` and `lsp.max_clients`. The scope limit is a
   backstop, not a budget: a server that grows to the scope limit takes the
   worker down with it.
2. **Too many scopes ran at once** — the cap and the limit disagreed. Verify
   with the invariant check in Operation above. Note that scopes from
   DIFFERENT boards share the host budget, and cron worker scopes consume the
   same slice.

**Do NOT raise the scope limit.** Raising it while the cap stays put just
moves the OOM later — the sum of the concurrent scopes still exceeds the host.

### Symptom: a card or cron job is deferred with "memory budget is exhausted"

This is the admission check working as designed, not a failure. The host
already had as many worker scopes as the budget allows, so no new scope was
created.

Check what is holding the scopes:

```bash
SL=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice
for d in "$SL"/hermes-worker-*.scope; do
  [ -d "$d" ] || continue
  echo "$(basename "$d"): max=$(( $(cat "$d/memory.max") /1048576 ))MiB cur=$(( $(cat "$d/memory.current") /1048576 ))MiB"
done
```

- **Cron scopes** are the most common cause: `cron.max_parallel_jobs` is unset
  by default, so cron parallelism is unbounded. Set it in `config.yaml`.
- **A leaked scope** — a scope whose task is already `done` but whose scope is
  still alive — holds a full claim against the budget. Check whether the
  worker process is actually still running.
- **Long-running workers** legitimately fill the budget. The card or job is
  retried on a later tick.

### Symptom: the scope limit looks too small

The limit is the MINIMUM of the measured footprint, the host budget, the
per-worker ceiling, and the slice limit. If it is below 1400 MiB on a host with
enough RAM, check whether the slice has a `memory.max` of its own:

```bash
cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/memory.max
```

`max` means unlimited. Anything else is a real bound and wins.

### Symptom: the derived cap is 2 on a large host

The cap is `floor(0.7 x MemTotal / per_worker_limit)` clamped to `[2, 8]`. A cap
of 2 means the host's budget only affords two workers at the footprint size —
the operator must set `kanban.max_in_progress` explicitly to raise it, and the
per-worker limit will shrink to compensate.

### Symptom: pyright still grows past `max_server_heap_mb`

`--max-old-space-size` caps the V8 heap, not total RSS. Node's baseline plus
the heap still land above the configured MiB. The ceiling is a GC trigger, not
a hard RSS limit — the scope's `MemoryMax` is the hard limit.

### Symptom: diagnostics stop appearing after the client cap is hit

The LRU client was evicted and is respawned on demand. The next request for
that workspace re-spawns the server (which costs a moment of latency). Raise
`lsp.max_clients` if the churn is visible, at the cost of one more server per
Hermes process.

## Dependencies

- `hermes_cli/memory_budget.py` — the shared budget module (new in this
  change). Imported by both `hermes_cli/kanban_db_dispatch.py` and
  `tools/process_registry.py`.
- `agent/lsp/manager.py`, `agent/lsp/servers.py`, `agent/lsp/eventlog.py` —
  the language-server client cap and the V8 heap ceiling.
- `hermes_cli/config_defaults.py` — the `lsp.max_server_heap_mb` and
  `lsp.max_clients` defaults.
- systemd cgroup v2 — the scope `MemoryMax` is a kernel-enforced limit.
- `~/.hermes/scripts/memory_pressure_watchdog.py` — the alert path whose
  remediation text was corrected here.

## See Also

- [Canonical Deploy Branch](canonical-deploy-branch.md) — the branch this
  change ships on.
- `t_4d12c8eb` — the SIBLING card: fixed a *mis-inherited* limit (the gateway
  unit's 1 GiB cap leaking into worker scopes, which pinned every worker to
  1 GiB). That fix is correct and raised the per-worker limit from 1 GiB to
  3971 MiB. **This card is about the next layer**: 3971 MiB x N workers is
  still more than the host's RAM.
- `t_5c5fd8c5` (rpc.gssd swap holder) and `t_3839c12d` (PVE virtiofs memfd) —
  about OTHER hosts (wolverine / ws4 / ws3). Do not conflate them with this
  host.
- t_8e8ec4de — this card.
