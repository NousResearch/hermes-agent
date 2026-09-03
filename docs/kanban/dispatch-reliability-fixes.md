# Kanban dispatch-reliability fixes (staged — review before arming)

Three fixes for the silent-failure modes found in the dispatch diagnosis.
**Every behavior change is inert until deliberately armed** (a config flag
flipped + a gateway restart). Nothing here restarts the dispatcher or
changes running behavior on merge alone. A read-only auditor (`hermes
kanban doctor`) reproduces every detection so each fix can be dry-run
against the live board *before* it is armed.

## What changed

| Area | File | Armed by | Default |
|------|------|----------|---------|
| Assignee routing classifier + probe | `hermes_cli/kanban_db.py` | always active (pure additive read helpers) | n/a |
| Create-time reject of unroutable assignee | `hermes_cli/kanban_db.py`, `hermes_cli/kanban.py` | `kanban.validate_assignee_on_create` | **off** (warns only) |
| Dispatcher `skipped_unroutable` split + STUCK telemetry | `hermes_cli/kanban_db.py`, `gateway/kanban_watchers.py` | gateway restart (picks up new code) | active on restart |
| Dispatcher heartbeat write | `gateway/kanban_watchers.py` | gateway restart | active on restart (harmless) |
| Hung-dispatcher standby/takeover | `gateway/kanban_watchers.py` | `kanban.dispatcher_takeover_enabled` | **off** |
| Read-only `doctor` (unroutable / heartbeat / stale) | `hermes_cli/kanban_doctor.py`, `hermes_cli/kanban.py` | nothing — always safe, read-only | n/a |

## Fix 1 — assignee validation + false-idle correction

- New `classify_assignee()` → `profile | pull_lane | unroutable | unassigned`,
  backed by a `KNOWN_PULL_LANES` registry (`fable`, `claude`, `claude-code`,
  `claudecode`, `fable-cc`, `orion-cc`, `orion-research`). Pull lanes have no
  profile on disk *by design* — a `fable` profile would make the dispatcher
  spawn a Hermes worker impersonating Claude Code.
- The dispatcher now splits non-profile ready tasks: known pull lane →
  `skipped_nonspawnable` (correctly idle, unchanged); anything else →
  new `skipped_unroutable` bucket → a distinct **STUCK** log warning. This
  ends the false "correctly idle" that hid the two `fable` cards for 55h.
- `create_task(validate_assignee=True)` refuses an unroutable assignee. The
  CLI wires this to `kanban.validate_assignee_on_create`; with the flag off
  it prints a warning but still creates (no workflow breaks).

**Dry-run now:** `hermes kanban doctor` — the `[assignees]` section lists any
unroutable ready/review task; `[pull-lanes]` lists pull-lane tasks aging past
threshold (what a terminal owes a pull, e.g. the fable cards).

**Arm:** set `kanban.validate_assignee_on_create: true` in `config.yaml`.
Takes effect for new `create` calls immediately (no restart needed — the CLI
reads the flag per call). The dispatcher-side STUCK telemetry arms on the
next gateway restart.

## Fix 2 — hung-dispatcher failover

The dispatcher holds `.dispatcher.lock` (an flock) for its whole process
life. systemd's `Restart=always` recovers a dispatcher that **dies** (flock
frees, a gateway re-acquires). It cannot recover one that **hangs** — the
flock stays held and dispatch stops fleet-wide with nothing watching.

- The lock holder now writes `<kanban>/kanban/.dispatcher.heartbeat`
  (`pid`, `boot_id`, `host`, `ts`, `tick`) every tick. Harmless; always on
  once the new code is running.
- `read_dispatcher_heartbeat()` + `doctor` classify: **healthy** (fresh),
  **hung** (stale + holder pid still alive → the systemd-blind case),
  **dead** (stale + pid gone → auto-recovers), **no_heartbeat** (pre-arm /
  old-code holder).
- When `kanban.dispatcher_takeover_enabled` is true, a gateway that loses the
  lock becomes a hot **standby**: it surfaces a hung holder (CRITICAL log)
  and takes over the moment the flock frees. **It never force-kills the
  holder** — surfacing + takeover-on-free only.

**Dry-run now:** `hermes kanban doctor` `[dispatcher]` line. Today it reports
`no_heartbeat: flock held by pid <N> but writes NO heartbeat — old-code
dispatcher; arms after this branch lands + gateway cycle` — the correct
pre-arm state.

**Arm:** restart gateways to start heartbeats (detection works immediately
after). To enable automatic takeover, also set
`kanban.dispatcher_takeover_enabled: true` (optionally
`dispatcher_takeover_stale_seconds`, default 5×interval, min 60s).

## Fix 3 — stale-task surfacing (no auto-delete, ever)

`recompute_ready` promotes a todo only when *all* parents are done/archived,
so a todo whose parent is `blocked` or itself a stalled `todo` waits forever.
17 such todos (18–40 days old) are on the board now.

- `doctor` `audit_stale_tasks()` surfaces: `deadlocked_todos` (todos past
  `--todo-days` gated behind ≥1 blocked/stalled parent, each with the
  offending parents + statuses) and `stale_ready` (routable-profile ready
  tasks past `--ready-days` that never spawned). **Read-only. Never mutates,
  never deletes** — resolution (unblock parent / cut link / archive) is a
  human call.

**Dry-run now:** `hermes kanban doctor` — the `[deadlocked-todos]` and
`[stale-ready]` sections. Fully complete today; needs no arming (surface-only).

## How to run the auditor

```
hermes kanban doctor            # text, exit 1 if findings
hermes kanban doctor --json     # structured
hermes kanban doctor --ready-days 2 --todo-days 7 --pull-lane-hours 6
```

It is strictly read-only (opens boards `mode=ro`) and safe against the live
fleet at any time — including as a scheduled monitoring pass.

## Landon's arming checklist

1. Review this branch / PR.
2. Merge (no behavior change yet — flags off, gateways still on old code).
3. `hermes kanban doctor` to confirm the pre-arm picture.
4. Restart gateways when ready → heartbeats + `skipped_unroutable` STUCK
   telemetry go live. Re-run `doctor`; the `[dispatcher]` line should flip
   to `healthy`.
5. Optionally set `kanban.validate_assignee_on_create: true` (reject bad
   assignees at create) and/or `kanban.dispatcher_takeover_enabled: true`
   (auto-failover), then restart gateways again.
6. Triage the 17 deadlocked todos surfaced by `doctor` (manual — the tool
   never deletes).
