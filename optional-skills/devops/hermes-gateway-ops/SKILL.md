---
name: hermes-gateway-ops
description: Diagnose and recover Hermes gateway and messaging faults.
version: 1.0.0
author: Pedro Relvas (@PRELVAS94) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [devops, gateway, telegram, troubleshooting, recovery, profiles]
    category: devops
    related_skills: [hermes-agent, hermes-s6-container-supervision]
---

# Hermes Gateway Ops

Diagnose and recover a Hermes gateway that is down, wedged, or answering with
provider errors — and tell a platform-side refusal apart from a local fault. It
covers the process/liveness ladder, provider-failure triage, multiplexing
basics, and hosted-instance (Nous Portal) lifecycle triage. It does not cover
adding a new messaging platform or the s6 supervision inside the Docker image
(`hermes-s6-container-supervision` owns that).

## When to Use

- A bot replies "API failed after N retries", "Rate limited after N retries",
  or "Billing or credits exhausted" — a model-provider or auth failure, not a
  gateway crash
- Messages sit unanswered, or the bot appears disconnected/unresponsive
- You suspect the gateway process crashed, stalled, or is restarting in a loop
- A dashboard or hosted-instance action refuses with "another operation is in
  progress", and you need to know whether the fault is on the box or upstream
- Multiple profiles: `gateway status` says "running via the default-profile
  multiplexer" and you need to know which process actually serves a profile

## Prerequisites

- A Hermes install whose `~/.hermes` (or `HERMES_HOME`) you can read — this
  skill only reads local files and CLI output; nothing here mutates state
  except the two clearly-marked recovery commands
- For the hosted-instance section: an agent OAuth token in `auth.json`
  (written by `hermes setup --portal`)

## How to Run

This skill is a runbook, not a script. Install it from the Skills Hub
(`hermes skills install hermes-gateway-ops`) and it loads when a gateway fault
is being diagnosed, or load it explicitly with `hermes -s hermes-gateway-ops`.

Everything here is read-only except the two clearly-marked recovery commands.
Inspect with the `terminal` tool, read state files with `read_file`, and search
logs with `search_files` — the local half needs no daemon, no credentials and
no network. The hosted-instance half (step 6) needs the platform's token, which
`auth.json` already carries.

## Quick Reference

| Signal | Where to look |
|---|---|
| Process alive? | `hermes gateway status`; `state/gateway.heartbeat` freshness |
| Platform link? | `platforms.<name>.state` in `gateway_state.json` (`connected` / `retrying`) |
| Provider/auth failures? | the `hermes` log, via `search_files`, for `401` / `403` / `429` |
| Restart loop? | Pairs of gateway starts seconds apart in `gateway-starts.log` |
| Clean or forced exit? | `Received <sig> — initiating shutdown` vs `did not exit after SIGTERM, sending SIGKILL` |
| Hosted-instance refusal | REST `/api/agents` `status` + `dashboardUrl` (see reference) |

## Procedure

### 1. Is the gateway process alive?

Run `hermes gateway status` (add `-p <profile>`). Healthy values are
`running`, `degraded` (alive, a platform adapter is failing), `starting`, and
`draining` (a restart or platform action is waiting for active turns to
finish). Anything else — `stopped`, `startup_failed` — means the process is
not serving; the exit reason is printed beside it.

The status verdict also considers *freshness*: a state file saying `running`
whose `updated_at` is minutes old is treated as stale, so a verdict of
`running` means the record is current. `state/gateway.heartbeat` is rewritten
every 30s on a healthy loop — a PID that exists while the heartbeat is frozen
is a wedged loop, not a healthy gateway; a missing PID plus a fresh heartbeat
is impossible (heartbeat thread dies with the process).

### 2. Is the messaging platform connected?

Read the platform entry in `gateway_state.json` — `platforms.<name>.state` is
`connected`, `retrying`, or `connecting` while the gateway is up. On a
multiplexing gateway, secondary profiles' adapters appear as
`<profile>:<platform>` entries beside `served_profiles`. In the log, the
adapter prints `[<name>] Connected to Telegram (polling mode)` (or
`webhook mode`); `Connected in degraded Telegram mode: gateway is alive,
polling will be retried in the background` is alive-but-retrying — see step 3
before touching the process.

### 3. What model/provider is the LIVE session using?

This is the most-missed step: a session inherits the model from the
conversation that started it, not from the current `config.yaml`. A session
that started under a since-expired OAuth provider or a since-changed key will
fail on every turn *even though config is now correct* — restarting the
gateway (step 5) re-reads config and fixes those. The user-facing error text
comes from the agent's turn recovery: `API failed after N retries`,
`Rate limited after N retries`, `Billing or credits exhausted`, each with the
final provider message after the dash. Search the `hermes` log for `401`,
`403`, `429` to classify:

- `401` / `403` / `unauthenticated:bad-credentials` → expired or revoked
  credential; fix the token in config/env, then restart the gateway so live
  sessions re-read it
- `429` with `at capacity` → provider-side overload, not your fault; the turn
  is retried and persisted, nothing to repair locally
- Billing/credits message → the account, not the box; fix upstream
- `Transient agent failure in session` in the gateway log → the agent turn
  crashed; the user message is persisted for redelivery, so a healthy gateway
  recovers it on the next pass

### 4. Read the shutdown/restart history

In the gateway log, `Received SIGTERM as a planned --replace takeover` or
`... as a planned gateway stop` is a *clean* exit somebody asked for;
`Received <sig> — initiating shutdown` is a plain signal; and `Old gateway
(PID N) did not exit after SIGTERM, sending SIGKILL` marks a forced takeover.
`gateway-starts.log` carries one monotonic stamp per start — pairs of starts
seconds apart are a restart loop; a *changing* PID with the same start-time
seconds apart is normal supervision (launchd/systemd/s6 respawning after the
watchdog confirms exit), while the same exit reason repeating means something
is crashing startup itself.

### 5. Recover

On a supervised setup (Docker's s6, systemd, launchd), **do not** start a
second gateway by hand — the new process refuses (one gateway per host
serves every profile) or races the supervisor. Use the lifecycle verbs:

- `hermes gateway restart` — graceful: signals the process, waits for active
  turns (bounded by the drain timeouts), and lets the supervisor respawn it
- `hermes gateway run --replace` — take over a wedged host owner: it asks the
  old PID to exit and escalates to SIGKILL only if it ignores SIGTERM
- `hermes gateway stop` / `start` — plain stop/start through the supervisor

Verify with `hermes gateway status` → `running` and a fresh heartbeat — not
just a zero exit code. On a multiplexing host, a *named* profile that should
run its own gateway but shows "running via the default-profile multiplexer"
means it is served by the host process; `hermes -p <name> gateway run` attaches
to the host rather than starting a second process, and
`hermes gateway migrate --multiplex` folds a fleet of per-profile gateways
into the one host gateway.

### 6. Hosted-instance / platform refusals

"Another operation is in progress for this agent", a portal start/stop/restart
that hangs, or a restart nobody asked for locally: the fault is usually
upstream of the box, and the worst move is to "repair" healthy local state.
The reference file carries the full ladder — attribute the message (upstream
Hermes vs your scripts), prove the instance healthy from outside (the
dashboard's `/api/health` and the login redirect), read the platform's own
view via the REST API, the drain-marker contract (`.drain_request.json`, with
its staleness rules — cancel through the contract, never by hand), the
restart-leftover forensics table, and the escalation ladder.

## Pitfalls

- **Trusting a stale `gateway_state.json` over the CLI verdict.** The status
  command already folds in freshness; a hand-read file can say `running` from
  minutes ago. Read the CLI verdict, then the file's `updated_at`.
- **Assuming `config.yaml`'s model is what a live session uses.** Sessions
  inherit from their start; an expired OAuth provider fails hours later with
  403s on every turn while config looks correct. Restart to re-read config.
- **Treating `degraded` as down.** It means alive with a failing adapter —
  diagnose that adapter (step 2–3) before restarting the whole process.
- **Treating the kanban dispatcher lock line as an error.**
  `kanban dispatcher: another gateway already holds the dispatcher lock` is
  informational: another profile's gateway owns cron dispatch by design.
- **Manually launching `gateway run` alongside a supervised gateway.** The
  host gate refuses the second process (or, with `--force`, splits your
  traffic). Always go through the supervisor's lifecycle verbs.
- **Clearing `.drain_request.json` by hand.** It is the platform's only
  in-band control channel; a leftover marker with a *different* epoch is
  ignored by design, and only the same-epoch contract clear
  (`gateway.drain_control.clear_drain_request()`) is safe. The reference has
  the exact recipe.
- **Missing that one gateway serves every profile.** `hermes -p <name> gateway
  run` on a live multiplexer attaches (prints the host PID and served set,
  exits 0) instead of starting a second process — that exit 0 is a report, not
  a failure.

## Verification

Recovery is complete only when all of these hold — read the CLI's verdict, not
a hand-read file:

1. `hermes gateway status` reports `running` (or `degraded` with one named,
   understood adapter) and the record is fresh.
2. `state/gateway.heartbeat` `updated_at` is seconds old — proof the loop is
   turning, not just that a PID exists.
3. The affected platform is `connected` and a test message round-trips.
4. `gateway-starts.log` gains no new pair of starts seconds apart (no restart
   loop) once the fix is in.
5. A turn completes without an `API failed after N retries` / `Rate limited
   after N retries` diagnostic reaching the chat.

If the fault was platform-side, verification is the platform's own view
agreeing (see the reference) plus a healthy `/api/health`.

## References

- `references/platform-lifecycle-and-dashboard-access.md` — hosted-instance
  lifecycle triage: attribution searches, the outside-in health proof, the
  portal REST read, the drain-marker contract and safe-cancel recipe,
  restart-leftover forensics, and the escalation ladder
