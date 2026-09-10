# Control Room — Operator Guide (CR-607)

## What it is

Control Room is the unified keyboard-first surface for what needs Sahil's
attention across CLI, Ink TUI, native Desktop and the Kensei Dashboard. It
shows the same contract snapshot everywhere: agents, tasks, peer messages,
system state — ranked deterministically by severity, with `needs-you` counts
surfaced in status bars and badges.

## Opening it

| Surface | Key / command |
|---------|---------------|
| CLI | `Ctrl+P` anywhere outside a prompt, or type `/control` |
| Ink TUI | `Ctrl+P` |
| Native Desktop | `Ctrl/Cmd+P` (palette stays `Cmd/Ctrl+K`) |
| Dashboard | `Ctrl+P` (command bar stays `Cmd/Ctrl+K`) |

Close with `Esc` (CLI/TUI/Desktop) or navigate back (Dashboard).

## What you see

- **Attention list** — ranked rows: critical (blocked/stalled), error
  (stopped gateway), warning (running but slow), info. `needs-you` counts
  items ranked critical or error.
- **Counts** — needs-you, active agents, running tasks, unread messages,
  system severity.
- **Sections** — Needs You / Agents / Tasks / Messages / System on Desktop
  and Dashboard; the TUI overlay shows the attention list + counts.
- **Capabilities** — anything without a live backend path renders as
  "unavailable", never as a dead button.

## Actions

All actions are confirmed before execution (the router returns a preview
first; you must confirm). Actions available in v1:

| Action | Effect | Confirmation |
|--------|--------|--------------|
| approval.respond | allow/deny a pending approval via the Peer/gateway path | required |
| process.kill | kill an owned background process | required |
| subagent.interrupt / steer | interrupt or steer a subagent | required |
| delegation.pause | pause/resume delegation spawning | required |
| peer.send / peer.inbox | send or release/refuse a peer message | required |
| kanban.create / comment / move | create, comment, or validated move (block/unblock/complete) | required |
| agent.run | start a background agent run (surface-injected runner) | required |

Stale rows are rejected with `refresh required`; duplicate submits are
rejected; cross-profile targets are rejected.

## Reading the badges

- TUI badge below the composer: `needs-you N` (blank until the first poll —
  never a fake zero). Polls every 5s.
- Desktop status bar: attention dot + count, click opens Control Room. Polls
  every 5s.
- Dashboard HealthStrip: `● N need you` pill (click opens Control Room),
  plus existing agent/gateway counts. Polls every 15s.
- CLI status segment: attention counts on the normal status-bar refresh,
  backed by a 2s cache (no extra polling).

## Troubleshooting

- **Badge shows nothing** — gateway RPC unreachable; check the gateway is
  running (`hermes gateway status`), then open Control Room and refresh.
- **Action says "unavailable"** — the backing plugin/route isn't present in
  this runtime (e.g. Hermes Peer plugin not installed). It is by design.
- **Action says "stale"** — the row changed since you loaded it. Refresh the
  snapshot and retry.
- **Cross-profile rejection** — the target names a profile outside the
  current scope. Switch scope or target the correct profile.

## Known gaps (v1)

- Live two-session peer lifecycle (message receipt → inbox action → request
  completion) is env-gated: the peer plugin suite covers creation, but a
  live cross-session run needs two running Hermes gateways.
- Desktop full vitest is env-blocked (missing vite plugins in the hoisted
  node_modules); the keybind contract is verified, the rest of the desktop
  tests are authored source.
- Dashboard BFF capabilities `approvals`/`peer_messages`/`delegation_control`
  are hardcoded unavailable in v1 — they become real when the gateway RPC
  client is added to the BFF.
