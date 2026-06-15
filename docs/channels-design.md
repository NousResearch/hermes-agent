# Sessions as Channels — multi-device, multi-participant design

Status: living design doc (2026-06-09). Owner: operator + desktop lane.
Related: `docs/session-presence.md`, mesh feature `F-003-multi-participant-channels`.

## Vision

A Hermes session stops being a private 1:1 thread between one human and the
agent and becomes a **channel**: synchronized across every device, joinable
from any client (desktop, TUI, CLI, phone), with every message attributed to
the human/device that sent it — the way a Discord channel works, with the
agent as a first-class participant. Long-term this scales outward: a device
mesh shares channels across its machines; an organization's mesh invites
another mesh into specific channels; eventually org↔org sharing.

## Hard constraint: zero-dependency core

Not every Hermes user runs MeshBoard (Syncthing under the hood) or Tailscale —
some have one, the other, or neither. **No layer of this design may REQUIRE
mesh infrastructure.** Every capability ships with a plain-Hermes default and
treats mesh/tailnet/cloud as optional accelerators, exactly like the existing
device-name resolver (config → MeshBoard → Tailscale → hostname fallback):

| Layer | Zero-dep default | Optional accelerators |
|---|---|---|
| Identity | resolved device name | MeshBoard label, Tailscale hostname, future accounts |
| Same-device co-viewing | gateway FanoutTransport (works today) | — |
| Cross-device attach | explicit `host:port` endpoint in presence record | Tailscale MagicDNS, mesh SSH tunnel |
| Session list sync | presence records + remote `session.list` RPC | Syncthing folder, cloud relay |
| Real-time fan-out across machines | hosting gateway broadcasts to remote ws clients | cloud relay (meshboard-cloud DO) for NAT-crossing |
| Orgs / invites / cross-mesh | n/a (single-user core) | meshboard-cloud accounts + teams |

## What exists today (verified 2026-06-09)

- **Transport fan-out is already built.** `tui_gateway/transport.py`
  `FanoutTransport`: a second `_attach_session_transport` *joins* the stream
  instead of stealing it; `write_json` resolves the session's transport
  per-write, so even a mid-turn attach starts receiving immediately; dead
  clients are pruned on write failure. Pinned by
  `tests/tui_gateway/test_concurrent_attach.py`.
- **Sender attribution** (`messages.sender_device`, schema v17): user
  messages record the device they were typed on; rewrites preserve it; the
  desktop renders the label. (F-003 slice 1.)
- **Presence discovery** (`hermes_cli/session_presence.py`): per-instance
  records under `~/.hermes/session-presence/active/` with TTL, host, client,
  profile, and an `endpoint` attach hint. Discovery only — not a transport.
- **Turn serialization**: one turn at a time per session
  (`session["running"]` + queued prompts), so concurrent writers are already
  safe — they interleave as turns, like chat.
- **Recoverable hosts**: graceful SIGTERM finalize, resume-from-state.db
  fallback, desktop ws rebind (#43004), and the self-host-kill guard.

## Phases

**Phase 1 — same-gateway channels (mostly done).** Two desktop windows / TUI
+ desktop on one machine co-view a session. Remaining: desktop UX for "also
open here" on live sessions (the sidebar Live section already lists them),
and showing the agent's busy state to the non-prompting viewer (statusbar
events already broadcast).

**Phase 2 — cross-device attach (the next big step).** A client on device B
attaches to a session hosted by device A's gateway:
1. Presence record's `endpoint` field carries a reachable ws endpoint
   (zero-dep: explicit `gateway.listen` host:port config; accelerators:
   Tailscale hostname, mesh-derived address).
2. Desktop B's session list merges local sessions + presence-discovered
   remote sessions (already rendered in the Live section); opening a remote
   one dials the remote gateway's ws and speaks the existing protocol —
   `session.activate` + fanout makes multi-client correct by construction.
3. Prompts from device B carry `sender_device` explicitly (protocol param on
   `session.prompt`); the hosting gateway's auto-stamp covers legacy clients.
4. Auth for the remote ws: reuse the gateway token mechanism the desktop
   already uses locally; pairing flow = show QR/token on host, paste on
   client (mirrors meshboard-cloud enrollment ergonomics without requiring it).

**Phase 3 — the channel UX.** Sessions ARE channels; the UI catches up:
flat session rows (done), sender labels (done), participant presence chips
(who's viewing — derive from fanout attach/detach events, surface as a
`session.participants` event), create-from-anywhere (remote `session.create`),
and channel naming = session titles (already synced via state.db + title RPCs).

**Phase 4 — mesh & org scale (meshboard-cloud).** For users who opt into the
cloud (or run their own relay): a Durable-Object per channel relays events
between gateways that can't reach each other directly, and persists the
message log for offline members. The substrate already exists in
meshboard-cloud: accounts, device pairing/bearers, AccountDO websocket
fan-out, teams + invites + roles + SSO + audit. Needed additions: `channels`,
`channel_messages`, `channel_participants` tables; per-channel DO; bridge
endpoints. Org→mesh invites reuse the teams model; mesh↔mesh and org↔org
sharing add a `peer_meshes` trust table (Ed25519 signing precedent already in
the marketplace tables).

## Decisions

- **Identity = device name now, accounts later.** `sender_device` is the
  attribution unit until accounts exist; when meshboard-cloud accounts join a
  channel, a `sender_user` column slots in beside it (additive migration).
- **The hosting gateway is the ordering authority** for its sessions. No
  CRDTs: one SQLite log per session, owned by the host; remote clients are
  thin. Cloud relay (Phase 4) replicates the host's log, never forks it.
- **Agent semantics in group chat:** one agent per channel; every user
  message is a turn input as today (turn queue serializes); @-mention gating
  becomes a config once multiple humans actively collide (not before).
- **File-sync (Syncthing) is never the real-time path.** It remains fine for
  presence-record propagation on meshes that have it, but message fan-out is
  always a live socket (local fanout or cloud DO) — the task-twin split-brain
  taught us what file-sync does to concurrent writers.

## Known gaps / follow-ups

- `session.prompt` lacks the explicit `sender_device` param (Phase 2, with
  the first remote client).
- Participant join/leave is not yet surfaced as an event (Phase 3).
- The gateway ws currently binds localhost-only by default; Phase 2 needs an
  opt-in listen address + token auth before any non-local exposure.
