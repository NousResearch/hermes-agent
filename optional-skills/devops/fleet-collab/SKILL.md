---
name: fleet-collab
description: "Multi-machine collaboration over Tailscale via shared board."
version: 1.0.0
author: JannLeo
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [kanban, multi-agent, fleet, tailscale, ssh, collaboration]
    category: devops
    related_skills: [kanban-orchestrator, kanban-worker]
---

# Fleet Collab Skill

Coordinate multiple machines (over Tailscale) on a shared Hermes kanban board. Any node can create tasks; a central hub dispatches ready tasks to remote workers via SSH.

## When to Use

- You have 2+ machines on a Tailscale tailnet, each running Hermes Agent.
- You want them to collaborate on a shared task board (e.g. compile on the beefy server, test on a small node, summarize on the laptop).
- You don't want the overhead of a shared filesystem (NFS/SSHFS) — the hub acts as the bridge.

## Prerequisites

1. **Tailscale** connecting all nodes (CGNAT range `100.64.0.0/10`).
2. **Hermes Agent** installed on each node (`hermes --version` works).
3. **Passwordless SSH** from the hub to each worker: hub's pubkey in each worker's `~/.ssh/authorized_keys`.
4. **Reverse SSH** from workers back to the hub: hub's sshd accepts each worker's key (so workers can create tasks on the shared board).
5. **Proxy bypass**: if any node runs a local HTTP proxy (Clash etc.), add Tailscale IPs to `NO_PROXY` in `~/.hermes/.env` so model calls to `100.x` aren't hijacked:
   ```
   NO_PROXY=localhost,127.0.0.1,::1,100.64.0.0/10,<hub-ip>,<worker-ip-1>,<worker-ip-2>
   no_proxy=localhost,127.0.0.1,::1,100.64.0.0/10,<hub-ip>,<worker-ip-1>,<worker-ip-2>
   ```

## How to Run

### 1. On the hub — create the shared board and worker profiles

```bash
# Create a dedicated board
hermes kanban boards create fleet-collab --name "Fleet Collaboration" --switch

# Create a profile per remote machine (clones hub config)
hermes profile create worker-<name> --clone
```

### 2. Deploy the scripts

Copy `scripts/fleet_create.sh` to **every** node (hub + workers).
Copy `scripts/fleet_dispatch.sh` to the **hub only**.

```bash
# On the hub, edit fleet_dispatch.sh FLEET_MAP to map assignee -> ssh target:
#   "worker-<name>|<ssh_user>|<tailscale_ip>|<remote_hermes_path>"

# On each node, edit fleet_create.sh HUB_USER / HUB_IP to point at the hub.
```

### 3. Create and dispatch tasks

```bash
# Any node can create a task (writes to the hub's shared board):
~/fleet_create.sh "Compile firmware and report size" --assignee worker-beefy --body "Run make and report binary size."

# List / inspect tasks from any node:
~/fleet_create.sh --list
~/fleet_create.sh --show t_xxxxx

# On the hub, dispatch ready tasks to remote workers:
~/fleet_dispatch.sh --task t_xxxxx        # single task
~/fleet_dispatch.sh --loop 30            # poll every 30s, continuously
```

## Quick Reference

| Script | Runs on | Purpose |
|---|---|---|
| `fleet_create.sh` | any node | create / list / show tasks on the shared board (SSH callback to hub) |
| `fleet_dispatch.sh` | hub only | send ready tasks to remote workers via SSH, write results back to kanban |

## Procedure

1. Hub: `hermes kanban boards create fleet-collab --switch`
2. Hub: `hermes profile create worker-<n> --clone` for each remote machine.
3. Establish passwordless SSH hub↔workers (both directions).
4. Fix `NO_PROXY` on any node running a local proxy.
5. Deploy `fleet_create.sh` to all nodes; `fleet_dispatch.sh` to the hub.
6. Edit the `FLEET_MAP` in `fleet_dispatch.sh` and `HUB_*` in `fleet_create.sh`.
7. Create a task on any node: `~/fleet_create.sh "title" --assignee worker-x --body "..."`
8. On hub: `~/fleet_dispatch.sh --loop 30` to auto-dispatch.

## Pitfalls

- **Proxy hijacks `100.x`**: a Clash/HTTP proxy on a node will return 502 for Tailscale IPs unless `NO_PROXY` includes `100.64.0.0/10` (the `100.*` glob is *not* recognized by Python httpx). The dispatch script also `unset` proxy env vars before calling the remote hermes as a belt-and-suspenders fix.
- **SSH arg quoting**: `fleet_create.sh` uses a tempfile + scp to move args to the hub, avoiding shell-quote nesting that breaks `ssh ... "cmd $*"`.
- **Don't use the SSH terminal backend** (`terminal.backend: ssh`) for these worker profiles: it force-syncs the hub's `~/.hermes/skills` (42 dirs) to the remote on every spawn — slow and overwrites the remote's own config. This skill's `fleet_dispatch.sh` SSHes into the remote's *own* hermes instead, no file sync.
- **macOS node via Tailscale SSH**: Tailscale SSH may require browser-based device approval; password SSH won't work until approved. Complete the approval first.
- **Worker "spoke and stopped" protocol violation**: some models (e.g. MiniMax) finish a task without calling `kanban_complete`/`kanban_block`, triggering `protocol violation` and auto-block after `failure_limit` (default 2) consecutive crashes. Mitigations: raise `kanban.failure_limit` in config, or use a stronger worker model. The orchestrator (below) can recover by re-decomposing.

## Recursive Decomposition (Orchestrator SOUL)

The built-in `auto_decompose` is a **one-shot** aux-LLM call that splits a triage task into ≤6 children. For large goals (multiple feature areas, multi-file work), a single child can still be too big — the worker burns its turn budget, gets blocked, and the loop stalls.

**Teach the orchestrator profile to recursively decompose.** Add this to the orchestrator profile's `SOUL.md` (or `personalities.<name>` in config.yaml):

```
When you wake up after children complete, inspect each child's result with kanban_show:
- done + real progress → accept, move on.
- blocked OR summary indicates too-large/turn-exhausted → re-decompose.

To re-decompose a too-large blocked task:
1. kanban_archive the blocked child.
2. Create 2-4 smaller child tasks via kanban_create --parent <root_id>, each with a
   single-file-or-single-feature body spec.
3. Always set --parent so the root stays gated until all children complete.

Sizing heuristic for manual fan-out:
- Small (1 turn, ~5 min): "implement X in file Y" → single task
- Medium (2-3 turns, ~15 min): "add endpoint + tests + verify" → single task
- Large (multi-file, >30 min): DO NOT send as one task — split into medium tasks
  with --parent linking in dependency order.

Anti-patterns:
- One huge task with 5+ feature areas → worker will fail.
- Blocking the root when children fail → re-decompose instead.
- Forgetting --parent → children run ungated, root never wakes.
```

This turns the orchestrator from a passive "wait for children" role into an active loop driver that keeps the board moving even when the one-shot decomposer under-splits.

## Verification

```bash
# From any node, create a smoke task:
~/fleet_create.sh "smoke: report hostname" --assignee worker-x --body "Run hostname, report it."

# On hub, dispatch and confirm:
~/fleet_dispatch.sh --task t_xxxxx
~/fleet_create.sh --show t_xxxxx   # status: done, comment contains hostname
```
