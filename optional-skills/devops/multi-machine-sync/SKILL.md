---
name: multi-machine-sync
description: Diagnose multi-machine sync with ZeroTier and Syncthing.
version: 1.0.0
author: ligl0325
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [syncthing, zerotier, sync, multi-machine, network]
    category: devops
    requires_toolsets: [terminal]
triggers: ['sync not working', 'files not syncing', 'syncthing problem', 'zerotier down', 'multi machine sync', 'folder conflict']
---

# Multi-Machine Sync Diagnosis

Diagnose and recover file synchronization across multiple machines using ZeroTier and Syncthing. Covers connectivity checks, pause/resume recovery, directory conflict resolution, and structure mismatch analysis.

## When to Use

- Files are not syncing between machines
- ZeroTier network is unreachable or unstable
- Syncthing shows disconnected devices or errors
- Sync paused unexpectedly on one or more peers
- Directory conflicts appear (`.sync-conflict-*` files)
- Large initial sync seems stuck or slow

## Prerequisites

- **ZeroTier** installed on all peers (`sudo zerotier-cli status` must work)
- **Syncthing** installed and running on all peers (`syncthing cli` or GUI)
- Network connectivity between peers (same ZeroTier network)
- `sudo` access for ZeroTier commands (or passwordless sudo configured)
- `terminal` tool available in Hermes

## How to Run

This skill is designed for interactive diagnosis. The agent walks through four phases in order:

1. **ZeroTier Connectivity** — verify the underlying network is healthy
2. **Syncthing Status** — check device and folder sync state
3. **Pause/Resume Recovery** — restart stalled syncs
4. **Directory Conflict Resolution** — handle conflicting files

Start with: "my files aren't syncing between machines" or use one of the trigger phrases above.

## Quick Reference

| Phase | Key Command | Purpose |
|-------|-------------|---------|
| ZeroTier | `sudo zerotier-cli status` | Check daemon status |
| ZeroTier | `sudo zerotier-cli listpeers` | List connected peers |
| Syncthing | `syncthing cli show system` | Show device status |
| Syncthing | `syncthing cli show errors` | View recent errors |
| Syncthing | `syncthing cli operations resume` | Resume paused sync |
| Conflicts | `find . -name '*.sync-conflict-*'` | Find conflict files |

## Procedure

### Phase 1: ZeroTier Connectivity

1. Check daemon status: `sudo zerotier-cli status`
2. List joined networks: `sudo zerotier-cli listnetworks`
3. List connected peers: `sudo zerotier-cli listpeers`
4. Verify peer reachability: `ping <peer-ip> -c 3`
5. If a peer is missing: restart ZeroTier service, re-join the network, or check authorization on my.zerotier.com

### Phase 2: Syncthing Status

1. Check device status: `syncthing cli show system`
2. List folder status: `syncthing cli show folder-status --folder <folder-id>`
3. View recent errors: `syncthing cli show errors`
4. Check active connections: `syncthing cli show connections`
5. Verify all expected devices are connected and folders are up-to-date

### Phase 3: Pause/Resume Recovery

1. Identify which peer paused sync
2. Check for conflict files: `find . -name '*.sync-conflict-*' 2>/dev/null`
3. Resume sync: `syncthing cli operations resume`
4. If resume fails, restart Syncthing: `systemctl --user restart syncthing`

### Phase 4: Directory Conflict Resolution

1. List all conflict files: `find . -name '*.sync-conflict-*' 2>/dev/null`
2. Compare conflicting versions with `diff` or `wc`
3. Resolution strategy: keep newest, merge manually, or ask the user
4. After resolution: delete conflict files and resume sync

## Pitfalls

- **ZeroTier identity persistence**: After reboot, `/var/lib/zerotier-one/` identity must be preserved or the node gets a new address
- **Large initial sync**: Can take hours for large directories; this is normal, not a failure
- **Firewall blocking**: Syncthing requires ports 22000 (TCP) and 21027 (UDP) open
- **Case sensitivity**: Syncthing on Linux is case-sensitive; files differing only in case may cause conflicts on mixed-OS networks
- **Pause cascade**: Pausing Syncthing on one machine may cause cascade failures on other peers; warn the user before pausing
- **Never auto-delete**: Never delete files automatically during conflict resolution — show the diff and ask

## Verification

After completing the procedure, confirm:

- `sudo zerotier-cli listpeers` shows all expected peers
- `syncthing cli show system` shows all devices as connected
- `syncthing cli show errors` shows no unresolved errors
- Target folders show synced status (no pending changes)
- No `.sync-conflict-*` files remain in shared directories