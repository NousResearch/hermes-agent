---
name: multi-machine-sync
description: Diagnose and recover file synchronization across multiple machines using ZeroTier and Syncthing. Covers connectivity checks, pause/resume recovery, directory conflict resolution, and structure mismatch analysis.
version: 1.0.0
author: ligl0325
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [syncthing, zerotier, sync, multi-machine, network]
    category: devops
triggers: ['sync not working', 'files not syncing', 'syncthing problem', 'zerotier down', 'multi machine sync', 'folder conflict']
toolsets: [terminal]
---

## Safety Guardrails
- Never delete files automatically during conflict resolution — show the diff and ask
- Pausing Syncthing on one machine may cause cascade failures; warn the user
- ZeroTier changes require network restart on all peers

## Phase 1: ZeroTier Connectivity
- Check ZeroTier status: sudo zerotier-cli status
- List networks: sudo zerotier-cli listnetworks
- List peers: sudo zerotier-cli listpeers
- Check moon/servers: sudo zerotier-cli listmoons
- Verify peer reachability: ping <peer-ip> -c 3
- Common fixes: restart service, re-join network, check auth on my.zerotier.com

## Phase 2: Syncthing Status
- Check status: syncthing cli operations
- List devices: syncthing cli show system
- Check folder status: syncthing cli show folder-status --folder <folder-id>
- View recent errors: syncthing cli show errors
- Check connections: syncthing cli show connections
- Verify all expected devices are connected

## Phase 3: Pause/Resume Recovery
- If sync paused: identify which peer paused
- Check for conflicts: look for .sync-conflict-* files
- Resume: syncthing cli operations resume or GUI
- Restart if needed: systemctl --user restart syncthing

## Phase 4: Directory Conflict Resolution
- List conflicts: find . -name '*.sync-conflict-*' 2>/dev/null
- Compare conflicting versions: diff or wc
- Resolution strategy: keep newest, merge manually, or ask user
- After resolution: delete conflict files, resume sync

## Pitfalls
- ZeroTier on restart needs /var/lib/zerotier-one/ identity to persist
- Syncthing on Windows paths vs Linux paths: case sensitivity differences
- Large initial sync can take hours; not a failure
- Firewall can block Syncthing ports (22000 TCP, 21027 UDP)
