# RFC: System Boundary Layer + Service Knowledge Layer

**Status:** Draft  
**Author:** Nikolay Gusev  
**Date:** 2026-05-09  
**Gist:** https://gist.github.com/NikolayGusev-astra/4e99af9e4dba05968db8dcc7313ecb93  

## Problem

The Hermes Agent currently cannot distinguish system-level file changes from user-level ones. It has no built-in concept of "system boundary" — `/etc/nginx/` and `/home/user/project/` are treated identically. The agent also lacks awareness of service dependencies: changing one config file may break several services that share ports or resources, but the agent has no way to know this.

Key issues:
- Agent can rewrite system configs without any confirmation
- The YOLO mode is a binary on/off switch — it disables all guards wholesale
- `write_file`, `patch`, `terminal` and `execute_code` have separate, inconsistent guard mechanisms
- After context compaction, the agent loses awareness of what it already learned about the system
- The only existing protection is pattern-matching on dangerous terminal commands (`rm -rf`, `dd`), which is trivially bypassed by switching tools

## Proposed Architecture

Two independent layers, neither relying on LLM memory or context:

### Layer 1 — System Boundary Layer

A deterministic (regex-based, not LLM) path classifier that categorizes every file write target:

- **SYSTEM**: paths managed by the OS or package manager (based on FHS conventions — `/etc/`, `/opt/`, `/usr/share/`, `/var/lib/`)
- **USER**: user-level paths (`/home/`, `/tmp/`, `/var/www/`, project directories)
- **UNKNOWN**: cannot classify → block by default

A single gate intercepts ALL write-capable tools (`write_file`, `patch`, `terminal`, `execute_code`) at the dispatch level (in `model_tools.py` or equivalent), normalizing each call to a unified `(path, intent)` form before the classifier runs. This prevents the agent from bypassing the gate by switching from `write_file` to `echo >` in a terminal.

### Layer 2 — Service Knowledge Layer

A persistent knowledge map stored on disk (LLM Wiki + HippoRAG), unaffected by context compaction:

- **Snapshot**: on first SYSTEM write (or on demand), the agent collects a system profile: running services, listening ports, config files at standard paths. The raw data is structured into entity files in the LLM Wiki.
- **Pre-write hook** (pure code, no LLM): looks up the target path in the knowledge map. If dependencies are found, the user sees: "this change will affect services X, Y (via port Z). Confirm?" If the map is empty, a snapshot is forced first.
- **Post-write diff**: after a successful write, the agent detects delta in resources (ports, files, sockets). New relationships are auto-added to the map; existing ones are never overwritten without confirmation.

## LLM Role

The LLM is explicitly excluded from the pre-write critical path. It is used only for:
- **Snapshot processing**: structuring raw system output (`systemctl`, `ss`, config parsers) into entity files
- **Post-write diff interpretation**: semantic understanding of what changed in config deltas

No LLM call is required — or allowed — in the pre-write hook.

## Non-Goals (v1)

- Docker, iptables, ufw, and network-level changes are **not** covered (v2 scope)
- Runtime/service reloads (`systemctl reload`, `kill -HUP`) are **not** intercepted
- Full fanotify/syscall-level interception is **not** implemented (v3 scope)

## Success Criteria

1. First SYSTEM write → automatic snapshot → guard displays affected services
2. Subsequent writes → map lookup, user gets a dependency warning before confirmation
3. After context compaction → knowledge map is intact (it lives on disk, not in LLM context)
4. Agent cannot bypass the gate by switching between `write_file`, `terminal`, or `execute_code`
5. YOLO mode disables the pre-write hook (single off switch)
6. No LLM call in the pre-write critical path

## Installation

```bash
bash <(curl -sSL https://install.hermes/sbl.sh)
```
