---
name: cadvp-verify
description: >-
  Cross-Agent Delivery Verification Protocol — a 13-dimension framework that
  validates knowledge injection across isolated agent execution contexts
  (interactive, cron/scheduled, gateway). Catches "channel fracture" failures
  where scheduled agents silently cannot write to memory.
version: 1.1.0
author: community
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [python3, sqlite3]
metadata:
  hermes:
    tags: [verification, cross-agent, memory, delivery, quality-gate, cron]
    homepage: https://github.com/tobiglevent001/agent-quality-gate
---

## Overview

CADVP (Cross-Agent Delivery Verification Protocol) is a verification framework for multi-agent systems where one agent must inject knowledge into another agent's persistent memory. It detects **channel fracture** — a silent failure mode where cron/scheduled agents cannot access memory tools due to hardcoded architectural guards.

This skill implements CADVP v1.1, which includes a **veto-level zero-check (CC-0: Channel Confirmation)** that prevents false-positive delivery assurance.

## 13-Dimension Checklist

| ID | Dimension | Level | Description |
|----|-----------|-------|-------------|
| **CC-0** | Channel Confirmation | 🔴 VETO | Target channel is architecturally available |
| PC-1 | Prerequisite Existence | Pre | Required source data exists |
| PC-2 | Prerequisite Format | Pre | Data is in correct format for injection |
| PC-3 | Prerequisite Access | Pre | Writer has permission to read source |
| WV-1 | Write Dispatch | Write | Write operation was dispatched |
| WV-2 | Write Acknowledgment | Write | Write was acknowledged by target system |
| WV-3 | Write Persistence | Write | Data persists beyond session scope |
| RV-1 | Read Availability | Read | Data is queryable by target agent |
| RV-2 | Read Correctness | Read | Retrieved data matches injected data |
| RV-3 | Read Completeness | Read | All injected items are retrievable |
| RV-4 | Read Timeliness | Read | Data is available within required timeframe |
| GR-1 | Fallback Path | Recovery | Alternative injection method exists |
| GR-2 | Error Propagation | Recovery | Failures are reported to operators |

**CC-0 (Channel Confirmation)** verifies before any write:
- Target execution context supports the required tools
- Memory subsystem is initialized in the target context
- No hardcoded guards (skip_memory, etc.) block the channel
- Tool registration is conditional on runtime state that will be present
- Profile permissions (cron_mode, etc.) allow the operation

If CC-0 fails, the operation is **vetoed** — alternative channels must be chosen.

## Channel Decision Tree

```
Cross-agent injection needed?
  ├─ Target cron context? → CC-0: cron agents CANNOT write memory
  │     └─ Use Channel A: direct DB write (SQLite)
  ├─ Target interactive context? → CC-0: memory() tool available
  │     └─ Use Channel B: target self-write via instruction
  └─ Target gateway context? → CC-0: verify memory tools registered
        └─ Use Channel A or B depending on tool availability
```

## Known Channel Availability

| Channel | Method | Reliability | Best For |
|---------|--------|-------------|----------|
| A | Direct SQLite INSERT into target memory_store.db | ✅ Highest | Batch injection, cron contexts |
| B | Target self-write via memory()/fact_store() tools | ✅ Reliable | Interactive, gateway contexts |
| ❌ C | Cron-delegated write (scheduled task → memory tools) | ⛔ Never | Not viable — blocked by skip_memory=True |

## Quick Start

```bash
# 1. Run the verification script against a target profile
python3 scripts/cadvp-verify.py <target-profile>

# 2. Read the JSON output
#    - All 13 dimensions: PASS / FAIL / VETO
#    - If CC-0 VETO: channel is unavailable, abort and switch
#    - If all PASS: delivery is confirmed

# 3. Use the decision tree above to select the correct channel
```

## References

- Full protocol & implementation: [agent-quality-gate](https://github.com/tobiglevent001/agent-quality-gate)
- Paper: *Channel Fracture: Architectural Blind Spots in Scheduled Cross-Agent Memory Injection*
