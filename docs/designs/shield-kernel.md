# Shield Kernel — Self-Repair Command for Hermes Agent

> **Type:** Feature Proposal (Design Document)
> **Status:** Draft — Requesting Community Discussion
> **Author:** @zealonexp
> **Date:** 2026-05-04

---

## 1. Problem Statement

When Hermes enters a fatal error loop — stuck in a repeating tool-call cycle, corrupted kernel file, or broken configuration — there is **no self-repair mechanism**. The user's only recourse is to manually inspect logs, identify the broken file, and fix it by hand. This is especially painful for users running Hermes as a headless gateway service (Telegram, Discord, Slack, etc.) where there is no direct terminal access.

**Key pain points:**
- Agent loops with no escape hatch (tool-call budget exhaustion, but no repair)
- Kernel file corruption (e.g., `run_agent.py`, `gateway/run.py`) breaks the agent entirely
- Configuration errors (broken skills, MCP servers, tool configs) cause repeated failures
- No diagnostic+repair tool available to the user when they need it most

## 2. Proposal: `/repair` Command

Shield Kernel is a **gateway-embedded `/repair` command** that spawns an independent LLM-driven MiniAgent subprocess. This MiniAgent:

1. Reads error logs and agent logs to diagnose the problem
2. Uses an LLM to decide what to fix (read files, write patches, run shell commands)
3. Optionally restores core files from verified "good snapshots"
4. Reports results back to the user via the gateway's message channel

```
User: /repair [optional problem description]
         |
         v
+------------------------------------------+
|  gateway/run.py                          |
|  _handle_repair_command()                |
|         |                                |
|         | asyncio.create_subprocess_exec |
|         v                                |
|  +------------------------------+       |
|  |  mini_agent.py (subprocess)  |       |
|  |  * Zero imports from Hermes  |       |
|  |  * Reads logs & config       |       |
|  |  * LLM decides read/patch/   |       |
|  |    shell/snapshot-restore    |       |
|  |  * Max 30 iterations         |       |
|  |  * 5-minute timeout          |       |
|  +----------+-------------------+       |
|             | stdout -> chat relay       |
|             v                            |
|     Repair result + diagnostic log      |
+------------------------------------------+
```

## 3. Why a Gateway Command (Not an External Process)

| Approach | Pros | Cons |
|----------|------|------|
| **External daemon process** | Full isolation | Extra systemd service; file-based IPC; cannot use gateway message relay; its own health management is yet another problem |
| **Gateway-embedded command** (proposed) | Zero extra services; natural message channel; works even when agent is stuck in a loop | If gateway itself crashes, `/repair` is unavailable (but systemd restart handles that) |

The key insight: **the gateway's command dispatch layer is far more stable than the agent itself.** When the agent is stuck in a fatal loop, `/repair` can still be triggered because it bypasses the active-agent guard.

## 4. Architecture

### 4.1 File Structure

```
~/.hermes/shield/
├── mini_agent.py          # MiniRepairAgent main logic
│                          # Zero Hermes imports, only openai + stdlib
│                          # 3 LLM tools: read_file, write_file, run_shell
│                          # + 1 snapshot tool: restore_snapshot
│
├── snapshot.py            # Snapshot creator
│                          # Copies core files + md5 verification + import test
│                          # LRU cleanup (max 5 snapshots)
│
├── snapshots/             # Snapshot storage
│   └── snap_YYYYMMDD_HHMMSS_HASH/
│       ├── run_agent.py
│       ├── model_tools.py
│       ├── cli.py
│       ├── gateway/run.py
│       ├── hermes_cli/commands.py
│       ├── tools/memory_tool.py
│       └── manifest.json  # md5 + git hash + import_test results
│
├── backups/               # Pre-repair file backups
└── repair_log.jsonl       # Repair audit log (one JSON per line)
```

### 4.2 Gateway Integration Points

| File | Change |
|------|--------|
| `hermes_cli/commands.py` | Add `CommandDef("repair", ...)` + add `"repair"` to `ACTIVE_SESSION_BYPASS_COMMANDS` |
| `gateway/run.py` | Add `_handle_repair_command()` (handler) |
| | Add `_run_repair_task()` (async subprocess execution) |
| | Add `_shield_trigger_snapshot()` (24h rate-limited auto-snapshot) |
| | Busy-agent routing: repair bypasses guard |
| | Idle routing: repair dispatch |
| | Stuck-loop log enhancement: hint `/repair` to user |

## 5. Core Mechanisms

### 5.1 MiniAgent Workflow

```
1. Read Hermes config.yaml + .env -> parse provider/apikey
2. Build initial prompt = error log summary + user problem description
3. LLM loop (max 30 iterations):
   a. LLM returns text -> display to user
   b. LLM returns tool_call -> execute tool (read_file / write_file / run_shell / restore_snapshot)
   c. Tool result appended to messages -> next iteration
   d. Stop condition: LLM no longer calls tools, or max iterations reached
4. Save repair log to repair_log.jsonl
```

### 5.2 API Key Resolution

MiniAgent resolves API keys independently (no dependency on Hermes code):

```
Priority:
1. providers.<provider_name>.api_key  (from config.yaml providers section)
2. model.api_key                     (top-level key)
3. model.apikey                      (top-level key, strip ':' prefix)
4. OPENAI_API_KEY env var            (loaded from .env)
```

### 5.3 Snapshot Mechanism

**Purpose:** Provide the MiniAgent's `restore_snapshot` tool with verified "known-good" versions of core files.

**Creation triggers:**
- Manual: `python ~/.hermes/shield/snapshot.py`
- Automatic: gateway triggers `_shield_trigger_snapshot()` after each successful conversation, **rate-limited to once per 24 hours**

**Why 24 hours?** A snapshot must be battle-tested. If the interval is too short (e.g., 1 hour), a broken state could be snapshotted before the problem is discovered. 24 hours ensures the snapshot survived at least a full day of normal operation.

**Snapshot contents:** 6 core Hermes files:
- `run_agent.py`, `model_tools.py`, `cli.py`
- `gateway/run.py`, `hermes_cli/commands.py`, `tools/memory_tool.py`

**Validation:** Each snapshot undergoes import testing + md5 checksum verification at creation time. Failed validations are discarded.

**Retention:** LRU — max 5 snapshots, oldest auto-deleted.

### 5.4 Repair Strategies

MiniAgent holds the `restore_snapshot` tool and can choose:

**Strategy A (Standard Repair):** Core files are intact -> MiniAgent uses `read_file`/`write_file` to fix external issues (config, skills, MCP, etc.)

**Strategy B (Snapshot Restore):** Core files are corrupted -> MiniAgent calls `restore_snapshot` to restore from the latest good snapshot, then restarts the gateway

## 6. Usage

```
/repair                          # Auto-diagnose: read recent error logs, LLM analyzes and repairs
/repair mempalace MCP connection failed  # Specify problem for better context
```

**Features:**
- Triggerable even while agent is running (bypasses active-agent guard)
- 5-minute timeout
- Results automatically relayed to the current chat
- Output auto-truncated if exceeding 4000 characters

## 7. Security & Safety Boundaries

```
MiniAgent CAN read:                     MiniAgent CANNOT (enforced by prompt):
--------------------                    -----------------------------------
+ errors.log / agent.log / gateway.log
+ config.yaml + .env                    - Modify .env secrets
+ ~/.hermes/logs/*                      - Send messages to user (stdout only)
+ Tool/skill/MCP files (repair targets)
+ 6 core files (via snapshot restore)
```

**MiniAgent is NOT a Hermes plugin** — it is a standalone Python script using only `openai` + stdlib, invoked via subprocess.

## 8. Known Limitations

| Limitation | Description |
|------------|-------------|
| Gateway crash = no repair | `/repair` depends on gateway command dispatch. If the gateway process crashes entirely, rely on systemd restart |
| First snapshot is manual | Until 24h auto-snapshot kicks in, the first snapshot must be created manually |
| Repair quality depends on LLM | No hard confidence threshold; relies on prompt-constrained minimal-change principle |
| No auto-repair (Phase 1) | Currently manual `/repair` only. Stuck-loop detection suspends + logs hint |
| Snapshot interval not hot-configurable | Hardcoded as class attribute in gateway/run.py; requires restart to change |

## 9. Future Phases

### Phase 2: Auto-Repair
- Stuck-loop detection automatically triggers MiniAgent (no manual `/repair` needed)
- Requires stricter safety boundaries (repair count limits + auto-rollback verification)

### Phase 3: Snapshot Quality Enhancement
- Run smoke tests before snapshot creation (not just import tests)
- Snapshot scoring: longer uptime = higher confidence

### Phase 4: GEP Integration
- Repair logs feed back into GEP Engine
- Error patterns -> Genes (automatic extraction of new repair strategies)
- Adaptive frequency control

## 10. Discussion Questions

We'd love community feedback on:

1. **Scope of snapshot files** — Are these 6 files the right set? Should we include more or fewer?
2. **24h snapshot interval** — Is this the right balance between freshness and safety?
3. **Auto-repair safety** — What guardrails would make automatic repair acceptable?
4. **MiniAgent permissions** — Should `run_shell` be included, or is that too dangerous?
5. **Integration path** — Should this be a built-in feature, an optional plugin, or a standalone tool?

---

*This is a design proposal. Implementation exists as a proof-of-concept and is not ready for merge. Seeking community discussion before proceeding with full implementation.*
