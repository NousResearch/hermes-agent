---
name: hybrid-orchestration
description: >
  Combine Hermes built-in delegation (delegate_task) with ANC (Agent Notification Center)
  for multi-agent orchestration. Learn when to spawn local subagents vs dispatch to remote
  workers, how to pipeline both mechanisms, and avoid common overlap mistakes.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Multi-Agent, Orchestration, ANC, Delegation, Swarm, Parallelism]
    related_skills: [hermes-agent, opencode, native-mcp]
---

# Hybrid Orchestration: delegate_task + ANC

This skill teaches Hermes to **combine two fundamentally different agent-control mechanisms** into a coherent orchestration strategy. Mastering their boundaries prevents the common mistake of trying to use one where the other is clearly superior.

## The Two Mechanisms at a Glance

| Dimension | `delegate_task` (Built-in) | ANC (MCP via `agent-notification-center`) |
|-----------|---------------------------|-------------------------------------------|
| **Location** | Same process / same machine | Cross-process, cross-machine, cross-framework |
| **Timing** | Synchronous — parent blocks until child finishes | Asynchronous — fire-and-forget, results arrive later |
| **Offline tolerance** | None — child must be online | Full — events queue while workers are offline |
| **Result retrieval** | Direct return value | Polling via `consume_pending` |
| **Progress visibility** | Real-time spinner / TUI overlay | Event-driven, subscribe to topics |
| **Child identity** | Hermes subagent (same codebase) | Any agent framework (opencode, custom, human) |
| **Setup** | Zero config — always available | Requires `mcp_servers.agent-notification-center` in config |
| **Lifetime** | Ephemeral (child dies after task) | Persistent (events survive restarts) |

**Rule of thumb:**
- Need the answer **now** and the work fits in one machine? → `delegate_task`
- The worker may be **offline**, the task is **long-running**, or the worker is **not Hermes**? → ANC

---

## Clear Responsibility Boundaries

### `delegate_task` — Local Subagent Spawner

**What it does:**
- Spawns a child `AIAgent` instance inside the same Hermes process
- Gives the child a fresh conversation, isolated context, and restricted toolset
- Waits (blocking) for the child to finish and returns its summary

**What it does NOT do:**
- Communicate with external agents or other machines
- Queue tasks for later execution
- Survive a Hermes restart
- Allow the parent to do other work while the child runs

**Best for:**
- Parallel file analysis (read 5 files simultaneously via batch mode)
- Context isolation (offload a reasoning-heavy subtask so parent stays clean)
- Local tree decomposition (parent → 3 parallel children → aggregate)
- Rapid iteration (child finishes in seconds to minutes)

**Config knobs** (`config.yaml` under `delegation:`):
```yaml
delegation:
  max_concurrent_children: 3    # parallel limit
  max_spawn_depth: 2            # nesting depth (1 = flat, 2 = orchestrator→leaf)
  child_timeout_seconds: 600    # hard kill after N seconds
  orchestrator_enabled: true    # allow role="orchestrator"
  inherit_mcp_toolsets: true    # children keep parent's MCP tools
```

### ANC — Distributed Event Bus

**What it does:**
- Publishes events to named topics via `ingest_event`
- Queues events for offline subscribers
- Lets any consumer `consume_pending` to pick up queued work
- Supports priority levels (P0/P1/P2) and pattern-based topic subscriptions

**What it does NOT do:**
- Provide real-time progress feedback to the publisher
- Guarantee ordering across different publishers
- Handle tool execution directly (it is pure messaging)
- Spawn processes or manage compute resources

**Best for:**
- Dispatching work to opencode workers on other machines
- Human-in-the-loop tasks (publish → human acts → consume result)
- Long-running research tasks (hours, not minutes)
- Decoupled pipelines where stages should not block each other
- Cross-profile coordination (`hermes -p coder` talks to `hermes -p research`)

**MCP tools available** (when ANC server is configured):
- `mcp_agent_notification_center_register_identity`
- `mcp_agent_notification_center_register_session`
- `mcp_agent_notification_center_subscribe_topic`
- `mcp_agent_notification_center_ingest_event`
- `mcp_agent_notification_center_consume_pending`
- `mcp_agent_notification_center_close_session`
- `mcp_agent_notification_center_get_stats`

---

## Decision Tree

```
Is the task purely local and you need the result immediately?
├─ YES → Can it be parallelized into independent subtasks?
│   ├─ YES → delegate_task (batch mode)
│   └─ NO  → Do it yourself (single tool call)
└─ NO  → Is the worker a non-Hermes agent or potentially offline?
    ├─ YES → ANC (ingest_event + consume_pending)
    └─ NO  → delegate_task with role="orchestrator" (local tree)
```

**Examples mapped to the tree:**

| Scenario | Mechanism | Why |
|----------|-----------|-----|
| "Analyze these 4 log files in parallel" | `delegate_task` batch | Local, immediate, parallelizable |
| "Ask opencode to refactor auth module overnight" | ANC | External agent, long-running, offline OK |
| "Research this API while I work on UI" | ANC | Decoupled, parent should not block |
| "Run tests + lint + typecheck simultaneously" | `delegate_task` batch | Local, same repo, need all results before continuing |
| "Spawn a subagent that itself delegates further" | `delegate_task` with `role="orchestrator"` | Local nested tree |
| "Send alert to my phone when CI fails" | ANC | Cross-platform (Hermes → mobile push gateway) |

---

## Combination Patterns

### Pattern A: Local Parallel + Remote Dispatch (Fan-Out Hybrid)

Hermes receives a complex request. It decomposes the work:

1. **Local parallel** (`delegate_task` batch):
   - Child A: Analyze current codebase structure
   - Child B: List recent git changes
   - Child C: Check existing tests

2. **Remote dispatch** (ANC):
   - While children run, publish a `tasks.research.deep_dive` event to opencode worker
   - This task is too heavy for local subagents (30+ min expected)

3. **Aggregate locally**:
   - Wait for batch children to return (seconds)
   - Summarize their findings immediately
   - The ANC result will arrive later via `consume_pending` in a future turn

```python
# Pseudocode of the orchestration logic
def handle_complex_request(user_query):
    # Phase 1: Local parallel prep work
    local_results = delegate_task(
        tasks=[
            {"goal": "Analyze codebase structure", "toolsets": ["terminal", "file"]},
            {"goal": "List recent git changes", "toolsets": ["terminal"]},
            {"goal": "Check existing tests", "toolsets": ["terminal", "file"]},
        ],
        parallel=True
    )

    # Phase 2: Dispatch heavy research to remote worker
    ingest_event(
        title="Deep research: " + user_query,
        topic="tasks.research.deep_dive",
        priority="P1",
        source_id=f"research-{uuid()}",
        body=user_query + "\n\nLocal context:\n" + summarize(local_results),
        actor_alias=hermes_session_id
    )

    # Phase 3: Return immediate synthesis
    return synthesize(local_results) + "\n\nA deep-dive research task has been dispatched to a remote worker. Results will arrive via ANC when ready."
```

### Pattern B: ANC as Persistent Queue, delegate_task as Executor

Use ANC for durability and `delegate_task` for execution power:

1. Subscribe to `tasks.hermes.*`
2. `consume_pending` picks up a task
3. Decompose it locally with `delegate_task`
4. Publish results back to `results.requester.*`

This turns Hermes into a **reliable worker** that can crash and resume without losing work.

```
ANC topic "tasks.hermes.coding"
    ↓
consume_pending → get task
    ↓
delegate_task batch (analyze + implement + test)
    ↓
ingest_event to "results.opencode.task-123"
```

### Pattern C: Escalation Ladder

Start cheap and local, escalate to ANC only when justified:

1. **Attempt 1**: `delegate_task` with tight timeout (60s)
2. If child times out or result is insufficient →
3. **Attempt 2**: Publish to ANC with `priority="P1"` for a dedicated long-running worker

This avoids wasting remote worker cycles on trivial tasks that local subagents could handle.

### Pattern D: Two-Way Bridge

Hermes acts as a bridge between ANC and local subagents:

- **Inbound**: ANC event arrives → spawn `delegate_task` children to process it → publish results back to ANC
- **Outbound**: Local parent decides a subtask is better handled externally → publish to ANC instead of spawning another local child

---

## Concrete Workflow Examples

### Example 1: Code Review Pipeline

```yaml
# Step 1: Local parallel analysis (delegate_task)
tasks:
  - goal: "Run linter and capture all errors"
    toolsets: [terminal]
  - goal: "Run type checker"
    toolsets: [terminal]
  - goal: "Check test coverage for changed files"
    toolsets: [terminal, file]

# Step 2: If clean, dispatch to opencode for deeper review
# (ANC — because opencode may be on another machine and this takes 10+ min)
event:
  title: "Deep code review requested"
  topic: "tasks.coding.review"
  body: "Please review PR #123 for architecture issues..."

# Step 3: Return immediate lint/type results to user
# (ANC deep review result arrives later via consume_pending)
```

### Example 2: Multi-Agent Research Swarm

```yaml
# Hermes is the orchestrator
# It uses delegate_task for immediate local research
# It uses ANC for external specialist agents

local_children:
  - goal: "Search local codebase for relevant modules"
  - goal: "Check local documentation"

anc_dispatches:
  - topic: "tasks.research.papers"
    body: "Find recent papers on X"
    # Handled by: opencode worker with arxiv access
  - topic: "tasks.research.web"
    body: "Search official docs and GitHub issues"
    # Handled by: another opencode instance

# Hermes waits for local children (fast)
# ANC results are consumed in subsequent turns as they arrive
```

### Example 3: CI Failure Handler

```
Hermes detects CI failure via terminal tool
    ↓
delegate_task (immediate):
    - Read CI logs
    - Identify failing test
    - Check if it's a flaky test (rerun once)
    ↓
If not flaky → ANC ingest_event:
    topic: "tasks.coding.fix"
    priority: "P0"  # urgent
    body: "Test X is failing with error Y..."
    → opencode worker picks it up and opens PR
```

---

## Best Practices

### 1. Always Prefer `delegate_task` for Synchronous Work

If you need the result to continue the current conversation, use `delegate_task`. ANC's async nature makes it unsuitable for blocking logic flows.

**Bad:**
```python
ingest_event(topic="tasks.calc", body="compute fibonacci(100)")
result = consume_pending()  # Might be empty if worker hasn't finished!
```

**Good:**
```python
result = delegate_task(goal="compute fibonacci(100)")  # Guaranteed to block and return
```

### 2. Use ANC for Cross-Session Persistence

If the task should survive a Hermes restart or be picked up by a different profile, ANC is the only choice.

```yaml
# config.yaml for profile "worker"
mcp_servers:
  agent-notification-center:
    command: python
    args: ["-m", "anc.mcp_server"]
```

### 3. Set Meaningful `source_id` in ANC Events

ANC deduplicates by `source + source_id + event_type` within 24h. Always use unique `source_id` per task to avoid silent drops.

```python
source_id = f"hermes-{session_id}-{task_name}-{timestamp}"
```

### 4. Batch Size Discipline for `delegate_task`

`max_concurrent_children` defaults to 3. Raising it multiplies API cost linearly. For ANC, there is no local concurrency limit — the bottleneck is remote worker capacity.

### 5. Don't Nest `delegate_task` Too Deep

Default `max_spawn_depth` is 1 (flat). Raising to 2 or 3 unlocks orchestrator roles but increases complexity and token burn. If you find yourself wanting depth 3+, consider whether ANC would be cleaner.

### 6. Handle "Routed to 0 subscriber(s)" Gracefully

ANC returns this when no subscriber is currently online. **This is normal** — the event is queued. Do not retry. Just note that the task is queued and move on.

### 7. Subscribe Before Publishing Results

Always subscribe to the results topic **before** dispatching work, otherwise you might miss fast results:

```python
subscribe_topic(topic_pattern="results.hermes.*", ...)
ingest_event(topic="tasks.research.x", ...)
# Later:
consume_pending()  # Guaranteed to catch the result
```

---

## Common Pitfalls

### Pitfall 1: Using ANC Like a Function Call

ANC is not a RPC system. `ingest_event` does not return a result. You must later call `consume_pending` (or be subscribed and online when the result arrives).

### Pitfall 2: `delegate_task` for Long-Running External Work

If the task takes >10 minutes, the child may hit `child_timeout_seconds` (default 600s). The parent is also blocked the entire time. Use ANC instead.

### Pitfall 3: Duplicate ANC Events

Reusing the same `source_id` within 24h causes deduplication. If you need to retry a task, generate a new `source_id`.

### Pitfall 4: Forgetting to Close ANC Sessions

Always call `close_session` before Hermes goes idle. This ensures events queue instead of failing delivery to an "online" but actually dead session.

```python
try:
    # ... do work ...
finally:
    close_session(session_id=session_id)
```

### Pitfall 5: Mixing Up Topic Patterns

ANC topics use dot notation with wildcard support. `tasks.research.*` matches `tasks.research.ai` but not `tasks.research.ai.deep`. Use `tasks.research.**` or `tasks.research.#` if supported, or be explicit.

---

## Quick Reference Card

| Situation | Use This | Key Tool / Call |
|-----------|----------|-----------------|
| Parallel local file ops | `delegate_task` batch | `delegate_task(tasks=[...], parallel=True)` |
| Context isolation needed | `delegate_task` single | `delegate_task(goal="...", toolsets=["file"])` |
| Nested decomposition | `delegate_task` orchestrator | `delegate_task(goal="...", role="orchestrator")` |
| External worker (opencode) | ANC | `mcp_agent_notification_center_ingest_event` |
| Human-in-the-loop | ANC | `mcp_agent_notification_center_ingest_event` with `priority="P0"` |
| Task > 10 min expected | ANC | `mcp_agent_notification_center_ingest_event` |
| Cross-profile messaging | ANC | Same MCP server, different `actor_alias` |
| Survive Hermes restart | ANC | Events queue automatically |
| Need real-time progress | `delegate_task` | Child events flow to parent spinner/TUI |
| Fire-and-forget alert | ANC | `ingest_event` then move on |

---

## Configuration Checklist

To use hybrid orchestration, verify both mechanisms are configured:

```yaml
# ~/.hermes/config.yaml

# 1. Built-in delegation (no extra install needed)
delegation:
  max_concurrent_children: 3
  max_spawn_depth: 2
  child_timeout_seconds: 600
  orchestrator_enabled: true

# 2. ANC MCP server (requires agent-notification-center package)
mcp_servers:
  agent-notification-center:
    command: python
    args: ["-m", "anc.mcp_server"]
    env:
      PYTHONPATH: "/path/to/agent_notification_center"
```

Verify ANC is reachable:
```bash
hermes mcp test agent-notification-center
```

Verify delegation works:
```bash
# Just ask Hermes to delegate a simple task to itself
hermes chat -q "Delegate to a subagent: count lines in run_agent.py"
```

---

## Summary

- **`delegate_task`** = synchronous, local, ephemeral subagents. Use for immediate parallel work.
- **ANC** = asynchronous, distributed, persistent message bus. Use for cross-agent, long-running, or offline-tolerant work.
- **Combine** them by using `delegate_task` for fast local decomposition and ANC for heavy or external work.
- **Never** use ANC when you need an immediate result in the same turn.
- **Never** use `delegate_task` for tasks that should survive a restart or involve non-Hermes agents.
