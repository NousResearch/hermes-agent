# Parallel Task Execution

Hermes now supports **parallel task execution**, allowing multiple independent tasks to run simultaneously instead of interrupting or queuing.

## Overview

When Hermes is busy processing a task and you send a new message, the system can now:

1. **Run in parallel** (NEW) - Independent tasks execute simultaneously
2. **Interrupt** - Stop the current task and start the new one
3. **Queue** - Wait for the current task to finish

## How It Works

### Task Classification

Hermes automatically classifies incoming tasks:

| Type | Description | Example | Behavior |
|------|-------------|---------|----------|
| **INDEPENDENT** | Can run safely in parallel | "Search for Python docs" | ⚡ Runs in parallel |
| **DEPENDENT** | Needs results from previous task | "Based on that search..." | ⏳ Waits for dependency |
| **SEQUENTIAL** | File/code operations | "Edit main.py" | 🔄 Runs sequentially |
| **BLOCKING** | Requires user confirmation | "Should I delete this?" | 🛑 Blocks everything |

### Example Usage

```
You: "Apply this code patch to the repo"           → Task 1: Running
You: "Search for Python async patterns"            → Task 2: ⚡ Running in parallel!
You: "Generate a diagram of the architecture"      → Task 3: ⚡ Running in parallel!
You: "Based on those search results, update docs"  → Task 4: ⏳ Waiting for Task 2
```

## Configuration

Enable parallel execution in `~/.hermes/config.yaml`:

```yaml
parallel_execution:
  enabled: true
  max_concurrent: 3
```

### Options

- `enabled`: Turn parallel execution on/off (default: `false`)
- `max_concurrent`: Maximum number of simultaneous tasks (default: `3`)

## Commands

### `/status`

Shows gateway status including parallel execution:

```
📊 Hermes Gateway Status

Session ID: abc123...
Agent Running: Yes ⚡

Parallel Execution: Enabled (max=3)
Running Tasks: 2
```

### `/tasks`

Lists all parallel tasks with their status:

```
📋 Running Tasks

1. ⚡ abc12345
   Search for Python async patterns
   Type: independent

2. ⚡ def67890
   Generate architecture diagram
   Type: independent

3. ✅ ghi11111
   Check weather in Tokyo
   Type: completed
```

### `/stop`

Stops the main running agent. Parallel tasks continue unless they fail.

## When Tasks Run in Parallel

Tasks run in parallel when:

1. Parallel execution is **enabled** in config
2. A task is classified as **INDEPENDENT**
3. No **resource conflicts** exist (e.g., two file operations on the same file)

### Independent Task Examples

- Web searches
- Image generation
- Calculations
- Weather/time lookups
- General questions

### Sequential Task Examples

- Code changes
- File edits
- Git operations
- Deployments

## Architecture

```
User Message
    ↓
[TaskClassifier] → INDEPENDENT/DEPENDENT/SEQUENTIAL/BLOCKING
    ↓
[ParallelTaskManager]
    ↓
┌─────────────────────────────────────────┐
│  Task Queue                             │
│  ├── Independent tasks → Parallel       │
│  ├── Dependent tasks → Wait             │
│  └── Sequential tasks → Queue           │
└─────────────────────────────────────────┘
    ↓
[AIAgent.run_conversation()] × N (concurrent)
    ↓
Responses sent to user
```

## Performance Considerations

### Resource Usage

- Each parallel task spawns its own `AIAgent` instance
- Memory usage scales with `max_concurrent` setting
- Tool calls from parallel tasks are thread-safe

### Recommended Settings

| System | max_concurrent | Notes |
|--------|----------------|-------|
| Low-end VM | 2 | Conservative, stable |
| Standard server | 3 | Balanced (default) |
| High-end server | 5 | Maximum throughput |

## Troubleshooting

### Tasks not running in parallel?

1. Check if enabled: `/status`
2. Verify task type: Check logs for classification
3. Resource conflicts: File operations are sequential

### High memory usage?

Reduce `max_concurrent`:

```yaml
parallel_execution:
  enabled: true
  max_concurrent: 2
```

### Cancel stuck tasks?

Use `/stop` to stop the main agent, or restart the gateway.

## Implementation Details

### Core Components

1. **TaskClassifier** (`gateway/task_classifier.py`)
   - Keyword-based classification
   - No ML/AI required
   - Confidence scoring

2. **ParallelTaskManager** (`gateway/parallel_task_manager.py`)
   - Async task queue
   - Session isolation
   - Lifecycle management

3. **ParallelExecutionIntegration** (`gateway/parallel_integration.py`)
   - Gateway integration
   - Task routing
   - Response handling

### Task Lifecycle

```
PENDING → QUEUED → RUNNING → COMPLETED
                          ↘ FAILED
                           ↘ CANCELLED
```

## Future Enhancements

- [ ] User-controlled task priority
- [ ] Visual task progress in UI
- [ ] Cross-session task sharing
- [ ] Task result caching

## Contributing

Found a bug or have an idea? Open an issue on GitHub!

---

**Version**: 1.0  
**Since**: Hermes v0.4.0  
**Issue**: #1468
