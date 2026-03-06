# SwarmMonitor

Telemetry and monitoring for the swarm system. No external dependencies.

## Quick start

```python
from swarm.monitor import SwarmMonitor

mon = SwarmMonitor()  # logs to ~/.hermes/swarm/logs

mon.record_api_call("claude-sonnet", 1200, 800, 2.3, 0.0138, task_id="abc123")
mon.record_tool_call("bash", 0.5, True, task_id="abc123")
mon.record_task_transition("abc123", "running", "completed")

mon.print_dashboard()
```

## API

### `SwarmMonitor(log_dir: Path = None)`

Creates a monitor. Default log directory is `~/.hermes/swarm/logs`.

### Recording

| Method | Description |
|--------|-------------|
| `record_event(event_type, task_id=, worker_id=, data=)` | Log a generic timestamped event |
| `record_api_call(model, input_tokens, output_tokens, duration_s, cost_usd, task_id=)` | Log an LLM API call |
| `record_tool_call(tool_name, duration_s, success, task_id=)` | Log a tool invocation |
| `record_task_transition(task_id, from_state, to_state)` | Log a task state change |

### Queries

| Method | Returns |
|--------|---------|
| `get_summary()` | `dict` -- total tasks, completed, failed, cost, tokens, avg duration, by-model and by-tool breakdowns |
| `get_timeline(last_n=100)` | `list[dict]` -- most recent N events |
| `get_cost_breakdown()` | `dict` -- cost split by model, role, and task |

### Export

| Method | Returns |
|--------|---------|
| `export_json(path=None)` | `Path` -- writes full telemetry to JSON; returns the file path |
| `print_dashboard()` | Prints a formatted text dashboard to stdout |

## Thread safety

All methods are safe to call from multiple threads. Internal state is protected by `threading.Lock`.

## Event schema

Every event is a dict with at minimum:

```json
{
  "timestamp": "2026-03-05T12:00:00+00:00",
  "type": "api_call"
}
```

Optional fields: `task_id`, `worker_id`, `data` (type-specific payload).
