---
name: live-interruptible-worker
description: Run batch tasks (web research, URL fetching) with real-time steering — STOP, SKIP, FOCUS, LIST — via PTY terminal stdin.
version: 1.0.0
author: argus-metis
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [pty, background, worker, interrupt, streaming, research, batch]
    category: devops
    requires_toolsets: [terminal]
    related_skills: [watchers]
---

# Live-Interruptible Worker

Run long batch tasks in a PTY terminal and **steer them in real time** — stop, skip, or narrow the focus mid-execution, within ~1s of your command.

## How It Works

The `live_worker.py` script runs in a background PTY terminal. It processes items **one at a time** and waits for commands between each. A background stdin reader thread sets a cancellation flag instantly when STOP, SKIP, or FOCUS arrives — the main thread checks it between every HTTP operation, so even a slow page load can be interrupted within ~1s.

### Cancellation Architecture (Three Layers)

| Layer | What it catches | Latency |
|-------|----------------|---------|
| **Between items** — 0.1s stdin poll loop in the main wait cycle | Commands arriving while the worker is idle | **Instant** |
| **Between HTTP operations** — `channel.check_cancel()` after each search/fetch/summarise step within a single item | Commands arriving mid-item | **Within current sub-step** |
| **Mid-HTTP-request** — `cancellable_fetch()` with adaptive timeouts (1s → 2s → 4s → 8s → 15s) | Commands arriving during a slow page load | **~1s** |

## Commands

| Command | Effect |
|---------|--------|
| `STOP` | Cancel remaining work and print final summary |
| `SKIP [N]` | Skip the next N items (default: 1) |
| `LIST` | Show remaining items and progress |
| `FOCUS <query>` | Keep only items matching `<query>` (case-insensitive) |
| `NEXT` | Proceed immediately without waiting for the auto-continue timer |

The worker auto-continues after **10 seconds** of inactivity if no command is received.

## When to Use

- Batch web research with many queries where you might want to narrow or redirect mid-way
- Crawling/checking a list of URLs where some turn out to be irrelevant
- Any task where you'd normally batch 50 items into a single delegation and hope it doesn't go wrong

**Skip for:** tasks that need full tool access (file writes, complex orchestration, tool chaining) — use `delegate_task` instead.

## Usage

### Launch from the terminal

```bash
# With --items
python3 scripts/live_worker.py --items "Query 1" "Query 2" "Query 3"

# Pipe JSON from stdin
echo '["Item 1", "Item 2"]' | python3 scripts/live_worker.py

# Load from a JSON file
python3 scripts/live_worker.py --file tasks.json
```

### Launch in a background PTY

```python
# Start the worker
terminal(
    command='python3 scripts/live_worker.py --items "X" "Y" "Z"',
    background=True,
    pty=True
)
```

### Send steering commands

```python
process(action='submit', session_id='proc_xxx', data='STOP')
process(action='submit', session_id='proc_xxx', data='SKIP 2')
process(action='submit', session_id='proc_xxx', data='FOCUS keyword')
process(action='submit', session_id='proc_xxx', data='LIST')
process(action='submit', session_id='proc_xxx', data='NEXT')
```

### Check output

```python
# Quick status poll
process(action='poll', session_id='proc_xxx')

# Full live output
process(action='log', session_id='proc_xxx')
```

## Task Types

The worker supports two built-in task handlers, selected with `--type`:

| `--type` | Handles items as | Behaviour |
|----------|-----------------|-----------|
| `search` (default) | Research queries | Searches the web, fetches top 3 result pages, extracts plain-text content |
| `fetch` | URLs or search queries | Fetches a single URL, or searches if the item isn't a URL |

## Task Handler API

The worker is designed to be extended. To add a custom task handler:

1. Write a function with signature `def my_handler(item: str, channel: CommandChannel) -> dict`
2. Call `channel.check_cancel()` **between every sub-operation**
3. Register it: `TASK_HANDLERS['my_type'] = my_handler`
4. Run with `--type my_type`

The `cancellable_fetch()` utility is available for HTTP tasks — it wraps `urllib.request` with adaptive timeouts and cancellation checks.

## Dependencies

**Zero.** The script uses only Python standard library — `urllib.request`, `threading`, `html.parser`, `json`, `select`, `socket`. No pip installs needed.

## Notes

- The web search endpoint defaults to a local SearXNG instance at `http://localhost:8080/search`. Override with the `SEARXNG_URL` environment variable.
- Reports are saved to the current working directory as `liveworker_report_<timestamp>.json`.
- The script works in any PTY-capable terminal, not just the Hermes agent. Run it directly in a terminal emulator and type commands interactively.
