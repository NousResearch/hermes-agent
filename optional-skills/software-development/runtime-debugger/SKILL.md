---
name: runtime-debugger
description: "Use when debugging runtime issues: Python (pdb/debugpy) or Node.js (--inspect/CDP), with recipes for local, remote, and post-mortem debugging."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [debugging, python, nodejs, pdb, debugpy, breakpoints, dap, cdp, node-inspect]
    related_skills: [systematic-debugging, debugging-hermes-tui-commands]
---

# Runtime Debugger

Unified debugging reference for Python and Node.js runtime inspection.

## Overview

Two tools under one umbrella:

- **Python Debugger** (`python-debugpy`): pdb REPL + debugpy remote debugging via DAP
- **Node.js Inspector** (`node-inspect-debugger`): `node inspect` + Chrome DevTools Protocol CLI

**Start with the simplest option.** Breakpoint in the source → run normally → inspect. Only escalate to remote debugging when the simple path doesn't work.

## When to Use

- A test fails and the traceback doesn't reveal the root cause
- You need to step through a function and watch state mutate
- A long-running process misbehaves and can't be restarted
- Post-mortem: an exception fired and you want to inspect locals at the crash site
- A subprocess/child is the actual bug site

**Don't use for:** things `print()` / `logging.debug` solve in under a minute.

## Python: Quick Reference

| Tool | When |
|------|------|
| `breakpoint()` + pdb | Local, interactive, simplest |
| `python -m pdb` | Launch existing script, no source edits |
| `debugpy` | Remote/headless, attach to running process |

### pdb Commands

| Command | Action |
|---------|--------|
| `h` / `h cmd` | help |
| `n` | next line (step over) |
| `s` | step into |
| `r` | return from current function |
| `c` | continue |
| `l` / `ll` | list source / full function |
| `w` | where (stack trace) |
| `u` / `d` | move up / down in stack |
| `p expr` / `pp expr` | print / pretty-print |
| `display expr` | auto-print on every stop |
| `b file:line` | set breakpoint |
| `!stmt` | execute arbitrary Python |
| `interact` | drop into full Python REPL |
| `q` | quit |

## Python: Common Recipes

### Recipe 1: Local breakpoint
```python
def compute(x, y):
    result = some_helper(x)
    breakpoint()  # drops into pdb here
    return result + y
```
Run normally. Remove before committing.

### Recipe 2: Launch script under pdb (no source edits)
```bash
python -m pdb path/to/script.py arg1 arg2
```

### Recipe 3: Post-mortem on any exception
```python
import pdb, sys
try:
    run_the_thing()
except Exception:
    pdb.post_mortem(sys.exc_info()[2])
```

### Recipe 4: Remote debug with debugpy
```bash
# Add near top of entry point:
import debugpy
debugpy.listen(("127.0.0.1", 5678))
print("debugpy listening on 5678, waiting for client...", flush=True)
debugpy.wait_for_client()
```

Start process; it blocks on `wait_for_client()`. Attach VS Code/Cursor with:
```json
{
  "name": "Attach to Hermes",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "127.0.0.1", "port": 5678 },
  "justMyCode": false
}
```

Or use `remote-pdb` (cleaner for terminal agents):
```python
from remote_pdb import set_trace
set_trace(host="127.0.0.1", port=4444)
# Then: nc 127.0.0.1 4444
```

## Node.js: Quick Reference

| Tool | When |
|------|------|
| `node inspect` | Built-in CLI REPL, zero install |
| `chrome-remote-interface` | Scriptable CDP, automation |

### node inspect Commands

| Command | Action |
|---------|--------|
| `c` / `cont` | continue |
| `n` / `next` | step over |
| `s` / `step` | step into |
| `o` / `out` | step out |
| `pause` | pause running code |
| `sb('file.js', 42)` | set breakpoint |
| `cb('file.js', 42)` | clear breakpoint |
| `bt` | backtrace |
| `repl` | drop into REPL in current scope |
| `.exit` | quit |

## Node.js: Common Recipes

### Recipe 1: Launch with inspector paused
```bash
node --inspect-brk script.js
# Then in another terminal:
node inspect -p <pid>
```

### Recipe 2: Attach to running process
```bash
kill -SIGUSR1 <pid>  # enables inspector
curl -s http://127.0.0.1:9229/json/list | jq -r '.[0].webSocketDebuggerUrl'
node inspect ws://127.0.0.1:9229/<uuid>
```

### Recipe 3: Programmatic CDP
```bash
npm i -g chrome-remote-interface
```
Use the CDP driver to set breakpoints, walk scopes, evaluate expressions.

## Python Pitfalls

1. **pdb under pytest-xdist silently does nothing.** Use `-p no:xdist` or `-n 0`.
2. **`breakpoint()` in CI/non-TTY hangs the process.** Never commit it.
3. **`PYTHONBREAKPOINT=0`** disables all `breakpoint()` calls.
4. **`debugpy.listen` blocks only with `wait_for_client()`.**
5. **Attach to PID fails on hardened kernels.** Fix: `echo 0 > /proc/sys/kernel/yama/ptrace_scope`
6. **Threads.** pdb only debugs current thread. Use debugpy for multithreaded.

## Node.js Pitfalls

1. **Wrong line numbers in TS.** Breakpoints hit emitted JS, not TS. Build first or use `--enable-source-maps`.
2. **`--inspect` vs `--inspect-brk`.** Without `-brk`, script races past breakpoints.
3. **Port collisions.** Default 9229. Pass `--inspect=0` for random port.
4. **Background kills.** `Ctrl+C` out of debugger leaves target paused. `cont` first or `kill` explicitly.

## Hermes-Specific Notes

- **Python tests:** `scripts/run_tests.sh tests/path/test.py --pdb -p no:xdist`
- **Python subprocesses (`_SlashWorker`):** Use `remote-pdb` with `set_trace()` inside worker exec path
- **Node TUI (`hermes --tui`):** `kill -SIGUSR1` on the Node PID, then `node inspect ws://...`
- **Vitest tests:** `node --inspect-brk ./node_modules/vitest/vitest.mjs run --no-file-parallelism src/test.test.tsx`

## Detailed References

See:
- `references/python-debugpy.md` — full Python debugger skill
- `references/node-debugger.md` — full Node.js inspector skill