# Python Debugging & Log Playbook

Load when the bug lives in a Python codebase. Maps directly onto `diagnose` phases: pytest loops for Phase 1, pdb/py-spy/faulthandler for Phase 4.

## Feedback loops (Phase 1)

| Goal | Command |
| --- | --- |
| Reproduce via test | `pytest path/to/test.py::test_name -x` |
| Loop a flaky test | `for i in {1..50}; do pytest test_name -x || break; done` |
| Drop into debugger on failure | `pytest --pdb` (post-mortem at the failure frame) or `pytest -x --pdb` |
| pdb at the *start* of a test | `pytest --trace` |
| Re-run only last failures | `pytest --lf` / `pytest --ff` |
| See live logs during tests | `pytest --log-cli-level=DEBUG -s` |
| Surface hidden warnings | `python -W error script.py` — warnings become errors (catches deprecations-turned-bugs) |
| Trace every call (last resort) | `python -m trace --trace script.py` or `python -X importtime` for slow imports |

## pdb — the canonical debugger

- `breakpoint()` at the suspect line. Redirect globally with `PYTHONBREAKPOINT=ipdb.set_trace`, or disable *every* breakpoint with `PYTHONBREAKPOINT=0` — never edit call sites.
- `python -m pdb script.py` — run under pdb; auto-enters post-mortem on uncaught exception.
- `python -m pdb -c continue script.py` — run normally, land in post-mortem only on crash.
- Attach to a running process: `python -m pdb -p <pid>` (3.14+).
- Commands that matter: `n` next, `s` step in, `c` continue, `w` stack, `u`/`d` move frames, `p`/`pp` print, `interact` full REPL in frame, `break func, x > 3` conditional, `display expr` auto-print.
- Habit: **interrogate expressions, not variables** — `p [s for s in adj if s not in counts]` answers a question; `p adj` only dumps data.
- Post-mortem after a crash: `import pdb; pdb.pm()` — or under pytest, exception info sits in `sys.last_traceback`.

## Production / can't-attach debugging

```bash
# faulthandler — stdlib, catches segfaults AND hangs
python -X faulthandler script.py
faulthandler.dump_traceback_later(30, repeat=True)   # dump all stacks if hung >30s
faulthandler.register(signal.SIGUSR1)                # `kill -USR1 <pid>` dumps stacks in prod

# py-spy — sampling profiler, zero code changes, safe on live processes
py-spy dump --pid <pid>                  # all thread stacks, like dotnet-stack
py-spy top --pid <pid>                   # live CPU per function
py-spy record -o profile.svg --pid <pid> # flamegraph
```

## Instrumentation (Phase 4)

- Tagged probe: `logger.debug("[DEBUG-a4f2] state=%r input=%r", state, input)` — grep the prefix to remove.
- Always `logger.exception(...)` (not `logger.error`) inside `except` — keeps the traceback.
- Structured logs: `structlog` or `dictConfig` with a JSON formatter → `jq`-queryable.
- `sys.excepthook` override to log *every* uncaught exception with full context before exit.
- asyncio bugs: `PYTHONASYNCIODEBUG=1` or `loop.set_debug(True)` + `loop.slow_callback_duration = 0.1` — surfaces never-awaited coroutines and blocking callbacks.

## Profiling

| Symptom | Tool |
| --- | --- |
| Slow function | `python -m cProfile -o out.pstats script.py` → `python -m pstats out.pstats` (`sort cumtime`, `stats 20`) — or `pyinstrument` for low overhead |
| Slow single function, line-level | `line_profiler` (`@profile` + `kernprof -l -v`) |
| Memory growth | `tracemalloc.start()` early → `snapshot()` before/after → `snapshot2.compare_to(snapshot1, 'lineno')` |
| Production CPU/mem | `py-spy record`, `memray` (`memray run` + `memray flamegraph`) |

## Log queries

```bash
# systemd / docker / k8s
journalctl -u myapp.service --since "1 hour ago" -p err
docker logs --since 30m --tail 500 <container>
kubectl logs deploy/myapp --since=1h --tail=-1

# traceback hunting — capture the frames above it
grep -n -B5 -A30 'Traceback (most recent call last)' app.log
grep -c 'Traceback' app.log          # frequency trend

# JSON/structlog
jq -c 'select(.level=="error")' app.log
jq -r '.event' app.log | sort | uniq -c | sort -rn | head
```

## Common failure patterns

| Symptom | Likely cause | Check |
| --- | --- | --- |
| Silent wrong output | `except: pass` / bare `except` swallowing the real error | grep `except` blocks; `logger.exception` everywhere |
| Hangs in prod, fine locally | deadlock, blocking call in event loop, GIL contention | `py-spy dump` or `SIGUSR1` + faulthandler |
| Memory climbs | unbounded cache, reference cycles, C-extension leak | `tracemalloc` diff; `memray` |
| Coroutine "ran" but nothing happened | missing `await` (returns coroutine object) | `PYTHONASYNCIODEBUG=1`; `RuntimeWarning: coroutine never awaited` |
| Mutable state leaks between calls | mutable default arg `def f(x=[])` | grep `def .*\(.*=(\[|{)` |
| Works locally, fails CI/prod | env vars, timezone, locale, missing `.env` | print `os.environ` diff; `python -W error` |

## Docs

- https://docs.python.org/3/library/pdb.html
- https://docs.pytest.org/en/stable/how-to/failures.html
- https://github.com/bloomberg/memray
