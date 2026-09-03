# RCA: `BlockingIOError: [Errno 35]` (EAGAIN) from `subprocess.Popen`/fork under host-wide concurrency

**Status:** resolved by `fix(code_kernel): add Popen retry-with-backoff for transient EAGAIN` (commit `b6955a6299`, merged to `main` at `4c9da9093a`)
**Severity:** P3 — transient, self-clearing, low event volume; no data loss or crash beyond the single tool/subprocess call that hit it.
**Sentry issues:** NICHE-BOTS-8 (this doc's primary target), plus siblings NICHE-BOTS-7, NICHE-BOTS-9, NICHE-BOTS-A — all the same underlying cause from different call sites, investigated and resolved in parallel kanban task chains on 2026-08-31.

## Summary

Between 2026-08-31T15:00Z and 16:30Z, four unrelated Sentry issues (NICHE-BOTS-7/8/9/A) fired the identical `BlockingIOError: [Errno 35] Resource temporarily unavailable` from four different code paths:

- NICHE-BOTS-8: `tools/code_kernel.py::_spawn()` (chat route `claude-sonnet-4-6`, booting a fresh code-kernel interpreter)
- NICHE-BOTS-9: `sentry_kanban_poller.py::create_kanban_card()` (`subprocess.run(['hermes', 'kanban', 'create', ...])`)
- NICHE-BOTS-7: agent terminal tool executing `echo "alive"`
- NICHE-BOTS-A: `sync_calendar.py`'s `_run_gog()` subprocess call

Errno 35 is `EAGAIN` — the OS refusing to `fork()`/`posix_spawn()` a new process at that instant, not a socket or non-blocking-I/O misconfiguration in application code.

## Root cause

This Mac mini runs a single-host deployment with no separate "production server": 21 always-on `launchd` Hermes gateway daemons (one per profile) plus concurrent kanban dispatch workers across ~15 boards all share one UID's process table and `RLIMIT_NPROC`. All four Sentry events are tagged `server_name: Mac`.

Investigation chain findings (kanban tasks t_ccf0aa0e, t_83f3ff04, t_4de12ae4, t_2ce72c66 and siblings):

- Both NICHE-BOTS-8 events coincided within seconds with **other, unrelated** subprocess calls (ssh, browser-use CLI launch, `read_file`) failing with the identical errno 35 in the same session breadcrumbs.
- Kanban task_events across boards confirm concurrent worker dispatch at both error timestamps (e.g. a sibling board fired 2 simultaneous task spawns right as the 16:30:08Z event hit).
- Load testing (up to 3000 concurrent synthetic `Popen()` calls) never reproduced `EAGAIN` in isolation. It only correlated with **real ambient host load** — load average spiking to 40–70 on a 10-physical-core Mac, swap climbing from ~2GB to ~4.4GB — during which synthetic child processes stalled 5–30s (a milder symptom one step short of outright fork refusal).
- Pinning the FD ulimit to `launchd`'s 256 soft limit produced a *different*, deterministic error (`OSError: [Errno 24] EMFILE`), ruling out FD exhaustion as the NICHE-BOTS-8/9 cause specifically.

Conclusion: **transient host-level fork/process-table pressure from concurrent multi-profile, multi-board Hermes agent activity on one consumer machine — not a code defect, resource leak, or socket bug** in any of the four call sites. Only 2 events in 90 minutes with no deterministic repro and no recurrence since is consistent with genuine environmental noise rather than a systemic leak.

## Fix

Defense-in-depth retry-with-backoff was added at each affected call site, mirroring the pattern `plugins/tools/terminal_tool.py` already used for its own subprocess calls:

- **`tools/code_kernel.py::_spawn()`** (NICHE-BOTS-8, this doc): wraps `subprocess.Popen()` with up to 3 retries, exponential backoff 2s/4s/8s, retrying only `EAGAIN`/`EWOULDBLOCK`/`ENOMEM`; any other `OSError` fails immediately without retry. 3 new unit tests (`TestPopenRetryOnTransientError`) cover retry-then-succeed, retry-exhaustion, and non-retryable-error paths — all passing. Commit `b6955a6299`, branch `fix/code-kernel-popen-retry-NICHE-BOTS-8`, merged to `main` at `4c9da9093a`.
- **`sentry_kanban_poller.py::_run_hermes_kanban_create()`** (NICHE-BOTS-9): same pattern, 3 attempts at 1.5s/3s/6s, degrades to `None` + logged warning after exhaustion instead of crashing the whole poll cycle. Deployed to both the git-tracked source and the live cron-deployed copy.
- NICHE-BOTS-7 and NICHE-BOTS-A received equivalent retry wrappers at their respective call sites (agent terminal tool and `sync_calendar.py`'s `_run_gog()`).

None of these fixes change socket blocking mode, connection pooling, or the chat/API client itself — the original task hypothesis (an actual non-blocking-socket bug in the Claude Sonnet route) was investigated and ruled out; every failure traced to `fork()`/`Popen()`, not a network socket.

## Recovery / what to watch

If this recurs at meaningfully higher frequency (not the observed 2-events-in-90-min baseline), the real lever is **capping concurrent kanban dispatch across boards on this host** or moving some profile gateways off this single Mac mini — not further patching individual call sites. A compound monitoring alert (BlockingIOError burst + dispatcher-stuck warnings + high VM compressor/swap) was recommended by the investigation but is a separate follow-up, not part of this fix.

## Related

- Sentry: NICHE-BOTS-7, NICHE-BOTS-8, NICHE-BOTS-9, NICHE-BOTS-A (https://sunny-day-llc.sentry.io/issues/7702592577/ for NICHE-BOTS-8)
- Kanban: t_57d929d4 (root), t_ccf0aa0e / t_83f3ff04 / t_7b152e6a / t_f0a97a6b (children), plus the parallel NICHE-BOTS-7/9/A task chains under `flatratefisp-email`.
