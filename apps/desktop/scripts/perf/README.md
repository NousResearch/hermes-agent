# Desktop perf harness

One systematized way to measure desktop rendering/interaction performance,
diff it against a committed baseline, and fail on regressions. It replaces the
dozen one-off `measure-*` / `profile-*` scripts that each reinvented the CDP
client, arg parsing, stats, and output (and never had a baseline).

## Quick start

```bash
# Isolated instance (recommended) — no running app or LLM credits needed.
# Its own --user-data-dir + HERMES_HOME means it never collides with `hgui`.
npm run perf -- --spawn

# Or: launch an isolated instance once, attach repeatedly (faster iteration).
npm run perf:serve            # leaves an instance on :9222
npm run perf                  # attaches, runs the CI suite, gates on baseline

# One scenario, with a CPU profile:
npm run perf -- stream --cpuprofile --tokens 800

# Representative PRODUCTION numbers (minified React, not the ~3x-slower dev build):
npm run perf -- cold-start stream keystroke transcript --spawn --prod

# Controlled session-open proof. These use only an isolated renderer, a
# synthetic transcript, and the local v3 cache fixture — no real sessions,
# LLM credits, or network telemetry.
npm run perf -- session-switch --spawn --prod --verified-cache --rounds 20
npm run perf -- session-switch --spawn --prod --delay-runtime 2000 --rounds 5

# Re-capture the baseline on your reference device, then commit baseline.json:
npm run perf -- cold-start stream keystroke transcript --spawn --prod --update-baseline
```

## Dev vs prod

By default the harness measures the **dev** renderer (fast to spin up, good for
relative regression checks). Pass `--prod` (with `--spawn`) to build a
production renderer *with the probe included* (`VITE_PERF_PROBE=1`) and measure
minified React — the representative shipped numbers. The committed baseline is
captured with `--prod`.

## Why isolation matters

The measurement this harness exists to run was historically blocked: a running
`hgui` holds the Electron single-instance lock, so a second instance quit
immediately. `--spawn` / `perf:serve` launch with their own `--user-data-dir`
(separate lock scope), their own `HERMES_HOME` (separate backend + sessions),
and their own `--remote-debugging-port`. Synthetic scenarios drive `$messages`
directly via `window.__PERF_DRIVE__`, so no LLM credits are spent.

## Scenarios

| scenario | tier | measures | replaces |
|---|---|---|---|
| `stream` | ci | streaming longtasks, frame p95/p99, mutation cadence | measure-synthetic-stream, profile-synth-stream, profile-long-stream |
| `stream --real` | backend | same, from a real LLM stream | measure-real-stream, profile-real-stream |
| `keystroke` | ci | composer keystroke → paint latency | measure-latency, profile-typing, leak-typing |
| `transcript` | ci | large-transcript mount + paint cost | (new) |
| `render-churn` | ci | per-component render attribution + store churn while N tabs stream | (new) |
| `idle-cost` | report | busy-but-silent tiles: idle commit rate, + fps while resizing / typing | (new) |
| `right-pane` | report | file tree + persistent xterm tabs under chat/terminal output and split dragging | (new) |
| `cold-start` | cold | launch → CDP → driver → first paint (fresh spawn/run) | (new) |
| `first-token` | backend | Enter → first assistant token painted (TTFT) | (new) |
| `submit` | backend | Enter → cleared → user msg painted, scroll jump | measure-submit, measure-jump |
| `session-switch` | backend / controlled | real route → first-paint → settle, or cache/REST/runtime open timing | profile-session-switch |
| `session-load` | backend | how far a session's transcript moves after first paint | (new) |
| `profile-switch` | backend | rail click → sidebar settled | measure-profile-switch |

`ci` + `cold` scenarios need no backend/credits and are gated against
`baseline.json` (`cold-start` requires `--spawn` since it measures a fresh
launch, and must be run in its own invocation). `backend` scenarios need a live
backend (and `--spawn` or a real session/credits) and are report-only.

CPU profiling is a cross-cutting `--cpuprofile` flag on any scenario (it wraps
the run in `Profiler.start/stop` and prints a top-self-time table), replacing
every standalone `profile-*` script.

### Controlled session-open modes

`session-switch` keeps its ordinary backend mode: provide `--a` and `--b` for
two real stored sessions, and it reports route-to-paint/settle timing. The two
controlled modes above do not need those ids. They run only in the isolated
`--spawn --prod` renderer and seed a valid v3 transcript-tail entry plus a
matching listed-row revision with deterministic mixed-markdown content. It
then drives the mounted production `useSessionActions.resumeSession` path; the
fixture injects only local gateway/REST responses and never calls a backend or
sends telemetry.

- `--verified-cache` records 20 (or more) cache-open rounds and fails when
  `cache_first_paint_p95_ms` exceeds 100 ms.
- `--delay-runtime <ms>` holds the synthetic `session.resume` response while
  real display hydration proceeds, then fails unless every round records REST
  commit strictly before resume readiness.

The output also includes REST/resume percentiles and `rest_before_resume_count`.
`agent_ready_p95_ms` is reported as unavailable when the active backend protocol
does not report deferred agent-prewarm completion; the fixture never invents a
timer-based agent-ready value. These controlled checks are explicit hard gates,
not additions to the cross-device `baseline.json` comparison.

## Adding a scenario

Create `scenarios/<name>.mjs` exporting `{ name, tier, description, run(cdp, opts) }`
where `run` returns `{ metrics, detail }` (metrics = flat numbers, lower is
better), then register it in `scenarios/index.mjs`. If it's `ci`, add a
`baseline.json` entry (or run `--update-baseline`).

## Layout

- `lib/cdp.mjs` — the one CDP client + target discovery + typing + CPU-profile wrapper + DOM selectors.
- `lib/stats.mjs` — percentiles, histograms, CPU-profile self-time ranking.
- `lib/baseline.mjs` — load/compare/update the baseline + regression gate.
- `lib/launch.mjs` — attach, or spawn a fully isolated instance.
- `scenarios/` — one module per measurement.
- `run.mjs` — entrypoint. `serve.mjs` — standalone isolated launcher.

## Not migrated (kept as dev utilities)

`eval.mjs`, `reload.mjs`, `reload-renderer.mjs`, `probe-renderer.mjs`,
`probe-thread.mjs`, `click-session.mjs`, `diag-*.mjs` are interactive dev
helpers, not benchmarks. They can adopt `lib/cdp.mjs` in a follow-up.
