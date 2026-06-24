# Performance Optimization Report — 2026-06-24

Scope: `/Users/shinchen-mac/.hermes/hermes-agent`

## Goal

Optimize Hermes runtime with evidence-backed changes and verify whether a measurable target improves by at least 50%.

## Research / Skill Search

- Existing local skills used:
  - `hermes-runtime-benchmark`
  - `performance-guard`
  - `find-skills`
- Skills ecosystem search via `npx skills find` found:
  - `wshobson/agents@python-performance-optimization` — 27.2K installs
  - `addyosmani/web-quality-skills@performance` — 18.7K installs
  - `samber/cc-skills-golang@golang-performance` — 28.1K installs
- Local profiler/tool availability checked:
  - installed: `ruff`, `orjson`, `uvloop`
  - not installed: `scalene`, `py-spy`, `memray`, `msgspec`, `rapidfuzz`

No new global skill/package was installed in this pass.

## Optimization Applied / Verified

### 1. Gateway import/startup hot path

High-confidence baseline from `performance-baseline.md`:

- `gateway.run` import median: `0.956991s`
- importtime hot spots included disabled/unused platform and account usage import paths.

Current verified measurement after rebasing onto `origin/main` (`6e88f7b6f`):

```text
import gateway.run median 0.1616s, min 0.1605s, max 0.1682s, n=5
```

Improvement against the original baseline:

- Absolute delta: `0.795391s`
- Relative speedup: ~`83.1%` lower import latency
- This still exceeds the ≥50% target for the gateway import/startup hot path after upstream integration.

### 2. Telegram short-message ingress latency

Telegram adaptive text batch defaults changed for normal short messages:

- Fast tier: `0.18s` → `0.015s` (~91.7% lower)
- Short tier: `0.24s` → `0.020s` (~91.7% lower)

This targets perceived Telegram response latency without changing long split-message delay semantics.

## Verification

Commands executed:

```bash
~/.hermes/hermes-agent/venv/bin/python ~/.hermes/skills/hermes-runtime-benchmark/scripts/benchmark.py --skip-codex-local
venv/bin/python -m pytest tests/gateway/test_telegram_text_batch_perf.py tests/gateway/test_usage_command.py -q
venv/bin/python -m pytest tests/gateway/test_slack.py tests/gateway/test_teams.py tests/gateway/test_feishu.py -q
venv/bin/python -X importtime -c 'import gateway.run'
```

Results:

- Runtime benchmark after integration: `/Users/shinchen-mac/.hermes/hermes-workspace/reports/hermes-runtime-benchmark/20260624-091157/REPORT.md`
- Focused tests after integration: `21 passed in 0.43s`
- Platform adapter tests after integration: `494 passed in 9.87s` with existing warnings
- Repeated subprocess import check after integration: `gateway.run` median `0.1616s`.

## Runtime Benchmark Snapshot

After-change benchmark:

| Command | Result | Time |
|---|---:|---:|
| `hermes --version` | ok | 1220.4ms |
| `hermes status` | ok | 1050.9ms |
| `hermes doctor` | ok | 8952.5ms |
| `hermes mcp test project-fs` | ok | 1713.5ms |
| `hermes curator status` | ok | 358.1ms |

## Boundaries

- I can claim ≥50% only for the measured gateway import/startup hot path and Telegram short-message batch delay, not for every runtime dimension.
- `hermes status`, `doctor`, and MCP discovery are mostly unchanged because they are different bottlenecks.
- No production gateway restart was performed in this pass.
- No new third-party skill/package was installed.

## Remaining High-ROI Targets

1. Codex-local end-to-end timeout stabilization.
2. State DB query/index profiling for session search and open-session cleanup.
3. Per-cron no-agent / model routing for simple repeated jobs.
4. Skill surface consolidation after a curator dry-run.
5. Optional install/evaluate `py-spy`, `scalene`, or `memray` for deeper CPU/memory profiling if Shin approves package installation.

## Rollback

Rollback code changes with:

```bash
git restore gateway/run.py gateway/slash_commands.py plugins/platforms/telegram/adapter.py plugins/platforms/slack/adapter.py plugins/platforms/teams/adapter.py plugins/platforms/feishu/adapter.py tests/gateway/test_telegram_text_batch_perf.py
```

If committed, rollback with:

```bash
git revert <commit_sha>
```
