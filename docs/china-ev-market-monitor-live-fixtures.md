# China EV Market Monitor — Fixture and Live-Smoke Strategy

## CI-safe fixtures

- Captured parser fixtures live under `tests/fixtures/market_monitor/`.
- Mandatory parser tests must use these checked-in fixtures.
- When a source layout changes, update the fixture and pin the regression with a parser test before changing production parser logic.

## Live smoke tests

- `tests/market_monitor/test_live_smoke.py` is marked `integration`.
- CI excludes it by default via the repo pytest configuration.
- To run live smoke tests locally:

```bash
MARKET_MONITOR_LIVE=1 pytest tests/market_monitor/test_live_smoke.py -q -m integration -n 0
```

## Update workflow

1. Capture or trim a real source page into a deterministic fixture.
2. Add or update a parser regression test against that fixture.
3. Run the normal `tests/market_monitor` suite.
4. Optionally run the live smoke test locally to confirm the source still responds.
