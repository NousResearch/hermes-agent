# Market Monitor fixtures

- `tests/fixtures/market_monitor/` stores CI-safe captured HTML/JSON snippets.
- Mandatory unit/integration tests only use these fixtures.
- Real-source smoke tests must be opt-in and marked `integration`.
- Update fixtures when source layouts change, then pin regressions in parser tests.
