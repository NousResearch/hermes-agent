# Source Intake Validation Report

**Evidence status:** `SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE`

No executable quote, RFQ, advice, or investment readiness is implied by this report. It only validates whether historical source packages are structurally replay-ready.

- Ready: `False`
- Coverage: `0/6 source groups structurally valid`
- Blocker count: `6`

## Blockers

- IBIT options history: fixture/manual_fixture sources cannot satisfy readiness
- Deribit options history: fixture/manual_fixture sources cannot satisfy readiness
- CME Bitcoin options history: source group missing
- BTC reference history: source group missing
- IBIT holdings history: source group missing
- Rates and fee curves: source group missing

## Source Results

- IBIT options history: blocked; rows=12; format=csv; provenance=manual_fixture; license=fixture
- Deribit options history: blocked; rows=12; format=jsonl; provenance=manual_fixture; license=fixture
