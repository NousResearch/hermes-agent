# Money Flow Radar Reliability Harness

## Purpose

The Money Flow Radar reliability harness evaluates already-generated report
payloads before future workflows promote, display, or compare them. It is meant
for Hermes-side cron, system doctor, and Obsidian promotion callers that need a
small offline gate for freshness, provenance, contradiction visibility,
calibration readiness, reproducibility, and safety-boundary language.

The implementation lives in `agent/money_flow_reliability.py`.

Primary entry points:

- `evaluate_money_flow_report(report, now=None)`
- `load_money_flow_report(path)`
- `evaluate_money_flow_report_path(path, now=None)`
- `fingerprint_money_flow_report(report)`

## Expected Report Shape

Callers pass a mapping or JSON object. The harness is permissive about exact
field layout, but expects these concepts:

- `report_id`: stable report identifier
- `generated_at`: ISO-8601 timestamp
- `source_freshness` or `sources`: source names plus status and `as_of`,
  `retrieved_at`, or another parseable timestamp
- `regime`: current liquidity/regime interpretation
- `rotation`: asset-class, region, sector, or theme rotation interpretation
- `evidence`: source-backed evidence for both `regime` and `rotation`
- `contradictions`: list or mapping, empty when none are known
- `watchlist`: monitoring-only items
- `review_ledger_summary`: reviewed report count and optional expansion status

Example:

```python
report = {
    "report_id": "mfr-2026-05-12",
    "generated_at": "2026-05-12T00:00:00Z",
    "source_freshness": {
        "global_liquidity": {"status": "fresh", "as_of": "2026-05-11T23:00:00Z"},
    },
    "regime": "risk-on liquidity impulse",
    "rotation": "technology and industrials leadership under review",
    "evidence": {
        "regime": [{"claim": "Liquidity improved.", "source_id": "liq-dashboard"}],
        "rotation": [{"claim": "Leadership broadened.", "citations": ["sector-flows"]}],
    },
    "contradictions": [],
    "watchlist": [
        "Monitor only: flow confirmation is for review; user decides whether any action is appropriate."
    ],
    "review_ledger_summary": {"reviewed_reports": 14, "expansion_eligible": True},
}
```

## Checks

The result contains `overall_status`, `summary`, `report_id`,
`report_fingerprint`, and `checks`. Each check has:

- `name`
- `status`: `OK`, `WARN`, or `FAIL`
- `detail`
- `remediation`
- `category`

Current checks:

- `report_identity`: requires `report_id` and parseable `generated_at`
- `source_freshness`: requires visible source metadata; stale sources warn or
  fail by age/status
- `regime_rotation_evidence`: requires regime and rotation evidence with visible
  provenance such as source IDs, citations, URLs, or source timestamps
- `contradictions_visible`: requires a contradictions field, even when empty
- `watchlist_safety_boundary`: rejects buy/sell/hold, position sizing, and
  order-timing language
- `calibration_gate`: requires 14 reviewed reports before expansion eligibility
- `report_fingerprint`: returns a deterministic SHA-256 fingerprint excluding
  volatile `generated_at`
- `result_json_serializable`: verifies the normalized result can be JSON encoded

Secret-looking strings in details and remediation text are redacted.

## Safety Boundary

The harness is not an investment advisor. It enforces watchlist language that
stays observational: monitor, review, evidence, risk, and user-decision wording.
It fails report payloads that emit direct recommendations or execution language,
including buy/sell/hold, position sizing, portfolio allocation, entry/exit
timing, order types, stop loss, take profit, or target price language.

Reports should describe evidence and uncertainty. Decisions remain with the
user.

## Calibration Gate

Money Flow Radar expansion is not eligible until at least 14 reports have been
reviewed. The harness reads `review_ledger_summary.reviewed_reports` and related
count fields. It warns below 14 reviewed reports and fails if a payload claims
`expansion_eligible=True` before the count reaches 14.

## Non-Goals

This harness does not:

- fetch live market data
- depend on `/Users/umbbi/projects/US_market`
- write to broker, order, or execution systems
- produce direct recommendations
- size portfolios or positions
- run a monitoring stack
- write to Obsidian, memory, or cron by default

It only evaluates local mappings or JSON files and returns a structured result.

## Example Usage

```python
from agent.money_flow_reliability import evaluate_money_flow_report_path

result = evaluate_money_flow_report_path("/tmp/mfr-report.json")
if result.overall_status.value == "OK":
    print(result.report_fingerprint)
else:
    for check in result.checks:
        if check.status.value != "OK":
            print(check.name, check.status.value, check.remediation)
```

Targeted verification:

```bash
HOME=/Users/umbbi /Users/umbbi/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts= tests/agent/test_money_flow_reliability.py -q
HOME=/Users/umbbi /Users/umbbi/.hermes/hermes-agent/venv/bin/python -m py_compile agent/money_flow_reliability.py tests/agent/test_money_flow_reliability.py
git diff --check
```
