# PRD-6 Phase-2 LCM Live Recovery Gate

Generated: 2026-06-18 04:29 UTC
Mode: live
Verdict: BLOCKED
Trials requested: 120; trials scored: 0
Sampling: temperature=0.0, seed=1729; fixture order shuffled by seed

## Gate summary

- Correct: 0/0
- Point recall: 0.000000 (required >= 0.950)
- Wilson 95% lower bound: 0.000000 (required >= 0.900)
- Confident-wrong: 0 (required 0)
- Missing tool-call evidence: 0
- VOID (no store lookup): 10/10 draws (rate 1.0000; max 0.20)
- estimated spend: $0.425350 / cap $25.000000
- observed spend: $0.352750 / cap $25.000000

## Wilson arithmetic

Arithmetic: successes=0, n=0, z=1.96, phat=successes/n=0.000000
denominator = 1 + z^2/n = 1.000000
centre = phat + z^2/(2n) = 0.000000
margin = z * sqrt((phat*(1-phat) + z^2/(4n))/n) = 0.000000
lower = (centre - margin) / denominator = 0.000000

## Judge calibration

- Semantic arms absent; judge calibration not required.

## Failures

- no trials scored
- live Phase-2 gate requires N=120 minimum; scored 0
- VOID rate 1.0000 exceeds max 0.2000 (10/10 draws produced no store lookup)
- VOID rate 1.000 exceeded max 0.200 after 10 draws (10 void) — bury is not reliably exercising the store path; this is a finding, not a redraw-around.

## Trial records

| prompt_id | buried_fact | tool_calls | answer | correct | confidence_wrong | estimated spend | observed spend |
|---|---|---|---|---|---|---:|---:|

## Harness notes

- Dry-run mode uses stubbed transcripts and responses; it exercises scoring, report, Wilson, judge, and budget gates without live spend.
- Live mode does not restart gateways or flip configs; it assumes Apollo has activated the Aegis profile and only invokes the configured Aegis command.
- Live mode records per-trial prompt id, buried fact, tool calls, answer, correctness, confidence-wrong adjudication, and spend.
