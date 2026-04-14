# Noise-Signal Heuristic Registry v1

Date: 2026-04-04  
Status: support-only registry (LT-01 bounded contract slice)  
Pack contract: `docs/ops/schemas/noise_signal_heuristic_pack.v1.schema.json`

## Purpose

Provide a reusable, auditable index for LT-01 / B8Q-003 noise-vs-signal heuristics so operator surfaces can consume the same heuristic IDs and suppression-counter naming.

## Registered heuristic IDs

| ID | Label | Primary consumer(s) | Counter key |
|---|---|---|---|
| H1 | stale-wave remediation residue | A6 layered health, A2 dispatch context | `h1_suppressed` |
| H2 | closed-blocked stagnation escalation | A6 layered health, A2 dispatch context | `h2_suppressed` |
| H3 | false-positive completion (artifact missing) | C1 worker state board | `h3_suppressed` |
| H4 | quota exhausted without reroute guidance | C1 worker state board | `h4_suppressed` |
| H5 | low-signal candidate opportunity | operator triage console | `h5_suppressed` |
| H6 | UI evidence without execution link | operator triage console, B8 packet runtime | `h6_suppressed` |
| H7 | stale-wave without remediation-health coupling | A6 layered health | `h7_suppressed` |
| H8 | undifferentiated regression risk | B8 packet runtime | `h8_suppressed` |
| H9 | drift truthfulness mismatch | C1 worker board, A2 dispatch context | `h9_suppressed` |
| H10 | candidate-cap overflow | operator triage console | `h10_suppressed` |

## Canonical example packet

- `reports/lt01_noise_signal_heuristic_pack_2026-04-04.example.json`

## Validation entrypoint

- `scripts/lt01_noise_signal_heuristic_pack_validate.py`

## Notes

- Registry is support-layer for LT-01; no canonical lane promotion is implied by this file.
- If a heuristic is retired/replaced, keep ID history and mark status in the packet (`status=retired`) instead of reusing IDs.
