<!--
Add to logs/ledger/errors/ERROR-LOG.md: a full block under "## Open", and a row in
the "Register (quick scan)" table. On close, move the block under "## Resolved".
Rules: logs/ledger/README.md
-->

### ERR-YYYY-MM-DD-NNN — <SEVERITY> — <area>

- **Opened:** YYYY-MM-DD · **Base:** hermes@<sha> (<N> behind upstream/main)
- **Source:** <AUDIT- id / test run / manual observation>
- **What:** <symptom — exact error text if short, command that triggered it>
- **Exposure / impact:** <blast radius; is anything committed or shipped?>
- **Status:** OPEN
- **Required action:** <numbered steps; who must act>

<!-- On resolution, append: -->
- **Resolved:** YYYY-MM-DD — <what fixed it>. Resolving change: CHG-YYYY-MM-DD-NNN.
- **Status:** RESOLVED   <!-- or WONTFIX / ACCEPTED-RISK with rationale -->

<!-- Register row:
| ERR-YYYY-MM-DD-NNN | YYYY-MM-DD | <SEV> | <area> | <summary> | OPEN | — |
-->
