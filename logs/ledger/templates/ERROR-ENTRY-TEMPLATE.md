<!--
Add to logs/ledger/errors/ERROR-LOG.md: a full block under "## Open", and a row in
the "Register (quick scan)" table. On close, move the block under "## Resolved".
Rules: logs/ledger/README.md

ERR- is for FAULTS (something is wrong). An open judgment call goes in
decisions/DECISION-LOG.md as a DECISION- instead — see README § ERR- vs DECISION-.
-->

### ERR-YYYY-MM-DD-NNN — <SEVERITY> — <area>

- **Opened:** YYYY-MM-DD · **Base:** hermes@<sha> (<N> behind upstream/main)
- **Run:** RUN-YYYY-MM-DD-NNN
- **Source:** <AUDIT- id / test run / manual observation>
- **What:** <symptom — exact error text if short, command that triggered it>
- **Confidence:** Confirmed Fact · Field-Reasoned · Unverified   <!-- confidence that the fault is real and as described. See README. -->
- **Exposure / impact:** <blast radius; is anything committed or shipped?>
- **Status:** OPEN
- **Required action:** <numbered steps; who must act>

<!-- On resolution, append: -->
- **Resolved:** YYYY-MM-DD — <what fixed it>. Resolving change: CHG-YYYY-MM-DD-NNN.
- **Status:** RESOLVED   <!-- or WONTFIX / ACCEPTED-RISK / SUPERSEDED (name the record that replaces it), with rationale -->

<!-- Register row:
| ERR-YYYY-MM-DD-NNN | YYYY-MM-DD | <SEV> | <area> | <summary> | OPEN | — |
-->
