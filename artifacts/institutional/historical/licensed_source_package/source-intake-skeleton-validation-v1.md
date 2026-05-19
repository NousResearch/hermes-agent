# Source Intake Validation Report

**Evidence status:** `SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE`

No executable quote, RFQ, advice, or investment readiness is implied by this report. It only validates whether historical source packages are structurally replay-ready.

- Ready: `False`
- Coverage: `0/6 source groups structurally valid`
- Blocker count: `18`

## Blockers

- IBIT options history: raw_sha256 missing or invalid
- IBIT options history: row_count must be positive
- IBIT options history: available_end missing or invalid
- Deribit options history: raw_sha256 missing or invalid
- Deribit options history: row_count must be positive
- Deribit options history: available_end missing or invalid
- CME Bitcoin options history: raw_sha256 missing or invalid
- CME Bitcoin options history: row_count must be positive
- CME Bitcoin options history: available_end missing or invalid
- BTC reference history: raw_sha256 missing or invalid
- BTC reference history: row_count must be positive
- BTC reference history: available_end missing or invalid
- IBIT holdings history: raw_sha256 missing or invalid
- IBIT holdings history: row_count must be positive
- IBIT holdings history: available_end missing or invalid
- Rates and fee curves: raw_sha256 missing or invalid
- Rates and fee curves: row_count must be positive
- Rates and fee curves: available_end missing or invalid

## Source Results

- IBIT options history: blocked; rows=0; format=csv; provenance=replace-with-licensed-provider-or-broker-export; license=replace-with-license-or-contract-label
- Deribit options history: blocked; rows=0; format=csv; provenance=replace-with-licensed-provider-or-broker-export; license=replace-with-license-or-contract-label
- CME Bitcoin options history: blocked; rows=0; format=csv; provenance=replace-with-licensed-provider-or-broker-export; license=replace-with-license-or-contract-label
- BTC reference history: blocked; rows=0; format=csv; provenance=replace-with-licensed-provider-or-broker-export; license=replace-with-license-or-contract-label
- IBIT holdings history: blocked; rows=0; format=csv; provenance=replace-with-licensed-provider-or-broker-export; license=replace-with-license-or-contract-label
- Rates and fee curves: blocked; rows=0; format=csv; provenance=replace-with-licensed-provider-or-broker-export; license=replace-with-license-or-contract-label
