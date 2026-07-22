# INGESTION_PIPELINE_DESIGN

## Purpose
Define ordered pipeline stages and interfaces for all data ingress into Hermes/BlackGold.

## Flow
source -> quarantine -> validation -> classification -> parsing -> schema validation -> duplicate detection -> approval -> storage

## Stage interfaces

### source
- Accepts url, path, message, inline payload.
- Rejects live confidential sources until Arthur authorizes live intake.

### quarantine
- Writes only to `data/vault/quarantine`.
- MIME and size limit enforced.

### validation
- Schema shape check.
- Encoding validation.
- Size limits and rate limits.

### classification
- Closed classifier: tier T0/T1/T2/T3/T4 only.
- PII pattern classifier fallback.
- No unrestricted LLM classification.

### parsing
- Deterministic mapping and flattening.
- No inference of missing fields.

### schema validation
- Required fields enforced.
- Type coercion limited to allowed fields.

### duplicate detection
- Content hash + primary key constraints.
- Dedupe in quarantine.

### approval
- Approval gate enforced via HUMAN_APPROVAL_GATES.
- No auto promotion from quarantine on failure level mismatch.

### storage
- Proceeds only if approval is recorded.
- Writes follow DATA_ACCESS_MATRIX rules.

## Observability
- Each stage emits event into AUDIT_LOG_STANDARD.
- Glanceable throughput and fail counts exposed via minimal CLI dashboard.
