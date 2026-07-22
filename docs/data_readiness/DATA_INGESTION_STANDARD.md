# DATA_INGESTION_STANDARD

## Purpose
Define the mandatory ingestion workflow and validation contract for all BlackGold-related data entering the platform.

## Pipeline stages
1. source acceptance
2. quarantine isolation
3. structural validation
4. classification + PII scan
5. parsing/normalization
6. schema validation
7. duplicate detection
8. human approval
9. storage in validated area

## Stage contracts

### source acceptance
- Only synthetic or Arthur-approved inputs accepted.
- Reject old confidential sources until authorized.

### quarantine isolation
- All inputs land in `data/vault/quarantine`.
- No agent may bypass quarantine.

### structural validation
- Valid UTF-8, valid JSON/YAML, size limits enforced.
- Hard failure on malformed input.

### classification + PII scan
- Closed-list classifier; no open-ended LLM classification.
- Hits on SSN/account/card/mask patterns trip hold for review.
- No live confidential data allowed under restoration freeze.

### parsing/normalization
- Stable field mapping; do not infer new fields.
- Missing required fields -> hold.

### schema validation
- Required fields and typed validation enforced.
- Reject before storage on mismatch.

### duplicate detection
- Content hash + key match.
- Exact duplicates skip re-processing.

### human approval
- Only Arthur Lyons or explicitly authorized humans approve live-data promotion.
- Audit logging required.

### storage in validated area
- Validated records are written to `data/vault/validated`.
- Encoded with tier marker.

## Failure behavior
Any stage failure returns error envelope with `stage`, `code`, `details`, `actionable_remediation`.
