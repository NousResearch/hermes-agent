# Truth Ledger schema contracts (T6)

This document freezes the versioned schema surface for deterministic validation.

## Draft and loading model

- JSON Schema draft: 2020-12 (`$schema: https://json-schema.org/draft/2020-12/schema`).
- Runtime loader: `plugins/truth-ledger/schemas.py`.
- Validation mode: fail-closed (`ValueError` on any schema mismatch).
- Unknown schema version behavior: reject.

## Versioned schema inventory (v1)

1. `truth-ledger.source-envelope.v1`
   - File: `plugins/truth-ledger/schemas/source-envelope-v1.schema.json`
   - Purpose: raw completed-turn capture envelope with bounded message fields.

2. `truth-ledger.fact-candidates.v1`
   - File: `plugins/truth-ledger/schemas/fact-candidates-v1.schema.json`
   - Purpose: extractor candidate facts with controlled vocabularies and confidence.

3. `ledger-event.v1` (schema name is implicit in file/validator key)
   - File: `plugins/truth-ledger/schemas/ledger-event-v1.schema.json`
   - Purpose: immutable append-only events for `assert|confirm|supersede|retract`.

4. `truth-ledger.spool-record.v1`
   - File: `plugins/truth-ledger/schemas/spool-record-v1.schema.json`
   - Purpose: pending/processing/dead-letter transition metadata.

5. `truth-ledger.dead-letter.v1`
   - File: `plugins/truth-ledger/schemas/dead-letter-v1.schema.json`
   - Purpose: bounded, sanitized failure records.

6. `truth-ledger.current-projection.v1`
   - File: `plugins/truth-ledger/schemas/current-projection-v1.schema.json`
   - Purpose: current-state materialization entries.

## Contract rules frozen by tests

- Compact UTF-8 JSONL serialization with trailing newline (`contracts.serialize_jsonl_record`).
- Unknown `schema_version` is rejected (`contracts.assert_schema_version`).
- Enum and operation checks reject invalid values.
- Oversized source-envelope payloads fail validation.
- Projection `value` key must be present (explicit null allowed, missing disallowed).
- Every v1 schema passes Draft 2020-12 metaschema validation.
