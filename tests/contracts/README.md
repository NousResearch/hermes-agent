# Noise-gate contract tests

This directory contains the standardized QA contract suite for the Hermes/OpenClaw `BE-REL-NOISE-001` / `BE-NG-QA-SUPPORT-001` noise-gate behavior.

## Purpose

The suite turns the QA checklist into reusable pytest tests for any backend implementation of:

```python
evaluate_event(event: dict) -> dict
```

It covers:

1. Duplicate replay: first event accepted, replay suppressed.
2. Canonicalization: formatting-only/key-order changes cannot bypass dedupe.
3. TTL/window: duplicate inside TTL suppressed; after TTL accepted as a new window.
4. Cross-profile independence: separate profiles do not suppress each other by default.
5. Invalid key behavior: malformed/unsafe events are rejected before side effects.
6. Observable metadata: audit-safe hashes only; no raw private identifiers.
7. Concurrency: parallel duplicates produce exactly one accepted result.

## Running against an implementation

Point the test suite at the backend callable:

```bash
HERMES_NOISE_GATE_EVALUATE_EVENT="package.module:evaluate_event" \
python -m pytest tests/contracts/test_noise_gate_contract.py -q -m contract
```

If `HERMES_NOISE_GATE_EVALUATE_EVENT` is unset, the tests skip. This lets the repository carry the standard before the backend target exists, while making future implementation proof one command away.

## Required event input shape

The callable receives one dict with these fields:

1. `profile`
2. `lane`
3. `event_type`
4. `source_scope_hash`
5. `semantic_payload`
6. `semantic_event_hash`
7. `ttl_seconds`
8. `created_at`

The suite uses unique `lane` values per test so a durable backend store does not need manual cleanup between tests.

## Required result shape

The callable should return a dict-like `NoiseGateResult` with at least:

1. `decision`
2. `side_effect_allowed`
3. `dedupe_key_hash`
4. `observable_metadata.source_scope_hash`
5. `observable_metadata.semantic_event_hash`
6. `observable_metadata.raw_private_fields_present`

Accepted decisions:

1. `accepted`
2. `accepted_after_window`

Suppression decision:

1. `suppressed_duplicate`

Rejected decisions:

1. `rejected_invalid_key`
2. `rejected_invalid_ttl`

## Privacy rules

Do not include raw chat IDs, thread IDs, email addresses, tokens, credentials, or private message/email contents in implementation results or logs. The tests assert that observable metadata stays hash-only and that invalid raw source scopes are rejected before side effects.
