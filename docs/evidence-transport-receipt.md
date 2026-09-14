# Hermes Evidence Transport Receipt v2

Status: implementation candidate. This protocol is a plain Python library; it
is not a model tool and does not change toolsets, presentation limits,
redaction, approvals, attachment storage, databases, or Kanban behavior.

## Purpose

Machine-generated evidence that requires exact-byte verification must travel as
a producer-owned durable file plus a bounded receipt. Terminal output,
line-numbered file presentation, excerpts, caches, and reconstructed prose are
not authoritative byte transports.

The transport invariant is:

```text
exclusive create -> write -> flush -> fsync -> close
                 -> hash finalized on-disk bytes
                 -> canonical bounded receipt
                 -> binding comparison
                 -> independent regular-file read and hash comparison
```

Only `VERIFIED` confirms that the bound file has the declared SHA-256 digest and
byte length. `UNAVAILABLE` covers a malformed or incomplete receipt/binding and
a missing, unreadable, symlink, or non-regular artifact. A parsed receipt whose
provenance, digest, or length does not match produces `INTEGRITY_FAILURE`.

## Canonical receipt

The schema is `hermes-evidence-receipt-v2`. Encode the complete object with:

```python
json.dumps(
    receipt,
    separators=(",", ":"),
    sort_keys=True,
    ensure_ascii=True,
)
```

The encoded UTF-8 receipt is at most 2048 bytes. It has no BOM, non-finite JSON
number, trailing whitespace, unknown key, nested value, metadata channel, prose,
artifact excerpt, artifact body, base64 payload, or receipt self-hash. Oversized
input is rejected rather than truncated.

Every receipt contains exactly these required keys:

| Key | Contract |
| --- | --- |
| `schema` | exactly `hermes-evidence-receipt-v2` |
| `status` | exactly `ready` |
| `artifact_path` | absolute string, 1..1024 characters, no NUL/C0 control |
| `sha256` | exactly 64 lowercase hexadecimal characters |
| `byte_length` | JSON integer (not Boolean), 0 through 2^63-1 |
| `created_utc` | 20..40-character RFC3339 UTC string ending in `Z` or `+00:00` |
| `profile` | 1..64 characters from `[A-Za-z0-9._-]` |
| `session_id` | 1..64 characters from `[A-Za-z0-9._-]` |
| `execution_kind` | exactly `standalone-single-query` |
| `execution_id` | same identifier grammar as `session_id`, and equal to it |
| `grant_id` | 1..128 characters from `[A-Za-z0-9._-]` |
| `cell_id` | 1..16 characters from `[A-Za-z0-9._-]` |

The v1 schema and any extension field, including `meta`, are invalid.

## Producer

`produce_receipt(path, data)` first requires and validates:

- `HERMES_PROFILE`
- `HERMES_SESSION_ID`
- `HERMES_EVIDENCE_GRANT_ID`
- `HERMES_EVIDENCE_CELL_ID`

It sets `execution_kind` to `standalone-single-query` and copies `session_id`
into `execution_id`. Missing or invalid provenance prevents artifact creation.
The producer uses `O_CREAT | O_EXCL` mode `0o600`, so an existing path or planted
symlink is not overwritten. It flushes and fsyncs the artifact, closes it, then
reopens the finalized regular file to derive SHA-256 and byte length without
normalization. No ready receipt is returned if any prior step fails.

`write_artifact_exclusive(path, data)` exposes the same exclusive durable write
primitive for callers that need it, but does not itself emit a receipt.

## Consumer

`verify_receipt(text, expected_binding=...)` requires a complete expected
binding with exactly `profile`, `session_id`, `execution_kind`, `execution_id`,
`grant_id`, and `cell_id`. These values come from parent-observed launcher
profile, post-termination session storage/footer, and parent-set grant/cell
environment. Missing or incomplete authority returns `UNAVAILABLE` even when the
artifact bytes happen to match.

The consumer parses the hostile receipt text without allowing exceptions to
escape. It compares every binding field before attempting the artifact read. A
binding mismatch returns `INTEGRITY_FAILURE`. It then opens the path without
following symlinks, requires a regular file, independently hashes the exact
bytes, and compares both digest and byte length. It never rewrites the expected
digest and never falls back to a neighboring file, terminal spill, cache,
excerpt, prose, or presentation reconstruction.

## Exact-byte coverage

Regression coverage includes 4096-byte data, a 61,453-byte long JSONL record,
UTF-8 multibyte text, CRLF, LF, BOM, final-LF and no-final-LF content, and binary
bytes. Mutation coverage includes bit flips, truncation, append, missing paths,
symlinks, directories, malformed types, non-canonical JSON, unknown fields,
oversized receipts, missing provenance, incomplete bindings, and stale or
swapped provenance.
