# NLS-184 — External exact-once approval protocol test contract

Lifecycle: `test-lane / RED / implementation pending`

## Scope

This is the headless approval path for dangerous terminal commands. It is
enabled only when `HERMES_EXEC_ASK=1` and
`HERMES_EXTERNAL_APPROVAL_MODE=exact-once`. It must not modify existing CLI or
gateway approvals, session YOLO, or the permanent allowlist.

## Exact v1 ABI

There is no generic Python approval adapter and no stdout/stderr/screen
protocol. Hermes configures the concrete FD transport only:

```python
configure_external_approval_fd_protocol(
    *, grant_input_fd: int | None = None, record_output_fd: int
) -> None
clear_external_approval_fd_protocol() -> None
build_external_approval_request(
    *, command: str, operation_kind: str, tool_identity: str,
    session_id: str, profile: str,
) -> dict
external_approval_tool_subprocess_kwargs() -> dict
```

`grant_input_fd` is optional. When set, it is a read-only, newline-delimited
JSON input from the trusted adapter. When omitted (record-only), Hermes emits
requests on the record FD and `_try_read_grant_line()` returns no grant —
never falling back to stdin or other generic channels. `record_output_fd` is a
write-only, newline-delimited JSON output for both requests and receipts. The
trusted adapter has the Ed25519 private key; Hermes receives only the 32-byte
raw public verification key. Each configured FD is non-inheritable, and
`external_approval_tool_subprocess_kwargs()` returns
`{"close_fds": True, "pass_fds": ()}` for every normal tool subprocess. The
transport owner must apply those kwargs at every terminal subprocess launch.

OS pipes are acceptable only in tests as a peer for this interface. Tests must
observe request/receipt bytes from `record_output_fd` and prove they did not
reach stdout/stderr; they must never install a generic in-process adapter.

## Canonical signed records

All records are JSON objects encoded as canonical UTF-8 JSON: recursive
lexicographic key order, `,`/`:` separators, `ensure_ascii=False`, and one
trailing newline on the FD. The Ed25519 signing message is exactly the
canonical grant object with `signature` omitted. `signature` is standard base64
of the 64-byte signature; `algorithm` is exactly `Ed25519`.

`tests/fixtures/external_approval_v1/schema.json` and the canonical request,
grant, and receipt examples are the normative v1 test fixtures.

Every request is:

```json
{"approval_id":"…","kind":"request","operation":{"fingerprint":"sha256(raw-command-utf8)","kind":"terminal.command","tool":"terminal"},"protocol":"hermes.external-approval","session":{"id":"…","profile":"…"},"version":1}
```

The request never transports the raw command, description, display text, or a
secret. Its fingerprint is SHA-256 of the exact raw command UTF-8 bytes. Do not
normalize, redact, strip, or substitute timestamps, UUIDs, temporary paths,
shell substitutions, whitespace, or any other dynamic bytes. The operation
kind/tool identity and every different byte produce a different fingerprint
and approval ID.

Every grant is signed and contains each field exactly once:

```json
{"algorithm":"Ed25519","approval_id":"…","choice":"approve_once","expires_at":0,"issued_at":0,"kind":"grant","operation":{"fingerprint":"…","kind":"terminal.command","tool":"terminal"},"protocol":"hermes.external-approval","session":{"id":"…","profile":"…"},"signature":"base64","version":1}
```

Only `choice: "approve_once"` is supported. Hermes must reject malformed JSON
(including duplicate members), unsupported versions/choices, missing fields,
invalid signatures, a wrong public key, any post-signing change, stale grants,
and any mismatch of approval ID, operation fingerprint/kind/tool, session ID,
or profile.

A receipt is emitted on `record_output_fd` only after durable consumption wins:

```json
{"approval_id":"…","choice":"approve_once","consumed_at":0,"kind":"receipt","operation":{"fingerprint":"…","kind":"terminal.command","tool":"terminal"},"protocol":"hermes.external-approval","session":{"id":"…","profile":"…"},"version":1}
```

The consumption key is persistent per profile and survives a resumed Hermes
process. A grant can win once only. Receipt write failure is fail closed at the
caller: it authorizes no subprocess; the consumed grant remains unavailable for
replay. An unsigned fixture or any receipt is never authorization evidence.

## Required RED assertions

- Requests/receipts are dedicated-FD records, absent from stdout/stderr, and
  normal tool subprocesses cannot inherit/read either protocol FD.
- A trusted Ed25519 signature authorizes one exact command once; invalid,
  tampered, wrong-key, unsupported-choice, and missing-field grants deny it.
- Fingerprints bind `terminal.command` + `terminal` and exact raw UTF-8 bytes;
  timestamps, UUIDs, temp paths, shell substitutions, and whitespace changes
  require a new approval.
- Reconfiguration for a resumed process cannot replay the consumed grant.
- Receipt failure and unsigned input fail closed with zero side effects.
- The path does not call or mutate session approval, YOLO, or permanent
  allowlist state.
