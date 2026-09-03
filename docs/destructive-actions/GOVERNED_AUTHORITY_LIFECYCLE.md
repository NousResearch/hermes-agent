# Governed destructive capability — authority lifecycle

This document describes the authority model used by `governed_mkfs` and the
one-shot destructive grant implementation.

## Threat model

Model-facing tools may execute code and write files under the same OS account
as Hermes. File permissions and an unkeyed digest therefore do not establish a
human/model trust boundary. A valid destructive capability must instead satisfy
both properties below:

1. issuance follows an explicit, correlated, one-shot human approval; and
2. the persisted grant is authenticated by authority unavailable to
   model-facing execution surfaces.

## Human approval receipt

Issuance consumes a process-local `HumanApprovalReceipt` created only after an
explicit `approve once` decision from the existing approval subsystem. The
receipt is correlated to the request and session context and is consumed
atomically when a grant is issued.

Session/permanent approval, smart/auxiliary-LLM approval, yolo, approval mode
off, cron auto-approval, unattended transports, and serialized receipt replay
must not mint destructive authority.

## Grant authentication

`GrantAuthorityProvider` defines the authentication boundary. The portable
baseline is `ProcessEphemeralAuthority`, which authenticates the canonical
immutable grant payload with HMAC-SHA256 using a random 256-bit process-local
secret.

The secret is not persisted, placed in the environment, serialized, or passed
to child execution processes. A child process therefore obtains a different
authority generation and cannot mint a parent-valid grant. Restarting Hermes
creates a new generation and intentionally invalidates outstanding grants.

The canonical authenticated payload includes the grant and receipt identities,
operation tuple, authorization metadata, validity window, nonce, session
correlation, and the authorized guest incarnation. Clone, TTL extension,
identifier substitution, authorization-source tamper, or receipt substitution
therefore fail verification.

The provider interface is also the extension point for an optional TPM/vTPM or
HSM-backed implementation. Hardware backing is a strengthening layer, not a
feature prerequisite, and must never expose a generic signing oracle to the
model.

## Target-generation fencing

A filesystem grant is bound to the observed guest incarnation:

- `vm_id` selects the logical target;
- `product_uuid` identifies the guest/VM incarnation;
- `boot_id` identifies the current boot generation;
- `hostname` is retained as an additional observed identity field.

The identity is captured before issuance and re-read at the mutation boundary.
A grant for generation A cannot mutate a rebooted or replaced generation B even
if the logical VM id and device path are reused.

## Claim and settlement

The lifecycle is:

```text
LIVE
  |
  | atomic claim (exactly one winner)
  v
CLAIMED(execution_id)
  |
  +--> FAILED_PRE_EFFECT
  |
  +--> EXECUTION_STARTED
          |
          +--> COMPLETED
          +--> INDETERMINATE
```

A claim happens before the first irreversible operation. Once execution may
have started, the capability never returns to the live pool. Lost postcondition
proof or ambiguous execution outcome settles as `indeterminate`; it is never
reported as a replayable denial.

## Structured execution edge

`governed_mkfs` keeps raw `mkfs` on the generic terminal hardline. The governed
path validates the exact grant tuple, generation fence and fail-closed storage
prechecks, then invokes `qga_create_filesystem` with a fixed argv derived only
from allowlisted filesystem binaries and validated fields.

The QGA transport is deployment-configured. No host address, SSH key path, or
permissive host-key policy is compiled into Hermes. The transport refuses to
operate without explicit control-plane configuration and uses strict SSH
host-key verification.

## Security invariants

- model surfaces may request authority but cannot approve themselves;
- a human approval receipt issues at most one grant;
- a grant is authenticated, short-lived, exact-scope and one-shot;
- `execute_code` and `write_file` cannot forge a parent-valid grant;
- stale guest generations are denied before mutation;
- concurrent replay has exactly one claim winner;
- uncertain post-effect outcomes are durably `indeterminate`;
- raw terminal filesystem formatting remains hardline-denied.
