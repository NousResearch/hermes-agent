# IVX Edge pilot foundation

Status: **simulator and security contract only**. This repository does not
currently ship a signed IVX Edge daemon, installer, OTA channel, cloud
enrollment service, or production fleet control plane.

## Product boundary

- **IVX Agency Desktop** delegates computer-based knowledge work through the
  existing Electron/Hermes runtime.
- **IVX Edge** is the proposed first-party endpoint runtime for granting a
  supported physical device narrow, revocable capabilities under policy.
- **KioskX** remains the separate phygital retail product. It may consume a
  future IVX Edge endpoint contract but is not renamed or replaced.
- The router repository's `edge/` service is the network/API layer, not this
  device runtime.

The smallest credible product is a signed, least-privilege device daemon with
one-time enrollment, per-device credentials, a capability manifest, bounded
adapters, signed commands, replay protection, audit, revocation, heartbeat,
offline idempotency, staged update, and rollback. Only the security contract
and deterministic safe adapters exist here.

## Current implementation

Run the local-only demo:

```bash
python -m ivx_edge
```

It performs no hardware, network, shell, filesystem, camera, microphone, GPIO,
printer, or actuator operation. The runtime currently implements:

- one-time enrollment codes with a maximum ten-minute lifetime;
- a random 256-bit per-device key returned only to the enrolling endpoint;
- canonical HMAC-SHA256 command signatures;
- exact workspace/App/device identity binding;
- five-minute maximum command lifetime and future-clock bound;
- capability allowlisting and registered-adapter enforcement;
- nonce replay protection and command-ID idempotency;
- offline queueing with verification again at execution time;
- revocation that fails closed and clears queued work;
- payload-digest audit records with no raw payload or key;
- a read-only mock temperature adapter;
- a simulated screen-content adapter with explicit confirmation and a bounded
  content identifier.

HMAC is suitable for this deterministic foundation, not the final key
architecture. Production design should move command signing to asymmetric
cloud keys and device identity to hardware-backed keys or certificates where
the supported platform provides them.

## Local/cloud split

| Local endpoint | Cloud control plane |
| --- | --- |
| Holds device credential in OS/hardware-backed storage | Issues short-lived enrollment challenges |
| Verifies signature, scope, lifetime, nonce, policy | Signs bounded commands and records intent |
| Runs only compiled-in or separately verified adapters | Hosts fleet inventory, RBAC, and rollout policy |
| Queues stable command IDs while disconnected | Reconciles heartbeat, audit, and revocation |
| Enforces confirmation and physical safety locally | Performs reasoning without bypassing local policy |
| Emits minimal, redacted events | Writes consented, scoped summaries to Memory |

No cloud response may weaken local policy. Loss of connectivity, identity
ambiguity, stale policy, expired command, or invalid signature fails closed.

## Threat model and required gates

| Threat | Foundation behavior | Production gate |
| --- | --- | --- |
| Enrollment theft | One-time short-lived code, exact workspace/App binding | Authenticated operator ceremony, rate limit, device display proof |
| Key extraction / cloning | Random per-device key | TPM/Secure Enclave/PKCS#11 storage, rotation, clone detection |
| Replay / offline duplicates | Nonce set plus stable command-ID result | Durable replay ledger across reboot and bounded retention |
| Command injection | Structured payload and adapter validation | Adapter sandbox, fuzzing, signed adapter catalog |
| Capability escalation | Explicit allowlist and adapter registry | Fleet policy signatures, RBAC, two-person high-risk grants |
| Cross-tenant command | Exact workspace/App/device match | Contract tests against production identity service |
| Compromised cloud | Local signature, policy, lifetime, confirmation | Separate signing authority, emergency kill switch |
| OTA compromise / rollback | Not implemented | TUF-style metadata, signed artifacts, staged rollout, anti-rollback |
| Camera/microphone abuse | No capture adapter | Explicit consent, visible indicator, bounded retention, legal review |
| Physical harm | No GPIO/actuator adapter | Per-adapter hazard analysis, interlocks, local emergency stop |
| Log leakage | Digest-only audit | Structured redaction tests and tenant-scoped audit export |
| Device loss | Runtime revoke fails closed | Remote wipe of app data, certificate revocation propagation |

## Support matrix

| Target | Status | Evidence required before support |
| --- | --- | --- |
| Python simulator on macOS arm64 | Development-tested | Unit tests and local CLI demo |
| Browser simulator | Implemented in router public-site branch | Browser, keyboard, responsive, theme, and Axe verification |
| Linux x86-64 mini-PC | Pilot candidate | Signed package, systemd sandbox, suspend/reboot, 72-hour soak |
| Raspberry Pi 4/5 64-bit Linux | Pilot candidate | Board-in-loop, thermal, storage wear, reconnect, update/rollback |
| NVIDIA Jetson | Evaluation | JetPack matrix, GPU process isolation, power/thermal evidence |
| Windows/macOS device daemon | Not supported | OS service, secure key store, signing/notarization, update channel |
| Android/iOS/industrial controller | Not supported | Platform-specific runtime, policy, distribution, certification |

No physical-hardware E2E evidence was available for this implementation.

## Memory and privacy contract

Memory writes must name workspace, App, device, and—when applicable—subject
scope. Subject-linked events require consent purpose, retention, deletion, and
export behavior. Runtime logs must never contain device keys, enrollment
secrets, raw camera/microphone content, arbitrary payloads, or unredacted PII.
The simulator emits summary-only events and stores no person-linked data.

## Release blockers

1. Select and procure the first reference hardware and immutable OS image.
2. Implement the authenticated enrollment and fleet control-plane APIs.
3. Choose hardware-backed key storage and asymmetric command/update signing.
4. Build signed packages, checksums, SBOM, provenance, staged OTA, rollback,
   anti-rollback, and kill switch.
5. Complete adapter-specific physical safety, privacy, accessibility, and
   legal review.
6. Run board-in-loop, disconnect/reconnect, power-loss, soak, cloning,
   cross-tenant, malicious-update, and recovery tests.
7. Obtain fresh security and reliability approval before any live-device CTA
   changes from “pilot.”
