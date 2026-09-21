# Hosted Output current-runtime cutover

A Bot can explicitly hand off a file to the Group without exposing a local path. The retained task freezes its recipients; Home verifies the bytes in Files before publishing a member message and acknowledging exactly that source output.

This is an additive local #99159 owner checkpoint against runtime `485d5f6848c2598ca00fa7704e50e8ae66d0983a`, **not a standalone runnable branch or complete replacement of the historical PR**. Classic exports (#104198) remain separate.

## Required composition

- Runtime #106742: `485d5f6848c2598ca00fa7704e50e8ae66d0983a`.
- Q quarantine: `43b9f02183444de0d48dc72170341346d58e09dc`.
- Replay #99960: `59a10872c92c8f9b9529e89b67505558523a0af3`.
- Core admission contract: `2fb90a347b5a0d7864367b221544dc058c6cf70c`.
- Route #100016: `64e68ccd2be3323497b5c9137a3e0fa89da1d80d`.
- Input custody #111362: `e671cb464b6e7c0d740855f2eabe1fd6bbe87eda`.
- Files #98072: `b8a10e38a8687280f0863cee56d37553abcdeb5f`; accepted minimal proof `ae7a62eb17ef827af80dde989712aee3a52a5dc9`.

Output consumes their real canonical owner registry, shared-first grant fence, current transaction connection, private input reconstruction and byte-verifying Files store. It does not carry those implementations. Output readiness is backed by the initialized real outbox and installed handler identities, not catalog metadata or detached legacy fallback.

## Separately owned transactional admission

Route owns the NEW-write seam in `gateway/session_api_turn.py` and
`gateway/session_peer_target.py`; the exact separately owned candidate is recorded
in the local seam handoff. Compose that Route delta with the lower/Files owners
and this Output owner using ordinary Git merges. Retain both sides of the shared
registration/dispatch/append contexts. No dependent patch is applied.

At HTTP consent capture, Output pins the concrete adapter-bound
`_room_output_admission` consumer in immutable transient evidence. API receives
that exact consumer through the server-only `_room_output_authorizer` argument,
not a new lookup that could accept replacement between capture and API entry.
Inside the existing shared-first/owner SQL
fence, `authorize_dispatch` verifies current dispatch, policy, catalog and both
grant stores, then requires that same consumer before and after its read-only
callback for signed artifact rights.
Missing, rebound, unavailable or incomplete Output consent refuses NEW instead
of degrading to text. Default four/five-right input grants require no consumer;
accepted replay bypasses the NEW check and creates no execution.

Output installs `authorize_output_consent` on that seam during real root setup.
Its readiness selection requires the exact installed consumer, initialized
outbox, owner/DB/epoch/process and real routes. The consumer uses the supplied
connections to recapture and compare exact adapter/owner/grant/policy/dispatch
consent, returning literal `True` only for authorized Output. It must not commit,
reacquire grant stores or do external work. The bearer remains transient.

The former `patches/hosted-output-route-admission.patch` packaging was removed
only after the ordinary owner composition passed the focused admission and real
root/default peer Output product witnesses. This closes that representation
hold only, not the receiver/lifecycle/publication holds below.

The proof also restores the exact Files-owned native staging cleanup test including Output setup and byte/row noninterference assertions. Its one prompt fixture adaptation uses the already-accepted composed expectation for NEW prompts and leaves the old frozen admission payload proof unchanged.

## Preserved and verified locally

- Actual root-local and canonical root/default peer explicit producer and bounded manifests; canonical retained Run/result and history/driver projection.
- Frozen recipients, Files byte verification before visibility, exact event/author/task/manifest/recipient/byte proof before ACK, lost-response replay and positive retirement evidence.
- Process/owner/epoch/admission/generation/grant/Run/result fences; absent output rights do not become authority.
- Original private outbox schema upgrades, quotas, indexed retention, unlink-retry obligations, generation trigger protections and stale-generation refusal. Named-profile *shared quota accounting* is a store regression, not proof of named-profile producer reachability.
- Existing root/peer document and PNG input, partial batches, frozen prompt payloads and native revoked-staging noninterference.

The canonical runner uses explicit safe selections, two workers, zero retries, a pinned existing interpreter and private command-scoped paths/CPU affinity. Fixtures use real temporary storage and direct drains with inert handlers/socket boundaries. No real executor/model/listener/coordinator/service is started.

## Withheld behavior and remaining gates

The historical standalone/named/managed producer and separate transcript receiver/bootstrap path is not established by the current runtime constructor chain. No readiness fallback is added. The original legacy service mixin's durable retry queue/backoff/unblocking, old terminal-generation upgrade replay, Stop/disband retirement orchestration, peer discard HTTP route and named/managed bridge are not imported or certified. These are real historical workflows, not declared obsolete. Existing bounded outbox cleanup remains present, but it does not certify those missing lifecycle callers.

Historical runtime/service/publication/recovery/API regression suites for those paths remain withheld. The original macOS-specific alias and shell-backed reader selections also remain outside this bounded Linux/inert gate. Parent source acceptance, full receiver parity, later lifecycle/Stop/fault/native/HIL gates and public-history replacement remain open. No force push, public PR update, main merge, deployment or live Barry change is included.

## Provenance

Selected Output hunks are taken from accepted composed endpoint `89535feffcecfce5678725a1b9a211f47f2d6251`, traced to root `9f205c357715dd34980c37f4dd1ce4a6b05b17cf` / `2c8ac08d53390efd6dd503168d1be3c6e7934d2c`, peer `4bc8dd598ed9704e503a409283c1d38ca06b3380`, and original #99159 `4fb3f28c1304e696d92b2d2c954cd361c3b1fbc1`. No whole mixed postimages or donor histories are merged into this owner. Original messages, authors/dates, hunk ledger and fresh logs/XML are retained in the external `PROTON_OUTPUT_*` evidence package.

These independent repositories have independent refs/indexes but borrow immutable local object stores and shallow boundaries. Preserve all donors; this is not a self-contained portable clone.
