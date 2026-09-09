# Discord Feature Parity & Alignment — current-state authority

> This is a source-of-truth reconciliation, not a completion claim.

## Snapshot

- Upstream main: `7b25941b0ecd1a2d367edc7b6ef89a0958c10822` at `2026-08-19T22:06:57Z`.
- Publication state rechecked at `2026-08-19T22:46:10Z`.
- Main Discord package: six files. `adapter.py` is 475,891 bytes.
- `tools/discord_api/` does not exist on main.
- No canonical capability row has terminal release evidence in this snapshot.

## Root finding

The campaign produced substantial candidate code and a locally green packet, but it did not preserve one executable semantic authority from the approved specification through publication and release. The packet's implementation map silently reassigned W-row meanings: canonical W1 is rejected native webhook administration, W3 is multiplex routing, W4 is proactive/home/cron delivery, and W5 is deferred/rejected OAuth. A packet that calls those IDs something else is a different contract, regardless of test count.

PR #90307 makes this class fail closed by digest-locking `(id, name, product_state)`, separating artifact evidence from delivery, and requiring one publication authority, runtime consumers, and terminal receipts.

## Corrected delivery state

- `candidate_open`: **0**
- `candidate_unwired`: **1** — M4
- `candidate_blocked`: **10** — M3, I4, I5, V2, V5, V6, A2, A5, W2, R3
- `gap`: **31**
- `on_main_unverified`: **0**
- `released`: **0**

The prior 19/16/7 tally treated closed packet-era PRs as live candidates. That was wrong. Closed candidate bytes remain provenance-bearing evidence, but they do not own current delivery state.

## Live publication authorities

| Row | Authority | State | Decisive gap |
|---|---:|---|---|
| M3 | #89405 | `candidate_blocked` | #89405 is the sole open M3 authority and #86419 is closed as superseded; merge remains sequenced behind the Discord extraction/consumer seam and exact-head acceptance. |
| M4 | #86324 | `candidate_unwired` | Open #86324 adds a typed embed module, but no production send/ingress consumer owns the projection. |
| I4 | #81388 | `candidate_blocked` | #81388 is the concrete live owner required by the canonical spec, but it must be retargeted/composed through the accepted component-authorization seam without growing forbidden god files. |
| I5 | #72742 | `candidate_blocked` | Only delayed-release cleanup is accepted; modal/free-form behavior remains contract-deferred, and #72742 still changes the forbidden Discord adapter surface. |
| V2 | #11359 | `candidate_blocked` | Retarget #11359 only after V1 establishes the native voice-message container contract; preserve #11358 credit and current main behavior. |
| V5 | #77998 | `candidate_blocked` | Arbitrate/compose #77998 with open #75078 while preserving TheSmokeDev and samuelBoucher credit; no encrypted unknown-SSRC frame may reach Opus first. |
| V6 | #78196 | `candidate_blocked` | Serialize open collision owners #78196/#78180 behind the required paired addendum; do not collapse distinct STT/TTS behaviors into one unreviewable train. |
| A2 | #86429 | `candidate_blocked` | Target guild/profile/requester authority remains unresolved; mutating permission-overwrite execution must fail closed until the product contract lands. |
| A5 | #86432 | `candidate_blocked` | Target guild/profile/requester authority and the model-callable execution boundary remain unresolved; request-builder tests alone are insufficient. |
| W2 | #70608 | `candidate_blocked` | Promote #70608 only after authenticated route/profile metadata, event-thread delivery, and profile-isolation acceptance compose with open #42439. |
| R3 | #87700 | `candidate_blocked` | #87700 is the current behavioral successor, but the full R3 paired addendum and sequencing with open #79651 recovery extraction remain unresolved. |

## Capability ledger

| ID | Canonical capability | Product state | Delivery state | Authority / decisive gap |
|---|---|---|---|---|
| M1 | Structured inbound message model — full doc alignment | `accepted` | `gap` | none: #86440 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| M2 | Agent-facing edit/delete | `accepted` | `gap` | none: #86449 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| M3 | Outbound reaction actions | `accepted` | `candidate_blocked` | #89405: #89405 is the sole open M3 authority and #86419 is closed as superseded; merge remains sequenced behind the Discord extraction/consumer seam and exact-head acceptance. |
| M4 | Rich embeds — typed outbound + ingress projection | `accepted` | `candidate_unwired` | #86324: Open #86324 adds a typed embed module, but no production send/ingress consumer owns the projection. |
| M5 | Poll read-projection | `accepted` | `gap` | none: #86451 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| M6 | Attachment contract — routing, preflight, bounded reads | `accepted` | `gap` | none: #86499 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| M7 | Streaming delivery correctness | `accepted` | `gap` | none: #86501 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| T1 | Thread lifecycle actions | `accepted` | `gap` | none: #86454 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| T2 | Thread session isolation + history | `accepted` | `gap` | none: #86503 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| T3 | Forum starter/tag/lifecycle | `accepted` | `gap` | none: #86458 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| T4 | Forum partial-delivery truth | `accepted` | `gap` | none: #86505 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| T5 | Thread permission correctness | `accepted` | `gap` | none: #86541 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| I1 | Command sync + registry parity | `accepted` | `gap` | none: #86550 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| I2 | Guild-scope + installation contexts | `accepted` | `gap` | none: #86475 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| I3 | Options, autocomplete, selected-value fidelity | `accepted` | `gap` | none: #86542 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| I4 | Component authorization seam | `accepted` | `candidate_blocked` | #81388: #81388 is the concrete live owner required by the canonical spec, but it must be retargeted/composed through the accepted component-authorization seam without growing forbidden god files. |
| I5 | Clarify lifecycle + UI + modal | `pair_gap` | `candidate_blocked` | #72742: Only delayed-release cleanup is accepted; modal/free-form behavior remains contract-deferred, and #72742 still changes the forbidden Discord adapter surface. |
| I6 | Interaction ACK + error discipline | `accepted` | `gap` | none: #86485 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| I7 | Sensitive-system-prompt privacy routing | `conditional` | `gap` | none: No implementation until an explicit sensitive-prompt privacy policy and paired confirmation define routing and fallback. |
| I8 | Deliverable approval | `pair_gap` | `gap` | none: Preserve #74471/#68789; choose one owner only after the approval contract and FILE-LIST gate are explicit. |
| I9 | Modals / context menus / cron buttons | `conditional` | `gap` | none: No modal/context-menu/cron-button implementation until a concrete consumer, callback state, authorization, timeout/restart, and accessibility contract exists. |
| V1 | Native voice-message container | `accepted` | `gap` | none: #86544 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| V2 | Waveform + duration | `existing` | `candidate_blocked` | #11359: Retarget #11359 only after V1 establishes the native voice-message container contract; preserve #11358 credit and current main behavior. |
| V3 | Pinned private receive transport seam | `accepted` | `gap` | none: No current implementation authority. |
| V4 | Hermes voice binding restoration | `accepted` | `gap` | none: No current implementation authority. |
| V5 | Unknown-SSRC encrypted-frame safety | `accepted` | `candidate_blocked` | #77998: Arbitrate/compose #77998 with open #75078 while preserving TheSmokeDev and samuelBoucher credit; no encrypted unknown-SSRC frame may reach Opus first. |
| V6 | STT/TTS reliability train | `pair_gap` | `candidate_blocked` | #78196: Serialize open collision owners #78196/#78180 behind the required paired addendum; do not collapse distinct STT/TTS behaviors into one unreviewable train. |
| A1 | Channel/category CRUD | `pair_gap` | `gap` | none: #86460 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| A2 | Permission overwrites | `pair_gap` | `candidate_blocked` | #86429: Target guild/profile/requester authority remains unresolved; mutating permission-overwrite execution must fail closed until the product contract lands. |
| A3 | Role CRUD + assignment | `pair_gap` | `gap` | none: #86462 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| A4 | Moderation primitives | `pair_gap` | `gap` | none: #86464 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| A5 | Scalar guild settings | `pair_gap` | `candidate_blocked` | #86432: Target guild/profile/requester authority and the model-callable execution boundary remain unresolved; request-builder tests alone are insufficient. |
| A6 | Audit retrieval + scheduled events | `pair_gap` | `gap` | none: #86466 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| W1 | Discord-native webhook operations | `rejected` | `gap` | none: Rejected absent a concrete Hermes consumer. A speculative tools/discord_api/webhooks.py is forbidden. |
| W2 | Generic Hermes webhook → Discord delivery | `pair_gap` | `candidate_blocked` | #70608: Promote #70608 only after authenticated route/profile metadata, event-thread delivery, and profile-isolation acceptance compose with open #42439. |
| W3 | Multiplex routing acceptance matrix | `accepted` | `gap` | none: #86545 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| W4 | Proactive/home/cron delivery | `accepted` | `gap` | none: #86487 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| W5 | OAuth2 authorization-code flow | `rejected` | `gap` | none: Paired-deferred/rejected-not-in-contract. No OAuth PR without a new product/security decision and paired acceptance. |
| R1 | Route-aware rate-limit contract | `existing` | `gap` | none: #86468 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| R2 | REST route + pagination conformance | `pair_gap` | `gap` | none: #86437 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |
| R3 | Recovery + reconnect correctness | `pair_gap` | `candidate_blocked` | #87700: #87700 is the current behavioral successor, but the full R3 paired addendum and sequencing with open #79651 recovery extraction remain unresolved. |
| R4 | Local reliability telemetry | `conditional` | `gap` | none: #86442 is closed and its declared candidate paths are absent from the pinned current-main snapshot; the bytes remain historical evidence, not an active delivery route. |

## Cross-cutting gaps

1. **Semantic authority:** row IDs were mutable prose labels rather than immutable contract identities. The W lane proves the failure.
2. **Publication authority:** most packet-era feature PRs are closed; only eleven current row authorities remain open, including promoted successors #81388 and #87700.
3. **Consumer authority:** of the live row authorities, only #89405, #81388, and #87700 name concrete runtime consumers; the rest are blocked or isolated request-builder/module candidates.
4. **Architecture:** `plugins/platforms/discord/adapter.py` remains a 475,891-byte monolith while #79650–#79654 are open. New feature work cannot be release-ready while its stable consumer seam is unsettled.
5. **Product gates:** I5/I7/I8/I9, V6, A1–A6, W1/W2/W5, and R2–R4 retain explicit decisions or paired-addendum gates that packets cannot bypass.
6. **Release proof:** there is no single integration SHA, exact-head full CI, live sandbox matrix, two independent reviews, merge receipt, and current-main re-verification for any row.

## Topology decisions encoded here

- M3 authority is #89405. #86419 is closed as superseded with provenance preserved.
- I4 authority is promoted to open #81388; closed #86543 is historical evidence.
- R3 authority is promoted to open #87700; closed #86547 is historical evidence and open #79651 remains the extraction dependency.
- W1 remains rejected and `tools/discord_api/webhooks.py` remains forbidden absent a concrete consumer.
- W3 remains multiplex profile routing; W4 remains proactive/home/cron delivery; W5 remains paired-deferred/rejected OAuth.
- A1–A6 remain blocked until target guild/profile/requester authority is explicit. Only A2 and A5 currently retain open candidate PRs.
- Packet files and green packet tests remain `artifact_evidence`; they do not advance `delivery_state`.

## Terminal condition

A row advances to `released` only with an exact merged commit, head-bound CI, live receipt when required, and two independent exact-head approvals. The campaign closes only after all 42 rows are released, intentionally rejected/deferred, or explicitly superseded with zero orphan publication and credit edges.
