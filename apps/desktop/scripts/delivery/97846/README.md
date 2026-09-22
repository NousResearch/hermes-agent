# #97846 dependent Desktop cancellation correction

This directory keeps the #97846-owned correction for Desktop Group Chat cancellation when the affected queue/runtime context is supplied by the already-public composition inputs.

## Dependency boundary

`desktop-cancellation-dependent-overlay.patch` is an exact six-path contextual Git patch. Its preimage is the runtime/lower composition **before any Files changes**:

- preimage tree: `e76b349554ed86a02f2470c31d3912fcd9b5b332`
- corrected lower-only tree: `12f75525101c8b0e533fa204914ec53d1435dd71`
- patch SHA-256: `59f066c0141cd3faa0a5ce3b35107d35820a79e5c1976d12bf1a3f2994f4c713`
- patch size: 47,702 bytes, 1,079 lines

Apply with `git apply --index` after composing the pinned runtime/lower inputs and **before** the Files owner/dependent overlays. The executable recipe in `apps/desktop/scripts/delivery/104199/compose.sh` verifies the preimage, digest and corrected lower tree. There is no production dependency on Files: this patch includes only the dependent cancellation/identity behavior and its regressions, not a runtime-owned whole source body. Three lines of context anchor test hunks instead of relying on offsets from a different composition.

## Owned behavior

- Bind an in-flight queue item to its exact command fence and thread.
- Remove a lease-lost pending item without mutating an unrelated active room epoch, turn, or session.
- On active lease loss, interrupt only the session scoped to that active item’s thread while preserving later local work.
- Make explicit Stop derive its target from the live queue item, not `latestActivity.thread`; retain the bare member-key fallback only for rooms with no thread-scoped session keys.
- Reject retired queue bindings while preserving identity across a room rename.
- Register existing-message recovery in that same queue so recovered work has the same active command/thread/session ownership as fresh work.
- Keep a reclaimed command pending while its cancelled predecessor unwinds, independent of the room's presentation `running` flag. Remove its exact pending item before releasing its fence; discard expired items before activation, rebind surviving items to the current epoch, and clear running state when the active item exits early.
- Keep local-reply and mailbox-claim queue entries distinct even on the same thread: local replies coalesce by thread, mailbox work only by fence generation. Lease loss removes only its claimant; explicit Stop cancels every queued claimant, without losing a local reply or allowing an erased claimant to requeue.
- Follow in-flight commands by stable room identity across rename, rechecking descriptor authority and permanently rejecting removed/replaced bindings. Resolve cancellation, settlement and persistence against the current name; dispose the binding when execution settles.

The patch also carries focused mailbox, Stop, session-migration, rename, retirement, and view-authority regressions for these behaviors. Recovery tests hold the cancelled backend across a new claim, check later activation, abort-driven removal and silent lease expiry, preserve unrelated queued work, and prove a fresh claim remains usable. A separate recovery case starts while an unrelated thread is active. Active recovery also exercises lease loss and UI Stop against the exact recovered backend session. Rename tests exercise lease loss, UI Stop, mailbox Stop and successful settlement.

## Lower-first verification

The corrected lower-only state passed 21 focused cases (99 excluded by name filter), all three Desktop TypeScript projects and changed-source ESLint with zero warnings. This includes two same-thread owner-collision regressions that failed on the previous representation, four queue-overlap cases and the prior recovery/rename/isolation controls. Reproduce with `DESKTOP_FILES_VERIFY=lower` using the existing #104199 recipe. `focused` composes Files afterward and runs the same affected-path checks plus renderer build. The recipe README records executed results and scope. Independent source/composition review accepted the owner-distinct correction at lower tree `12f75525101c8b0e533fa204914ec53d1435dd71` and composed tree `f86ea20723955476e1d94297fe0a334ab9199dba`, including both same-thread enqueue orders, recovery/rename continuity and exact Stop/lease-loss ownership. This is not native Desktop or hosted-CI acceptance.
