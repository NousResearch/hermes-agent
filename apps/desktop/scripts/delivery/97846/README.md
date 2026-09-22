# #97846 dependent Desktop cancellation correction

This directory keeps the #97846-owned correction for Desktop Group Chat cancellation when the affected queue/runtime context is supplied by the already-public composition inputs.

## Dependency boundary

`desktop-cancellation-dependent-overlay.patch` is an exact six-path contextual Git patch. Its preimage is the runtime/lower composition **before any Files changes**:

- preimage tree: `e76b349554ed86a02f2470c31d3912fcd9b5b332`
- corrected lower-only tree: `02ef3e6596c87239d6b1076d11b850022e7a1ab4`
- patch SHA-256: `8ad9da6a90a4de55c07b8d7e15054aa67db3c98db3a107dd9374d771550cbb52`
- patch size: 34,185 bytes, 752 lines

Apply with `git apply --index` after composing the pinned runtime/lower inputs and **before** the Files owner/dependent overlays. The executable recipe in `apps/desktop/scripts/delivery/104199/compose.sh` verifies the preimage, digest and corrected lower tree. There is no production dependency on Files: this patch includes only the dependent cancellation/identity behavior and its regressions, not a runtime-owned whole source body. Three lines of context anchor test hunks instead of relying on offsets from a different composition.

## Owned behavior

- Bind an in-flight queue item to its exact command fence and thread.
- Remove a lease-lost pending item without mutating an unrelated active room epoch, turn, or session.
- On active lease loss, interrupt only the session scoped to that active item’s thread while preserving later local work.
- Make explicit Stop derive its target from the live queue item, not `latestActivity.thread`; retain the bare member-key fallback only for rooms with no thread-scoped session keys.
- Reject retired queue bindings while preserving identity across a room rename.
- Register existing-message recovery in that same queue so recovered work has the same active command/thread/session ownership as fresh work.
- Follow in-flight commands by stable room identity across rename, rechecking descriptor authority and permanently rejecting removed/replaced bindings. Resolve cancellation, settlement and persistence against the current name; dispose the binding when execution settles.

The patch also carries focused mailbox, Stop, session-migration, rename, retirement, and view-authority regressions for these behaviors. The new recovery tests abandon one lease, reclaim the same durable message, then exercise both lease loss and UI Stop against the exact recovered backend session. Rename tests exercise lease loss, UI Stop, mailbox Stop and successful settlement.

## Lower-first verification

Before Files was added, the corrected lower tree above passed 15 focused causal tests (99 excluded by name filter), all three Desktop TypeScript projects, and six-path ESLint with zero warnings. Reproduce that intermediate state with `DESKTOP_FILES_VERIFY=lower` using the existing #104199 recipe; `focused` composes Files afterward and runs the changed-path checks plus renderer build. The recipe README records exact commands and scope. Native desktop acceptance is separate.
