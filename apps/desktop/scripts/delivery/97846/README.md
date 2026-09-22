# #97846 dependent Desktop cancellation correction

This directory keeps the #97846-owned correction for Desktop Group Chat cancellation when the affected queue/runtime context is supplied by the already-public composition inputs.

## Dependency boundary

`desktop-cancellation-dependent-overlay.patch` is an exact five-path Git patch whose required preimage is the existing #104199 composed production tree:

- preimage tree: `19b08e1c6be010cbc05716bc24bb4de163313d71`
- patch SHA-256: `ac5ca70a360d9059d2918c9a9561a60c5091c6e7640026f3c0e79a59839cb39a`
- patch size: 21,162 bytes, 409 lines

The patch must be applied with `git apply --unidiff-zero --index` only after #104199 has merged the pinned public runtime/lower inputs and applied the pinned Files owner/dependent overlays. The #104199 composition recipe verifies both the preimage tree and this patch digest before applying it. This file contains only the dependent behavioral and regression hunks; it does not copy a runtime-owned whole source body.

## Owned behavior

- Bind an in-flight queue item to its exact command fence and thread.
- Remove a lease-lost pending item without mutating an unrelated active room epoch, turn, or session.
- On active lease loss, interrupt only the session scoped to that active item’s thread while preserving later local work.
- Make explicit Stop derive its target from the live queue item, not `latestActivity.thread`; retain the bare member-key fallback only for rooms with no thread-scoped session keys.
- Reject retired queue bindings while preserving identity across a room rename.

The patch also carries focused mailbox, Stop, session-migration, rename, retirement, and view-authority regressions for these behaviors.
