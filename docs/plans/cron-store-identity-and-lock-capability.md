# Cron Store Identity And Lock Capability

## Problem

Cron mutations currently open `cron/.jobs.lock` with `a+` and then perform
`jobs.json` and output I/O through mutable pathnames. A missing lock can be
created by ordinary acquisition, and a parent or Cron-directory replacement
can move later writes outside the directory identity that was locked.

Plugins that persist a pending decision during `pre_cron_job_persist` also
need to reuse the active Core lock. Mutable module booleans or a caller-shaped
object are not proof that the current thread still owns the real lock inode.

## Proposed Owner Contract

1. Provision `cron/.jobs.lock` only during explicit Core Cron-store setup.
   Ordinary reads and mutations open the existing regular, non-symlink lock
   without creating it.
2. Open and retain the full Profile/Cron directory identity before taking the
   lock. Keep `jobs.json` replacement and Job output create/prune/delete on
   that retained Cron directory through fsync and commit.
3. While the real lock and Cron-directory guard are held, Core may issue one
   opaque same-thread capability. The capability carries no Job or tool
   authority; it only proves the owner thread and exact Cron/lock device and
   inode identities.
4. Core alone validates the capability against its active owner state and
   invalidates it before unlocking. Plugins cannot construct, copy, or revive
   it from matching fields.
5. Missing, stale, replaced, symlinked, wrong-thread, or unverifiable state
   fails closed before any Job, output, Frame, or external Profile write.

## Compatibility And Tests

- Preserve the public Cron API and the existing `cron/.jobs.lock` location.
- Cover fresh explicit bootstrap, existing missing lock, parent/Cron/lock
  replacement, FIFO/symlink inputs, cross-process contention, stale and forged
  capabilities, wrong thread, release invalidation, pinned `jobs.json` atomic
  replacement, and pinned output writes/pruning.
- Keep the capability optional for plugins. Callers without the canonical
  Core object must acquire the existing lock independently or fail closed.

This proposal intentionally defines one ownership behavior rather than a new
scheduler, persistence layer, policy engine, or authorization model.
