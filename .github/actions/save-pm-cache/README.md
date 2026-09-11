# Bundle dependency caches

`setup-pm` restores uv's real cache, not the installed virtual environment.
Its fallback keys retain the native target, OS version, Python version, and
pruning mode. A metadata or dependency change can reuse compatible wheels;
uv still resolves and installs from the frozen project lock.

Bundle jobs set `save-python-cache: false` and call `save-pm-cache` after their
build step with the `uv-path`, `uv-cache-path`, and `python-cache-key` outputs.
Both the caller and this composite use `!cancelled()` so a failed build does
not suppress the save. Cancellation is excluded to avoid racing a dying uv
process. A runner crash or job timeout can still prevent the save.

Snapshots use a run ID, attempt, and producer job suffix because Actions caches
are immutable. Sibling bundle workflows in one caller run must not race to save
different contents under the same key.
Restore tries the current dependency set's rolling snapshots first, then the
compatible v2 prefix. There is no fallback to older cache formats. This lets
a retry add wheels to a snapshot saved by a partially failed build. The ordinary
automatic cache path and its exact-hit smoke tests remain available.

Smoke namespaces precede the native/dependency identity, outside production's
restore prefix. PM Toolchain cleanup removes its run-scoped snapshots.

Before saving, `uv cache prune` removes dangling entries. It does not use
`--ci`: that option discards downloaded wheels, while bundles copy the full
cache for offline dependency installation. Pruning happens after payload staging
and packaging, and it does not change the staged payload.

This is not a lockfile-aware or size-bounded cache. Old, still-referenced package
versions can remain inside a snapshot and be copied forward. GitHub evicts whole
cache snapshots under its retention and repository quota policies; that does
not remove old versions inside the newest snapshot. A future size policy must
account for the offline payload too, rather than deleting uv internals or
promising that `prune` keeps only the current lock.
