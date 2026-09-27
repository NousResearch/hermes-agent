# Bundle dependency caches

`setup-pm` restores uv's real cache, not the installed virtual environment.
Its fallback keys retain the native target, OS version, and Python version.
A metadata or dependency change can reuse compatible wheels;
uv still resolves and installs from the frozen project lock.

Bundle jobs set `save-python-cache: false` and call `save-pm-cache` after their
build step with the `python-path`, `uv-cache-path`, and `python-cache-key` outputs.
Both the caller and this composite use `!cancelled()` so a failed build does
not suppress the save. Cancellation is excluded to avoid racing a dying uv
process. A runner crash or job timeout can still prevent the save.

Snapshots use a run ID, attempt, and producer job suffix because Actions caches
are immutable. Sibling bundle workflows in one caller run must not race to save
different contents under the same key.
Restore tries the current dependency set's rolling snapshots first, then the
compatible v2 prefix. There is no fallback to older cache formats. This lets
a retry add wheels to a snapshot saved by a partially failed build. The ordinary
automatic cache path remains available.

Suffixed namespaces precede the native/dependency identity, outside production's
restore prefix.

Before saving, `python -m pm.build_env --exact-lock --cache PATH --lock-source REPO`
deletes every entry the project's `uv.lock` cannot resolve. It keeps downloaded
wheels the lock resolves (bundles copy the full cache for offline dependency
installation) — `uv cache prune --ci` would discard them. Pruning happens after
payload staging and packaging, and it does not change the staged payload.
The same lock-exactness contract governs the bundle ship gate
(`stage_uv_cache`) and CI's rolling snapshots, so no snapshot accumulates
sediment for superseded pins.

This is not a lockfile-aware or size-bounded cache. Old, still-referenced package
versions can remain inside a snapshot and be copied forward. GitHub evicts whole
cache snapshots under its retention and repository quota policies; that does
not remove old versions inside the newest snapshot. A future size policy must
account for the offline payload too, rather than deleting uv internals or
promising that `prune` keeps only the current lock.
