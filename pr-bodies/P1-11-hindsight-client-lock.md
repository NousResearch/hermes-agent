## Summary

Fix a thread-safety bug in the Hindsight memory plugin: `_get_client()` used an unprotected check-then-act pattern that allowed two concurrent threads to race and create duplicate clients, wasting resources and potentially causing runtime errors.

## Bug

`_get_client()` at `plugins/memory/hindsight/__init__.py:1012` checks `if self._client is None` and then creates the client without holding a lock. In concurrent scenarios (gateway serving multiple sessions), two threads could both see `_client is None`, both create a new client, and one client's resources would leak.

## Fix

Added a `_client_lock` (threading.Lock) to `HindsightMemoryProvider.__init__()` and a double-checked locking pattern in `_get_client()`:

1. Fast path: if `_client is not None`, return it (no lock needed)
2. Acquire `_client_lock`
3. Double-check: if `_client is not None` (another thread finished creating), return it
4. Create client while holding the lock

## Impact

- **Severity**: P1 — resource leak in concurrent gateway deployments
- **Scope**: Hindsight memory plugin (`plugins/memory/hindsight/__init__.py`)
- **Risk**: Minimal — lock is per-provider instance, contention is near-zero (client created once)

## Testing

- `tests/plugins/memory/test_hindsight_client_lock.py`: 3 regression tests
  - `test_concurrent_clients_created_once`: Two threads racing `_get_client` create only one client
  - `test_second_call_returns_cached`: After creation, subsequent calls return cached client
  - `test_lock_acquired_during_creation`: Lock is held during client creation
