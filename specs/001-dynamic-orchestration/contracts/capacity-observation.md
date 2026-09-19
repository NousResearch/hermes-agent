# Contract: Capacity Observation and View v1

Status: PROPOSED.

`CapacityObservation{observation_id,route_id?,quota_pool_id,billing_pool_id?,metric,value?,unit?,source,captured_at,valid_until,freshness,confidence,is_estimated,plan?,entitlement?,reset_at?,price?,max_concurrency?,reason_code?}`. Identity/provenance are mandatory; other values may be unknown; null never means zero/unlimited.

`CapacityView{view_id,built_at,pools:sorted tuple[PoolCapacity],source_observation_ids[]}`.

OBSERVED: CredentialPool is mutable/persistent (`agent/credential_pool.py:582-614`, `632-637`) and availability may mutate (`1567-1696`). PROPOSED `snapshot_for_capacity()` acquires the pool lock and copies only bounded secret-free state. Its sentinel scope is absolute: it MUST NOT call `select()`, `_available_entries(refresh=True)`, `_available_entries(clear_expired=True)`, `_persist()`, OAuth, or network paths. A no-argument, read-only `_available_entries()` or `has_available()` MAY be used only while holding the pool lock after its purity (no refresh, expiry clearing, persistence, OAuth, network, or mutation) is demonstrated by focused tests; otherwise snapshotting reads copied state directly. Sentinel tests prove both the prohibited calls and the permitted-read purity boundary.

Reservations are distinct leases keyed to route/quota/billing pool. `derived_remaining` is derived only when a fresh, unit-compatible `CapacityObservation` has `metric=remaining`: `derived_remaining=CapacityObservation.value`; for a missing, stale, incompatible, or other metric it is `unknown`. Compatibility-only arithmetic is `dispatchable=derived_remaining-active_reservations-protected_reserve-safety_margin`. Breakers scope to route/pool, not generic model/provider. Same model/different pool stays separate; stale/unknown never becomes healthy; snapshots never contain secret/raw provider response.
