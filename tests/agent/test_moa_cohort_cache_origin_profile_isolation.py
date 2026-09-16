"""Regression coverage: MoA cohort/slot pinning caches must be scoped by origin profile
(Hermes home), not by ``execution_id`` alone (plans/2026-09-15_141016-guided-model-routing.md
§6 "MoA"; root AGENTS.md named bug class: "module globals... hold the launch profile's
state... a silent default-profile leak").

``resolve_moa_cohort_pinned``/``resolve_moa_slot_route_pinned`` cache in a bare module-level
dict. Two DIFFERENT origin profiles that happen to mint the SAME ``execution_id`` (plausible:
a multiplexed host process serving several Hermes homes concurrently, or simply two profiles
racing a UUID collision) must never share, evict, or leak into each other's pinned
cohort/receipt/endpoint -- covered here as an A -> B -> A sequence under one literal
``execution_id`` across two real, independently-published policies/homes.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _policy(route_id: str, endpoint: str, revision: int = 1) -> dict:
    return {
        "schema_version": 1, "policy_id": "kanban-default", "revision": revision,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": route_id, "route_revision": 1, "provider": "custom",
            "model": f"model-for-{route_id}", "endpoint": endpoint, "maker": f"maker-{route_id}",
            "model_family": "test-model", "status": "approved",
            "allowed_roles": ["moareference", "moaaggregator"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {
            "moareference": {"deep": [route_id], "shallow": [route_id]},
            "moaaggregator": {"deep": [route_id], "shallow": [route_id]},
        },
    }


@pytest.fixture()
def two_profiles():
    """Two independent, real Hermes homes -- distinct on-disk profiles, each with its own
    published+activated policy pinning a DIFFERENT route/endpoint, so a cross-profile cache
    leak is trivially observable (profile B would see profile A's route/endpoint)."""
    from agent.model_selection_store import activate_policy, publish_policy

    homes = {}
    for name, route_id in (("A", "route-A"), ("B", "route-B")):
        test_home = tempfile.mkdtemp(prefix=f"hermes_moa_profile_{name}_")
        hermes_home = os.path.join(test_home, ".hermes")
        os.makedirs(hermes_home)
        endpoint = f"http://127.0.0.1:9/{name.lower()}"
        policy = _policy(route_id, endpoint)
        record = publish_policy(hermes_home, policy, approval_ref="operator:test")
        activate_policy(hermes_home, "kanban-default", record["revision"])
        homes[name] = {"hermes_home": hermes_home, "test_home": test_home,
                       "route_id": route_id, "endpoint": endpoint}
    try:
        yield homes
    finally:
        for info in homes.values():
            shutil.rmtree(info["test_home"], ignore_errors=True)


def _slot(role: str) -> dict:
    return {"provider": "does-not-matter", "model": "does-not-matter", "routing_role": role}


def test_same_execution_id_across_profiles_never_shares_cohort(two_profiles):
    """A -> B -> A under the IDENTICAL literal execution_id: B's resolution must never be A's,
    and A's SECOND resolution (after B ran) must still be A's original pinned resolution --
    ending B's cohort work must never evict or corrupt A's pinned entry."""
    from agent.moa_model_routing import _forget_cohort, resolve_moa_cohort_pinned

    same_execution_id = "exec-collision-0001"
    reference_slots = [_slot("moareference")]
    aggregator = _slot("moaaggregator")

    home_a = two_profiles["A"]["hermes_home"]
    home_b = two_profiles["B"]["hermes_home"]
    try:
        resolved_a1 = resolve_moa_cohort_pinned(
            reference_slots, aggregator, execution_id=same_execution_id, hermes_home=home_a,
        )
        resolved_b = resolve_moa_cohort_pinned(
            reference_slots, aggregator, execution_id=same_execution_id, hermes_home=home_b,
        )
        resolved_a2 = resolve_moa_cohort_pinned(
            reference_slots, aggregator, execution_id=same_execution_id, hermes_home=home_a,
        )

        assert resolved_a1["aggregator"]["provider"] == "custom"
        assert resolved_a1["aggregator"]["model"] == "model-for-route-A"
        assert resolved_a1["aggregator"]["endpoint"] == two_profiles["A"]["endpoint"]

        assert resolved_b["aggregator"]["model"] == "model-for-route-B"
        assert resolved_b["aggregator"]["endpoint"] == two_profiles["B"]["endpoint"]
        # Receipt ids are content-addressed from (execution key, policy_id, revision) --
        # deterministically identical here since both profiles publish revision=1 under the
        # same policy_id; the real isolation guarantee is that each id is stored in and
        # resolved from its OWN profile's on-disk store, never a shared one (proven by the
        # model/endpoint assertions above resolving to each profile's own route).

        # Ending B's cohort work must never evict or mutate A's pinned entry: the SAME
        # execution_id resolved again under home_a is byte-identical to the first A resolution,
        # not re-resolved and not silently swapped to B's route.
        assert resolved_a2["aggregator"]["receipt_id"] == resolved_a1["aggregator"]["receipt_id"], (
            "profile A's pinned cohort must survive an intervening resolution for the SAME "
            "execution_id under profile B -- B ending its work must never evict A's entry"
        )
        assert resolved_a2["aggregator"]["model"] == "model-for-route-A"
    finally:
        _forget_cohort(same_execution_id, hermes_home=home_a)
        _forget_cohort(same_execution_id, hermes_home=home_b)


def test_same_execution_id_across_profiles_slot_cache_isolated(two_profiles):
    """The per-slot pinned cache (``resolve_moa_slot_route_pinned``) must be isolated the same
    way as the cohort cache: a colliding ``(execution_id, slot_id)`` across two profiles must
    resolve/return each profile's own route, never cross-contaminate."""
    from agent.moa_model_routing import resolve_moa_slot_route_pinned

    same_execution_id = "exec-collision-0002"
    home_a = two_profiles["A"]["hermes_home"]
    home_b = two_profiles["B"]["hermes_home"]

    slot_a = resolve_moa_slot_route_pinned(
        _slot("moareference"), execution_id=same_execution_id, slot_id="reference-0",
        hermes_home=home_a,
    )
    slot_b = resolve_moa_slot_route_pinned(
        _slot("moareference"), execution_id=same_execution_id, slot_id="reference-0",
        hermes_home=home_b,
    )
    slot_a_again = resolve_moa_slot_route_pinned(
        _slot("moareference"), execution_id=same_execution_id, slot_id="reference-0",
        hermes_home=home_a,
    )

    assert slot_a["model"] == "model-for-route-A"
    assert slot_b["model"] == "model-for-route-B"
    assert slot_a_again["receipt_id"] == slot_a["receipt_id"], (
        "profile A's pinned per-slot entry must survive an intervening resolution for the "
        "SAME (execution_id, slot_id) under profile B"
    )


def test_concurrent_profiles_same_execution_id_no_race(two_profiles):
    """Thread-safety: many concurrent resolutions for the SAME colliding execution_id, split
    across the two profiles, must never corrupt either profile's cache entry (no exception, no
    cross-assignment) -- the lock in ``agent.moa_model_routing`` must key on the FULL
    ``(execution_id, origin_home)`` tuple, not merely serialize access to a single
    execution_id-keyed slot that both profiles would share."""
    from agent.moa_model_routing import _forget_cohort, resolve_moa_cohort_pinned

    same_execution_id = "exec-collision-0003"
    reference_slots = [_slot("moareference")]
    aggregator = _slot("moaaggregator")
    home_a = two_profiles["A"]["hermes_home"]
    home_b = two_profiles["B"]["hermes_home"]

    results: dict = {"A": [], "B": []}
    errors: list = []

    def _run(label, home):
        try:
            for _ in range(20):
                resolved = resolve_moa_cohort_pinned(
                    reference_slots, aggregator, execution_id=same_execution_id, hermes_home=home,
                )
                results[label].append(resolved["aggregator"]["model"])
        except Exception as exc:  # pragma: no cover - failure path surfaced via errors list
            errors.append(exc)

    try:
        threads = [
            threading.Thread(target=_run, args=("A", home_a)),
            threading.Thread(target=_run, args=("B", home_b)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"concurrent resolution under a colliding execution_id raised: {errors}"
        assert set(results["A"]) == {"model-for-route-A"}, (
            "profile A must never observe profile B's model under concurrent access to a "
            "colliding execution_id"
        )
        assert set(results["B"]) == {"model-for-route-B"}, (
            "profile B must never observe profile A's model under concurrent access to a "
            "colliding execution_id"
        )
    finally:
        _forget_cohort(same_execution_id, hermes_home=home_a)
        _forget_cohort(same_execution_id, hermes_home=home_b)


def test_forget_cohort_evicts_slot_cache_entries(two_profiles):
    """Turn-end cleanup (``agent.agent_runtime_helpers.note_turn_persisted``) calls
    ``_forget_cohort(turn_id)`` exactly once at the end of every turn. It must evict not
    only the joint cohort cache entry but every per-slot entry it seeded
    (``resolve_moa_slot_route_pinned``'s ``__slots__`` backing dict) -- otherwise every
    finished MoA run leaves its slot resolutions permanently cached (a real per-turn memory
    leak, since a turn's execution_id is never reused)."""
    from agent.moa_model_routing import _cohort_cache, _forget_cohort, resolve_moa_cohort_pinned

    execution_id = "exec-turn-end-cleanup"
    reference_slots = [_slot("moareference")]
    aggregator = _slot("moaaggregator")
    home_a = two_profiles["A"]["hermes_home"]

    resolve_moa_cohort_pinned(
        reference_slots, aggregator, execution_id=execution_id, hermes_home=home_a,
    )
    home_key = os.path.realpath(home_a)
    per_slot = _cohort_cache.get("__slots__", {})
    seeded_keys = [k for k in per_slot if k[0] == execution_id and k[2] == home_key]
    assert seeded_keys, "cohort resolution must seed the per-slot cache"

    _forget_cohort(execution_id, hermes_home=home_a)

    assert (execution_id, home_key) not in _cohort_cache
    per_slot_after = _cohort_cache.get("__slots__", {})
    assert not [k for k in per_slot_after if k[0] == execution_id and k[2] == home_key], (
        "_forget_cohort must evict every per-slot entry it seeded for this "
        "(execution_id, origin_home), not just the joint cohort entry"
    )


def test_forget_cohort_scoped_slot_eviction_does_not_touch_other_execution(two_profiles):
    """Ending one turn's (execution_id A) cohort must never evict a DIFFERENT still-live
    turn's (execution_id B) per-slot cache entries under the same profile."""
    from agent.moa_model_routing import _cohort_cache, _forget_cohort, resolve_moa_cohort_pinned

    reference_slots = [_slot("moareference")]
    aggregator = _slot("moaaggregator")
    home_a = two_profiles["A"]["hermes_home"]
    home_key = os.path.realpath(home_a)

    resolve_moa_cohort_pinned(
        reference_slots, aggregator, execution_id="turn-ending", hermes_home=home_a,
    )
    resolve_moa_cohort_pinned(
        reference_slots, aggregator, execution_id="turn-still-live", hermes_home=home_a,
    )

    try:
        _forget_cohort("turn-ending", hermes_home=home_a)

        per_slot = _cohort_cache.get("__slots__", {})
        assert [k for k in per_slot if k[0] == "turn-still-live" and k[2] == home_key], (
            "a different, still-live turn's per-slot cache entries must survive "
            "another turn's end-of-turn cleanup"
        )
        assert not [k for k in per_slot if k[0] == "turn-ending" and k[2] == home_key]
    finally:
        _forget_cohort("turn-still-live", hermes_home=home_a)

