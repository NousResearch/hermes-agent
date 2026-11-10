"""Contract suite for the profile_identity capability.

Every case runs a real assembled backend, seeds a profile's identity through the public sessions, routing, bindings and maintenance operations, and observes through public reads. The behaviour pinned for ``rekey_profile`` / ``purge_profile``: names stripped, empty or equal names a no-op with empty counts, collisions refused before any write, one count per facet the store has, the API binding and local creation receipts moved with the name.
"""

from __future__ import annotations

import dataclasses
import json
import time

import pytest

from persistence import Conflict, ProfileScope, SessionId
from persistence.authority.bindings import NewApiBinding, NewSessionBinding
from persistence.conversation.sessions import NewSession
from persistence.housekeeping.profile_identity import ProfileIdentityCollision, ProfilePurgeCounts, ProfileRekeyCounts, profile_namespace
from tests.contracts.authority.test_bindings_api_contract import STAMP, STARTED
from tests.contracts.authority.test_bindings_contract import begin

NS = "telegram"
NS_ALT = "discord"

OLD, NEW, BYSTANDER = "alpha", "beta", "alpha_x"


def _entry(key: str, session_id: str, profile: str | None = None) -> str:
    return json.dumps({"session_key": key, "session_id": session_id, "origin": {"profile": profile, "platform": NS} if profile else None})


def seed_profile(backend, scope: ProfileScope, profile: str, tag: str) -> None:
    """One session (owner tag and origin naming the profile), one routing entry in NS, one in NS_ALT, one topic binding, all keyed under the profile's namespace."""
    store = backend.open(scope)
    key = profile_namespace(profile) + f"{NS}:{tag}"
    sid = f"{profile}-{tag}"
    store.sessions.create(NewSession(session_id=SessionId(sid), source=NS, session_key=key, profile_name=profile, origin={"profile": profile}))
    store.routing.upsert_routing_entry(key, _entry(key, sid, profile), routing_scope=NS)
    alt = profile_namespace(profile) + f"{NS_ALT}:{tag}"
    store.routing.upsert_routing_entry(alt, _entry(alt, sid), routing_scope=NS_ALT)
    store.routing.bind_topic(chat_id=f"chat-{profile}", thread_id=f"thread-{tag}", user_id="u1", session_key=key, session_id=sid, profile_name=profile)


def keys_of(backend, scope: ProfileScope, profile: str, tag: str, *, owner: str | None = None) -> dict:
    """Every place the identity shows: the session's key, owner and origin, the two routing scopes' keys and payloads, the binding under ``owner``."""
    store = backend.open(scope)
    session = store.sessions.get(SessionId(f"{profile}-{tag}"))
    main = store.routing.load_routing_entries(routing_scope=NS)
    binding = store.routing.get_topic_binding(chat_id=f"chat-{profile}", thread_id=f"thread-{tag}", profile_name=owner or profile)
    return {
        "session": (session.session_key, session.profile_name, dict(session.origin or {})) if session else None,
        "main_keys": sorted(main),
        "alt_keys": sorted(store.routing.load_routing_entries(routing_scope=NS_ALT)),
        "payloads": sorted((json.loads(text)["session_key"], (json.loads(text)["origin"] or {}).get("profile") or "") for text in main.values()),
        "binding_key": binding.session_key if binding else None,
    }


def test_rekey_moves_every_facet_and_counts_them(backend, scope_a):
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    seed_profile(backend, scope_a, BYSTANDER, "t1")
    store_a.maintenance.register_backend_heartbeat(backend_id="hb-old", pid=1, started_at=time.time(), profile=OLD)
    counts = store_a.profile_identity.rekey_profile(OLD, NEW)
    # CONTRACT: one count per facet the store has, in one dict; the delivery ledger is a facet this store may lack (then its keys are absent).
    assert isinstance(counts, ProfileRekeyCounts)
    facts = counts.as_dict()
    assert {key: facts[key] for key in ("sessions_profile_name", "gateway_heartbeats_profile", "sessions_session_key", "sessions_origin_json", "gateway_routing")} == {
        "sessions_profile_name": 1, "gateway_heartbeats_profile": 1, "sessions_session_key": 1, "sessions_origin_json": 1, "gateway_routing": 2,
    }
    assert (facts["telegram_dm_topic_mode_profile_name"], facts["telegram_dm_topic_bindings_profile_name"], facts["telegram_dm_topic_bindings_session_key"]) == (0, 1, 1)
    new_key = profile_namespace(NEW) + f"{NS}:t1"
    after = keys_of(backend, scope_a, OLD, "t1", owner=NEW)
    # CONTRACT: the session's key, owner and origin, both routing scopes' keys, the payload's key and origin profile, and the binding all moved.
    assert after["session"] == (new_key, NEW, {"profile": NEW})
    assert after["main_keys"] == sorted([new_key, profile_namespace(BYSTANDER) + f"{NS}:t1"])
    assert after["alt_keys"] == sorted([profile_namespace(NEW) + f"{NS_ALT}:t1", profile_namespace(BYSTANDER) + f"{NS_ALT}:t1"])
    assert after["payloads"] == sorted([(new_key, NEW), (profile_namespace(BYSTANDER) + f"{NS}:t1", BYSTANDER)])
    assert after["binding_key"] == new_key
    assert store_a.routing.get_topic_binding(chat_id=f"chat-{OLD}", thread_id="thread-t1", profile_name=OLD) is None
    # CONTRACT: the moved heartbeat row is now the new name's.
    assert store_a.profile_identity.purge_profile(NEW).gateway_heartbeats == 1


def test_rekey_is_literal_prefix_matching_and_leaves_bystanders(backend, scope_a):
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    seed_profile(backend, scope_a, BYSTANDER, "t1")
    store_a.profile_identity.rekey_profile(OLD, NEW)
    # CONTRACT: ``agent:alpha:`` is a literal prefix; ``alpha_x`` (an underscore, a LIKE wildcard elsewhere) is untouched.
    b_key = profile_namespace(BYSTANDER) + f"{NS}:t1"
    bystander = keys_of(backend, scope_a, BYSTANDER, "t1")
    assert bystander["session"] == (b_key, BYSTANDER, {"profile": BYSTANDER}) and bystander["binding_key"] == b_key
    # CONTRACT: a key merely CONTAINING the old namespace is not under it.
    odd = f"x:{profile_namespace(OLD)}y"
    store_a.routing.upsert_routing_entry(odd, _entry(odd, "other"), routing_scope=NS)
    assert store_a.profile_identity.rekey_profile(OLD, NEW).gateway_routing == 0
    assert odd in store_a.routing.load_routing_entries(routing_scope=NS)


def test_rekey_names_are_stripped_and_degenerate_names_are_no_ops(backend, scope_a):
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    before = keys_of(backend, scope_a, OLD, "t1")
    # CONTRACT: empty or equal (after stripping) names return empty counts and touch nothing; nothing is refused, ``default`` included.
    for old, new in (("", NEW), (OLD, ""), (OLD, OLD), (f" {OLD} ", OLD), (None, NEW)):
        assert store_a.profile_identity.rekey_profile(old, new) == ProfileRekeyCounts()
        assert store_a.profile_identity.rekey_profile(old, new).as_dict() == {}
    for name in ("", "  ", None):
        assert store_a.profile_identity.purge_profile(name) == ProfilePurgeCounts()
    assert keys_of(backend, scope_a, OLD, "t1") == before
    assert store_a.profile_identity.rekey_profile("default", "other").sessions_session_key == 0
    # CONTRACT: padded names are stripped before matching.
    assert store_a.profile_identity.rekey_profile(f" {OLD} ", f" {NEW}\n").sessions_session_key == 1
    assert profile_namespace(f" {NEW} ") == "agent:beta:"


def test_rekey_is_idempotent_and_reversible(backend, scope_a):
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    first = store_a.profile_identity.rekey_profile(OLD, NEW)
    assert first.sessions_session_key == 1
    # CONTRACT: repeating a completed rename counts zero everywhere; a name nothing holds counts zero, never an error.
    again = store_a.profile_identity.rekey_profile(OLD, NEW)
    assert set(again.as_dict()) == set(first.as_dict()) and set(again.as_dict().values()) == {0}
    assert set(store_a.profile_identity.rekey_profile("ghost", "phantom").as_dict().values()) == {0}
    # CONTRACT: the inverse call restores the identity with the same counts.
    assert store_a.profile_identity.rekey_profile(NEW, OLD) == first
    assert keys_of(backend, scope_a, OLD, "t1")["session"][0] == profile_namespace(OLD) + f"{NS}:t1"


@pytest.mark.parametrize("routing_scope", [NS, "42", "", "  "])
def test_rekey_routing_collision_is_typed_with_its_text_and_writes_nothing(backend, scope_a, routing_scope):
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    target = profile_namespace(NEW) + "telegram:literal"
    for key in (profile_namespace(OLD) + "telegram:literal", target):
        store_a.routing.upsert_routing_entry(key, _entry(key, "squatter"), routing_scope=routing_scope)
    before = keys_of(backend, scope_a, OLD, "t1")
    # CONTRACT: the collision refuses with ProfileIdentityCollision (a Conflict and a ValueError) carrying its text...
    with pytest.raises(ProfileIdentityCollision) as caught:
        store_a.profile_identity.rekey_profile(OLD, NEW)
    assert isinstance(caught.value, Conflict) and isinstance(caught.value, ValueError)
    assert str(caught.value) == f"profile routing collision in scope {routing_scope!r}: {target!r}"
    # ...and NOTHING moved (atomic per store).
    assert keys_of(backend, scope_a, OLD, "t1") == before
    assert json.loads(store_a.routing.load_routing_entries(routing_scope=routing_scope)[target])["session_id"] == "squatter"


def test_rekey_topic_binding_collision_writes_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    store_a.sessions.create(NewSession(session_id=SessionId("squatter"), source=NS))
    store_a.routing.bind_topic(chat_id=f"chat-{OLD}", thread_id="thread-t1", user_id="u2", session_key="agent:beta:telegram:sq", session_id="squatter", profile_name=NEW)
    before = keys_of(backend, scope_a, OLD, "t1")
    # CONTRACT: a binding the new name already holds for the same chat and thread refuses the rename before any write.
    with pytest.raises(ProfileIdentityCollision, match="profile identity collision in telegram_dm_topic_bindings"):
        store_a.profile_identity.rekey_profile(OLD, NEW)
    assert keys_of(backend, scope_a, OLD, "t1") == before


def test_purge_deletes_bookkeeping_keeps_sessions_and_is_idempotent(backend, scope_a):
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    seed_profile(backend, scope_a, OLD, "t2")
    seed_profile(backend, scope_a, BYSTANDER, "t1")
    for backend_id, profile in (("hb-old", OLD), ("hb-other", "gamma"), ("hb-unowned", "")):
        store_a.maintenance.register_backend_heartbeat(backend_id=backend_id, pid=1, started_at=time.time(), profile=profile)
    counts = store_a.profile_identity.purge_profile(OLD)
    # CONTRACT: routing entries under the namespace in EVERY routing scope, the owner's heartbeat rows and topic bindings are gone.
    assert isinstance(counts, ProfilePurgeCounts)
    assert (counts.gateway_routing, counts.gateway_heartbeats, counts.telegram_dm_topic_mode, counts.telegram_dm_topic_bindings) == (4, 1, 0, 2)
    assert sorted(store_a.routing.load_routing_entries(routing_scope=NS)) == [profile_namespace(BYSTANDER) + f"{NS}:t1"]
    assert sorted(store_a.routing.load_routing_entries(routing_scope=NS_ALT)) == [profile_namespace(BYSTANDER) + f"{NS_ALT}:t1"]
    assert store_a.routing.get_topic_binding(chat_id=f"chat-{OLD}", thread_id="thread-t1", profile_name=OLD) is None
    assert store_a.routing.get_topic_binding(chat_id=f"chat-{BYSTANDER}", thread_id="thread-t1", profile_name=BYSTANDER) is not None
    assert store_a.maintenance.clear_backend_heartbeat("hb-other") is True and store_a.maintenance.clear_backend_heartbeat("hb-old") is False
    # CONTRACT: session rows are never deleted or altered by a purge (identity, not history).
    for tag in ("t1", "t2"):
        assert store_a.sessions.get(SessionId(f"{OLD}-{tag}")).session_key == profile_namespace(OLD) + f"{NS}:{tag}"
    # CONTRACT: idempotent; a name nothing holds purges to zeros, never an error.
    assert set(store_a.profile_identity.purge_profile(OLD).as_dict().values()) == {0}
    assert set(store_a.profile_identity.purge_profile("ghost").as_dict().values()) == {0}


def test_purge_after_rekey_finds_nothing_under_the_old_name(backend, scope_a):
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    store_a.profile_identity.rekey_profile(OLD, NEW)
    # CONTRACT: rename then purge-the-old-name is a no-op; purge-the-new-name settles the moved rows.
    assert set(store_a.profile_identity.purge_profile(OLD).as_dict().values()) == {0}
    counts = store_a.profile_identity.purge_profile(NEW)
    assert (counts.gateway_routing, counts.telegram_dm_topic_bindings) == (2, 1)


def bind_receipts(store, profile: str, tag: str) -> tuple[SessionId, SessionId]:
    """One API binding and one local creation receipt minted by ``profile``: route and entry key under its namespace, origin naming it, ``profile_id`` the minting home."""
    epoch = begin(store)
    api_sid, local_sid = SessionId(f"api-{profile}-{tag}"), SessionId(f"local-{profile}-{tag}")
    route = profile_namespace(profile) + f"api_server:dm:{api_sid}"
    origin = {"platform": "api_server", "chat_id": api_sid, "user_id": "api", "chat_type": "dm", "profile": profile}
    entry = {"session_key": route, "session_id": api_sid, "created_at": STAMP, "updated_at": STAMP, "origin": origin}
    store.bindings.bind_api(epoch=epoch, binding=NewApiBinding(
        session_id=api_sid, route=route, storage_source="api_server", title=None, declared_key=None, caller_key=None, origin=origin, entry=entry,
        receipt={"profile_id": f"/homes/{profile}", "session_id": api_sid, "route": route, "entry": entry}, started_at=STARTED))
    route = profile_namespace(profile) + f"local:dm:{local_sid}"
    origin = {"platform": "local", "chat_id": local_sid, "user_id": "owner", "chat_type": "dm", "profile": profile}
    store.bindings.reserve(epoch=epoch, owner=None, binding=NewSessionBinding(
        profile_id=f"/homes/{profile}", principal_id="owner", request_id=f"req-{tag}", session_id=local_sid, route=route,
        entry={"session_key": route, "session_id": local_sid, "created_at": STAMP, "updated_at": STAMP, "origin": origin},
        policy={"source": "cli", "model": "model-a", "cwd": "/workspace", "request_json": "{}"}, created_source="cli"))
    return api_sid, local_sid


def _moved(receipt, profile: str):
    """``receipt`` as a rename to ``profile`` leaves it: route and entry key in the new namespace, origin naming the new name, every other field as stored."""
    route = profile_namespace(profile) + receipt.route.split(":", 2)[2]
    return dataclasses.replace(receipt, route=route, entry={**receipt.entry, "session_key": route, "origin": {**receipt.entry["origin"], "profile": profile}})


def test_rekey_moves_api_and_local_receipts_and_counts_them(backend, scope_a):
    store_a = backend.open(scope_a)
    api_sid, local_sid = bind_receipts(store_a, OLD, "t1")
    api_keep, local_keep = bind_receipts(store_a, BYSTANDER, "t1")
    api, local = store_a.bindings.api_binding(api_sid), store_a.bindings.receipt(local_sid)
    kept = (store_a.bindings.api_binding(api_keep), store_a.bindings.receipt(local_keep))
    counts = store_a.profile_identity.rekey_profile(OLD, NEW)
    # CONTRACT: each receipt's route and entry key move to the new namespace and its origin to the new name; profile_id (the minting home), the id and the lineage stay as stored.
    assert (counts.api_binding_receipts, counts.local_policy_receipts) == (1, 1)
    assert store_a.bindings.api_binding(api_sid) == _moved(api, NEW)
    assert store_a.bindings.receipt(local_sid) == _moved(local, NEW)
    assert (store_a.bindings.api_binding(api_sid).profile_id, store_a.bindings.receipt(local_sid).profile_id) == (f"/homes/{OLD}", f"/homes/{OLD}")
    # CONTRACT: another profile's receipts stay, ``alpha_x`` included (a whole-namespace match only).
    assert (store_a.bindings.api_binding(api_keep), store_a.bindings.receipt(local_keep)) == kept
    # CONTRACT: a repeated rename counts zero receipts and changes none.
    moved = (store_a.bindings.api_binding(api_sid), store_a.bindings.receipt(local_sid))
    again = store_a.profile_identity.rekey_profile(OLD, NEW)
    assert (again.api_binding_receipts, again.local_policy_receipts) == (0, 0)
    assert (store_a.bindings.api_binding(api_sid), store_a.bindings.receipt(local_sid)) == moved


def test_rekey_counts_receipts_even_when_none_are_stored(backend, scope_a):
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    # CONTRACT: once the names are valid both receipt counts are present, zero when nothing moved.
    facts = store_a.profile_identity.rekey_profile(OLD, NEW).as_dict()
    assert (facts["api_binding_receipts"], facts["local_policy_receipts"]) == (0, 0)
    assert list(facts)[-3:] == ["gateway_routing", "api_binding_receipts", "local_policy_receipts"]


def test_profile_identity_is_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    seed_profile(backend, scope_a, OLD, "t1")
    seed_profile(backend, scope_b, OLD, "t1")
    # CONTRACT: the handle names the STORE; the other store's identical identity is untouched by both verbs.
    store_a.profile_identity.rekey_profile(OLD, NEW)
    assert keys_of(backend, scope_b, OLD, "t1")["session"][0] == profile_namespace(OLD) + f"{NS}:t1"
    assert store_a.profile_identity.purge_profile(NEW).gateway_routing == 2
    assert keys_of(backend, scope_b, OLD, "t1")["binding_key"] == profile_namespace(OLD) + f"{NS}:t1"
    assert store_b.profile_identity.purge_profile(OLD).telegram_dm_topic_bindings == 1
