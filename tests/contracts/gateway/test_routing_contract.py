"""Contract cases for the routing-index and Telegram DM topic verbs of ``persistence.gateway.routing.RoutingRepository``.

Every case runs on every registered backend through the ``backend`` fixture and seeds through public verbs.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from persistence import Conflict, PermanentStoreError, ProfileScope, SessionId, SessionRowMissing
from persistence.conversation.sessions import NewSession
from persistence.gateway.routing import TOPIC_SCHEMA_META_KEY, TopicAlreadyLinked, TopicBinding
from tests.contracts._waiting import wait_past
from tests.contracts._scopes import MEDIA_ROOT

HORIZON_S = 7 * 86400


def entry(session_id: Any, **extra: Any) -> str:
    return json.dumps({"session_id": session_id, **extra})


def _sessions(backend, scope: ProfileScope, *session_ids: str) -> None:
    store = backend.open(scope)
    for sid in session_ids:
        store.sessions.create(NewSession(session_id=SessionId(sid), source="telegram"))


# ── routing index ─────────────────────────────────────────────────────────────


def test_upsert_stores_the_text_as_given_and_the_latest_write_wins(backend, scope_a):
    store_a = backend.open(scope_a)
    odd = '{"session_id":"s1",  "z": 1, "a": [1,2]}'
    store_a.routing.upsert_routing_entry("k1", odd)
    store_a.routing.upsert_routing_entry("k2", entry("s2"))
    assert store_a.routing.load_routing_entries() == {"k1": odd, "k2": entry("s2")}
    store_a.routing.upsert_routing_entry("k1", entry("s1b", note="later"))
    assert store_a.routing.load_routing_entries()["k1"] == entry("s1b", note="later")
    # Text that names no session is stored as well: an array, an object without session_id.
    store_a.routing.upsert_routing_entry("k3", "[1,2]")
    store_a.routing.upsert_routing_entry("k4", '{"a": 1}')
    assert {key: text for key, text in store_a.routing.load_routing_entries().items() if key in ("k3", "k4")} == {"k3": "[1,2]", "k4": '{"a": 1}'}


def test_empty_keys_and_entries_are_dropped_silently(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.routing.upsert_routing_entry("", entry("s"))
    store_a.routing.upsert_routing_entry("k", "")
    assert store_a.routing.load_routing_entries() == {}
    assert store_a.routing.replace_routing_entries({"": entry("s"), "e": "", "k": entry("s")}) == set()
    assert store_a.routing.load_routing_entries() == {"k": entry("s")}


def test_malformed_entry_text_is_refused_and_writes_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.routing.upsert_routing_entry("keep", entry("s"), routing_scope="r")
    # The save reads the entry's session_id and raises ``malformed JSON``.
    with pytest.raises(PermanentStoreError, match="malformed JSON"):
        store_a.routing.upsert_routing_entry("k", "not json")
    with pytest.raises(PermanentStoreError, match="malformed JSON"):
        store_a.routing.replace_routing_entries({"a": "x{", "b": entry("s")}, routing_scope="r")
    with pytest.raises(PermanentStoreError, match="malformed JSON"):
        store_a.routing.route_retired("k", "nope")
    assert store_a.routing.load_routing_entries() == {}
    assert store_a.routing.load_routing_entries(routing_scope="r") == {"keep": entry("s")}


def test_routing_scopes_partition_the_index_and_unknown_reads_empty(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.routing.upsert_routing_entry("k", entry("s-default"))
    store_a.routing.upsert_routing_entry("k", entry("s-other"), routing_scope="/srv/other")
    assert store_a.routing.load_routing_entries() == {"k": entry("s-default")}
    assert store_a.routing.load_routing_entries(routing_scope="") == {"k": entry("s-default")}
    assert store_a.routing.load_routing_entries(routing_scope="/srv/other") == {"k": entry("s-other")}
    assert store_a.routing.load_routing_entries(routing_scope="/srv/none") == {}


def test_replace_swaps_one_routing_scope_only_and_empty_is_deletion(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.routing.upsert_routing_entry("old", entry("s-old"), routing_scope="r")
    store_a.routing.upsert_routing_entry("other", entry("s-other"), routing_scope="q")
    assert store_a.routing.replace_routing_entries({"new": entry("s-new"), "old2": entry("s-old2")}, routing_scope="r") == set()
    assert store_a.routing.load_routing_entries(routing_scope="r") == {"new": entry("s-new"), "old2": entry("s-old2")}
    assert store_a.routing.load_routing_entries(routing_scope="q") == {"other": entry("s-other")}
    assert store_a.routing.replace_routing_entries({}, routing_scope="r") == set()
    assert store_a.routing.load_routing_entries(routing_scope="r") == {}
    assert store_a.routing.load_routing_entries(routing_scope="q") == {"other": entry("s-other")}


def test_routing_index_two_profile_isolation(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.routing.upsert_routing_entry("k", entry("s-a"))
    store_b.routing.upsert_routing_entry("k", entry("s-b"))
    store_b.routing.replace_routing_entries({}, routing_scope="")
    assert store_a.routing.load_routing_entries() == {"k": entry("s-a")}
    assert store_b.routing.load_routing_entries() == {}
    assert store_b.routing.routing_entry_for_session("s-a") is None


def test_routing_entry_for_session_searches_every_routing_scope(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.routing.upsert_routing_entry("bad", "[1,2]")
    store_a.routing.upsert_routing_entry("k1", entry("gr-e1", suspended=True), routing_scope="launch")
    store_a.routing.upsert_routing_entry("k2", entry("gr-e2", origin={"chat_id": "9"}), routing_scope="profile")
    store_a.routing.upsert_routing_entry("k3", entry(5))
    # The parsed entry is returned; non-object entries are skipped.
    found = store_a.routing.routing_entry_for_session("gr-e2")
    assert found == {"session_id": "gr-e2", "origin": {"chat_id": "9"}}
    assert store_a.routing.routing_entry_for_session("gr-e1").get("suspended") is True
    assert store_a.routing.routing_entry_for_session("gr-none") is None
    # A returned entry is a copy.
    found["origin"]["chat_id"] = "changed"
    assert store_a.routing.routing_entry_for_session("gr-e2")["origin"] == {"chat_id": "9"}
    # An entry that moved to another session no longer names the old one.
    store_a.routing.upsert_routing_entry("k1", entry("gr-e1b"), routing_scope="launch")
    assert store_a.routing.routing_entry_for_session("gr-e1") is None


# ── route fences and retired routes ───────────────────────────────────────────


def test_fence_retired_routes_fences_the_namespace_and_main_and_drops_the_routes(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "live")
    store_a.routing.upsert_routing_entry("agent:main:telegram:dm:1", entry("gone"))
    store_a.routing.upsert_routing_entry("agent:work:telegram:dm:1", entry("gone"), routing_scope="other")
    store_a.routing.upsert_routing_entry("agent:main:telegram:dm:2", entry("live"))
    store_a.routing.upsert_routing_entry("agent:work:telegram:dm:2", entry("live"))
    store_a.routing.upsert_routing_entry("plain-key", entry("live"))
    store_a.routing.fence_retired_routes(["gone", "live", ""], namespace="work", expiring=["live"])
    # Routes in the fenced namespaces of each id go, in every routing scope; a main route to a session this store holds stays.
    assert store_a.routing.load_routing_entries() == {"agent:main:telegram:dm:2": entry("live"), "plain-key": entry("live")}
    assert store_a.routing.load_routing_entries(routing_scope="other") == {}
    # The fences hold these values: permanent for a delete, ``expires`` for a sweep; ``main`` only for an id this store holds no session of.
    gone_work = json.loads(store_a.state_meta.get("gateway.retired_route.v1.work:gone"))
    assert set(gone_work) == {"retired_at"} and isinstance(gone_work["retired_at"], float)
    assert set(json.loads(store_a.state_meta.get("gateway.retired_route.v1.main:gone"))) == {"retired_at"}
    assert json.loads(store_a.state_meta.get("gateway.retired_route.v1.work:live"))["expires"] is True
    assert store_a.state_meta.get("gateway.retired_route.v1.main:live") is None
    # Each route is judged against its own namespace.
    cases = {
        ("agent:main:x:y", "gone"): True,
        ("agent:work:x:y", "gone"): True,
        ("agent:main:x:y", "live"): False,
        ("agent:work:x:y", "live"): True,
        ("agent:other:x:y", "gone"): False,
    }
    for (key, sid), retired in cases.items():
        assert store_a.routing.route_retired(key, entry(sid)) is retired, (key, sid)


def test_routing_saves_skip_fenced_routes(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "live")
    store_a.routing.fence_retired_routes(["gone"], namespace="work")
    store_a.routing.upsert_routing_entry("agent:work:telegram:dm:9", entry("gone"))
    assert store_a.routing.load_routing_entries() == {}
    skipped = store_a.routing.replace_routing_entries(
        {"agent:work:a:b": entry("gone"), "agent:main:a:b": entry("live"), "plain": "[1]"}, routing_scope="z"
    )
    # The skipped keys are returned so the sessions.json mirror leaves them out too.
    assert skipped == {"agent:work:a:b"}
    assert store_a.routing.load_routing_entries(routing_scope="z") == {"agent:main:a:b": entry("live"), "plain": "[1]"}


def test_this_stores_own_retirement_fences_only_its_main_routes(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "mine")
    store_a.routing.upsert_routing_entry("agent:main:telegram:dm:1", entry("mine"))
    assert store_a.routing.route_retired("agent:main:telegram:dm:1", entry("mine")) is False
    assert store_a.sessions.delete(SessionId("mine"), media_root=MEDIA_ROOT) == ("mine",)
    # The delete retired the id: a main route to it is skipped; a secondary profile's route is judged by that profile's fence only.
    assert store_a.routing.route_retired("agent:main:q:r", entry("mine")) is True
    assert store_a.routing.route_retired("agent:work2:q:r", entry("mine")) is False
    store_a.routing.upsert_routing_entry("agent:main:telegram:dm:1", entry("mine"))
    store_a.routing.upsert_routing_entry("agent:work2:telegram:dm:1", entry("mine"))
    assert store_a.routing.load_routing_entries() == {"agent:work2:telegram:dm:1": entry("mine")}


def test_a_sweep_fence_ages_out_at_the_horizon_and_never_shortens_a_permanent_one(backend, scope_a, monkeypatch):
    store_a = backend.open(scope_a)
    store_a.routing.fence_retired_routes(["swept"], namespace="work", expiring=["swept"])
    store_a.routing.fence_retired_routes(["deleted"], namespace="work")
    store_a.routing.fence_retired_routes(["deleted"], namespace="work", expiring=["deleted"])
    later = time.time() + HORIZON_S + 60
    monkeypatch.setattr(time, "time", lambda: later)
    assert store_a.routing.route_retired("agent:work:a:b", entry("swept")) is False
    assert store_a.routing.route_retired("agent:work:a:b", entry("deleted")) is True
    # An empty id set writes nothing: the aged sweep fence is still stored.
    store_a.routing.fence_retired_routes(["", None], namespace="work")
    assert store_a.state_meta.get("gateway.retired_route.v1.work:swept") is not None
    # The next fence write reclaims every aged sweep fence; permanent fences stay.
    store_a.routing.fence_retired_routes(["other"], namespace="work")
    assert store_a.state_meta.get("gateway.retired_route.v1.work:swept") is None
    assert store_a.state_meta.get("gateway.retired_route.v1.main:swept") is None
    assert store_a.state_meta.get("gateway.retired_route.v1.work:deleted") is not None


# ── Telegram DM topic mode ────────────────────────────────────────────────────


def test_a_store_without_topic_tables_reads_empty_and_cleanups_create_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s")
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u") is False
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t") is None
    assert store_a.routing.list_topic_bindings_for_chat(chat_id="c") == ()
    assert store_a.routing.get_topic_binding_for_session(session_id="s") is None
    assert store_a.routing.is_session_linked_to_topic(session_id="s") is False
    assert store_a.routing.delete_topic_binding(chat_id="c", thread_id="t") == 0
    store_a.routing.disable_topic_mode(chat_id="c")
    assert store_a.state_meta.get(TOPIC_SCHEMA_META_KEY) is None


def test_topic_mode_is_keyed_to_the_enabling_user_and_survives_a_disable_cycle(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.routing.enable_topic_mode(chat_id="c", user_id="u1", has_topics_enabled=True, allows_users_to_create_topics=False)
    assert store_a.state_meta.get(TOPIC_SCHEMA_META_KEY) == "3"
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u1") is True
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u2") is False
    store_a.routing.disable_topic_mode(chat_id="c")
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u1") is False
    # A re-enable by another user moves the row to that user.
    store_a.routing.enable_topic_mode(chat_id="c", user_id="u2")
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u2") is True
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u1") is False


def test_topic_rows_are_keyed_by_the_routed_profile(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s-default", "s-work")
    store_a.routing.enable_topic_mode(chat_id="c", user_id="u", profile_name="work")
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u") is False
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u", profile_name=" work ") is True
    # An empty or blank profile is the default profile.
    store_a.routing.enable_topic_mode(chat_id="c", user_id="u", profile_name="  ")
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u", profile_name="default") is True
    store_a.routing.bind_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="s-default", profile_name="")
    store_a.routing.bind_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="s-work", profile_name="work")
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t").session_id == "s-default"
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t", profile_name="work").session_id == "s-work"
    assert [b.session_id for b in store_a.routing.list_topic_bindings_for_chat(chat_id="c", profile_name="work")] == ["s-work"]
    # Disabling one profile's chat touches only that profile's rows.
    store_a.routing.disable_topic_mode(chat_id="c", profile_name="work")
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t", profile_name="work") is None
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t").session_id == "s-default"
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u") is True
    # The session lookups ignore the profile.
    assert store_a.routing.get_topic_binding_for_session(session_id="s-default").profile_name == "default"


def test_topic_state_two_profile_isolation(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    _sessions(backend, scope_a, "s")
    store_a.routing.enable_topic_mode(chat_id="c", user_id="u")
    store_a.routing.bind_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="s")
    assert store_b.routing.is_topic_mode_enabled(chat_id="c", user_id="u") is False
    assert store_b.routing.get_topic_binding(chat_id="c", thread_id="t") is None
    assert store_b.routing.is_session_linked_to_topic(session_id="s") is False
    assert store_b.state_meta.get(TOPIC_SCHEMA_META_KEY) is None


# ── Telegram DM topic bindings ────────────────────────────────────────────────


def test_bind_and_lookups_round_trip_the_row(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s1")
    before = time.time()
    store_a.routing.bind_topic(chat_id="c", thread_id="t", user_id="", session_key="agent:main:telegram:dm:c:t", session_id="s1")
    binding = store_a.routing.get_topic_binding(chat_id="c", thread_id="t")
    assert isinstance(binding, TopicBinding)
    assert (binding.profile_name, binding.chat_id, binding.thread_id, binding.user_id, binding.session_key, binding.session_id, binding.managed_mode) == (
        "default", "c", "t", "", "agent:main:telegram:dm:c:t", "s1", "auto",
    )
    assert before <= binding.linked_at == binding.updated_at
    assert store_a.routing.get_topic_binding_for_session(session_id="s1") == binding
    assert store_a.routing.is_session_linked_to_topic(session_id="s1") is True
    assert store_a.routing.list_topic_bindings_for_chat(chat_id="c") == (binding,)


def test_restored_managed_mode_round_trips(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s1")
    store_a.routing.bind_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="s1", managed_mode="restored")
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t").managed_mode == "restored"


def test_rebinding_the_same_topic_refreshes_it_and_a_cross_topic_relink_is_refused(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s1")
    store_a.routing.bind_topic(chat_id="c", thread_id="t1", user_id="u", session_key="k", session_id="s1")
    first = store_a.routing.get_topic_binding(chat_id="c", thread_id="t1")
    store_a.routing.bind_topic(chat_id="c", thread_id="t1", user_id="u2", session_key="k2", session_id="s1", managed_mode="restored")
    again = store_a.routing.get_topic_binding(chat_id="c", thread_id="t1")
    # The same (profile, chat, thread) is an upsert: linked_at kept, the rest refreshed.
    assert (again.linked_at, again.user_id, again.session_key, again.managed_mode) == (first.linked_at, "u2", "k2", "restored")
    assert again.updated_at >= first.updated_at
    # Another thread, chat or profile for a linked session raises its ValueError text; nothing changes.
    for other in ({"thread_id": "t2"}, {"chat_id": "c2", "thread_id": "t1"}, {"thread_id": "t1", "profile_name": "work"}):
        kwargs = {"chat_id": "c", "user_id": "u", "session_key": "k", "session_id": "s1", **other}
        with pytest.raises(TopicAlreadyLinked, match="session is already linked to another Telegram topic") as caught:
            store_a.routing.bind_topic(**kwargs)
        assert isinstance(caught.value, ValueError) and isinstance(caught.value, Conflict)
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t2") is None
    assert store_a.routing.get_topic_binding_for_session(session_id="s1") == again


def test_bind_moves_an_existing_topic_to_a_new_session(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s-old", "s-new")
    store_a.routing.bind_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="s-old")
    store_a.routing.bind_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="s-new")
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t").session_id == "s-new"
    assert store_a.routing.is_session_linked_to_topic(session_id="s-old") is False
    # The old session is free to link elsewhere.
    store_a.routing.bind_topic(chat_id="c", thread_id="t9", user_id="u", session_key="k9", session_id="s-old")
    assert store_a.routing.get_topic_binding_for_session(session_id="s-old").thread_id == "t9"


def test_an_expected_session_id_rebinds_only_a_topic_still_naming_it(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s-old", "s-other", "s-new")
    bind = {"chat_id": "c", "thread_id": "t", "user_id": "u", "session_key": "k"}
    assert store_a.routing.bind_topic(**bind, session_id="s-old") is True
    # Another session took the topic after the caller read it: the compare-and-swap refuses and writes nothing.
    store_a.routing.bind_topic(**bind, session_id="s-other")
    taken = store_a.routing.get_topic_binding(chat_id="c", thread_id="t")
    assert store_a.routing.bind_topic(**bind, session_id="s-new", expected_session_id="s-old") is False
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t") == taken
    assert store_a.routing.is_session_linked_to_topic(session_id="s-new") is False
    # The stored row still names the expected id: the rebind goes through.
    assert store_a.routing.bind_topic(**bind, session_id="s-new", expected_session_id="s-other") is True
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t").session_id == "s-new"


def test_an_expected_session_id_refuses_a_topic_with_no_binding(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s1", "s2")
    bind = {"chat_id": "c", "thread_id": "t", "user_id": "u", "session_key": "k"}
    # A binding cleared since the caller read it (a /topic off, a delete) is not recreated.
    assert store_a.routing.bind_topic(**bind, session_id="s1", expected_session_id="s0") is False
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t") is None
    assert store_a.routing.is_session_linked_to_topic(session_id="s1") is False
    # The compared row is the (profile, chat, thread) being bound: another profile's row does not count.
    store_a.routing.bind_topic(**bind, session_id="s2", profile_name="work")
    assert store_a.routing.bind_topic(**bind, session_id="s1", expected_session_id="s2") is False
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t") is None


def test_bind_needs_an_existing_session_and_a_deleted_session_drops_its_binding(backend, scope_a):
    store_a = backend.open(scope_a)
    # A missing session row refuses the bind; the tables are still prepared by the first write.
    with pytest.raises(SessionRowMissing):
        store_a.routing.bind_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="ghost")
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t") is None
    assert store_a.state_meta.get(TOPIC_SCHEMA_META_KEY) == "3"
    assert store_a.sessions.get(SessionId("ghost")) is None
    _sessions(backend, scope_a, "s")
    store_a.routing.bind_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="s")
    assert store_a.sessions.delete(SessionId("s"), media_root=MEDIA_ROOT) == ("s",)
    # The binding cascades with its session row.
    assert store_a.routing.get_topic_binding(chat_id="c", thread_id="t") is None
    assert store_a.routing.is_session_linked_to_topic(session_id="s") is False


def test_list_bindings_scopes_by_chat_newest_first(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s1", "s2", "s3")
    store_a.routing.bind_topic(chat_id="c", thread_id="t1", user_id="u", session_key="k1", session_id="s1")
    store_a.routing.bind_topic(chat_id="other", thread_id="t2", user_id="u", session_key="k2", session_id="s2")
    store_a.routing.bind_topic(chat_id="c", thread_id="t3", user_id="u", session_key="k3", session_id="s3")
    # A rebind refreshes updated_at, moving the binding to the front.
    wait_past(store_a.routing.get_topic_binding(chat_id="c", thread_id="t3").updated_at)
    store_a.routing.bind_topic(chat_id="c", thread_id="t1", user_id="u", session_key="k1", session_id="s1")
    assert [b.thread_id for b in store_a.routing.list_topic_bindings_for_chat(chat_id="c")] == ["t1", "t3"]
    assert [b.thread_id for b in store_a.routing.list_topic_bindings_for_chat(chat_id="other")] == ["t2"]
    assert store_a.routing.list_topic_bindings_for_chat(chat_id="none") == ()


def test_delete_binding_counts_and_the_last_binding_disables_the_mode(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s1", "s2")
    store_a.routing.enable_topic_mode(chat_id="c", user_id="u")
    store_a.routing.bind_topic(chat_id="c", thread_id="t1", user_id="u", session_key="k1", session_id="s1")
    store_a.routing.bind_topic(chat_id="c", thread_id="t2", user_id="u", session_key="k2", session_id="s2")
    assert store_a.routing.delete_topic_binding(chat_id="c", thread_id="t1") == 1
    assert store_a.routing.delete_topic_binding(chat_id="c", thread_id="t1") == 0
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u") is True
    # Another profile's binding on the same chat does not keep this profile's mode on.
    store_a.routing.enable_topic_mode(chat_id="c", user_id="u", profile_name="work")
    store_a.routing.bind_topic(chat_id="c", thread_id="t3", user_id="u", session_key="k3", session_id="s1", profile_name="work")
    assert store_a.routing.delete_topic_binding(chat_id="c", thread_id="t2") == 1
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u") is False
    assert store_a.routing.is_topic_mode_enabled(chat_id="c", user_id="u", profile_name="work") is True


def test_disable_clears_bindings_by_default_and_keeps_them_on_request(backend, scope_a):
    store_a = backend.open(scope_a)
    _sessions(backend, scope_a, "s1", "s2")
    store_a.routing.enable_topic_mode(chat_id="c1", user_id="u")
    store_a.routing.bind_topic(chat_id="c1", thread_id="t", user_id="u", session_key="k1", session_id="s1")
    store_a.routing.enable_topic_mode(chat_id="c2", user_id="u")
    store_a.routing.bind_topic(chat_id="c2", thread_id="t", user_id="u", session_key="k2", session_id="s2")
    store_a.routing.disable_topic_mode(chat_id="c1")
    assert store_a.routing.list_topic_bindings_for_chat(chat_id="c1") == ()
    store_a.routing.disable_topic_mode(chat_id="c2", clear_bindings=False)
    assert store_a.routing.is_topic_mode_enabled(chat_id="c2", user_id="u") is False
    assert [b.session_id for b in store_a.routing.list_topic_bindings_for_chat(chat_id="c2")] == ["s2"]
