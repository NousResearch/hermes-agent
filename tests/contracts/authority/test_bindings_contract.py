"""Contract suite for ``SessionBindingRepository.reserve`` and ``receipt``.

Backend cases exercise public repository operations; record validation constructs commands without a backend, and failure injection uses reference state. The reservation is one write: the positive cases assert the reserved row through the sessions capability and the route through the routing capability together with the receipt, and the crash cases prove neither-or-both. Refusals are ``AuthorityRefusal`` with their reason codes.

Validation order: the client-request and encoding checks precede the epoch; the epoch precedes the replay (a stale-epoch duplicate is a fence refusal); the replay precedes the retirement, row and route checks (an exact duplicate replays though its own row and route exist).
"""

from __future__ import annotations

import json

import pytest

from persistence import AuthorityRefusal, FenceStale, SessionId, StaleFenceKind, TurnLeaseLost
from persistence.authority.bindings import NewSessionBinding
from persistence.conversation.sessions import NewSession
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts._reference.authority.bindings import BETWEEN_ROUTE_AND_RECEIPT, BETWEEN_SESSION_AND_ROUTE
from tests.contracts.conftest import require_capability, require_failpoints
from tests.contracts._scopes import MEDIA_ROOT

CREATED_AT = "2026-09-16T10:00:00+00:00"


def make_binding(
    session_id: str = "s-bind-1",
    *,
    route: str = "route-bind-1",
    principal_id: str = "user-1",
    request_id: str = "req-1",
    request_json: str = '{"kind":"launch","n":1}',
    profile_id: str = "default",
    policy: dict | None = None,
    created_source: str | None = "cli",
) -> NewSessionBinding:
    return NewSessionBinding(
        profile_id=profile_id,
        principal_id=principal_id,
        request_id=request_id,
        session_id=SessionId(session_id),
        route=route,
        entry={
            "session_key": route,
            "session_id": session_id,
            "created_at": CREATED_AT,
            "updated_at": CREATED_AT,
            "origin": {"chat_id": session_id, "chat_type": "dm", "user_id": principal_id, "platform": "local"},
        },
        policy={"source": "cli", "model": "model-a", "cwd": "/workspace", "request_json": request_json, **(policy or {})},
        created_source=created_source,
    )


def begin(store, instance: str = "instance-a") -> int:
    return store.runtime.begin_epoch(instance_id=instance)


def assert_refusal(call, reason: str) -> None:
    with pytest.raises(AuthorityRefusal) as caught:
        call()
    assert caught.value.reason == reason


def _kwargs(binding: NewSessionBinding) -> dict:
    return {
        "profile_id": binding.profile_id, "principal_id": binding.principal_id, "request_id": binding.request_id,
        "session_id": binding.session_id, "route": binding.route, "entry": dict(binding.entry), "policy": dict(binding.policy),
        "created_source": binding.created_source,
    }


# ── record validation at the contract boundary ────────────────────────────────


def test_new_binding_rejects_missing_identity_and_payload_fields():
    # CONTRACT: identity fields and the store-read payload keys are validated before any backend is touched.
    good = make_binding()
    for field, value in (("route", ""), ("principal_id", ""), ("request_id", ""), ("profile_id", ""), ("created_source", 1)):
        with pytest.raises(ValueError):
            NewSessionBinding(**{**_kwargs(good), field: value})
    # CONTRACT: a creation surface of None or empty is accepted.
    for value in (None, ""):
        NewSessionBinding(**{**_kwargs(good), "created_source": value})
    for policy in (
        {"model": "m", "cwd": "/w", "request_json": "{}"},
        {"source": "cli", "model": "m", "cwd": "/w"},
        {"source": "cli", "request_json": "{}"},
    ):
        with pytest.raises(ValueError):
            NewSessionBinding(**{**_kwargs(good), "policy": policy})
    for entry in (
        {**dict(good.entry), "session_id": "someone-else"},
        {**dict(good.entry), "created_at": "not-a-timestamp"},
        {**dict(good.entry), "origin": {}},
    ):
        with pytest.raises(ValueError):
            NewSessionBinding(**{**_kwargs(good), "entry": entry})
    with pytest.raises(ValueError):
        NewSessionBinding(**_kwargs(good), lineage=())
    NewSessionBinding(**{**_kwargs(good), "entry": {**dict(good.entry), "created_at": "2026-09-16T10:00:00"}})


# ── the composite reservation ─────────────────────────────────────────────────


def test_reserve_publishes_session_route_and_receipt_together(backend, scope_a):
    store_a = backend.open(scope_a)
    binding = make_binding("s-bind-atomic", route="route-atomic")
    reserved = store_a.bindings.reserve(epoch=begin(store_a), binding=binding, owner=None)
    # CONTRACT: the creating call reports created; the binding carries the stored receipt.
    assert reserved.created is True
    stored = reserved.binding
    assert (stored.session_id, stored.route, stored.principal_id, stored.request_id, stored.profile_id) == (
        "s-bind-atomic", "route-atomic", "user-1", "req-1", "default")
    assert stored.policy["request_json"] == binding.policy["request_json"]
    assert stored.current_target == binding.session_id
    assert stored.lineage == (binding.session_id,)
    # CONTRACT: the reserved row: source, the caller's created_source, principal, route, chat id, dm, model, cwd.
    record = store_a.sessions.get(binding.session_id)
    assert (record.source, record.created_source, record.session_key, record.user_id, record.chat_type) == ("cli", "cli", "route-atomic", "user-1", "dm")
    assert (record.chat_id, record.model, record.cwd) == ("s-bind-atomic", "model-a", "/workspace")
    # CONTRACT: the routing index holds the entry as raw JSON text in the "" namespace.
    entries = store_a.routing.load_routing_entries()
    assert json.loads(entries["route-atomic"]) == dict(binding.entry)
    read = store_a.bindings.receipt(binding.session_id)
    assert read == stored


def test_a_source_that_is_no_surface_stores_no_created_source(backend, scope_a):
    from session_rules.surfaces import created_surface

    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    cases = [(source, created_surface(source)) for source in ("local", "unknown")] + [("cli", ""), ("cli", None)]
    for n, (source, created) in enumerate(cases):
        binding = make_binding(f"s-nosurface-{n}", route=f"route-nosurface-{n}", request_id=f"req-nosurface-{n}",
                               policy={"source": source}, created_source=created)
        assert store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None).created is True
        record = store_a.sessions.get(binding.session_id)
        # CONTRACT: a created_surface of None is stored as NULL: the row keeps its policy source and no created_source.
        assert (record.source, record.created_source) == (source, None), (source, created)


def test_row_cwd_follows_the_create_request(backend, scope_a):
    # CONTRACT: the request's ``row_cwd`` names the row cwd ("" = none); the policy cwd otherwise.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.bindings.reserve(epoch=epoch, binding=make_binding("s-cwd-1", route="r-cwd-1", request_json='{"row_cwd":"/rows"}'), owner=None)
    store_a.bindings.reserve(epoch=epoch, binding=make_binding("s-cwd-2", route="r-cwd-2", request_json='{"row_cwd":""}'), owner=None)
    assert store_a.sessions.get(SessionId("s-cwd-1")).cwd == "/rows"
    assert store_a.sessions.get(SessionId("s-cwd-2")).cwd is None


def test_exact_duplicate_replays_the_stored_receipt(backend, scope_a):
    # CONTRACT: an exact duplicate returns the STORED receipt with created False; replay precedes the row and route checks.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    binding = make_binding("s-bind-dup", route="route-dup")
    first = store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    again = store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    assert (again.binding, again.created) == (first.binding, False)


def test_identity_or_request_mismatch_under_the_same_id_is_invalid_params(backend, scope_a):
    # CONTRACT: the replay identity is profile, principal, request, session and route plus the same create retry; any other is invalid_params and the first writer's receipt stays.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.bindings.reserve(epoch=epoch, binding=make_binding("s-bind-ident", route="route-ident"), owner=None)
    for changed in (
        make_binding("s-bind-ident", route="route-other"),
        make_binding("s-bind-ident", route="route-ident", principal_id="user-2"),
        make_binding("s-bind-ident", route="route-ident", request_id="req-2"),
        make_binding("s-bind-ident", route="route-ident", profile_id="other"),
        make_binding("s-bind-ident", route="route-ident", request_json='{"kind":"launch","n":2}'),
    ):
        assert_refusal(lambda changed=changed: store_a.bindings.reserve(epoch=epoch, binding=changed, owner=None), "invalid_params")
    stored = store_a.bindings.receipt(SessionId("s-bind-ident"))
    assert (stored.route, stored.principal_id, stored.request_id) == ("route-ident", "user-1", "req-1")


def test_retry_matches_the_request_as_the_client_sent_it(backend, scope_a):
    # CONTRACT: ``desktop_local`` and a resolved model under ``model_requested`` do not split a retry.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    saved = '{"model":"resolved","model_requested":"alias","base_url":"https://a","desktop_local":true}'
    store_a.bindings.reserve(epoch=epoch, binding=make_binding("s-bind-req", route="r-req", request_json=saved), owner=None)
    retry = make_binding("s-bind-req", route="r-req", request_json='{"model":"other","model_requested":"alias","base_url":"https://b"}')
    assert store_a.bindings.reserve(epoch=epoch, binding=retry, owner=None).created is False


def test_client_env_retry_is_judged_by_the_frozen_rules(backend, scope_a):
    # CONTRACT: a policy that froze a client env needs ``client_request`` (else invalid_params before the epoch); the retry's map, filtered by the frozen rules, must equal the stored map.
    store_a = backend.open(scope_a)
    rules = {"behaviour": [], "names": ["PATH"], "prefixes": [], "secret": ["TOKEN"]}
    frozen = {"client_env_json": json.dumps({"PATH": "/bin"}), "client_env_rules_json": json.dumps(rules)}
    binding = make_binding("s-bind-env", route="r-env", policy=frozen)
    assert_refusal(lambda: store_a.bindings.reserve(epoch=0, binding=binding, owner=None), "invalid_params")
    epoch = begin(store_a)
    assert store_a.bindings.reserve(epoch=epoch, binding=binding, client_request={"client_env": {"PATH": "/bin"}}, owner=None).created is True
    kept = {"client_env": {"PATH": "/bin", "API_TOKEN": "dropped", "OTHER": "dropped"}}
    assert store_a.bindings.reserve(epoch=epoch, binding=binding, client_request=kept, owner=None).created is False
    for different in ({"client_env": {"PATH": "/usr/bin"}}, {}, {"client_env": ["PATH"]}):
        assert_refusal(lambda different=different: store_a.bindings.reserve(epoch=epoch, binding=binding, client_request=different, owner=None), "invalid_params")


def test_route_collision_is_admission_conflict(backend, scope_a):
    # CONTRACT: a route held in the routing index refuses admission_conflict; the loser publishes nothing.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.bindings.reserve(epoch=epoch, binding=make_binding("s-bind-r1", route="route-shared"), owner=None)
    assert_refusal(lambda: store_a.bindings.reserve(epoch=epoch, binding=make_binding("s-bind-r2", route="route-shared"), owner=None), "admission_conflict")
    assert_refusal(lambda: store_a.bindings.receipt(SessionId("s-bind-r2")), "storage_unavailable")
    assert store_a.sessions.get(SessionId("s-bind-r2")) is None


def test_existing_unbound_session_id_is_storage_unavailable(backend, scope_a):
    # CONTRACT: the reservation cannot vouch for a row it did not create.
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=SessionId("s-bind-taken"), source="cli"))
    assert_refusal(lambda: store_a.bindings.reserve(epoch=begin(store_a), binding=make_binding("s-bind-taken"), owner=None), "storage_unavailable")


def test_retired_session_id_is_not_found(backend, scope_a):
    # CONTRACT: a deleted id stays retired: a create retry must not resurrect it.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.sessions.create(NewSession(session_id=SessionId("s-bind-gone"), source="cli"))
    assert store_a.sessions.delete(SessionId("s-bind-gone"), media_root=MEDIA_ROOT)
    assert_refusal(lambda: store_a.bindings.reserve(epoch=epoch, binding=make_binding("s-bind-gone"), owner=None), "not_found")


def test_reserve_is_fenced_fail_closed_on_epoch(backend, scope_a):
    # CONTRACT: a never-issued zero, the previous epoch, a future value and non-int values are the epoch fence.
    store_a = backend.open(scope_a)
    with pytest.raises(FenceStale) as fenced:
        store_a.bindings.reserve(epoch=0, binding=make_binding("s-bind-e0"), owner=None)
    assert fenced.value.kind is StaleFenceKind.STALE_EPOCH
    first = begin(store_a)
    second = begin(store_a, "instance-b")
    for stale in (first, second + 1, True, float(second)):
        with pytest.raises(FenceStale):
            store_a.bindings.reserve(epoch=stale, binding=make_binding("s-bind-e1"), owner=None)
    assert_refusal(lambda: store_a.bindings.receipt(SessionId("s-bind-e1")), "storage_unavailable")


def test_epoch_precedes_replay(backend, scope_a):
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    binding = make_binding("s-bind-order", route="route-order")
    first = store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    current = begin(store_a, "instance-b")
    with pytest.raises(FenceStale):
        store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    assert store_a.bindings.reserve(epoch=current, binding=binding, owner=None).binding == first.binding


def test_unencodable_payload_is_invalid_params_even_with_a_stale_epoch(backend, scope_a):
    # CONTRACT: the receipt is encoded before the write, so encoding precedes the epoch fence.
    store_a = backend.open(scope_a)
    binding = make_binding("s-bind-nan", route="route-nan")
    poisoned = NewSessionBinding(**{**_kwargs(binding), "policy": {**dict(binding.policy), "budget": float("nan")}})
    assert_refusal(lambda: store_a.bindings.reserve(epoch=0, binding=poisoned, owner=None), "invalid_params")


# ── receipt lookup ────────────────────────────────────────────────────────────


def test_receipt_is_storage_unavailable_without_a_receipt(backend, scope_a):
    # CONTRACT: ``receipt`` refuses a missing receipt, including a row that exists without one.
    store_a = backend.open(scope_a)
    assert_refusal(lambda: store_a.bindings.receipt(SessionId("s-bind-absent")), "storage_unavailable")
    store_a.sessions.create(NewSession(session_id=SessionId("s-bind-plain"), source="cli"))
    assert_refusal(lambda: store_a.bindings.receipt(SessionId("s-bind-plain")), "storage_unavailable")


def test_corrupt_stored_receipt_is_storage_unavailable(backend, scope_a):
    # CONTRACT: an undecodable receipt, one that is not an object, or one missing a field refuses storage_unavailable, on read and on the replay. Seeded through the reference meta space.
    store_a = backend.open(scope_a)
    require_failpoints(backend)
    state = authority_state(backend, scope_a)
    for sid, value in (("s-c1", "{not json"), ("s-c2", "[1, 2]"), ("s-c3", {"session_id": "s-c3"})):
        with state.transaction() as staged:
            staged.authority["meta"]["binding:" + sid] = value
        assert_refusal(lambda sid=sid: store_a.bindings.receipt(SessionId(sid)), "storage_unavailable")
    assert_refusal(lambda: store_a.bindings.reserve(epoch=begin(store_a), binding=make_binding("s-c1"), owner=None), "storage_unavailable")


def test_bindings_and_epochs_are_scope_isolated(backend, scope_a, scope_b):
    store_a, store_b = backend.open(scope_a), backend.open(scope_b)
    epoch_a = begin(store_a)
    binding = make_binding("s-bind-iso", route="route-iso")
    store_a.bindings.reserve(epoch=epoch_a, binding=binding, owner=None)
    assert_refusal(lambda: store_b.bindings.receipt(SessionId("s-bind-iso")), "storage_unavailable")
    assert store_b.bindings.reserve(epoch=begin(store_b, "instance-b"), binding=binding, owner=None).created is True
    assert store_a.bindings.receipt(binding.session_id).route == "route-iso"


@pytest.mark.parametrize("window", [BETWEEN_SESSION_AND_ROUTE, BETWEEN_ROUTE_AND_RECEIPT, "publish"])
def test_reserve_crash_windows_publish_nothing_then_retry_exactly_once(backend, scope_a, window):
    store_a = backend.open(scope_a)
    require_failpoints(backend)
    failpoints = authority_state(backend, scope_a).failpoints
    epoch = begin(store_a)
    binding = make_binding("s-bind-crash", route="route-crash")
    failpoints.arm(window)
    with pytest.raises(InjectedFailure):
        store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    assert store_a.sessions.get(binding.session_id) is None
    assert "route-crash" not in store_a.routing.load_routing_entries()
    assert_refusal(lambda: store_a.bindings.receipt(binding.session_id), "storage_unavailable")
    retried = store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    assert retried.created is True
    assert store_a.sessions.get(binding.session_id) is not None
    assert "route-crash" in store_a.routing.load_routing_entries()
    assert store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None).created is False


def test_reservation_survives_reopen_and_replays(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "reopen_durability")
    epoch = begin(store_a)
    binding = make_binding("s-bind-durable", route="route-durable")
    first = store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    backend.reopen()
    assert store_a.bindings.receipt(binding.session_id) == first.binding
    assert store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None).binding == first.binding


# ── the legacy adoption arm ──


def _legacy(session_id: str, **overrides) -> NewSessionBinding:
    base = make_binding(session_id, route="route-" + session_id)
    return NewSessionBinding(**_kwargs(base), legacy_session_id=overrides.pop("legacy_session_id", session_id), lineage=(SessionId(session_id),))


def test_legacy_arm_binds_an_unrouted_local_row(backend, scope_a):
    # CONTRACT: the old row keeps its source and gains the route, chat id, principal and origin; a legacy binding record is stored and the receipt carries its lineage.
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=SessionId("old-1"), source="tui"))
    reserved = store_a.bindings.reserve(epoch=begin(store_a), binding=_legacy("old-1"), owner=None)
    assert (reserved.created, reserved.binding.legacy_session_id, reserved.binding.lineage) == (True, "old-1", ("old-1",))
    record = store_a.sessions.get(SessionId("old-1"))
    assert (record.source, record.session_key, record.chat_id, record.user_id) == ("tui", "route-old-1", "old-1", "user-1")
    assert store_a.bindings.binding_marks(SessionId("old-1")).legacy is True
    assert store_a.bindings.validate_lineage(reserved.binding) == "old-1"


def test_legacy_arm_refusals(backend, scope_a):
    # CONTRACT: the refusals in order: a legacy id that is not the session id, a missing row, a row that is not unrouted local history of this principal, a row with a compression child, a live turn lease.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.sessions.create(NewSession(session_id=SessionId("old-2"), source="tui"))
    assert_refusal(lambda: store_a.bindings.reserve(epoch=epoch, binding=_legacy("old-2", legacy_session_id="other"), owner=None), "storage_unavailable")
    assert_refusal(lambda: store_a.bindings.reserve(epoch=epoch, binding=_legacy("old-missing"), owner=None), "not_found")
    store_a.sessions.create(NewSession(session_id=SessionId("old-api"), source="api_server"))
    assert_refusal(lambda: store_a.bindings.reserve(epoch=epoch, binding=_legacy("old-api"), owner=None), "not_found")
    store_a.sessions.create(NewSession(session_id=SessionId("old-foreign"), source="cli", user_id="someone-else"))
    assert_refusal(lambda: store_a.bindings.reserve(epoch=epoch, binding=_legacy("old-foreign"), owner=None), "not_found")
    assert store_a.coordination.try_acquire_turn_lease(session_id=SessionId("old-2"), holder="live", ttl_seconds=300.0) is True
    # The transcript guards run before the arm's own lease check, so a live turn lease refuses as the guard's TurnLeaseLost.
    with pytest.raises(TurnLeaseLost):
        store_a.bindings.reserve(epoch=epoch, binding=_legacy("old-2"), owner=None)
    store_a.coordination.release_turn_lease(session_id=SessionId("old-2"), holder="live")
    assert store_a.sessions.get(SessionId("old-2")).session_key is None
    assert store_a.bindings.reserve(epoch=epoch, binding=_legacy("old-2"), owner=None).created is True
