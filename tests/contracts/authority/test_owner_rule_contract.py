"""Contract cases for the owner rule: the pure rule in ``session_rules.legacy_local`` over an ``OwnerRule``, the authority operations that take one (``authorize_history``, the ``reserve`` legacy arm, ``legacy_adoption_scan``, ``mutate``'s history claim and delete walk, ``delete_footprint``), and that every operation reading a stored owner takes the rule from its caller.

The rule's host data arrives as an ``OwnerRule``; without one only an equal stamp or the subject's own login label is its owner and no label is a client app's.
"""

from __future__ import annotations

import dataclasses
import inspect
import json

import pytest

from persistence import AuthorityRefusal, ProofRejected, SessionId
from persistence.authority import runtime as rt
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.sessions import NewSession
from persistence.gateway.hosted_rooms import HostedOwnerClaim
from session_rules.legacy_local import (
    binding_names, bindings_agree, client_label, legacy_creation_source, legacy_history_row, not_client_labels, owners_agree,
    same_owner, validate_legacy_row,
)
from tests.contracts.authority.test_bindings_contract import _legacy, begin

OWNER = "uid:501"
STRANGER = "uid:999999"
WEBAPP_LOGIN = 'auth:v1:["basic","","admin"]'
RULE = OwnerRule(host_subject=OWNER, login_providers=frozenset({"basic", "nous", "self-hosted", "session-token", "server-internal"}),
                 messaging_labels=frozenset({"telegram", "discord", "teams"}))


@pytest.mark.parametrize("stamp, owned", [
    ("basic:admin", True), (f"local:{OWNER}", True), ('auth:v1:["self-hosted","https://idp","sub"]', True),
    ('auth:v1:["local","","' + OWNER + '"]', True),
    ("local:uid:999999", False), ('auth:v1:["local","","uid:999999"]', False),
    ("telegram:42", False), ("uid:999999", False), ("nous:account", False), ('auth:v1:["","","x"]', False),
])
def test_only_sign_in_stamps_and_this_machines_account_read_as_the_owners(stamp, owned):
    # CONTRACT: while the subject is the host owner, a sign-in stamp or a local stamp of this machine's account is its; any other subject owns only its own stamp.
    assert same_owner(stamp, OWNER, RULE) is owned
    assert same_owner(stamp, STRANGER, RULE) is (stamp == STRANGER)


def test_without_a_rule_only_the_stamp_itself_or_the_subjects_login_label_is_the_owners():
    # CONTRACT: no OwnerRule grants no sign-in stamp; a subject's own login label still names it.
    assert same_owner("basic:admin", OWNER, None) is False
    assert same_owner(OWNER, OWNER, None) is True
    assert same_owner("github:alice", 'auth:v1:["github","","alice"]', None) is True


def test_bindings_and_stored_stamps_compare_through_the_rule():
    # CONTRACT: a binding names a subject when its owner half is the subject's under the rule; two stored stamps agree either way round; two bindings agree when both are absent or name one owner.
    webapp_binding, owner_binding = json.dumps(["default", WEBAPP_LOGIN]), json.dumps(["default", OWNER])
    assert binding_names(webapp_binding, OWNER, RULE) is True
    assert binding_names(webapp_binding, STRANGER, RULE) is False
    assert binding_names("not json", OWNER, RULE) is False
    assert binding_names(json.dumps([None, OWNER]), OWNER, RULE) is False
    assert owners_agree("basic:admin", OWNER, RULE) and owners_agree(OWNER, "basic:admin", RULE)
    assert not owners_agree("telegram:42", OWNER, RULE)
    assert bindings_agree(None, None, RULE) and not bindings_agree(None, owner_binding, RULE)
    assert bindings_agree(webapp_binding, owner_binding, RULE) and not bindings_agree(webapp_binding, owner_binding, None)
    assert not bindings_agree(owner_binding, "not json", RULE)


def test_a_binding_follows_its_store_after_the_home_moved():
    # CONTRACT: the profile half of a stored binding is the home path that wrote it into this store, so it decides nothing: a binding written under an older home still names its owner and agrees with one written under the new home; only the owner is compared.
    departed, moved = json.dumps(["/mac/home", OWNER]), json.dumps(["/cloud", OWNER])
    assert binding_names(departed, OWNER, RULE) is True
    assert bindings_agree(departed, moved, RULE) is True
    assert binding_names(departed, STRANGER, RULE) is False
    assert bindings_agree(departed, json.dumps(["/cloud", STRANGER]), RULE) is False


def test_a_client_apps_label_is_neither_a_surface_a_platform_nor_automation():
    # CONTRACT: a client label is a non-empty label outside the local surfaces, the refused labels and the rule's messaging labels; a routeless row under one is legacy history resumed as gui.
    assert {"cli", "desktop", "cron", "webui", "telegram", "teams"} <= not_client_labels(RULE)
    assert [client_label(label, RULE) for label in ("mobile", "ios", "telegram", "cron", "cli", "", None)] == [True, True, False, False, False, False, False]
    assert client_label("mobile", None) is False
    assert legacy_history_row({"source": "mobile"}, RULE) is True
    assert legacy_history_row({"source": "mobile", "session_key": "agent:main:mobile:dm:1"}, RULE) is False
    assert legacy_history_row({"source": "mobile"}, None) is False
    assert [legacy_creation_source({"source": source}, RULE) for source in ("mobile", "desktop", "cli")] == ["gui", "gui", "cli"]


def test_validate_legacy_row_reads_owner_and_label_through_the_rule():
    # CONTRACT: a client app's routeless row is adoptable under the rule; a sign-in stamp adopts only for the host owner; no rule keeps both strict.
    validate_legacy_row("default", {"source": "mobile"}, subject=OWNER, rule=RULE)
    validate_legacy_row("default", {"source": "cli", "user_id": "basic:admin"}, subject=OWNER, rule=RULE)
    for row, subject, rule in (({"source": "mobile"}, OWNER, None), ({"source": "cli", "user_id": "basic:admin"}, STRANGER, RULE),
                               ({"source": "cli", "user_id": "basic:admin"}, OWNER, None), ({"source": "telegram"}, OWNER, RULE)):
        with pytest.raises(AuthorityRefusal, match="^not_found$"):
            validate_legacy_row("default", row, subject=subject, rule=rule)



@pytest.mark.parametrize("repository, operation", [
    ("bindings", "reserve"), ("bindings", "authorize_history"), ("runtime", "mutate"), ("runtime", "delete_footprint"),
    ("worker_executions", "reserve_for_admission"), ("hosted_rooms", "claim_owner"), ("sessions", "roster_sessions"),
])
def test_every_operation_that_reads_a_stored_owner_takes_the_rule_from_its_caller(backend, scope_a, repository, operation):
    # CONTRACT: ``owner`` has no default: each caller passes the rule, or None for the strict reading.
    bound = getattr(getattr(backend.open(scope_a), repository), operation)
    assert inspect.signature(bound).parameters["owner"].default is inspect.Parameter.empty
    assert next(field for field in dataclasses.fields(HostedOwnerClaim) if field.name == "rule").default is dataclasses.MISSING


# ── the operations ──────────────────────────────────────────────────────


def _authorize(store, sid: str, subject: str = OWNER, owner: OwnerRule | None = RULE, native_transport=None) -> bool:
    return store.bindings.authorize_history(session_id=SessionId(sid), profile_id="default", subject=subject, may_create=True,
                                            native_transport=native_transport, owner=owner)


def test_authorize_history_reads_stamps_bindings_and_client_rows_through_the_rule(backend, scope_a):
    store = backend.open(scope_a)
    epoch = begin(store)
    store.sessions.create(NewSession(session_id=SessionId("o-login"), source="cli", user_id="basic:admin"))
    store.sessions.create(NewSession(session_id=SessionId("o-phone"), source="mobile"))
    writer = rt.HistoryActor("default", WEBAPP_LOGIN, None, True)
    store.runtime.mutate(epoch, principal_id="p", session_id="o-imp", request_id="imp", expected_revision=0,
                         command=rt.Import(({"id": "o-imp", "source": "webui", "messages": []},)), history=rt.ImportControl(writer, ("o-imp",)), owner=None)
    # CONTRACT: a row stamped with a sign-in and a binding a Webapp login stored are the host owner's under the rule, so the claim check passes them without the registry's answer; a client app's routeless row is history under the rule.
    assert [_authorize(store, sid, native_transport=False) for sid in ("o-login", "o-imp")] == [True, True]
    assert _authorize(store, "o-phone") is True
    for sid in ("o-login", "o-imp"):
        with pytest.raises(ProofRejected):
            _authorize(store, sid, subject=STRANGER)
        with pytest.raises(ProofRejected):
            _authorize(store, sid, owner=None)
    assert _authorize(store, "o-phone", owner=None) is False


def test_the_legacy_arm_adopts_a_client_apps_chat_as_desktop_under_the_rule(backend, scope_a):
    store = backend.open(scope_a)
    epoch = begin(store)
    store.sessions.create(NewSession(session_id=SessionId("old-phone"), source="mobile"))
    store.sessions.create(NewSession(session_id=SessionId("old-login"), source="cli", user_id="basic:admin"))
    owned = dataclasses.replace(_legacy("old-phone"), principal_id=OWNER)
    # CONTRACT: without the rule a client app's row is no legacy history (not_found); with it the row binds and is relabelled desktop.
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store.bindings.reserve(epoch=epoch, binding=owned, owner=None)
    assert store.bindings.reserve(epoch=epoch, binding=owned, owner=RULE).created is True
    record = store.sessions.get(SessionId("old-phone"))
    assert (record.source, record.user_id) == ("desktop", OWNER)
    login = dataclasses.replace(_legacy("old-login"), principal_id=OWNER)
    # CONTRACT: a sign-in stamp is the host owner's under the rule; the row keeps its source.
    assert store.bindings.reserve(epoch=epoch, binding=login, owner=RULE).created is True
    assert store.sessions.get(SessionId("old-login")).source == "cli"


def test_the_adoption_scan_takes_routeless_client_rows_under_the_rule(backend, scope_a):
    store = backend.open(scope_a)
    scan = store.bindings.legacy_adoption_scan
    store.maintenance.import_sessions([{"id": sid, "source": source, "started_at": started, "messages": []} for sid, source, started in (
        ("a-cli", "cli", 100.0), ("b-phone", "mobile", 200.0), ("c-cron", "cron", 50.0), ("d-tg", "telegram", 150.0))])
    store.sessions.create(NewSession(session_id=SessionId("e-routed"), source="mobile", session_key="agent:main:mobile:dm:1"))
    # CONTRACT: the candidates are the local sources and every routeless row outside the refused labels.
    assert scan(RULE).session_ids == ("a-cli", "b-phone")


def _delete(store, epoch, sid, request_id, owner=None, **kwargs):
    row = store.sessions.get(SessionId(sid))
    return store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id=request_id, expected_revision=row.runtime_revision,
                                expected_generation=row.runtime_generation, command=rt.Delete(), owner=owner, **kwargs)


def _stamped_chain(store, root: str, tip: str) -> None:
    store.sessions.create(NewSession(session_id=SessionId(root), source="cli", user_id=OWNER))
    store.sessions.create(NewSession(session_id=SessionId(tip), source="cli", user_id="basic:admin", parent_session_id=SessionId(root)))
    store.sessions.end(SessionId(root), "compression")


def test_the_delete_walk_follows_a_member_whose_stamp_agrees_under_the_rule(backend, scope_a):
    store = backend.open(scope_a)
    footprint = store.runtime.delete_footprint
    epoch = begin(store)
    _stamped_chain(store, "w-root", "w-tip")
    # CONTRACT: a routeless member whose owner stamp agrees with the anchor's under the rule is the anchor's history; without the rule it is not.
    assert tuple(footprint(SessionId("w-root"), owner=RULE).targets) == ("w-root", "w-tip")
    assert tuple(footprint(SessionId("w-root"), owner=None).targets) == ("w-root",)
    _delete(store, epoch, "w-root", "del", owner=RULE)
    assert store.sessions.get(SessionId("w-tip")) is None


def test_a_history_claim_reads_the_rows_stamp_through_the_rule(backend, scope_a):
    store = backend.open(scope_a)
    epoch = begin(store)
    store.sessions.create(NewSession(session_id=SessionId("c-login"), source="cli", user_id="basic:admin"))
    claim = rt.ClaimHistory(rt.HistoryActor("default", OWNER, None, True))
    # CONTRACT: a claim of a row stamped with a sign-in refuses without the rule and lands for the host owner with it.
    with pytest.raises(ProofRejected):
        store.runtime.mutate(epoch, principal_id="p", session_id="c-login", request_id="claim", expected_revision=0, command=rt.Rename("t"), history=claim, owner=None)
    store.runtime.mutate(epoch, principal_id="p", session_id="c-login", request_id="claim", expected_revision=0, command=rt.Rename("t"), history=claim, owner=RULE)
    assert store.sessions.get(SessionId("c-login")).title == "t"
