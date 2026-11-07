"""``anon_auth.run_sign_in``: the one plain device-code sign-in every surface renders.

Driven directly against the same fake portal ``hermes auth upgrade`` is tested with, so the states,
the persistence rules, the cancel rules and the free tier's handoff are exercised on the real wire
rather than mocked away. Each test asserts a single ruled property of the flow.
"""

from __future__ import annotations

import contextlib
import ssl
import threading
import time

import pytest

from hermes_cli import anon_auth, auth_device_flow
from hermes_cli.auth import _auth_file_path, _load_auth_store
from tests.hermes_cli.test_anon_upgrade import (
    EMAIL, FREE_PICK, PORTAL, WELCOME, _write_model_config, free_account, portal)

__all__ = ["free_account", "portal"]


def _drain(**kwargs):
    """Run a sign-in to its end and return every state it yielded."""
    return list(anon_auth.run_sign_in(**kwargs))


def _seed_free_tier() -> dict:
    return anon_auth.ensure_portal_identity(explicit=True)


def _auth_bytes() -> bytes:
    path = _auth_file_path()
    return path.read_bytes() if path.exists() else b""


def test_a_completed_sign_in_yields_code_waiting_then_completed(portal, free_account):
    _seed_free_tier()
    _write_model_config({"provider": "nous", "default": anon_auth.GUEST_MODEL, "base_url": WELCOME})

    states = _drain()

    assert [s.kind for s in states] == ["code", "waiting", "completed"]
    code, completed = states[0], states[-1]
    assert code.link == f"{PORTAL}/device?user_code={portal.user_code}"
    assert code.code == portal.user_code
    assert completed.email == EMAIL
    assert completed.model == FREE_PICK
    assert completed.model_changed is True


def test_no_surface_calls_the_promotion_endpoints(portal, free_account):
    _seed_free_tier()

    _drain()

    assert not [p for _, p in portal.calls if "promotion" in p]


def test_the_email_comes_from_the_access_token_not_the_claim_answer(portal, free_account):
    _seed_free_tier()
    portal.claim_reply = {**portal.claim_reply, "account_email": "someone-else@example.test"}

    assert _drain()[-1].email == EMAIL


def test_the_handoff_result_rides_the_completed_state(portal, free_account):
    _seed_free_tier()
    portal.claim_reply = {**portal.claim_reply, "connectors": {
        "kind": "moved", "moved_toolkits": ["gmail"], "dropped_toolkits": ["slack"]}}

    completed = _drain()[-1]

    assert completed.handoff == (f"Your apps moved to {EMAIL}: Gmail.", f"Reconnect in {EMAIL}: Slack.")
    assert completed.copy.splitlines()[:3] == [f"Signed in as {EMAIL}.", *completed.handoff]


def test_no_identity_on_disk_signs_in_without_minting_or_claiming(portal, free_account, monkeypatch):
    monkeypatch.setattr(anon_auth, "mint_guest",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_sign_in must not mint")))

    states = _drain()

    assert states[-1].kind == "completed" and states[-1].handoff == ()
    assert portal.claims == []
    assert not anon_auth.is_guest_state(_load_auth_store()["providers"]["nous"])


def test_an_expired_code_yields_timed_out_and_keeps_the_enriched_detail(portal):
    _seed_free_tier()
    before = _auth_bytes()
    portal.token_pending, portal.expires_in = True, 1

    state = _drain()[-1]

    assert state.kind == "timed_out"
    assert state.copy == anon_auth.UPGRADE_TIMED_OUT
    assert state.detail == auth_device_flow._nous_device_auth_timeout_message(PORTAL)
    assert portal.token_grants == 0
    assert _auth_bytes() == before


def test_an_expired_token_answer_yields_timed_out(portal):
    _seed_free_tier()
    portal.token_error = "expired_token"

    state = _drain()[-1]

    assert state.kind == "timed_out" and state.copy == anon_auth.UPGRADE_TIMED_OUT
    assert portal.token_grants == 0


def test_the_portal_override_is_read_inside_the_profile_scope(portal, free_account, monkeypatch):
    in_scope = []

    @contextlib.contextmanager
    def _scope():
        in_scope.append(True)
        try:
            yield None
        finally:
            in_scope.pop()

    from hermes_cli import auth_nous
    reads = []
    real_override = auth_nous._nous_portal_env_override
    monkeypatch.setattr(auth_nous, "_nous_portal_env_override",
                        lambda: (reads.append(bool(in_scope)), real_override())[1])

    assert _drain(scope=_scope)[-1].kind == "completed"
    assert reads and all(reads[:1])


def test_the_portal_override_wins_over_a_saved_portal(portal, monkeypatch):
    guest = _seed_free_tier()
    from hermes_cli.auth import _save_auth_store, _save_provider_state
    store = _load_auth_store()
    _save_provider_state(store, "nous", {**guest, "portal_base_url": "https://portal.nousresearch.com"})
    _save_auth_store(store)
    asked = []

    def _record(client, portal_base_url, client_id, scope):
        asked.append(portal_base_url)
        raise RuntimeError("stop after the code request")
    monkeypatch.setattr(auth_device_flow, "_request_device_code", _record)

    _drain()

    assert asked == [PORTAL]


def test_a_transport_error_yields_failed_without_leaking_the_detail_into_chat_copy(portal, monkeypatch):
    _seed_free_tier()
    before = _auth_bytes()
    detail = "boom at https://portal.example.test/api/oauth/device/code"

    def _boom(*a, **kw):
        raise RuntimeError(detail)
    monkeypatch.setattr(auth_device_flow, "_request_device_code", _boom)

    state = _drain()[-1]

    assert state.kind == "failed"
    assert state.copy == anon_auth.UPGRADE_NOT_COMPLETED
    assert "https://" not in state.copy.lower()
    assert detail in state.copy_terminal
    assert _auth_bytes() == before


def test_a_persist_failure_yields_failed_rather_than_raising(portal, free_account, monkeypatch):
    _seed_free_tier()
    settles = []
    from hermes_cli import auth_nous
    monkeypatch.setattr(
        auth_nous, "persist_nous_credentials", lambda *a, **kw: (_ for _ in ()).throw(OSError("read-only home")))
    monkeypatch.setattr(anon_auth, "settle_after_upgrade", lambda state: settles.append(state) or {})

    states = _drain()

    assert states[-1].kind == "failed"
    assert states[-1].terminal is True
    assert "read-only home" in states[-1].copy_terminal
    assert settles == []


def test_a_settle_failure_yields_failed_rather_than_raising(portal, free_account, monkeypatch):
    _seed_free_tier()
    persists = []
    from hermes_cli import auth_nous
    real_persist = auth_nous.persist_nous_credentials
    monkeypatch.setattr(
        auth_nous, "persist_nous_credentials",
        lambda state, **kw: (persists.append(state), real_persist(state, **kw))[1])

    def _boom(state):
        raise RuntimeError("config is read-only")
    monkeypatch.setattr(anon_auth, "settle_after_upgrade", _boom)

    states = _drain()

    assert states[-1].kind == "failed"
    assert "config is read-only" in states[-1].copy_terminal
    assert len(persists) == 1


def test_already_signed_in_short_circuits_before_any_network_unless_replacing(portal):
    from hermes_cli.auth import _save_auth_store, _save_provider_state
    store = _load_auth_store()
    _save_provider_state(store, "nous", {"auth_method": "oauth_device_code", "access_token": "x"})
    _save_auth_store(store)
    portal.calls.clear()

    states = _drain()

    assert [s.kind for s in states] == ["already_signed_in"]
    assert states[0].ok is True and states[0].precondition is True
    assert portal.calls == []
    gen = anon_auth.run_sign_in(replace_account=True)
    assert next(gen).kind == "code"
    gen.close()


def test_cancelling_before_the_wait_persists_nothing(portal):
    _seed_free_tier()
    before = _auth_bytes()
    stop = []

    # Driven by hand so the flag flips exactly between the code and the wait.
    gen = anon_auth.run_sign_in(cancelled=lambda: bool(stop))
    assert next(gen).kind == "code"
    stop.append(True)

    assert [s.kind for s in gen] == ["cancelled"]
    assert portal.token_polls == 0
    assert _auth_bytes() == before


def test_cancelling_during_the_token_poll_stops_it_within_a_second(portal):
    _seed_free_tier()
    before = _auth_bytes()
    portal.token_pending = True
    stop = threading.Event()

    gen = anon_auth.run_sign_in(cancelled=stop.is_set)
    assert next(gen).kind == "code"
    timer = threading.Timer(0.2, stop.set)
    timer.start()
    try:
        started = time.monotonic()
        rest = [s.kind for s in gen]
        elapsed = time.monotonic() - started
    finally:
        timer.cancel()
    polls = portal.token_polls

    assert rest == ["waiting", "cancelled"]
    assert elapsed < 3.0
    assert polls >= 1 and portal.token_polls == polls     # the generator is done: nothing polls on
    assert _auth_bytes() == before


def test_a_cancel_that_lands_with_the_token_grant_still_saves_nothing(portal, free_account):
    _seed_free_tier()
    before = _auth_bytes()
    stop = threading.Event()
    portal.on_token = stop.set

    states = _drain(cancelled=stop.is_set)

    assert states[-1].kind == "cancelled"
    assert portal.token_grants == 1
    assert _auth_bytes() == before
    assert portal.claims == []


def test_a_persist_guard_that_refuses_persists_nothing_and_never_settles(portal, free_account, monkeypatch):
    _seed_free_tier()
    before = _auth_bytes()
    settles = []
    monkeypatch.setattr(anon_auth, "settle_after_upgrade", lambda state: settles.append(state) or {})

    @contextlib.contextmanager
    def _refuse():
        yield False

    states = _drain(persist_guard=_refuse)

    assert states[-1].kind == "cancelled"
    assert settles == []
    assert _auth_bytes() == before


def test_persistence_happens_only_after_the_token_grant(portal, free_account):
    _seed_free_tier()
    before = _auth_bytes()
    seen = {state.kind: _auth_bytes() for state in anon_auth.run_sign_in()}

    assert seen["code"] == before
    assert seen["waiting"] == before
    assert seen["completed"] != before


def test_a_token_without_a_scope_keeps_the_requested_scope(portal, free_account, monkeypatch):
    from hermes_cli import auth_nous
    from hermes_cli.auth import PROVIDER_REGISTRY
    real_poll = auth_device_flow._poll_for_token
    monkeypatch.setattr(auth_device_flow, "_poll_for_token",
                        lambda **kw: {k: v for k, v in real_poll(**kw).items() if k != "scope"})
    saved = []
    real_persist = auth_nous.persist_nous_credentials
    monkeypatch.setattr(auth_nous, "persist_nous_credentials",
                        lambda state, **kw: (saved.append(state), real_persist(state, **kw))[1])

    assert _drain()[-1].kind == "completed"
    assert saved[0]["scope"] == PROVIDER_REGISTRY["nous"].scope


def test_the_ca_bundle_is_honoured_for_the_whole_sign_in(portal, free_account, monkeypatch):
    import certifi
    monkeypatch.setenv("HERMES_CA_BUNDLE", certifi.where())
    seen = []
    from hermes_cli import auth_nous
    real_client = auth_nous._nous_http_client

    def _recording(timeout_seconds, verify):
        seen.append(verify)
        return real_client(timeout_seconds, verify)

    assert _drain(client_factory=_recording)[-1].kind == "completed"
    assert seen and isinstance(seen[0], ssl.SSLContext)


def test_the_scope_is_entered_for_the_precondition_and_the_persist_but_never_around_a_wait(
        portal, free_account, monkeypatch):
    _seed_free_tier()
    events = []

    @contextlib.contextmanager
    def _scope():
        events.append(("enter", threading.get_ident()))
        try:
            yield None
        finally:
            events.append(("exit", threading.get_ident()))

    real_poll = auth_device_flow._poll_for_token

    def _poll(**kwargs):
        events.append(("poll", threading.get_ident()))
        return real_poll(**kwargs)
    monkeypatch.setattr(auth_device_flow, "_poll_for_token", _poll)

    states = []
    for state in anon_auth.run_sign_in(scope=_scope):
        events.append(("yield", threading.get_ident()))
        states.append(state)

    assert states[-1].kind == "completed"
    kinds = [kind for kind, _ in events]
    assert kinds == ["enter", "exit", "yield", "yield", "poll", "enter", "exit", "yield"]
