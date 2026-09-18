"""``anon_auth.run_sign_in``: the one sign-in composition every surface renders.

Driven directly against the same fake account service ``hermes auth upgrade`` is tested with, so the
states, the persistence rules and the cancellation rules are exercised on the real wire rather than
mocked away. Each test asserts a single ruled property of the flow.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from hermes_cli import anon_auth
from hermes_cli.auth import _auth_file_path
from hermes_cli.auth_constants import AuthError
from tests.hermes_cli.test_anon_upgrade import (  # noqa: F401  (fixtures used by name)
    EMAIL, FREE_PICK, PORTAL, WELCOME, _shared_store, _write_model_config, free_account, portal)

__all__ = ["free_account", "portal"]


def _drain(**kwargs):
    """Run a sign-in to its end and return every state it yielded."""
    return list(anon_auth.run_sign_in(**kwargs))


def _seed_free_tier() -> dict:
    return anon_auth.ensure_portal_identity(explicit=True)


def _stub_wait(monkeypatch, outcome, *, before=None):
    """Replace the promotion wait with one that returns *outcome* (running *before* first)."""
    def _wait(client, portal_base_url, claim_code, *, expires_in, interval, cancelled=None):
        if before is not None:
            before()
        return dict(outcome)
    monkeypatch.setattr(anon_auth, "wait_for_promotion", _wait)


def _voided(reason: str) -> dict:
    return {"status": "voided", "reason": reason}


def test_a_completed_sign_in_yields_code_waiting_then_completed(portal, free_account):
    _seed_free_tier()
    _write_model_config({"provider": "nous", "default": anon_auth.GUEST_MODEL, "base_url": WELCOME})

    states = _drain()

    assert [s.kind for s in states] == ["code", "waiting", "completed"]
    code, completed = states[0], states[-1]
    assert code.link.startswith(PORTAL)
    assert code.code == "clm_1"
    assert completed.email == EMAIL
    assert completed.model == FREE_PICK
    assert completed.model_changed is True


def test_declined_yields_declined_and_persists_nothing(portal, tmp_path):
    _seed_free_tier()
    before = _auth_file_path().read_bytes()
    shared_before = _shared_store(tmp_path)
    portal.status_sequence = [{"status": "pending"}, _voided("user_declined")]

    terminal = [s for s in _drain() if s.terminal]

    assert len(terminal) == 1
    assert terminal[0].kind == "declined"
    assert terminal[0].copy == anon_auth.UPGRADE_REASON_COPY["user_declined"]
    assert portal.token_grants == 0
    assert _auth_file_path().read_bytes() == before
    assert _shared_store(tmp_path) == shared_before


def test_a_timeout_yields_timed_out_and_keeps_the_enriched_detail(portal, monkeypatch):
    _seed_free_tier()
    _stub_wait(monkeypatch, {"status": "timeout"})

    state = _drain()[-1]
    assert state.kind == "timed_out"
    assert state.copy == anon_auth.UPGRADE_TIMED_OUT
    assert portal.token_grants == 0

    # The token poll can time out too, and its guidance is enriched at the source.
    from hermes_cli import auth_device_flow
    enriched = auth_device_flow._nous_device_auth_timeout_message(PORTAL)
    _stub_wait(monkeypatch, {"status": "completed", "account_email": EMAIL})

    def _timeout(**kwargs):
        raise TimeoutError(enriched)
    monkeypatch.setattr(auth_device_flow, "_poll_for_token", _timeout)

    state = _drain()[-1]
    assert state.kind == "timed_out"
    assert state.detail == enriched
    assert state.copy == anon_auth.UPGRADE_TIMED_OUT
    assert portal.token_grants == 0


def test_a_retired_identity_yields_retired_and_clears_the_free_tier(portal, monkeypatch):
    guest = _seed_free_tier()
    cleared = []
    real_clear = anon_auth.clear_dead_guest

    def _spy(reason, *, dead_token=None):
        cleared.append((reason, dead_token))
        real_clear(reason, dead_token=dead_token)
    monkeypatch.setattr(anon_auth, "clear_dead_guest", _spy)
    portal.status_sequence = [_voided("account_not_anonymous")]

    state = _drain()[-1]

    assert state.kind == "retired"
    assert state.copy == anon_auth.UPGRADE_REASON_COPY["account_retired"]
    assert cleared == [("retired", guest["anon_token"])]
    from hermes_cli.auth import _load_auth_store
    assert "nous" not in _load_auth_store().get("providers", {})


def test_a_server_superseded_outcome_yields_superseded(portal, tmp_path):
    _seed_free_tier()
    before = _auth_file_path().read_bytes()
    portal.status_sequence = [_voided("superseded")]

    state = _drain()[-1]

    assert state.kind == "superseded"
    assert state.copy == anon_auth.UPGRADE_REASON_COPY["superseded"]
    assert portal.token_grants == 0
    assert _auth_file_path().read_bytes() == before


def test_a_retired_identity_yields_retired_even_when_cleanup_fails(portal, monkeypatch):
    _seed_free_tier()

    def _retired(*args, **kwargs):
        raise anon_auth.AnonCredentialDead("retired")

    def _read_only(*args, **kwargs):
        raise OSError("read-only store")

    monkeypatch.setattr(anon_auth, "register_promotion_intent", _retired)
    monkeypatch.setattr(anon_auth, "clear_dead_guest", _read_only)

    states = _drain()

    assert [s.kind for s in states] == ["retired"]


@pytest.mark.parametrize(
    "reason,expected",
    [("account_busy", anon_auth.UPGRADE_REASON_COPY["account_busy"]),
     ("wat", anon_auth.UPGRADE_NOT_COMPLETED)])
def test_an_unknown_reason_yields_failed_with_the_generic_copy(portal, reason, expected):
    _seed_free_tier()
    portal.status_sequence = [_voided(reason)]

    state = _drain()[-1]

    assert state.kind == "failed"
    assert state.copy == expected


def test_a_transport_error_yields_failed_without_leaking_the_detail_into_chat_copy(
        portal, monkeypatch, tmp_path):
    _seed_free_tier()
    before = _auth_file_path().read_bytes()
    detail = "boom at https://portal.example.test/api/anonymous/promotion-intent"

    def _boom(*a, **kw):
        raise RuntimeError(detail)
    monkeypatch.setattr(anon_auth, "register_promotion_intent", _boom)

    state = _drain()[-1]

    assert state.kind == "failed"
    assert state.copy == anon_auth.UPGRADE_NOT_COMPLETED
    lowered = state.copy.lower()
    for banned in ("http://", "https://", "/anonymous/", "anonymous"):
        assert banned not in lowered
    assert state.copy_terminal == f"Sign-in failed: {detail}"
    assert portal.token_grants == 0
    assert _auth_file_path().read_bytes() == before


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


def test_already_signed_in_short_circuits_before_any_network(portal):
    from hermes_cli.auth import _load_auth_store, _save_auth_store, _save_provider_state
    store = _load_auth_store()
    _save_provider_state(store, "nous", {"auth_method": "oauth_device_code", "access_token": "x"})
    _save_auth_store(store)
    portal.calls.clear()

    states = _drain()

    assert [s.kind for s in states] == ["already_signed_in"]
    assert states[0].ok is True
    assert states[0].precondition is True
    assert portal.calls == []


def test_free_tier_off_yields_unavailable(portal, monkeypatch):
    monkeypatch.setattr(anon_auth, "guest_enabled", lambda: False)
    portal.calls.clear()

    states = _drain()

    assert [s.kind for s in states] == ["unavailable"]
    assert states[0].copy == anon_auth.UPGRADE_UNAVAILABLE_CHAT
    assert states[0].copy_terminal == anon_auth.UPGRADE_UNAVAILABLE
    assert portal.calls == []


def test_no_identity_on_disk_yields_unavailable_without_touching_the_portal(portal, monkeypatch):
    """A sign-in never creates the identity it signs in from: the boot bootstrap is the only creator.
    With nothing on disk the flow yields ``Unavailable`` and makes no portal call and no mint attempt."""
    monkeypatch.setattr(anon_auth, "ensure_portal_identity",
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_sign_in must not mint")))
    monkeypatch.setattr(anon_auth, "mint_guest",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_sign_in must not mint")))

    states = _drain()

    assert [s.kind for s in states] == ["unavailable"]
    assert states[-1].copy_terminal == anon_auth.UPGRADE_UNAVAILABLE
    assert states[-1].copy == anon_auth.UPGRADE_UNAVAILABLE_CHAT
    assert portal.calls == []


def test_cancelling_before_the_wait_persists_nothing(portal, tmp_path):
    _seed_free_tier()
    before = _auth_file_path().read_bytes()
    stop = []

    # Driven by hand so the flag flips exactly between the code and the wait.
    gen = anon_auth.run_sign_in(cancelled=lambda: bool(stop))
    first = next(gen)
    assert first.kind == "code"
    stop.append(True)
    rest = list(gen)

    assert [s.kind for s in rest] == ["superseded"]
    assert portal.token_grants == 0
    assert _auth_file_path().read_bytes() == before


def test_cancelling_during_the_wait_ends_it_within_a_second(portal, tmp_path):
    _seed_free_tier()
    before = _auth_file_path().read_bytes()
    portal.status_sequence = [{"status": "pending"}]
    stop = threading.Event()
    outcomes = []
    real_wait = anon_auth.wait_for_promotion

    def _recording_wait(*args, **kwargs):
        outcome = real_wait(*args, **kwargs)
        outcomes.append(outcome)
        return outcome

    gen = anon_auth.run_sign_in(cancelled=stop.is_set)
    first = next(gen)
    assert first.kind == "code"
    anon_auth.wait_for_promotion = _recording_wait
    try:
        timer = threading.Timer(0.2, stop.set)
        timer.start()
        started = time.monotonic()
        rest = list(gen)
        elapsed = time.monotonic() - started
    finally:
        anon_auth.wait_for_promotion = real_wait
        timer.cancel()

    assert elapsed < 3.0
    assert outcomes == [{"status": "cancelled"}]
    assert [s.kind for s in rest] == ["waiting", "superseded"]
    assert portal.token_grants == 0
    assert _auth_file_path().read_bytes() == before


@pytest.mark.parametrize("cancel_wins", [False, True])
def test_cancelling_during_a_completed_status_request_obeys_the_surface_policy(
        portal, free_account, cancel_wins):
    _seed_free_tier()
    calls = 0

    def _cancelled():
        nonlocal calls
        calls += 1
        return calls > 2

    states = _drain(cancelled=_cancelled, cancel_wins_after_promotion=cancel_wins)

    assert states[-1].kind == ("superseded" if cancel_wins else "completed")
    assert portal.token_grants == (0 if cancel_wins else 1)
    from hermes_cli.auth import _load_auth_store
    state = _load_auth_store()["providers"]["nous"]
    assert anon_auth.is_guest_state(state) is cancel_wins


def _cancel_after_a_completed_promotion(portal, monkeypatch, *, cancel_wins: bool):
    _seed_free_tier()
    stop = threading.Event()

    def _wait(client, portal_base_url, claim_code, *, expires_in, interval, cancelled=None):
        stop.set()   # the surface cancels while this call is blocked
        return {"status": "completed", "user_id": "nas_user:9", "account_email": EMAIL}
    monkeypatch.setattr(anon_auth, "wait_for_promotion", _wait)
    return list(anon_auth.run_sign_in(
        cancelled=stop.is_set, cancel_wins_after_promotion=cancel_wins))


def test_a_desktop_style_cancel_after_a_completed_promotion_persists_nothing(
        portal, free_account, monkeypatch, tmp_path):
    _seed_free_tier()
    before = _auth_file_path().read_bytes()

    states = _cancel_after_a_completed_promotion(portal, monkeypatch, cancel_wins=True)

    assert states[-1].kind == "superseded"
    assert portal.token_grants == 0
    assert _auth_file_path().read_bytes() == before


def test_a_gateway_style_supersede_after_a_completed_promotion_still_signs_in(
        portal, free_account, monkeypatch):
    states = _cancel_after_a_completed_promotion(portal, monkeypatch, cancel_wins=False)

    assert states[-1].kind == "completed"
    assert portal.token_grants == 1
    from hermes_cli.auth import _load_auth_store
    state = _load_auth_store()["providers"]["nous"]
    assert not anon_auth.is_guest_state(state)


def test_a_persist_guard_that_refuses_persists_nothing_and_never_settles(
        portal, free_account, monkeypatch, tmp_path):
    import contextlib
    _seed_free_tier()
    before = _auth_file_path().read_bytes()
    settles = []
    monkeypatch.setattr(anon_auth, "settle_after_upgrade", lambda state: settles.append(state) or {})

    @contextlib.contextmanager
    def _refuse():
        yield False

    states = list(anon_auth.run_sign_in(persist_guard=_refuse))

    assert states[-1].kind == "superseded"
    assert settles == []
    assert _auth_file_path().read_bytes() == before


def test_settle_runs_exactly_once_per_completion(portal, free_account, monkeypatch):
    _seed_free_tier()
    settles, persists = [], []
    real_settle = anon_auth.settle_after_upgrade
    from hermes_cli import auth_nous
    real_persist = auth_nous.persist_nous_credentials
    monkeypatch.setattr(
        anon_auth, "settle_after_upgrade",
        lambda state: (settles.append(state), real_settle(state))[1])
    monkeypatch.setattr(
        auth_nous, "persist_nous_credentials",
        lambda state, **kw: (persists.append(state), real_persist(state, **kw))[1])

    states = list(anon_auth.run_sign_in())

    assert states[-1].kind == "completed"
    assert len(settles) == 1
    assert len(persists) == 1


def test_persistence_happens_only_after_the_promotion_and_the_token_grant(portal, free_account):
    _seed_free_tier()
    before = _auth_file_path().read_bytes()
    seen = {}
    for state in anon_auth.run_sign_in():
        seen[state.kind] = _auth_file_path().read_bytes()

    assert seen["code"] == before
    assert seen["waiting"] == before
    assert seen["completed"] != before
    paths = [p for _, p in portal.calls]
    assert "/api/oauth/token" in paths


def test_the_budget_shrinks_across_the_two_waits(portal, free_account, monkeypatch):
    """One absolute deadline: what the promotion wait spends, the token poll no longer has."""
    _seed_free_tier()
    captured = {}
    from hermes_cli import auth_device_flow
    real_request, real_poll = auth_device_flow._request_device_code, auth_device_flow._poll_for_token

    def _short_lived_code(client, portal_base_url, client_id, scope):
        return {**real_request(client, portal_base_url, client_id, scope), "expires_in": 10}
    monkeypatch.setattr(auth_device_flow, "_request_device_code", _short_lived_code)

    def _wait(client, portal_base_url, claim_code, *, expires_in, interval, cancelled=None):
        time.sleep(3.0)
        return {"status": "completed", "account_email": EMAIL}
    monkeypatch.setattr(anon_auth, "wait_for_promotion", _wait)

    def _poll(**kwargs):
        captured.update(kwargs)
        return real_poll(**kwargs)
    monkeypatch.setattr(auth_device_flow, "_poll_for_token", _poll)

    states = list(anon_auth.run_sign_in())

    assert states[-1].kind == "completed"
    assert 1 <= captured["expires_in"] <= 7


def test_the_scope_is_entered_for_the_preconditions_and_the_persist_but_never_around_a_wait(
        portal, free_account, monkeypatch):
    import contextlib
    _seed_free_tier()
    events = []

    @contextlib.contextmanager
    def _scope():
        events.append(("enter", time.monotonic(), threading.get_ident()))
        try:
            yield None
        finally:
            events.append(("exit", time.monotonic(), threading.get_ident()))

    def _wait(client, portal_base_url, claim_code, *, expires_in, interval, cancelled=None):
        events.append(("wait-start", time.monotonic(), threading.get_ident()))
        time.sleep(0.2)
        events.append(("wait-end", time.monotonic(), threading.get_ident()))
        return {"status": "completed", "account_email": EMAIL}
    monkeypatch.setattr(anon_auth, "wait_for_promotion", _wait)

    states = []
    yields = []
    for state in anon_auth.run_sign_in(scope=_scope):
        yields.append(time.monotonic())
        states.append(state)

    assert states[-1].kind == "completed"
    pairs = [e for e in events if e[0] in ("enter", "exit")]
    assert [e[0] for e in pairs] == ["enter", "exit", "enter", "exit"]
    for first, second in (pairs[0:2], pairs[2:4]):
        assert first[2] == second[2]      # one thread per scope
    wait_start = next(e[1] for e in events if e[0] == "wait-start")
    wait_end = next(e[1] for e in events if e[0] == "wait-end")
    for first, second in (pairs[0:2], pairs[2:4]):
        assert not (first[1] <= wait_start and wait_end <= second[1])
        # and never held across a yield, which would hand the scope to the consumer's thread
        assert not any(first[1] <= at <= second[1] for at in yields)


def test_wait_for_promotion_without_a_cancel_hook_is_unchanged(portal):
    import httpx
    _seed_free_tier()
    portal.status_sequence = [{"status": "completed", "account_email": EMAIL}]
    portal.calls.clear()
    with httpx.Client(transport=httpx.MockTransport(portal.handler), base_url=PORTAL) as client:
        outcome = anon_auth.wait_for_promotion(
            client, PORTAL, "clm_1", expires_in=30, interval=0)

    assert outcome == {"status": "completed", "account_email": EMAIL}
    assert [p for _, p in portal.calls] == ["/api/anonymous/promotion-status"]
