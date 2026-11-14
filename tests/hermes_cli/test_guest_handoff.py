"""The guest handoff step: a Nous account replacing the free tier moves the free tier's apps into it.

Driven end to end through Hermes' real writers, stores and HTTP client against a fake account
service (``anon_portal.FakePortal`` plus the claim and token-refresh routes).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

from tests.hermes_cli.anon_portal import PORTAL, FakePortal, install_portal, make_jwt

DESTINATION = "user_acct_1"


def _account(sub: str = DESTINATION, refresh: str = "rt_1") -> dict:
    expires = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    return {
        "auth_method": "oauth_device_code", "portal_base_url": PORTAL, "client_id": "hermes-cli",
        "inference_base_url": "https://inference-api.nousresearch.com/v1", "scope": "inference:invoke",
        "token_type": "Bearer", "access_token": make_jwt(sub=sub, client_id="hermes-cli", account_tier="standard"),
        "refresh_token": refresh, "obtained_at": datetime.now(UTC).isoformat(), "expires_at": expires,
        "expires_in": 3600}


CLAIMED = {"status": "claimed", "destination_org_id": "org_1", "account_email": "sid@example.com",
           "connectors": {"kind": "moved", "moved_toolkits": ["gmail"], "dropped_toolkits": []},
           "cohort_carried": False}


class ClaimPortal(FakePortal):
    """FakePortal plus ``POST /api/anonymous/claim`` and the OAuth refresh route."""

    def __init__(self):
        super().__init__()
        self.claim_answers: list = []          # each: httpx.Response, or an exception to raise
        self.claims: list[dict] = []
        self.refreshes = 0
        self.locks_held_during_claim: list[bool] = []
        self.on_claim = None                    # side effect run as each claim arrives

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/anonymous/claim":
            self.locks_held_during_claim.append(holds_auth_locks())
            self.claims.append({"body": json.loads(request.content),
                                "authorization": request.headers.get("authorization")})
            if self.on_claim:
                self.on_claim()
            answer = self.claim_answers.pop(0) if self.claim_answers else httpx.Response(200, json=CLAIMED)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if request.url.path == "/api/oauth/token":
            self.refreshes += 1
            return httpx.Response(200, json={
                "access_token": make_jwt(sub=DESTINATION, client_id="hermes-cli", account_tier="standard",
                                         rotation=self.refreshes),
                "refresh_token": f"rt_{self.refreshes + 1}", "expires_in": 3600, "token_type": "Bearer"})
        return super().handler(request)


def holds_auth_locks() -> bool:
    from hermes_cli import auth, auth_nous
    if getattr(auth_nous._nous_shared_lock_holder, "depth", 0):
        return True
    with auth._auth_target_lock_holders_guard:
        holders = list(auth._auth_target_lock_holders.values())
    return any(getattr(holder, "depth", 0) for holder in holders)


@pytest.fixture
def portal(monkeypatch, tmp_path):
    fake = install_portal(monkeypatch, tmp_path, ClaimPortal())
    # The settle reads the recommended default from the Portal; that is not under test here.
    monkeypatch.setattr("hermes_cli.anon_auth.settle_after_upgrade", lambda _s: {"model": "", "changed": False})
    return fake


def _mint_guest() -> str:
    from hermes_cli import anon_auth
    state = anon_auth.ensure_portal_identity(explicit=True)
    assert anon_auth.is_guest_state(state)
    return state["anon_token"]


def _records() -> dict:
    from hermes_cli import guest_handoff
    return guest_handoff._load(guest_handoff._records_path())["records"]


def _only_record() -> dict:
    records = _records()
    assert len(records) == 1, records
    return next(iter(records.values()))


# --- Writer coverage: every way an account replaces the guest records the move -------------------

def _save(account):
    from hermes_cli.auth import persist_nous_credentials
    persist_nous_credentials(account)


def _persist(account):
    """A surface's save: the account, then the handoff outside the locks."""
    from hermes_cli import guest_handoff
    _save(account)
    guest_handoff.drain()


def _step_up(account, monkeypatch):
    from hermes_cli import auth
    monkeypatch.setattr(auth, "_nous_device_code_login", lambda **_kw: account)
    auth.step_up_nous_billing_scope(open_browser=False)


def _login_nous(account, monkeypatch):
    from hermes_cli import auth, auth_nous
    monkeypatch.setattr(auth, "_nous_device_code_login", lambda **_kw: account)
    monkeypatch.setattr(auth_nous, "_pick_nous_model_after_login", lambda *_a: None)
    auth_nous._login_nous(SimpleNamespace(), auth.PROVIDER_REGISTRY["nous"])


def _subscription_login(account, monkeypatch):
    from hermes_cli import auth, nous_subscription
    monkeypatch.setattr(auth, "_nous_device_code_login", lambda **_kw: account)
    monkeypatch.setattr(nous_subscription, "_confirm", lambda _p: True)
    assert nous_subscription._run_nous_portal_login_only(capability="Web search")


def _sibling_adoption(account, monkeypatch, tmp_path):
    """Another profile signed in (account in the shared store); this one adopts it on next use."""
    import os
    from hermes_cli import anon_auth, auth_nous
    mine, sibling = os.environ["HERMES_HOME"], tmp_path / "sibling-profile"
    sibling.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(sibling))
    auth_nous._write_shared_nous_state(account, renewal=False)
    monkeypatch.setenv("HERMES_HOME", mine)
    anon_auth.ensure_portal_identity(explicit=True)


WRITERS = {
    "persist_nous_credentials": lambda a, mp, tp: _persist(a),
    "step_up_nous_billing_scope": lambda a, mp, tp: _step_up(a, mp),
    "_login_nous": lambda a, mp, tp: _login_nous(a, mp),
    "nous_subscription_login": lambda a, mp, tp: _subscription_login(a, mp),
    "shared_store_adoption": _sibling_adoption,
}


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_every_writer_that_replaces_the_guest_hands_it_over(portal, monkeypatch, tmp_path, writer):
    anon = _mint_guest()
    WRITERS[writer](_account(), monkeypatch, tmp_path)

    assert [c["body"] for c in portal.claims] == [{"token": anon}]
    assert portal.claims[0]["authorization"].startswith("Bearer ")
    assert portal.locks_held_during_claim == [False]
    record = _only_record()
    assert record["state"] == "claimed" and "anon_token" not in record
    from hermes_cli import guest_handoff
    assert guest_handoff.handoff_status()["moved_toolkits"] == ["gmail"]


def test_an_account_replacing_an_account_is_not_a_handoff(portal):
    _persist(_account(refresh="rt_a"))
    _persist(_account(sub="user_acct_2", refresh="rt_b"))
    assert portal.claims == [] and _records() == {}


# --- Each answer of the claim endpoint ------------------------------------------------------------

def _err(status, code, **headers):
    return httpx.Response(status, json={"error": code, "error_description": "x"}, headers=headers)


@pytest.mark.parametrize("answer, state, probably_moved", [
    (httpx.Response(200, json=CLAIMED), "claimed", False),
    (_err(404, "unknown_token"), "gone", False),
    (_err(410, "already_retired"), "gone", True),
    (_err(401, "invalid_token"), "retrying", False),
    (_err(409, "account_busy"), "retrying", False),
    (_err(429, "rate_limited", **{"Retry-After": "120"}), "retrying", False),
    (_err(404, "not_found"), "retrying", False),
    (_err(503, "temporarily_disabled"), "retrying", False),
    (_err(503, "temporarily_unavailable", **{"Retry-After": "5"}), "retrying", False),
    (_err(500, "internal_error"), "retrying", False),
    (_err(409, "no_destination"), "retrying", False),
    (_err(409, "unexpected_account_state"), "retrying", False),
    (_err(422, "a_code_this_version_does_not_know"), "retrying", False),
    (_err(403, "client_not_allowed"), "failed", False),
    (_err(403, "account_locked"), "failed", False),
], ids=lambda v: v if isinstance(v, str) else None)
def test_each_claim_answer_lands_in_its_state(portal, answer, state, probably_moved):
    anon = _mint_guest()
    portal.claim_answers = [answer]
    _persist(_account())

    assert len(portal.claims) == 1
    record = _only_record()
    assert record["state"] == state
    assert bool(record.get("probably_moved")) is probably_moved
    # A retrying or failed record keeps the credential; a finished one drops it.
    assert (record.get("anon_token") == anon) is (state in ("retrying", "failed"))
    if state == "retrying":   # honours Retry-After, else the first retry delay
        wait = float(record["next_attempt_at"]) - time.time()
        assert wait >= float(answer.headers.get("Retry-After", 60)) - 1


def _later(monkeypatch, seconds: float) -> None:
    from hermes_cli import guest_handoff
    now = guest_handoff._now() + seconds
    monkeypatch.setattr(guest_handoff, "_now", lambda: now)


def test_lost_success_then_unknown_token_reads_as_probably_moved(portal, monkeypatch):
    from hermes_cli import guest_handoff
    portal.claim_answers = [httpx.ReadTimeout("lost"), _err(404, "unknown_token")]
    _mint_guest()
    _persist(_account())
    _later(monkeypatch, 120)
    guest_handoff.drain()
    record = _only_record()
    assert len(portal.claims) == 2
    assert record["state"] == "gone" and record["probably_moved"] is True and record["attempts"] == 2


@pytest.mark.parametrize("carried, refreshes", [(True, 1), (False, 0)])
def test_account_token_refreshes_only_when_the_cohort_carried(portal, carried, refreshes):
    portal.claim_answers = [httpx.Response(200, json={**CLAIMED, "cohort_carried": carried})]
    _mint_guest()
    _persist(_account())
    assert _only_record()["state"] == "claimed"
    assert portal.refreshes == refreshes


def test_transient_failures_retry_a_few_times_then_stop(portal, monkeypatch):
    from hermes_cli import guest_handoff
    attempts = len(guest_handoff._RETRY_DELAYS_SECONDS) + 1
    portal.claim_answers = [_err(503, "temporarily_disabled")] * (attempts + 1)
    _mint_guest()
    _persist(_account())
    for _ in range(attempts):
        _later(monkeypatch, 7 * 3600)
        guest_handoff.drain()
    assert len(portal.claims) == attempts
    assert _only_record()["state"] == "failed" and _only_record()["anon_token"]


def test_a_retry_not_yet_due_sends_nothing(portal):
    from hermes_cli import guest_handoff
    portal.claim_answers = [_err(503, "temporarily_disabled")]
    _mint_guest()
    _persist(_account())
    assert guest_handoff.drain() == [] and len(portal.claims) == 1


# --- Sharing, durability --------------------------------------------------------------------------

def test_two_profiles_sharing_one_guest_claim_once(portal, monkeypatch, tmp_path):
    from hermes_cli import anon_auth, guest_handoff
    import os
    first = os.environ["HERMES_HOME"]
    second = tmp_path / "second-profile"
    second.mkdir()
    anon = _mint_guest()
    monkeypatch.setenv("HERMES_HOME", str(second))
    assert anon_auth.ensure_portal_identity(explicit=True)["anon_token"] == anon   # adopts the guest

    monkeypatch.setenv("HERMES_HOME", first)
    _persist(_account())
    monkeypatch.setenv("HERMES_HOME", str(second))
    anon_auth.ensure_portal_identity(explicit=True)      # adopts the account, replacing the guest

    assert len(portal.claims) == 1
    assert guest_handoff.handoff_status()["state"] == "claimed"
    assert len(_only_record()["profiles"]) == 2


def test_a_crash_between_save_and_claim_is_claimed_on_the_next_run(portal):
    from hermes_cli import guest_handoff
    anon = _mint_guest()
    _save(_account())                                    # the process dies before the claim
    assert portal.claims == [] and _only_record()["state"] == "pending"

    guest_handoff.drain()                                # the next process's retry worker tick
    assert [c["body"] for c in portal.claims] == [{"token": anon}]
    assert _only_record()["state"] == "claimed"


def test_a_claimed_record_is_never_reopened(portal):
    from hermes_cli import guest_handoff
    _mint_guest()
    _persist(_account())
    record = _only_record()
    guest_handoff.note_replacement({"auth_method": "anonymous", "anon_token": "anon_0001"}, _account())
    guest_handoff.drain()
    assert len(portal.claims) == 1 and _only_record()["state"] == record["state"] == "claimed"


def test_a_late_answer_never_reopens_a_finished_move(portal, monkeypatch):
    from hermes_cli import guest_handoff
    _mint_guest()
    _save(_account())
    # A's answer is lost; while A hangs past its lease, B hears unknown_token and finishes the move.
    portal.claim_answers = [_err(404, "unknown_token"), httpx.ReadTimeout("lost")]

    def b_runs_while_a_hangs():
        portal.on_claim = None
        _later(monkeypatch, guest_handoff._LEASE_SECONDS + 1)
        guest_handoff.drain()
    portal.on_claim = b_runs_while_a_hangs
    guest_handoff.drain()
    _later(monkeypatch, 7 * 3600)
    assert guest_handoff.drain() == []
    assert len(portal.claims) == 2 and _only_record()["state"] == "gone"


def test_a_busy_store_lock_leaves_the_move_for_a_later_run(portal, monkeypatch):
    from hermes_cli import guest_handoff
    _mint_guest()
    _save(_account())
    real_bearer = guest_handoff._bearer
    monkeypatch.setattr(guest_handoff, "_bearer", lambda **_kw: (_ for _ in ()).throw(TimeoutError("busy")))
    assert guest_handoff.drain() == []
    assert portal.claims == [] and _only_record()["state"] == "pending"
    monkeypatch.setattr(guest_handoff, "_bearer", real_bearer)
    guest_handoff.drain()
    assert _only_record()["state"] == "claimed"


def test_the_retry_worker_binds_the_launch_profile(portal, monkeypatch):
    import threading
    from agent.secret_scope import current_secret_scope, is_multiplex_active, set_multiplex_active
    from hermes_cli import guest_handoff
    ticked, scopes = threading.Event(), []
    monkeypatch.setattr(guest_handoff, "_WORKER_FIRST_TICK_SECONDS", 0.0)
    monkeypatch.setattr(guest_handoff, "drain", lambda: (scopes.append(current_secret_scope()), ticked.set()))
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        guest_handoff.start_worker()
        assert ticked.wait(5)
    finally:
        guest_handoff.stop_worker()
        set_multiplex_active(previous)
    assert scopes[0] is not None


def test_a_run_while_an_attempt_is_in_flight_sends_nothing(portal):
    from hermes_cli import guest_handoff
    _mint_guest()
    _save(_account())
    portal.on_claim = lambda: portal.__dict__.setdefault("inner", guest_handoff.drain())
    guest_handoff.drain()
    assert len(portal.claims) == 1 and portal.inner == [] and _only_record()["state"] == "claimed"


@pytest.mark.parametrize("code", ["client_not_allowed", "account_locked", "invalid_request", "some_new_code", None])
def test_failure_copy_never_prints_a_server_code(code):
    from hermes_cli import guest_handoff
    lines = guest_handoff.status_lines({"state": "failed", "error": code})
    assert lines and lines[0].startswith("Your free-tier apps could not be moved.")
    assert code is None or code not in " ".join(lines)


def test_a_claim_answer_without_connectors_reads_as_neutral(portal):
    from hermes_cli import guest_handoff
    portal.claim_answers = [httpx.Response(200, json={
        "status": "claimed", "destination_org_id": "org_1", "account_email": "sid@example.com"})]
    _mint_guest()
    _persist(_account())
    text = " ".join(guest_handoff.status_lines(guest_handoff.handoff_status()))
    assert "sid@example.com" in text and "no free-tier apps" not in text


def test_gone_on_a_first_attempt_reads_as_neutral(portal):
    from hermes_cli import guest_handoff
    portal.claim_answers = [_err(404, "unknown_token")]
    _mint_guest()
    _persist(_account())
    assert guest_handoff.status_lines(guest_handoff.handoff_status()) == ["Nothing left to move from the free tier."]


def test_a_long_lived_process_with_a_guest_still_runs_the_handoff_worker(portal, monkeypatch):
    from hermes_cli import guest_handoff, nous_auth_keepalive
    monkeypatch.setattr(nous_auth_keepalive, "get_provider_auth_state",
                        lambda _p: {"auth_method": "anonymous", "anon_token": "anon_x"})
    try:
        nous_auth_keepalive.start_nous_auth_keepalive()
        assert guest_handoff._worker_thread is not None and guest_handoff._worker_thread.is_alive()
    finally:
        nous_auth_keepalive.stop_nous_auth_keepalive()
    assert guest_handoff._worker_thread is None


def test_a_guest_token_still_cached_from_a_connector_call_does_not_skip_the_move(portal):
    from hermes_cli import auth
    from hermes_constants import hermes_home_key
    _mint_guest()
    auth._RESOLVE_TOKEN_CACHE[hermes_home_key()] = (time.monotonic(), make_jwt())   # the guest's own token
    _persist(_account())
    assert len(portal.claims) == 1 and _only_record()["state"] == "claimed"


def test_auth_status_shows_what_moved(portal, capsys):
    from hermes_cli import auth_commands
    _mint_guest()
    _persist(_account())
    capsys.readouterr()
    auth_commands.auth_status_command(SimpleNamespace(provider="nous"))
    out = capsys.readouterr().out
    assert "nous: logged in" in out and "Your apps moved to sid@example.com: Gmail." in out


@pytest.mark.parametrize("kind", ["moved", "swapped"])
def test_a_move_with_no_named_apps_reads_as_nothing_moved(portal, kind):
    """A free tier that only read its connectors gets a vendor id row on the service, so the claim
    answers ``moved``/``swapped`` with an empty list; that empty list is the truth."""
    from hermes_cli import guest_handoff
    portal.claim_answers = [httpx.Response(200, json={
        **CLAIMED, "connectors": {"kind": kind, "moved_toolkits": [], "dropped_toolkits": []}})]
    _mint_guest()
    _persist(_account())
    assert guest_handoff.status_lines(guest_handoff.handoff_status()) == [
        "Signed in as sid@example.com. There were no free-tier apps to move."]


def test_status_copy_never_names_the_mechanism(portal):
    from hermes_cli import guest_handoff
    for status in (
        {"state": "claimed", "connectors_kind": "moved", "moved_toolkits": [], "account_email": "a@b.c"},
        {"state": "claimed", "connectors_kind": "conflict", "dropped_toolkits": ["notion"]},
        {"state": "gone", "probably_moved": True}, {"state": "retrying"}, {"state": "failed", "error": "x"},
    ):
        text = " ".join(guest_handoff.status_lines(status)).lower()
        assert text and not any(word in text for word in ("guest", "anonymous", "claim"))
    moved = guest_handoff.status_lines({"state": "claimed", "connectors_kind": "moved", "account_email": "a@b.c",
                                        "moved_toolkits": ["gmail", "googlecalendar"], "dropped_toolkits": ["linear"]})
    assert moved == ["Your apps moved to a@b.c: Gmail, Google Calendar.", "Reconnect in a@b.c: Linear."]
