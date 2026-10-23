"""The guest handoff step: a Nous account replacing the free tier moves the free tier's apps into it.

Driven end to end through Hermes' real writers, stores and HTTP client against a fake account
service (``anon_portal.FakePortal`` plus the claim and token-refresh routes).
"""

from __future__ import annotations

import json
import threading
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
            from hermes_cli import guest_handoff
            self.locks_held_during_claim.append(guest_handoff._holds_auth_locks())
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


@pytest.fixture
def portal(monkeypatch, tmp_path):
    from hermes_cli import guest_handoff
    fake = install_portal(monkeypatch, tmp_path, ClaimPortal())
    # A runner's backoff sleep moves the handoff's own clock instead of the wall clock.
    skew = [0.0]
    monkeypatch.setattr(guest_handoff, "_now", lambda: time.time() + skew[0])
    monkeypatch.setattr(guest_handoff, "_sleep", lambda s: skew.__setitem__(0, skew[0] + s))
    monkeypatch.setattr(guest_handoff, "_local", threading.local())
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


# --- Writer coverage: every way an account replaces the guest fires the hook ----------------------

def _persist(account):
    from hermes_cli.auth import persist_nous_credentials
    persist_nous_credentials(account)


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
    from hermes_cli import anon_auth, auth_nous, guest_handoff
    mine, sibling = os.environ["HERMES_HOME"], tmp_path / "sibling-profile"
    sibling.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(sibling))
    monkeypatch.setattr(auth_nous, "drain", lambda: None)   # the sibling's own run is not this test
    auth_nous._write_shared_nous_state(account)
    monkeypatch.setattr(auth_nous, "drain", guest_handoff.drain)
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
    assert record["state"] == "claimed" and record["destination"] == DESTINATION
    assert "anon_token" not in record
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
def test_each_claim_answer_lands_in_its_state(portal, monkeypatch, answer, state, probably_moved):
    from hermes_cli import guest_handoff
    monkeypatch.setattr(guest_handoff, "SYNC_BUDGET_SECONDS", 3.0)   # one attempt: every retry waits longer
    monkeypatch.setattr(guest_handoff, "_BACKOFF_SECONDS", (60.0,))
    anon = _mint_guest()
    portal.claim_answers = [answer]
    _persist(_account())

    record = _only_record()
    assert record["state"] == state
    assert bool(record.get("probably_moved")) is probably_moved
    # A retrying or failed record keeps the credential; a finished one drops it.
    assert (record.get("anon_token") == anon) is (state in ("retrying", "failed"))
    if state == "retrying":   # honours Retry-After, else the first backoff step
        wait = float(record["next_attempt_at"]) - time.time()
        assert wait >= float(answer.headers.get("Retry-After", 1)) - 1


def test_a_refused_account_token_is_refreshed_once_then_retried(portal):
    portal.claim_answers = [_err(401, "invalid_token"), httpx.Response(200, json=CLAIMED)]
    _mint_guest()
    _persist(_account())
    assert len(portal.claims) == 2 and portal.refreshes == 1
    assert portal.claims[0]["authorization"] != portal.claims[1]["authorization"]
    assert _only_record()["state"] == "claimed" and _only_record()["attempts"] == 1


def test_lost_success_then_unknown_token_reads_as_probably_moved(portal):
    portal.claim_answers = [httpx.ReadTimeout("lost"), _err(404, "unknown_token")]
    _mint_guest()
    _persist(_account())
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


def test_transient_failures_keep_retrying_past_the_old_attempt_cap(portal, monkeypatch):
    from hermes_cli import guest_handoff
    portal.claim_answers = [_err(503, "temporarily_disabled")] * 12   # the ops breaker: no Retry-After
    monkeypatch.setattr(guest_handoff, "_BACKOFF_SECONDS", (0.0,))
    _mint_guest()
    _persist(_account())
    assert len(portal.claims) == 13 and _only_record()["state"] == "claimed"


def test_a_long_outage_retries_on_the_slowest_rung(portal, monkeypatch):
    from hermes_cli import guest_handoff
    _pending_for(monkeypatch, _account())
    guest_handoff._update(guest_handoff._records_path(), next(iter(_records())),
                          lambda r: {**r, "state": "retrying", "attempts": 40})
    portal.claim_answers = [_err(503, "temporarily_disabled")]
    guest_handoff.drain()
    record = _only_record()
    assert record["state"] == "retrying" and record["anon_token"]
    assert float(record["next_attempt_at"]) - guest_handoff._now() >= guest_handoff._BACKOFF_SECONDS[-1] - 5


# --- Binding, sharing, durability -----------------------------------------------------------------

def test_a_retry_only_claims_into_the_destination_account(portal, monkeypatch):
    from hermes_cli import guest_handoff
    monkeypatch.setattr(guest_handoff, "_BACKOFF_SECONDS", (60.0,))
    portal.claim_answers = [_err(409, "account_busy")]
    _mint_guest()
    _persist(_account())
    assert _only_record()["state"] == "retrying"

    # The profile now holds another account: the record is not claimed into it.
    _persist(_account(sub="someone_else", refresh="rt_other"))
    monkeypatch.setattr(guest_handoff, "_now", lambda: time.time() + 10_000)
    guest_handoff.drain()
    assert len(portal.claims) == 1 and _only_record()["destination"] == DESTINATION

    _persist(_account(refresh="rt_back"))
    guest_handoff.drain()
    assert len(portal.claims) == 2 and _only_record()["state"] == "claimed"


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


def test_a_crash_between_save_and_claim_is_claimed_on_next_boot(portal, monkeypatch):
    from hermes_cli import auth_nous, guest_handoff
    anon = _mint_guest()
    monkeypatch.setattr(auth_nous, "drain", lambda: None)     # the process dies before the claim
    _persist(_account())
    assert portal.claims == [] and _only_record()["state"] == "pending"

    monkeypatch.setattr(auth_nous, "drain", guest_handoff.drain)
    monkeypatch.setattr(guest_handoff, "_local", threading.local())   # a fresh process
    guest_handoff.retry_all_profiles()                                 # its retry worker's first tick

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


def test_a_surface_lock_defers_the_claim_until_it_is_released(portal):
    from hermes_cli import guest_handoff
    _mint_guest()
    with guest_handoff.deferred() as batch:
        _persist(_account())
        assert portal.claims == [] and _only_record()["state"] == "pending"
    assert len(portal.claims) == 1
    assert batch.account_email == "sid@example.com"


# --- Review round 1: the apps never land in an account the record is not bound to ----------------

def _pending_for(monkeypatch, account) -> None:
    """A saved account and its pending record, the claim not yet run."""
    _mint_guest()
    _pending_for_current_guest(monkeypatch, account)


def _pending_for_current_guest(monkeypatch, account) -> None:
    from hermes_cli import auth_nous, guest_handoff
    monkeypatch.setattr(auth_nous, "drain", lambda: None)
    _persist(account)
    monkeypatch.setattr(auth_nous, "drain", guest_handoff.drain)
    assert _only_record()["state"] == "pending"


def test_a_sign_in_racing_the_claim_never_moves_apps_into_it(portal, monkeypatch):
    from hermes_cli import auth_nous, guest_handoff
    _pending_for(monkeypatch, _account())
    real_bearer = guest_handoff._bearer

    def bearer_after_another_sign_in(**kw):
        monkeypatch.setattr(auth_nous, "drain", lambda: None)
        _persist(_account(sub="intruder", refresh="rt_intruder"))   # lands after the bound check
        return real_bearer(**kw)

    monkeypatch.setattr(guest_handoff, "_bearer", bearer_after_another_sign_in)
    guest_handoff.drain()
    assert portal.claims == []
    assert _only_record()["state"] == "pending" and _only_record()["attempts"] == 0


def test_a_destination_rebound_before_the_attempt_is_not_claimed_into_the_old_account(portal, monkeypatch):
    from hermes_cli import guest_handoff
    _pending_for(monkeypatch, _account())
    real_update, fired = guest_handoff._update, []

    def sibling_wins_the_records_lock_first(path, key, change):
        if not fired:   # the runner's first locked write; a sibling's rebind lands just before it
            fired.append(True)
            guest_handoff.note_replacement(
                {"auth_method": "anonymous", "anon_token": "anon_0001"}, _account(sub="sibling_acct"))
        return real_update(path, key, change)

    monkeypatch.setattr(guest_handoff, "_update", sibling_wins_the_records_lock_first)
    guest_handoff.drain()
    assert portal.claims == []
    assert _only_record()["destination"] == "sibling_acct" and _only_record()["state"] == "pending"


def _age_record(days: float, **fields) -> None:
    from hermes_cli import guest_handoff
    created = guest_handoff._iso(guest_handoff._now() - days * 86400)
    guest_handoff._update(guest_handoff._records_path(), next(iter(_records())),
                          lambda r: {**r, "created_at": created, **fields})


def test_retries_stop_safely_before_the_free_tier_is_reaped(portal, monkeypatch):
    from hermes_cli import guest_handoff
    _pending_for(monkeypatch, _account())
    _age_record(13.5, state="retrying", attempts=300)
    guest_handoff.drain()
    assert portal.claims == []
    assert _only_record()["state"] == "failed" and _only_record()["error"] == "retry_window_ended"


@pytest.mark.parametrize("trigger", ["auth_status", "auth_upgrade", "sign_in"])
def test_a_record_an_older_build_failed_on_a_transient_error_is_rearmed(portal, monkeypatch, capsys, trigger):
    from hermes_cli import anon_auth, auth_commands
    anon = _mint_guest()
    _pending_for_current_guest(monkeypatch, _account())
    _age_record(2, state="failed", error="retries_exhausted", attempts=8)
    if trigger == "auth_status":
        auth_commands.auth_status_command(SimpleNamespace(provider="nous"))
    elif trigger == "auth_upgrade":
        anon_auth.upgrade_guest(SimpleNamespace(no_browser=True, timeout=None))
    else:
        _persist(_account(refresh="rt_again"))    # the account saved again (re-login)
    assert [c["body"] for c in portal.claims] == [{"token": anon}]
    assert _only_record()["state"] == "claimed"


def test_a_final_refusal_is_not_rearmed(portal, monkeypatch):
    from hermes_cli import auth_commands
    _pending_for(monkeypatch, _account())
    _age_record(1, state="failed", error="account_locked", attempts=1)
    auth_commands.auth_status_command(SimpleNamespace(provider="nous"))
    assert portal.claims == [] and _only_record()["state"] == "failed"


@pytest.mark.parametrize("code", ["client_not_allowed", "account_locked", "invalid_request",
                                  "retry_window_ended", "some_new_code", None])
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


def test_a_slow_refusal_at_the_budget_edge_starts_no_refresh_or_second_claim(portal, monkeypatch):
    from hermes_cli import guest_handoff
    clock = [1000.0]
    monkeypatch.setattr(guest_handoff.time, "monotonic", lambda: clock[0])
    portal.claim_answers = [_err(401, "invalid_token")]
    portal.on_claim = lambda: clock.__setitem__(0, clock[0] + guest_handoff.SYNC_BUDGET_SECONDS + 5)
    _mint_guest()
    _persist(_account())
    assert len(portal.claims) == 1 and portal.refreshes == 0
    assert _only_record()["state"] == "pending" and _only_record()["attempts"] == 0


def test_corrupt_records_neither_block_a_sign_in_nor_stop_the_retry_worker(portal, monkeypatch):
    from hermes_cli import guest_handoff
    path = guest_handoff._records_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert guest_handoff.retry_all_profiles() == []

    _mint_guest()
    _persist(_account())
    assert _only_record()["state"] == "claimed"
    assert path.with_suffix(".json.corrupt").read_text(encoding="utf-8") == "{not json"


def test_a_move_bound_to_another_account_reads_as_paused(portal, monkeypatch):
    from hermes_cli import guest_handoff
    portal.claim_answers = [_err(409, "account_busy")]
    monkeypatch.setattr(guest_handoff, "_BACKOFF_SECONDS", (60.0,))
    _mint_guest()
    _persist(_account())
    assert guest_handoff.handoff_status()["state"] == "retrying"
    _persist(_account(sub="someone_else", refresh="rt_other"))
    status = guest_handoff.handoff_status()
    assert status["state"] == "paused"
    assert "another Nous account" in guest_handoff.status_lines(status)[0]


# --- Review round 2: retries reach every profile; the first attempt is not lost -------------------

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


def test_the_worker_retries_a_named_profiles_move_from_the_launch_profile(portal, monkeypatch, tmp_path):
    import os
    from hermes_cli import guest_handoff
    anon = _mint_guest()
    _pending_for_current_guest(monkeypatch, _account())
    launch = tmp_path / "launch-profile"
    launch.mkdir()
    named_home = os.environ["HERMES_HOME"]
    monkeypatch.setenv("HERMES_HOME", str(launch))      # the long-lived process runs as another profile
    guest_handoff.retry_all_profiles()
    assert [c["body"] for c in portal.claims] == [{"token": anon}]
    monkeypatch.setenv("HERMES_HOME", named_home)
    assert _only_record()["state"] == "claimed"


def test_a_guest_token_still_cached_from_a_connector_call_does_not_skip_the_move(portal):
    from hermes_cli import auth
    from hermes_constants import hermes_home_key
    _mint_guest()
    auth._RESOLVE_TOKEN_CACHE[hermes_home_key()] = (time.monotonic(), make_jwt())   # the guest's own token
    _persist(_account())
    assert len(portal.claims) == 1 and _only_record()["state"] == "claimed"


def test_a_held_auth_lock_cannot_stretch_the_sync_budget(portal, monkeypatch):
    from hermes_cli import auth, guest_handoff
    _pending_for(monkeypatch, _account())
    monkeypatch.setattr(guest_handoff, "SYNC_BUDGET_SECONDS", 1.0)
    held, release = threading.Event(), threading.Event()

    def hold_auth_lock():
        with auth._auth_store_lock():
            held.set()
            release.wait(30)

    holder = threading.Thread(target=hold_auth_lock, daemon=True)
    holder.start()
    held.wait(5)
    started = time.monotonic()
    try:
        guest_handoff.drain()
    finally:
        release.set()
        holder.join(5)
    assert time.monotonic() - started < 6
    assert portal.claims == [] and _only_record()["state"] == "pending" and _only_record()["attempts"] == 0


def test_auth_status_shows_what_moved(portal, capsys):
    from hermes_cli import auth_commands
    _mint_guest()
    _persist(_account())
    capsys.readouterr()
    auth_commands.auth_status_command(SimpleNamespace(provider="nous"))
    out = capsys.readouterr().out
    assert "nous: logged in" in out and "Your apps moved to sid@example.com: gmail." in out


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
    moved = guest_handoff.status_lines(
        {"state": "claimed", "connectors_kind": "moved", "moved_toolkits": ["gmail"], "account_email": "a@b.c"})
    assert moved == ["Your apps moved to a@b.c: gmail."]
