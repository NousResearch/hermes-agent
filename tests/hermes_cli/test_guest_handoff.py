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

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/anonymous/claim":
            from hermes_cli import guest_handoff
            self.locks_held_during_claim.append(guest_handoff._holds_auth_locks())
            self.claims.append({"body": json.loads(request.content),
                                "authorization": request.headers.get("authorization")})
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
    monkeypatch.setattr(guest_handoff, "_sleep", lambda _s: None)
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
    (_err(409, "no_destination"), "failed", False),
    (_err(409, "unexpected_account_state"), "failed", False),
    (_err(403, "client_not_allowed"), "failed", False),
    (_err(403, "account_locked"), "failed", False),
], ids=lambda v: v if isinstance(v, str) else None)
def test_each_claim_answer_lands_in_its_state(portal, monkeypatch, answer, state, probably_moved):
    from hermes_cli import guest_handoff
    monkeypatch.setattr(guest_handoff, "SYNC_BUDGET_SECONDS", 0.0)   # one attempt, no in-run retry
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


def test_retries_give_up_after_the_cap(portal, monkeypatch):
    from hermes_cli import guest_handoff
    portal.claim_answers = [_err(409, "account_busy")] * (guest_handoff.MAX_ATTEMPTS + 2)
    monkeypatch.setattr(guest_handoff, "_BACKOFF_SECONDS", (0.0,))
    _mint_guest()
    _persist(_account())
    record = _only_record()
    assert record["state"] == "failed" and record["error"] == "account_busy"
    assert len(portal.claims) == guest_handoff.MAX_ATTEMPTS


# --- Binding, sharing, durability -----------------------------------------------------------------

def test_a_retry_only_claims_into_the_destination_account(portal, monkeypatch):
    from hermes_cli import guest_handoff
    monkeypatch.setattr(guest_handoff, "SYNC_BUDGET_SECONDS", 0.0)
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
    from hermes_cli import auth_nous, guest_handoff, nous_auth_keepalive
    anon = _mint_guest()
    monkeypatch.setattr(auth_nous, "drain", lambda: None)     # the process dies before the claim
    _persist(_account())
    assert portal.claims == [] and _only_record()["state"] == "pending"

    monkeypatch.setattr(auth_nous, "drain", guest_handoff.drain)
    monkeypatch.setattr(guest_handoff, "_local", threading.local())   # a fresh process
    monkeypatch.setattr(nous_auth_keepalive, "_refresh_selected_pool_entry", lambda **_kw: True)
    nous_auth_keepalive.refresh_nous_auth_keepalive_once()

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
    assert batch.account_email == "sid@example.com" and batch.settled == {"model": "", "changed": False}


def test_auth_status_shows_what_moved(portal, capsys):
    from hermes_cli import auth_commands
    _mint_guest()
    _persist(_account())
    capsys.readouterr()
    auth_commands.auth_status_command(SimpleNamespace(provider="nous"))
    out = capsys.readouterr().out
    assert "nous: logged in" in out and "Your apps moved to sid@example.com: gmail." in out


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
        {"state": "claimed", "connectors_kind": "moved", "moved_toolkits": [], "account_email": "a@b.c"})
    assert moved == ["Your apps moved to a@b.c."]
