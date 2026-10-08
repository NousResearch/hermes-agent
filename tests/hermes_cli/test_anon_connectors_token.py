"""Nous free tier: the connectors-only guest token (NAS hermes-portal#1516).

A guest who only uses connectors or managed tools never meets the free-inference browser challenge:
those callers (``resolve_nous_access_token``) hold or mint a ``purpose: "connectors"`` token, and an
inference token is minted (inside ``run_with_challenge``) only when the free model is called
(``resolve_nous_runtime_credentials``). Driven through the fake NAS with Hermes' real client code;
no request leaves the process.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from hermes_cli import anon_auth, anon_challenge
from hermes_cli import auth as auth_mod
from hermes_cli.auth_nous import resolve_nous_runtime_credentials

from tests.hermes_cli.anon_portal import WELCOME, install_portal, make_jwt


@pytest.fixture
def nas(monkeypatch, tmp_path):
    fake = install_portal(monkeypatch, tmp_path)
    assert anon_auth.is_guest_state(anon_auth.ensure_portal_identity(explicit=True))
    return fake


@pytest.fixture
def presented(monkeypatch):
    seen: list[anon_challenge.BrowserChallenge] = []
    monkeypatch.setattr(anon_challenge, "present", seen.append)
    return seen


def _purposes(nas) -> list:
    return [r["body"].get("purpose") for r in nas.token_requests]


def _scope(token: str) -> str:
    return anon_auth._decode_jwt_claims(token)["scope"]


def _forget_memo(monkeypatch):
    monkeypatch.setattr(auth_mod, "_RESOLVE_TOKEN_CACHE", {})


def _persist_guest_tokens(*, inference: int | None = None, connectors: int | None = None) -> dict:
    """Write held tokens expiring in the given seconds. An inference-only write is exactly what
    current releases persist; ``connectors`` adds the new slot beside it."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    at = lambda s: (datetime.now(timezone.utc) + timedelta(seconds=s)).isoformat()  # noqa: E731
    tokens = {}
    with _auth_store_lock():
        store = _load_auth_store()
        nous = store["providers"]["nous"]
        if inference is not None:
            tokens["inference"] = make_jwt(exp=int(time.time()) + inference)
            nous.update(access_token=tokens["inference"], scope="inference:invoke", token_type="Bearer",
                        inference_base_url=WELCOME, expires_at=at(inference))
        if connectors is not None:
            tokens["connectors"] = make_jwt(scope="connectors:invoke", exp=int(time.time()) + connectors)
            nous["connectors_token"] = {"access_token": tokens["connectors"], "expires_at": at(connectors)}
        _save_auth_store(store)
    return tokens


class TestConnectorsCallers:
    def test_never_challenged_even_while_the_inference_mint_would_be(self, nas, presented):
        from tools.managed_tool_gateway import read_nous_access_token
        nas.challenge_required = True
        nas.challenge_never_clears = True
        token = auth_mod.resolve_nous_access_token()
        assert _scope(token) == "connectors:invoke"
        assert read_nous_access_token() == token
        assert _purposes(nas) == ["connectors"]
        assert presented == [] and "/api/anonymous/challenge/status" not in [p for _, p in nas.calls]

    @pytest.mark.parametrize("inference, connectors, serves", [
        (-60, 900, "connectors"),     # expired inference, live connectors: no mint
        (900, -60, "inference"),      # live inference, expired connectors: no mint
        (-60, -60, None),             # both expired: one connectors mint, never an inference one
        (-60, None, None),            # only an expired inference token (a current release's state)
    ])
    def test_picks_the_live_slot_or_mints_connectors_only(self, nas, presented, monkeypatch,
                                                          inference, connectors, serves):
        from tools.managed_tool_gateway import read_nous_access_token
        held = _persist_guest_tokens(inference=inference, connectors=connectors)
        nas.challenge_required = True
        token = auth_mod.resolve_nous_access_token()
        _forget_memo(monkeypatch)
        assert read_nous_access_token() == token
        if serves:
            assert token == held[serves] and nas.token_requests == []
        else:
            assert _scope(token) == "connectors:invoke" and _purposes(nas) == ["connectors"]
            assert anon_auth.current_nous_state()["connectors_token"]["access_token"] == token
        assert presented == []

    def test_the_connectors_token_is_persisted_beside_the_inference_slot(self, nas, monkeypatch):
        token = auth_mod.resolve_nous_access_token()
        state = anon_auth.current_nous_state()
        assert state["connectors_token"]["access_token"] == token
        assert not state.get("access_token")      # filled only when the free model is used
        _forget_memo(monkeypatch)
        assert auth_mod.resolve_nous_access_token() == token
        assert len(nas.token_requests) == 1

    def test_account_reads_use_the_connectors_token(self, nas, presented):
        from hermes_cli.nous_account import get_nous_portal_account_info, reset_nous_portal_account_info_cache
        nas.challenge_required = True
        auth_mod.resolve_nous_access_token()
        reset_nous_portal_account_info_cache()
        info = get_nous_portal_account_info()
        assert info.logged_in and info.source == "jwt" and info.account_tier == "anonymous"
        assert len(nas.token_requests) == 1 and presented == []

    @pytest.mark.parametrize("error", ["signin_required", "access_denied"])
    def test_a_refusal_surfaces_as_it_does_today(self, nas, error):
        nas.token_response = httpx.Response(403, json={"error": error})
        with pytest.raises(anon_auth.AuthError) as exc:
            auth_mod.resolve_nous_access_token()
        assert exc.value.code == anon_auth.ANON_SIGNIN_REQUIRED and exc.value.retryable is False
        assert str(exc.value) == anon_auth.ANON_FAILURE_COPY[anon_auth.ANON_SIGNIN_REQUIRED]


class TestInferenceCallers:
    def test_still_work_the_challenge_and_never_take_the_connectors_token(self, nas, presented):
        connectors = auth_mod.resolve_nous_access_token()
        nas.challenge_required = True
        nas.challenge_statuses = ["pending"]
        creds = resolve_nous_runtime_credentials()
        assert creds["api_key"] != connectors and _scope(creds["api_key"]) == "inference:invoke"
        assert [c.url for c in presented] == [nas.challenge_url]
        assert _purposes(nas) == ["connectors", "inference", "inference"]

    def test_a_held_inference_token_satisfies_connectors_callers(self, nas, monkeypatch):
        creds = resolve_nous_runtime_credentials()
        _forget_memo(monkeypatch)
        assert auth_mod.resolve_nous_access_token() == creds["api_key"]
        assert len(nas.token_requests) == 1

    def test_a_connectors_mint_leaves_a_pending_inference_challenge_alone(self, nas, presented):
        """The connectors grant says nothing about the inference check: the desktop's hidden window
        must keep the ticket it was handed."""
        nas.challenge_required = True
        nas.challenge_never_clears = True
        with anon_challenge.background_caller(), pytest.raises(anon_auth.AuthError):
            resolve_nous_runtime_credentials()
        assert anon_challenge.pending_challenge()["url"] == nas.challenge_url
        auth_mod.resolve_nous_access_token()
        assert anon_challenge.pending_challenge()["url"] == nas.challenge_url

    def test_a_connectors_caller_never_queues_behind_an_inference_challenge(self, nas, monkeypatch):
        """The wait runs outside the auth locks and its work lock is the inference mint's alone."""
        nas.challenge_required = True
        nas.challenge_statuses = ["pending"]
        during_wait: list[str] = []

        def present(_challenge):
            reader = threading.Thread(
                target=lambda: during_wait.append(auth_mod.resolve_nous_access_token()), daemon=True)
            reader.start()
            reader.join(timeout=5)
        monkeypatch.setattr(anon_challenge, "present", present)
        assert resolve_nous_runtime_credentials()["api_key"]
        assert [_scope(t) for t in during_wait] == ["connectors:invoke"]

    def test_an_inference_give_up_cooldown_does_not_fail_connectors_callers_fast(
            self, nas, presented, monkeypatch):
        nas.challenge_required = True
        nas.challenge_never_clears = True
        monkeypatch.setattr(anon_challenge, "CHALLENGE_WAIT_SECONDS", 0)
        with pytest.raises(anon_auth.AuthError):
            resolve_nous_runtime_credentials()
        assert anon_challenge._gave_up_recently()
        assert _scope(auth_mod.resolve_nous_access_token()) == "connectors:invoke"


class TestNoInferenceIntent:
    """Paths that are not a free-model call never mint inference in the foreground or on a timer."""

    def test_the_keepalive_never_mints_for_a_guest(self, nas):
        from hermes_cli.nous_auth_keepalive import refresh_nous_auth_keepalive_once
        auth_mod.resolve_nous_access_token()
        assert refresh_nous_auth_keepalive_once() is False
        assert _purposes(nas) == ["connectors"]

    def test_a_model_catalog_read_never_waits_on_a_challenge(self, nas, presented):
        from hermes_cli.models import _nous_catalog, get_curated_nous_model_ids
        nas.challenge_required = True
        nas.challenge_never_clears = True
        assert _nous_catalog("nous", False) == (get_curated_nous_model_ids() or None)
        assert "/api/anonymous/challenge/status" not in [p for _, p in nas.calls]


class TestOlderNas:
    """A NAS from before #1516 drops the unknown ``purpose`` key and mints today's token."""

    def test_an_inference_token_for_a_connectors_mint_fills_the_inference_slot(self, nas, monkeypatch):
        nas.honours_purpose = False
        token = auth_mod.resolve_nous_access_token()
        assert _scope(token) == "inference:invoke"
        state = anon_auth.current_nous_state()
        assert state["access_token"] == token and "connectors_token" not in state
        assert resolve_nous_runtime_credentials()["api_key"] == token
        assert len(nas.token_requests) == 1

    def test_a_challenged_connectors_mint_falls_back_to_the_challenge_flow(self, nas, presented):
        nas.honours_purpose = False
        nas.challenge_required = True
        nas.challenge_statuses = ["pending"]
        assert _scope(auth_mod.resolve_nous_access_token()) == "inference:invoke"
        assert len(presented) == 1


class TestPersistedState:
    def test_state_written_by_current_releases_loads_unchanged(self, nas):
        from tools.managed_tool_gateway import peek_nous_access_token
        token = _persist_guest_tokens(inference=900)["inference"]
        assert peek_nous_access_token() == token
        assert auth_mod.resolve_nous_access_token() == token
        assert resolve_nous_runtime_credentials()["api_key"] == token
        assert nas.token_requests == []
