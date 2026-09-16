"""Nous free tier: every way the account service or the wire can refuse the free tier, and what
Hermes does with each (the failure-mode contract behind the desktop's onboarding copy).

Driven through a fake NAS whose responses are the ones the real service sends (see the code table
in ``hermes_cli.anon_auth``), never through mocked-away client code.
"""

from __future__ import annotations

import json

import httpx
import pytest

from hermes_cli import anon_auth, anon_sign_in, free_tier_bootstrap
from hermes_cli.auth import _load_auth_store

from tests.hermes_cli.anon_portal import PORTAL, WELCOME, install_portal  # noqa: F401


@pytest.fixture
def nas(monkeypatch, tmp_path):
    return install_portal(monkeypatch, tmp_path)


def _mint_error(nas) -> anon_auth.AuthError:
    with pytest.raises(anon_auth.AuthError) as exc:
        anon_auth.ensure_portal_identity(explicit=True)
    return exc.value


def _exchange_error(nas) -> anon_auth.AuthError:
    """Mint (the credential is persisted before any exchange), then exchange it at first use."""
    from hermes_cli.auth_nous import resolve_nous_runtime_credentials
    assert anon_auth.is_guest_state(anon_auth.ensure_portal_identity(explicit=True))
    with pytest.raises(anon_auth.AuthError) as exc:
        resolve_nous_runtime_credentials()
    return exc.value


# --- What NAS sends -> which code ------------------------------------------------------------------


class TestNasRefusalCodes:
    def test_surface_not_enabled_is_a_terminal_gate(self, nas):
        nas.create_response = httpx.Response(404, json={"error": "not_found"})
        err = _mint_error(nas)
        assert err.code == anon_auth.ANON_GATE_CLOSED and err.retryable is False
        assert "Nous account" in str(err) and "free" in str(err)
        # Terminal: no later attempt this process, whatever the clock says.
        assert anon_auth.ensure_portal_identity(explicit=True) is None
        assert nas.creates() == 1
        assert anon_auth.last_mint_failure() == {
            "error_code": anon_auth.ANON_GATE_CLOSED, "error": str(err), "retryable": False, "retry_after": 0}

    def test_ops_breaker_is_a_retryable_pause_with_a_floor(self, nas):
        nas.create_response = httpx.Response(
            503, json={"error": "temporarily_disabled", "error_description": "switched off"})
        err = _mint_error(nas)
        assert err.code == anon_auth.ANON_GATE_PAUSED
        failure = anon_auth.last_mint_failure()
        assert failure["retryable"] is True
        assert failure["retry_after"] >= 59      # never polled faster than the paused floor

    def test_too_many_sign_ups_honours_retry_after(self, nas):
        nas.create_response = httpx.Response(
            429, json={"error": "temporarily_unavailable"}, headers={"Retry-After": "42"})
        err = _mint_error(nas)
        assert err.code == anon_auth.ANON_RATE_LIMITED and err.retry_after == 42
        assert "about a minute" in str(err)
        assert anon_auth.last_mint_failure()["retry_after"] == 42

    def test_proof_of_work_is_deferred_with_the_agreed_sentence(self, nas):
        nas.token_response = httpx.Response(428, json={"error": "pow_required", "pow": {"bits": 31}})
        err = _exchange_error(nas)
        assert err.code == anon_auth.ANON_POW_REQUIRED and err.retryable is False
        assert str(err).startswith("The Nous server asked for a proof of work, but that isn't implemented")
        # The credential from ``create`` is kept, so a later NAS without PoW exchanges it instead
        # of minting again.
        assert anon_auth.is_guest_state(_load_auth_store()["providers"]["nous"])
        assert nas.creates() == 1

    def test_locked_account_is_dead_and_never_replaced(self, nas):
        nas.token_response = httpx.Response(403, json={"error": "account_locked"})
        err = _exchange_error(nas)
        assert isinstance(err, anon_auth.AnonCredentialDead)
        assert err.code == anon_auth.ANON_ACCOUNT_LOCKED
        # Retired, and NOT replaced by a fresh identity: the way forward is a sign-in.
        assert "nous" not in _load_auth_store().get("providers", {})
        assert nas.creates() == 1

    def test_a_locked_account_is_never_replaced_through_connectors_either(self, nas):
        from hermes_cli.auth import _auth_store_lock, _save_auth_store
        from tests.hermes_cli.anon_portal import make_jwt
        from tools import managed_tool_gateway as mtg
        anon_auth.ensure_portal_identity(explicit=True)
        with _auth_store_lock():
            store = _load_auth_store()
            store["providers"]["nous"]["expires_at"] = "2000-01-01T00:00:00+00:00"
            store["providers"]["nous"]["access_token"] = make_jwt(exp=1)
            _save_auth_store(store)
        nas.token_response = httpx.Response(403, json={"error": "account_locked"})
        assert mtg.read_nous_access_token() is None
        assert "nous" not in _load_auth_store().get("providers", {})
        assert nas.creates() == 1

    def test_unknown_token_is_replaced_once_at_first_use(self, nas):
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        first = anon_auth.ensure_portal_identity(explicit=True)
        original = nas.handler

        def _first_only(request):
            # Only the FIRST credential is dead; its replacement exchanges normally.
            if request.url.path == "/api/anonymous/token" and json.loads(request.content)["token"] == first["anon_token"]:
                nas.calls.append((request.method, request.url.path))
                return httpx.Response(404, json={"error": "unknown_token"})
            return original(request)
        nas.handler = _first_only  # type: ignore[method-assign]
        try:
            creds = resolve_nous_runtime_credentials()
        finally:
            nas.handler = original  # type: ignore[method-assign]
        assert creds["base_url"].startswith(WELCOME)
        assert nas.creates() == 2
        assert _load_auth_store()["providers"]["nous"]["anon_token"] != first["anon_token"]

    def test_a_5xx_or_non_json_body_is_a_retryable_server_error(self, nas):
        nas.create_response = httpx.Response(500, text="<html>oops</html>")
        err = _mint_error(nas)
        assert err.code == anon_auth.ANON_SERVER_ERROR and err.retryable is not False
        assert "hiccup" in str(err)

    def test_a_bare_401_on_sign_up_rides_the_ladder_rather_than_dying_for_the_process(self, nas):
        nas.create_response = httpx.Response(401, json={})
        err = _mint_error(nas)
        assert err.code == anon_auth.ANON_CREDENTIAL_DEAD
        assert anon_auth.last_mint_failure()["retryable"] is True

    def test_the_wire_failing_is_unreachable(self, nas):
        nas.raise_transport = httpx.ConnectTimeout("no route")
        err = _mint_error(nas)
        assert err.code == anon_auth.ANON_UNREACHABLE
        assert isinstance(err.__cause__, httpx.ConnectTimeout)
        assert "internet connection" in str(err)


# --- The cooldown memo ---------------------------------------------------------------------------


class TestMintCooldown:
    def test_a_transient_failure_is_retried_after_its_cooldown_not_never(self, nas, monkeypatch):
        nas.raise_transport = httpx.ConnectTimeout("no route")
        _mint_error(nas)
        assert anon_auth.ensure_portal_identity(explicit=True) is None     # inside the cooldown
        assert nas.creates() == 1
        failure = anon_auth._mint_failure_for_profile()
        assert failure.retry_after == anon_auth._MINT_RETRY_LADDER[0]
        monkeypatch.setattr(anon_auth.time, "monotonic", lambda: failure.not_before + 1)
        nas.raise_transport = None
        state = anon_auth.ensure_portal_identity(explicit=True)             # the cooldown passed
        assert anon_auth.is_guest_state(state)
        assert nas.creates() == 2
        assert anon_auth.last_mint_failure() is None                        # success clears the memo

    def test_repeated_transient_failures_climb_the_ladder(self, nas, monkeypatch):
        nas.raise_transport = httpx.ConnectTimeout("no route")
        waits = []
        for _ in range(4):
            _mint_error(nas)
            failure = anon_auth._mint_failure_for_profile()
            waits.append(failure.retry_after)
            monkeypatch.setattr(anon_auth.time, "monotonic", lambda f=failure: f.not_before + 1)
        assert waits == [15.0, 60.0, 300.0, 300.0]

    def test_the_users_own_retry_bypasses_the_cooldown_once(self, nas):
        nas.raise_transport = httpx.ConnectTimeout("no route")
        _mint_error(nas)
        nas.raise_transport = None
        assert anon_auth.ensure_portal_identity(explicit=True) is None
        assert anon_auth.is_guest_state(anon_auth.ensure_portal_identity(explicit=True, force=True))
        assert nas.creates() == 2

    def test_a_terminal_code_ignores_even_a_forced_retry_by_making_one_attempt_only(self, nas):
        nas.create_response = httpx.Response(404, json={"error": "not_found"})
        _mint_error(nas)
        with pytest.raises(anon_auth.AuthError):
            anon_auth.ensure_portal_identity(explicit=True, force=True)     # the click = one attempt
        assert nas.creates() == 2
        assert anon_auth.ensure_portal_identity(explicit=True) is None      # and still terminal


# --- The boot record and its retries -------------------------------------------------------------


class TestBootstrapRecord:
    def test_the_record_carries_the_code_and_the_wait(self, nas):
        nas.create_response = httpx.Response(
            429, json={"error": "temporarily_unavailable"}, headers={"Retry-After": "30"})
        record = free_tier_bootstrap.run_bootstrap(announce=False)
        assert record.has_identity is False and record.free_tier is False
        assert record.failure["error_code"] == anon_auth.ANON_RATE_LIMITED
        assert record.failure["retryable"] is True and 28 <= record.failure["retry_after"] <= 30
        assert record.failure_fields() == {
            "error": record.error, "error_code": anon_auth.ANON_RATE_LIMITED, "retryable": True,
            "retry_after": record.failure["retry_after"]}

    def test_a_clean_boot_carries_no_failure_block(self, nas):
        record = free_tier_bootstrap.run_bootstrap(announce=False)
        assert record.free_tier is True and record.failure_fields() == {}

    def test_the_background_loop_retries_a_transient_failure_until_it_settles(self, nas, monkeypatch):
        nas.raise_transport = httpx.ConnectTimeout("no route")
        slept = []

        def _sleep(seconds):
            slept.append(seconds)
            # The cooldown passes while we "slept".
            failure = anon_auth._mint_failure_for_profile()
            monkeypatch.setattr(anon_auth.time, "monotonic", lambda f=failure: f.not_before + 1)
            if len(slept) == 2:
                nas.raise_transport = None          # the network comes back on the second wait
        monkeypatch.setattr(free_tier_bootstrap, "_sleep", _sleep)
        free_tier_bootstrap._bootstrap_then_retry()
        record = free_tier_bootstrap.current_record()
        assert record.has_identity is True and record.free_tier is True
        assert slept == [15, 60]
        assert nas.creates() == 3

    def test_the_background_loop_never_retries_a_terminal_code(self, nas, monkeypatch):
        nas.create_response = httpx.Response(404, json={"error": "not_found"})
        monkeypatch.setattr(free_tier_bootstrap, "_sleep", lambda s: pytest.fail("must not sleep"))
        free_tier_bootstrap._bootstrap_then_retry()
        assert free_tier_bootstrap.current_record().failure["error_code"] == anon_auth.ANON_GATE_CLOSED
        assert nas.creates() == 1

    def test_the_background_loop_is_bounded(self, nas, monkeypatch):
        nas.raise_transport = httpx.ConnectTimeout("no route")

        def _sleep(seconds):
            failure = anon_auth._mint_failure_for_profile()
            monkeypatch.setattr(anon_auth.time, "monotonic", lambda f=failure: f.not_before + 1)
        monkeypatch.setattr(free_tier_bootstrap, "_sleep", _sleep)
        free_tier_bootstrap._bootstrap_then_retry()
        assert nas.creates() == 1 + free_tier_bootstrap.BOOTSTRAP_RETRY_ATTEMPTS
        assert free_tier_bootstrap.current_record().failure["error_code"] == anon_auth.ANON_UNREACHABLE

    def test_a_retry_re_inventories_so_a_provider_connected_meanwhile_keeps_inference(self, nas, monkeypatch):
        nas.raise_transport = httpx.ConnectTimeout("no route")
        free_tier_bootstrap.run_bootstrap(announce=False)
        # The user connected their own provider during the cooldown.
        monkeypatch.setattr(free_tier_bootstrap, "_inventory_other_providers", lambda: True)
        nas.raise_transport = None
        record = free_tier_bootstrap.retry_bootstrap_mint(force=True, announce=False)
        assert record.has_identity is True and record.other_providers is True
        assert _load_auth_store().get("active_provider") != "nous"

    def test_a_late_failed_build_never_overwrites_a_success_that_landed_meanwhile(self, nas, monkeypatch):
        """The background loop and the user's click can race: the loop's build (no identity, still
        in cooldown) must not replace the record the click just wrote."""
        nas.raise_transport = httpx.ConnectTimeout("no route")
        free_tier_bootstrap.run_bootstrap(announce=False)
        real_build = free_tier_bootstrap._build_record

        def slow_build(**kw):
            stale = real_build(**kw)                       # no identity: still inside the cooldown
            monkeypatch.setattr(free_tier_bootstrap, "_build_record", real_build)
            nas.raise_transport = None                    # ...meanwhile the click's forced mint succeeds
            free_tier_bootstrap.retry_bootstrap_mint(force=True, announce=False)
            return stale
        monkeypatch.setattr(free_tier_bootstrap, "_build_record", slow_build)
        record = free_tier_bootstrap.retry_bootstrap_mint(force=False, announce=False)
        assert record.has_identity is True
        assert free_tier_bootstrap.current_record().has_identity is True

    def test_the_desktop_retry_refreshes_the_boot_record(self, nas):
        nas.raise_transport = httpx.ConnectTimeout("no route")
        free_tier_bootstrap.run_bootstrap(announce=False)
        nas.raise_transport = None
        record = free_tier_bootstrap.retry_bootstrap_mint(force=True, announce=False)
        assert record.free_tier is True and record.failure_fields() == {}
        assert free_tier_bootstrap.current_record() is record


# --- Signing in while the service misbehaves ------------------------------------------------------


class TestSignInFailures:
    @pytest.mark.parametrize("code,retry_after,retryable,needle", [
        (anon_auth.ANON_RATE_LIMITED, 45, True, "busy"),
        (anon_auth.ANON_GATE_PAUSED, 0, True, "busy"),
        (anon_auth.ANON_UNREACHABLE, 0, True, "internet connection"),
        (anon_auth.ANON_GATE_CLOSED, 0, False, "Nous account"),
        (anon_auth.ANON_POW_REQUIRED, 0, False, "proof of work"),
    ])
    def test_a_service_verdict_keeps_its_code_wait_and_copy(self, code, retry_after, retryable, needle):
        state = anon_sign_in._failed_from_exception(
            anon_auth._anon_err("x", code, retry_after=retry_after or None))
        assert state.kind == "failed" and state.reason == code
        assert state.retryable is retryable
        assert state.retry_after == retry_after
        assert needle in state.copy
        assert "Hermes " not in state.copy and "http" not in state.copy

    def test_the_wire_failing_reads_as_unreachable(self):
        state = anon_sign_in._failed_from_exception(httpx.ReadTimeout("slow"))
        assert state.reason == anon_auth.ANON_UNREACHABLE and state.retryable is True

    def test_an_unnamed_local_failure_keeps_the_generic_copy_and_its_terminal_detail(self):
        state = anon_sign_in._failed_from_exception(RuntimeError("bad CA bundle"))
        assert state.reason == "" and state.copy == anon_sign_in.UPGRADE_NOT_COMPLETED
        assert state.copy_terminal == "Sign-in failed: bad CA bundle"

    def test_account_busy_is_retryable(self):
        state = anon_sign_in.Failed(reason="account_busy")
        assert state.retryable is True and "few seconds" in state.copy


# --- The dev-only host override ------------------------------------------------------------------


def test_extra_welcome_hosts_make_a_local_stand_in_the_welcome_host(monkeypatch):
    """``HERMES_EXTRA_WELCOME_HOSTS`` (env-only) extends the ROUTE predicate so a rehearsal against a
    local stand-in gets the free tier's rules — including the route-keyed dark-tier 403."""
    monkeypatch.delenv("HERMES_EXTRA_WELCOME_HOSTS", raising=False)
    assert anon_auth.route_is_welcome_host("http://127.0.0.1:8765/v1") is False
    monkeypatch.setenv("HERMES_EXTRA_WELCOME_HOSTS", "127.0.0.1, localhost")
    assert anon_auth.route_is_welcome_host("http://127.0.0.1:8765/v1") is True
    assert anon_auth.route_is_welcome_host("http://localhost:9/v1") is True
    assert anon_auth.welcome_route_refusal(403, "You tried to access something", "http://127.0.0.1:8765/v1") == "tier_disabled"
    assert anon_auth.route_is_welcome_host("https://inference-api.nousresearch.com/v1") is False


# --- The spoken wait -----------------------------------------------------------------------------


@pytest.mark.parametrize("seconds,expected", [
    (0, "a few seconds"), (5, "a few seconds"), (30, "about a minute"), (89, "about a minute"),
    (300, "about 5 minutes"), (3599, "about 60 minutes"), (3600, "about an hour"), (7200, "about 2 hours"),
    ("nope", "a few seconds"),
])
def test_friendly_wait_never_prints_raw_seconds(seconds, expected):
    assert anon_auth.friendly_wait(seconds) == expected


def test_retry_after_is_the_whole_wait_not_the_wait_plus_float_dust(monkeypatch):
    """``(now + 60) - now`` is 60.000000000000455 for this ``now``; the payload must still say 60."""
    now = 4080.78551427515
    monkeypatch.setattr(anon_auth.time, "monotonic", lambda: now)
    failure = anon_auth.MintFailure(code=anon_auth.ANON_SERVER_ERROR, message="x", retryable=True,
                                    retry_after=60.0, not_before=now + 60.0)
    assert failure.as_payload()["retry_after"] == 60
