"""Nous free tier: the browser challenge in front of a token exchange (``hermes_cli.anon_challenge``).

Driven through the fake NAS with Hermes' real client code: the exchange hits a 428, the challenge is
presented and polled OUTSIDE the auth locks, and the exchange runs once more.
"""

from __future__ import annotations

import httpx
import pytest

from hermes_cli import anon_auth, anon_challenge

from tests.hermes_cli.anon_portal import PORTAL, install_portal  # noqa: F401


@pytest.fixture
def nas(monkeypatch, tmp_path):
    return install_portal(monkeypatch, tmp_path)


@pytest.fixture
def presented(monkeypatch):
    seen: list[anon_challenge.BrowserChallenge] = []
    monkeypatch.setattr(anon_challenge, "_presenter", seen.append)
    return seen


def _resolve():
    from hermes_cli.auth_nous import resolve_nous_runtime_credentials
    assert anon_auth.is_guest_state(anon_auth.ensure_portal_identity(explicit=True))
    return resolve_nous_runtime_credentials()


class TestWhatTheClientSends:
    def test_the_exchange_reports_surface_version_and_the_browser_capability(self, nas):
        _resolve()
        sent = nas.token_requests[-1]
        assert sent["body"]["client"]["capabilities"] == ["browser-v1"]
        assert sent["body"]["client"]["surface"] == "cli"
        assert sent["body"]["client"]["name"] == "hermes-agent"
        assert sent["user_agent"].startswith("hermes-agent/") and "(cli;" in sent["user_agent"]

    def test_the_desktop_backend_reports_the_desktop_surface(self, nas, monkeypatch):
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        _resolve()
        assert nas.token_requests[-1]["body"]["client"]["surface"] == "desktop"
        assert "(desktop;" in nas.token_requests[-1]["user_agent"]

    def test_the_messaging_gateway_reports_the_gateway_surface(self, nas, monkeypatch):
        """The gateway process has no console and no desktop; it must not read as ``cli`` (whose
        rules may expect a human at a terminal) or print challenge links into its logs."""
        import gateway.status as status
        monkeypatch.setattr(status, "owns_gateway_runtime_lock", lambda: True)
        _resolve()
        assert nas.token_requests[-1]["body"]["client"]["surface"] == "gateway"


class TestRequiredChallenge:
    def test_a_host_failure_wakes_the_waiter_but_still_requires_an_authoritative_mint(self, nas, monkeypatch):
        nas.challenge_required = True
        nas.challenge_never_clears = True

        def failed(challenge):
            assert anon_challenge.record_host_outcome(challenge.url, challenge.attempt, "error")
        monkeypatch.setattr(anon_challenge, "_presenter", failed)
        with pytest.raises(anon_auth.AuthError, match="couldn't finish"):
            _resolve()
        assert [p for _, p in nas.calls].count("/api/anonymous/token") == 2
        assert [p for _, p in nas.calls].count("/api/anonymous/challenge/status") == 1
        # A window that crashed is not a reason to let a boot burst of callers each re-mint and
        # reopen a window: the second 428 starts the same fail-fast cooldown as a timeout.
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        with pytest.raises(anon_auth.AuthError):
            resolve_nous_runtime_credentials()
        assert [p for _, p in nas.calls].count("/api/anonymous/challenge/status") == 1

    def test_a_poll_the_service_refuses_ends_the_wait_instead_of_running_out_the_clock(self, nas, presented):
        """404 ``unknown_token`` / 400 on the status poll: the credential is gone (or the surface is
        off), so no amount of waiting clears it. The wait ends and the re-mint says what happened."""
        nas.challenge_required = True
        nas.challenge_never_clears = True
        nas.challenge_statuses = ["pending"]          # the "is it already settled?" read
        original = nas.handler

        def refuse_once_pending_is_spent(request):
            if request.url.path.endswith("/challenge/status") and not nas.challenge_statuses:
                nas.calls.append((request.method, request.url.path))
                return httpx.Response(404, json={"error": "unknown_token"})
            return original(request)
        nas.handler = refuse_once_pending_is_spent
        with pytest.raises(anon_auth.AuthError) as exc:
            _resolve()
        assert exc.value.code == anon_auth.ANON_CHALLENGE_REQUIRED
        assert [p for _, p in nas.calls].count("/api/anonymous/challenge/status") == 2

    def test_the_status_poll_trusts_what_the_credential_was_minted_with(self, nas, presented, monkeypatch):
        """A guest whose ``auth.json`` carries a ``tls`` block polls under it (a private CA, say),
        not under a fresh default context that would fail every read and run out the clock."""
        from hermes_cli import auth_nous
        from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
        assert anon_auth.is_guest_state(anon_auth.ensure_portal_identity(explicit=True))
        with _auth_store_lock():
            store = _load_auth_store()
            store["providers"]["nous"]["tls"] = {"insecure": True}
            _save_auth_store(store)
        polled_with = []
        real = auth_nous._nous_http_client
        monkeypatch.setattr(auth_nous, "_nous_http_client",
                            lambda timeout, verify: polled_with.append(verify) or real(timeout, verify))
        nas.challenge_required = True
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        resolve_nous_runtime_credentials()
        assert [p for _, p in nas.calls].count("/api/anonymous/challenge/status") >= 1
        assert polled_with and all(verify is False for verify in polled_with)

    def test_a_stale_status_cannot_hide_a_clearance_from_the_final_mint(self, nas, presented, monkeypatch):
        nas.challenge_required = True
        nas.challenge_never_clears = True
        monkeypatch.setattr(anon_challenge, "CHALLENGE_WAIT_SECONDS", 0)
        original = nas.handler

        def cleared_after_status(request):
            response = original(request)
            if request.url.path.endswith("/challenge/status"):
                nas.challenge_required = False
            return response
        nas.handler = cleared_after_status
        assert _resolve()["api_key"]
        assert [p for _, p in nas.calls].count("/api/anonymous/token") == 2
        assert anon_challenge.pending_challenge() is None
    def test_a_challenge_is_presented_polled_and_the_exchange_retried(self, nas, presented):
        nas.challenge_required = True
        nas.challenge_statuses = ["pending", "pending", "needs_interaction"]
        creds = _resolve()
        assert creds["api_key"]
        assert [c.url for c in presented] == [nas.challenge_url]
        assert presented[0].required and presented[0].message == "A quick check first."
        paths = [p for _, p in nas.calls]
        assert paths.count("/api/anonymous/token") == 2
        # the three scripted states (the first answers "is it already settled?") + the settling "none"
        assert paths.count("/api/anonymous/challenge/status") == 4
        assert anon_challenge.pending_challenge() is None

    def test_the_wait_happens_outside_the_auth_locks(self, nas, monkeypatch):
        """A sibling process must be able to take the shared-store lock while the page runs."""
        from hermes_cli.auth_nous import _nous_shared_store_lock
        nas.challenge_required = True
        nas.challenge_statuses = ["pending"]

        def present(_challenge):
            with _nous_shared_store_lock(timeout_seconds=0.5):
                pass
        monkeypatch.setattr(anon_challenge, "_presenter", present)
        assert _resolve()["api_key"]

    def test_status_shows_the_pending_challenge_while_it_is_worked(self, nas, monkeypatch):
        nas.challenge_required = True
        nas.challenge_statuses = ["pending"]
        seen = []
        monkeypatch.setattr(anon_challenge, "_presenter", lambda _c: seen.append(anon_challenge.pending_challenge()))
        _resolve()
        assert seen == [{"type": "browser", "url": nas.challenge_url, "required": True,
                         "expires_in": 600, "message": "A quick check first.", "attempt": 0}]

    def test_a_challenge_that_never_clears_is_a_retryable_error_not_a_hang(self, nas, presented, monkeypatch):
        nas.challenge_required = True
        nas.challenge_never_clears = True
        monkeypatch.setattr(anon_challenge, "CHALLENGE_WAIT_SECONDS", 0.05)
        monkeypatch.setattr(anon_challenge, "_sleep", lambda _seconds: __import__("time").sleep(0.02))
        with pytest.raises(anon_auth.AuthError) as exc:
            _resolve()
        assert exc.value.code == anon_auth.ANON_CHALLENGE_REQUIRED and exc.value.retryable is True
        assert "browser" in str(exc.value)
        # The callers queued behind it fail fast instead of each parking for a full wait.
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        with pytest.raises(anon_auth.AuthError):
            resolve_nous_runtime_credentials()
        assert len(presented) == 1

    def test_a_second_challenge_after_the_retry_is_not_looped_on(self, nas, presented):
        nas.challenge_required = True
        nas.challenge_statuses = ["failed"]          # the page ruled; the ticket left the waiting states

        original = nas.handler

        def still_challenged(request):
            response = original(request)
            nas.challenge_required = True            # NAS keeps answering 428
            return response
        nas.handler = still_challenged
        with pytest.raises(anon_auth.AuthError) as exc:
            _resolve()
        assert exc.value.code == anon_auth.ANON_CHALLENGE_REQUIRED
        assert [p for _, p in nas.calls].count("/api/anonymous/token") == 2

    def test_the_managed_tool_token_path_works_a_challenge_too(self, nas, presented):
        from hermes_cli.auth import resolve_nous_access_token
        assert anon_auth.is_guest_state(anon_auth.ensure_portal_identity(explicit=True))
        nas.challenge_required = True
        nas.challenge_statuses = ["pending"]
        assert resolve_nous_access_token()
        assert len(presented) == 1


class TestWhoWaits:
    def test_a_background_caller_announces_and_returns_at_once(self, nas, presented):
        """A keepalive tick or a status paint has nobody waiting on it: no poll, no parked thread."""
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        assert anon_auth.is_guest_state(anon_auth.ensure_portal_identity(explicit=True))
        nas.challenge_required = True
        nas.challenge_never_clears = True
        with anon_challenge.background_caller():
            with pytest.raises(anon_auth.AuthError) as exc:
                resolve_nous_runtime_credentials()
        assert exc.value.code == anon_auth.ANON_CHALLENGE_REQUIRED and exc.value.retryable is True
        assert len(presented) == 1
        assert "/api/anonymous/challenge/status" not in [p for _, p in nas.calls]
        assert anon_challenge.pending_challenge()["url"] == nas.challenge_url

    def test_a_background_caller_never_prints_or_opens_a_browser(self, monkeypatch, capsys):
        import webbrowser
        monkeypatch.setattr(webbrowser, "open", lambda _url: pytest.fail("opened a browser in the background"))
        challenge = anon_challenge.BrowserChallenge(f"{PORTAL}/challenge?code=t", True, 600, 2, "A quick check.")
        with anon_challenge.background_caller():
            anon_challenge.present(challenge)
        assert capsys.readouterr().err == ""

    def test_the_keepalive_and_the_status_paint_are_background_callers(self, nas, presented):
        from hermes_cli.auth_nous import _compute_nous_auth_status
        from hermes_cli.nous_auth_keepalive import refresh_nous_auth_keepalive_once
        assert anon_auth.is_guest_state(anon_auth.ensure_portal_identity(explicit=True))
        nas.challenge_required = True
        nas.challenge_never_clears = True
        assert refresh_nous_auth_keepalive_once() is False
        _compute_nous_auth_status()
        assert "/api/anonymous/challenge/status" not in [p for _, p in nas.calls]

    def test_a_poll_error_is_a_blip_not_a_verdict(self, nas, presented):
        nas.challenge_required = True
        nas.challenge_status_errors = [429]              # the "is it already settled?" read
        nas.challenge_statuses = ["pending"]
        nas.challenge_status_errors += [500]
        assert _resolve()["api_key"]
        assert len(presented) == 1                       # the 429 did not skip presenting

    def test_a_desktop_flag_without_a_gateway_falls_back_to_the_terminal(self, monkeypatch, capsys):
        """HERMES_DESKTOP is inherited by CLIs the desktop spawns; with no gateway in this process
        there is nobody to announce to, so the link is printed instead of silently waiting."""
        import sys
        import webbrowser
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv("SSH_TTY", "/dev/pts/0")
        monkeypatch.delitem(sys.modules, "tui_gateway.server", raising=False)
        monkeypatch.setattr(webbrowser, "open", lambda _url: False)
        anon_challenge.present(anon_challenge.BrowserChallenge(f"{PORTAL}/challenge?code=t", True, 600, 2, "m"))
        assert f"{PORTAL}/challenge?code=t" in capsys.readouterr().err


class TestOtherEndpointsAreUnchanged:
    def test_a_428_on_sign_up_is_still_the_proof_of_work_verdict(self, nas):
        nas.create_response = httpx.Response(428, json={"error": "challenge_required", "challenges": []})
        with pytest.raises(anon_auth.AuthError) as exc:
            anon_auth.ensure_portal_identity(explicit=True)
        assert exc.value.code == anon_auth.ANON_POW_REQUIRED

    def test_a_403_on_sign_up_stays_retryable(self, nas):
        nas.create_response = httpx.Response(403, json={"error": "access_denied"})
        with pytest.raises(anon_auth.AuthError) as exc:
            anon_auth.ensure_portal_identity(explicit=True)
        assert exc.value.code == anon_auth.ANON_SERVER_ERROR and exc.value.retryable is True


class TestWhatIsNeverOpened:
    def test_a_challenge_url_off_the_portal_origin_is_refused(self, nas, presented):
        nas.challenge_required = True
        nas.challenge_url = "https://evil.example/challenge?code=x"
        with pytest.raises(anon_auth.AuthError) as exc:
            _resolve()
        assert exc.value.code == anon_auth.ANON_SIGNIN_REQUIRED and exc.value.retryable is False
        assert presented == []

    @pytest.mark.parametrize("url", [
        "javascript:alert(1)", "file:///etc/passwd", f"{PORTAL}.evil.example/challenge",
        PORTAL.replace("https://", "http://") + "/challenge", "", None])
    def test_same_origin_means_scheme_and_host(self, url):
        payload = {"challenges": [{"type": "browser", "url": url}]}
        assert anon_challenge.parse_browser_challenge(payload, PORTAL) is None

    def test_an_unknown_challenge_type_falls_back_to_sign_in(self, nas, presented):
        nas.token_response = httpx.Response(428, json={
            "error": "challenge_required", "message": "Attest your device.",
            "challenges": [{"type": "attestation-v9", "url": f"{PORTAL}/x"}]})
        with pytest.raises(anon_auth.AuthError) as exc:
            _resolve()
        assert exc.value.code == anon_auth.ANON_SIGNIN_REQUIRED
        assert "Sign in" in str(exc.value) and "Attest" not in str(exc.value)

    def test_an_unknown_428_is_a_sign_in_not_the_proof_of_work_sentence(self, nas):
        nas.token_response = httpx.Response(428, json={"error": "something_new"})
        with pytest.raises(anon_auth.AuthError) as exc:
            _resolve()
        assert exc.value.code == anon_auth.ANON_SIGNIN_REQUIRED
        assert "proof of work" not in str(exc.value)

    @pytest.mark.parametrize("error", ["signin_required", "access_denied"])
    def test_a_refusal_uses_the_services_own_words(self, nas, error):
        nas.token_response = httpx.Response(403, json={"error": error, "message": "Not from here, sorry."})
        with pytest.raises(anon_auth.AuthError) as exc:
            _resolve()
        assert exc.value.code == anon_auth.ANON_SIGNIN_REQUIRED and str(exc.value) == "Not from here, sorry."

    def test_network_numbers_are_kept_finite_and_sane(self):
        def parsed(expires_in):
            return anon_challenge.parse_browser_challenge(
                {"challenges": [{"type": "browser", "url": f"{PORTAL}/challenge?code=t",
                                 "expires_in": expires_in}]}, PORTAL)
        assert parsed(float("inf")).expires_in == 600
        assert parsed(10 ** 12).expires_in == 900
        assert parsed(0).expires_in == 30
        assert parsed(True).expires_in == 600
        assert parsed(float("inf")).as_payload()["expires_in"] == 600

    def test_control_characters_never_reach_the_terminal(self):
        assert anon_challenge.server_message("Check\x1b[2J this") == "Check [2J this"
        payload = {"challenges": [{"type": "browser", "url": f"{PORTAL}/challenge?code=\x1b]0;x"}]}
        assert anon_challenge.parse_browser_challenge(payload, PORTAL) is None

    def test_server_copy_is_only_used_when_it_looks_like_copy(self):
        assert anon_challenge.server_message("  Two\n lines  ") == "Two lines"
        assert anon_challenge.server_message("x" * 301) == ""
        assert anon_challenge.server_message({"not": "text"}) == ""


class TestOptionalChallenge:
    def test_a_terminal_never_opens_anything_for_an_optional_challenge(self, nas, presented, monkeypatch):
        announced = []
        monkeypatch.setattr(anon_challenge, "_announce", announced.append)
        nas.optional_challenge = True
        assert _resolve()["api_key"]
        assert presented == [] and announced == []

    def test_the_desktop_backend_announces_it_and_waits_on_nothing(self, nas, presented, monkeypatch):
        announced = []
        monkeypatch.setattr(anon_challenge, "_announce", announced.append)
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        nas.optional_challenge = True
        assert _resolve()["api_key"]
        assert [c.required for c in announced] == [False]
        assert "/api/anonymous/challenge/status" not in [p for _, p in nas.calls]
        assert anon_challenge.pending_challenge()["url"] == nas.challenge_url

    def test_replayed_offers_keep_the_original_expiry_and_old_results_cannot_end_a_retry(self, nas, monkeypatch):
        clock = [100.0]
        monkeypatch.setattr(anon_challenge.time, "monotonic", lambda: clock[0])
        challenge = anon_challenge.BrowserChallenge(nas.challenge_url, False, 60, 2, "check")
        anon_challenge._set_pending(challenge)
        clock[0] += 20
        anon_challenge._set_pending(challenge)
        assert anon_challenge.pending_challenge()["expires_in"] == 40
        retry = anon_challenge._set_pending(challenge, new_attempt=True)
        assert not anon_challenge.record_host_outcome(challenge.url, 0, "done")
        assert anon_challenge.pending_challenge()["attempt"] == retry.attempt
        clock[0] += 41
        assert anon_challenge.pending_challenge() is None


class TestPresenting:
    def test_the_desktop_backend_announces_and_never_opens_a_browser(self, monkeypatch):
        announced, opened = [], []
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setattr(anon_challenge, "_announce", lambda c: announced.append(c) or True)
        monkeypatch.setattr(anon_challenge, "_present_in_terminal", opened.append)
        challenge = anon_challenge.BrowserChallenge(f"{PORTAL}/challenge?code=t", True, 600, 2, "m")
        anon_challenge.present(challenge)
        assert announced == [challenge] and opened == []

    def test_a_terminal_opens_one_tab_per_ticket_however_many_attempts_resume_it(self, monkeypatch, capsys):
        import webbrowser
        from hermes_cli import auth_device_flow
        opened = []
        monkeypatch.setattr(auth_device_flow, "_is_remote_session", lambda: False)
        monkeypatch.setattr(auth_device_flow, "_can_open_graphical_browser", lambda: True)
        monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
        challenge = anon_challenge.BrowserChallenge(f"{PORTAL}/challenge?code=once", True, 600, 2, "m")
        anon_challenge.present(challenge)
        anon_challenge.present(challenge)
        assert opened == [challenge.url]
        assert capsys.readouterr().err.count(challenge.url) == 2     # the link is always shown

    def test_a_terminal_prints_the_link_and_skips_the_browser_over_ssh(self, monkeypatch, capsys):
        import webbrowser
        monkeypatch.setenv("SSH_TTY", "/dev/pts/0")
        monkeypatch.setattr(webbrowser, "open", lambda _url: pytest.fail("opened a browser over SSH"))
        challenge = anon_challenge.BrowserChallenge(f"{PORTAL}/challenge?code=t", True, 600, 2, "A quick check.")
        anon_challenge.present(challenge)
        err = capsys.readouterr().err
        assert "A quick check." in err and f"{PORTAL}/challenge?code=t" in err
