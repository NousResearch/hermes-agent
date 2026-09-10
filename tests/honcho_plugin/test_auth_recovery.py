"""Tests for OAuth 401 recovery: prompt exchange retry, invalid_grant handling,
forced refresh + single retry on sync and dialectic, backoff exemption, and
the one-time user-facing notice."""

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho import oauth
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import (
    HonchoAuthError,
    HonchoSession,
    HonchoSessionManager,
)
from plugins.memory.honcho.session_auth import _is_auth_error


def _host_block(refresh="hch-rt-old", expires_at=100):
    return {
        "apiKey": "hch-at-old",
        "oauth": {
            "refreshToken": refresh,
            "expiresAt": expires_at,
            "clientId": "hermes-desktop",
            "tokenEndpoint": "http://localhost:8000/oauth/token",
            "scope": "write",
            "tokenType": "Bearer",
        },
    }


def _write(path: Path, raw: dict) -> None:
    path.write_text(json.dumps(raw), encoding="utf-8")


def _rotated_body(n=1):
    return {
        "access_token": f"hch-at-new{n}",
        "refresh_token": f"hch-rt-new{n}",
        "expires_in": 3600,
        "scope": "write",
        "token_type": "Bearer",
    }


@pytest.fixture(autouse=True)
def _reset_oauth_module_state():
    """Module-level oauth dicts persist across tests in one process; reset so
    dead grants / cooldowns / memoized verdicts can't leak between tests."""
    yield
    oauth._dead_grants.clear()
    oauth._refresh_failure_at.clear()
    oauth._reauth_check_cache.clear()
    oauth._expiry_cache.clear()


# ---------------------------------------------------------------------------
# oauth: transient vs permanent exchange failures
# ---------------------------------------------------------------------------


class TestExchangeRetry:
    def test_transient_failure_recovers_on_immediate_retry(self, tmp_path, monkeypatch):
        """A timed-out exchange retries right away — the server honors the
        replayed refresh token only within its rotation grace window."""
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block()}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)

        calls = []

        def flaky(url, data, timeout):
            calls.append(data["refresh_token"])
            if len(calls) == 1:
                raise TimeoutError("token exchange timed out")
            return 200, _rotated_body()

        monkeypatch.setattr(oauth, "_http_post_form_status", flaky)
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)

        assert token == "hch-at-new1" and refreshed is True
        assert calls == ["hch-rt-old", "hch-rt-old"]
        saved = json.loads(path.read_text())["hosts"]["hermes"]
        assert saved["oauth"]["refreshToken"] == "hch-rt-new1"

    def test_invalid_grant_stops_retries_and_marks_reauth_required(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block()}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)

        calls = []

        def revoked(url, data, timeout):
            calls.append(1)
            return 400, {"error": "invalid_grant", "error_description": "grant revoked"}

        monkeypatch.setattr(oauth, "_http_post_form_status", revoked)
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)

        # Fail-open return, but no retry of a permanently rejected grant.
        assert token == "hch-at-old" and refreshed is False
        assert len(calls) == 1
        assert oauth.reauth_required(path, "hermes") is True

        # Later refresh attempts skip the endpoint entirely.
        token2, refreshed2 = oauth.ensure_fresh_token(path, "hermes", now=2000)
        assert token2 == "hch-at-old" and refreshed2 is False
        assert len(calls) == 1

        # The forced (post-401) path refuses a dead grant too.
        assert oauth.force_refresh_token(path, "hermes") is None
        assert len(calls) == 1

    def test_relogin_clears_the_dead_grant(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block()}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)
        monkeypatch.setattr(
            oauth, "_http_post_form_status",
            lambda *a, **k: (400, {"error": "invalid_grant"}),
        )
        oauth.ensure_fresh_token(path, "hermes", now=1000)
        assert oauth.reauth_required(path, "hermes") is True

        oauth.install_grant(
            path, "hermes",
            {"access_token": "hch-at-fresh", "refresh_token": "hch-rt-fresh", "expires_in": 3600},
            client_id="hermes-desktop",
            token_endpoint="http://localhost:8000/oauth/token",
            now=2000,
        )
        assert oauth.reauth_required(path, "hermes") is False
        token, _ = oauth.ensure_fresh_token(path, "hermes", now=2000)
        assert token == "hch-at-fresh"

    def test_error_body_is_logged(self, tmp_path, monkeypatch, caplog):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block()}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)
        monkeypatch.setattr(
            oauth, "_http_post_form_status",
            lambda *a, **k: (400, {"error": "invalid_grant", "error_description": "grant revoked"}),
        )
        with caplog.at_level(logging.WARNING, logger="plugins.memory.honcho.oauth"):
            oauth.ensure_fresh_token(path, "hermes", now=1000)
        assert "invalid_grant" in caplog.text
        assert "grant revoked" in caplog.text

    def test_redaction_strips_token_values(self):
        redacted = oauth.redact_tokens(
            "exchange failed for hch-rt-supersecret123 got hch-at-alsosecret456"
        )
        assert "supersecret123" not in redacted
        assert "alsosecret456" not in redacted
        assert "hch-rt-[redacted]" in redacted
        assert "hch-at-[redacted]" in redacted


class TestForceRefreshToken:
    def test_rotates_despite_local_validity(self, tmp_path, monkeypatch):
        """A server-side 401 forces a rotation even when the local clock says
        the token is still live."""
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(expires_at=time.time() + 3600)}})
        monkeypatch.setattr(
            oauth, "_http_post_form_status", lambda *a, **k: (200, _rotated_body())
        )
        token = oauth.force_refresh_token(path, "hermes")
        assert token == "hch-at-new1"
        saved = json.loads(path.read_text())["hosts"]["hermes"]
        assert saved["apiKey"] == "hch-at-new1"

    def test_adopts_concurrent_rotation_without_exchange(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        far = time.time() + 7200
        _write(path, {"hosts": {"hermes": _host_block(expires_at=far)}})
        # Seed the expiry cache with the old token.
        oauth.ensure_fresh_token(path, "hermes")
        # Another process rotated the credential on disk.
        rotated = _host_block(refresh="hch-rt-2", expires_at=far)
        rotated["apiKey"] = "hch-at-2"
        _write(path, {"hosts": {"hermes": rotated}})
        monkeypatch.setattr(
            oauth, "_http_post_form_status",
            lambda *a, **k: pytest.fail("must adopt the on-disk rotation, not exchange"),
        )
        assert oauth.force_refresh_token(path, "hermes") == "hch-at-2"

    def test_transient_failure_returns_none(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(expires_at=time.time() + 3600)}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)

        def boom(*a, **k):
            raise ConnectionError("network down")

        monkeypatch.setattr(oauth, "_http_post_form_status", boom)
        assert oauth.force_refresh_token(path, "hermes") is None
        # Not permanent: a later attempt may exchange again.
        assert oauth.reauth_required(path, "hermes") is False

    def test_static_api_key_is_noop(self, tmp_path):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": {"apiKey": "hch-v3-static"}}})
        assert oauth.force_refresh_token(path, "hermes") is None


# ---------------------------------------------------------------------------
# session: auth error detection
# ---------------------------------------------------------------------------


class TestAuthErrorDetection:
    def test_matches_honcho_token_message(self):
        assert _is_auth_error(Exception("Invalid or expired access token"))

    def test_matches_status_code_attr(self):
        exc = Exception("boom")
        exc.status_code = 401
        assert _is_auth_error(exc)

    def test_matches_401_text(self):
        assert _is_auth_error(Exception("HTTP 401 Unauthorized"))

    def test_ignores_other_errors(self):
        assert not _is_auth_error(Exception("connection reset by peer"))
        assert not _is_auth_error(Exception("HTTP 500 internal error"))

    def test_bare_401_digits_are_not_auth_errors(self):
        """A false positive spends a token rotation and can revoke the grant;
        digits appearing in latency figures, request ids, or identifiers must
        never classify as auth failures."""
        for msg in (
            "Rate limited, retry after 4010 ms",
            "500 Internal Server Error (request id req-4012ab)",
            "connection timeout to workspace ws-401-prod",
            "peer 401k-planning not found",
        ):
            assert not _is_auth_error(Exception(msg)), msg

    def test_401_with_http_context_matches(self):
        assert _is_auth_error(Exception("HTTP 401"))
        assert _is_auth_error(Exception("status 401"))
        assert _is_auth_error(Exception("status_code: 401"))
        assert _is_auth_error(Exception("401 Unauthorized"))

    def test_concrete_non_auth_status_wins_over_text(self):
        exc = Exception("authentication failed")
        exc.status = 429
        assert not _is_auth_error(exc)

    def test_bare_authentication_word_is_not_enough(self):
        assert not _is_auth_error(Exception("authentication service unreachable"))


# ---------------------------------------------------------------------------
# session: dialectic 401 recovery
# ---------------------------------------------------------------------------


class _FlakyPeer:
    """chat() raises an auth error N times, then succeeds."""

    def __init__(self, failures: int, result: str = "synthesized answer"):
        self.failures = failures
        self.result = result
        self.calls = 0

    def chat(self, query, **kw):
        self.calls += 1
        if self.calls <= self.failures:
            raise Exception("Invalid or expired access token")
        return self.result


def _make_manager(peer, *, reauth_ok=True):
    cfg = HonchoClientConfig(host="hermes", api_key="hch-at-x", enabled=True)
    mgr = HonchoSessionManager(config=cfg)
    session = HonchoSession(
        key="k", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s"
    )
    mgr._cache["k"] = session
    mgr._get_or_create_peer = lambda peer_id: peer
    mgr._force_reauth = lambda: reauth_ok
    return mgr


class TestDialecticAuthRetry:
    def test_401_forces_refresh_and_retries_once(self):
        peer = _FlakyPeer(failures=1)
        mgr = _make_manager(peer)
        assert mgr.dialectic_query("k", "who is this user?") == "synthesized answer"
        assert peer.calls == 2  # original + one retry

    def test_persistent_401_raises_auth_error(self):
        peer = _FlakyPeer(failures=99)
        mgr = _make_manager(peer)
        with pytest.raises(HonchoAuthError):
            mgr.dialectic_query("k", "q")
        assert peer.calls == 2  # exactly one retry, no loop

    def test_failed_reauth_raises_without_retry(self):
        peer = _FlakyPeer(failures=99)
        mgr = _make_manager(peer, reauth_ok=False)
        with pytest.raises(HonchoAuthError):
            mgr.dialectic_query("k", "q")
        assert peer.calls == 1  # no retry without a fresh token

    def test_non_auth_errors_stay_fail_open(self):
        class _BrokenPeer:
            def chat(self, *a, **kw):
                raise Exception("connection reset by peer")

        mgr = _make_manager(_BrokenPeer())
        assert mgr.dialectic_query("k", "q") == ""

    def test_success_after_failure_clears_auth_state(self):
        peer = _FlakyPeer(failures=99)
        mgr = _make_manager(peer, reauth_ok=False)
        with pytest.raises(HonchoAuthError):
            mgr.dialectic_query("k", "q")
        assert mgr._auth_failure is not None

        peer.failures = 0
        mgr._force_reauth = lambda: True
        assert mgr.dialectic_query("k", "q") == "synthesized answer"
        assert mgr._auth_failure is None
        assert mgr.pop_auth_notice() is None


class TestForceReauth:
    def test_rotates_and_applies_to_live_client(self, tmp_path, monkeypatch):
        from plugins.memory.honcho import client as client_mod
        from plugins.memory.honcho import session as session_mod

        fake_client = object()
        applied = {}
        monkeypatch.setattr(session_mod, "get_honcho_client", lambda *a, **k: fake_client)
        monkeypatch.setattr(client_mod, "resolve_config_path", lambda: tmp_path / "honcho.json")
        monkeypatch.setattr(oauth, "force_refresh_token", lambda p, h: "hch-at-new")

        def apply(client, token):
            applied["client"] = client
            applied["token"] = token
            return True

        monkeypatch.setattr(oauth, "apply_token_to_client", apply)

        mgr = HonchoSessionManager(config=HonchoClientConfig(host="hermes"))
        assert mgr._force_reauth() is True
        assert applied == {"client": fake_client, "token": "hch-at-new"}

    def test_returns_false_when_refresh_yields_nothing(self, tmp_path, monkeypatch):
        from plugins.memory.honcho import client as client_mod

        monkeypatch.setattr(client_mod, "resolve_config_path", lambda: tmp_path / "honcho.json")
        monkeypatch.setattr(oauth, "force_refresh_token", lambda p, h: None)
        mgr = HonchoSessionManager(config=HonchoClientConfig(host="hermes"))
        assert mgr._force_reauth() is False


# ---------------------------------------------------------------------------
# session: message sync 401 recovery
# ---------------------------------------------------------------------------


class _FlakyHonchoSession:
    """add_messages() raises an auth error N times, then succeeds."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def add_messages(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise Exception("Invalid or expired access token")


def _make_sync_manager(flaky_session, *, reauth_ok=True):
    cfg = HonchoClientConfig(host="hermes", api_key="hch-at-x", enabled=True)
    mgr = HonchoSessionManager(config=cfg)
    peer = MagicMock()
    peer.message.side_effect = lambda content: content
    mgr._get_or_create_peer = lambda peer_id: peer
    mgr._sessions_cache["s"] = flaky_session
    mgr._force_reauth = lambda: reauth_ok
    session = HonchoSession(
        key="k", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s"
    )
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    return mgr, session


class TestSyncAuthRetry:
    def test_401_forces_refresh_and_retries_once(self):
        flaky = _FlakyHonchoSession(failures=1)
        mgr, session = _make_sync_manager(flaky)
        assert mgr._flush_session(session) is True
        assert flaky.calls == 2
        assert all(m["_synced"] for m in session.messages)

    def test_persistent_401_fails_and_records_auth_failure(self):
        flaky = _FlakyHonchoSession(failures=99)
        mgr, session = _make_sync_manager(flaky)
        assert mgr._flush_session(session) is False
        assert flaky.calls == 2  # exactly one retry, no loop
        assert not any(m.get("_synced") for m in session.messages)
        assert mgr._auth_failure is not None

    def test_failed_reauth_fails_without_retry(self):
        flaky = _FlakyHonchoSession(failures=99)
        mgr, session = _make_sync_manager(flaky, reauth_ok=False)
        assert mgr._flush_session(session) is False
        assert flaky.calls == 1
        assert mgr._auth_failure is not None

    def test_later_success_recovers_and_clears_auth_state(self):
        flaky = _FlakyHonchoSession(failures=2)
        mgr, session = _make_sync_manager(flaky, reauth_ok=False)
        assert mgr._flush_session(session) is False
        assert mgr._auth_failure is not None

        mgr._force_reauth = lambda: True
        assert mgr._flush_session(session) is True
        assert all(m["_synced"] for m in session.messages)
        assert mgr._auth_failure is None


# ---------------------------------------------------------------------------
# dead grant: skip calls entirely until re-login
# ---------------------------------------------------------------------------


def _kill_grant(tmp_path, monkeypatch) -> Path:
    """Revoke the grant on a tmp config and point the manager's path at it."""
    from plugins.memory.honcho import client as client_mod

    path = tmp_path / "honcho.json"
    _write(path, {"hosts": {"hermes": _host_block()}})
    monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        oauth, "_http_post_form_status",
        lambda *a, **k: (400, {"error": "invalid_grant"}),
    )
    oauth.ensure_fresh_token(path, "hermes", now=1000)
    assert oauth.reauth_required(path, "hermes") is True
    monkeypatch.setattr(client_mod, "resolve_config_path", lambda: path)
    return path


def _relogin(path: Path) -> None:
    oauth.install_grant(
        path, "hermes",
        {"access_token": "hch-at-fresh", "refresh_token": "hch-rt-fresh", "expires_in": 3600},
        client_id="hermes-desktop",
        token_endpoint="http://localhost:8000/oauth/token",
    )


class TestDeadGrantSkipsCalls:
    def test_dead_grant_issues_no_dialectic_call(self, tmp_path, monkeypatch):
        _kill_grant(tmp_path, monkeypatch)
        peer = _FlakyPeer(failures=0)
        mgr = _make_manager(peer)
        with pytest.raises(HonchoAuthError):
            mgr.dialectic_query("k", "q")
        assert peer.calls == 0
        assert mgr.pop_auth_notice() is not None

    def test_relogin_resumes_dialectic_without_waiting(self, tmp_path, monkeypatch):
        path = _kill_grant(tmp_path, monkeypatch)
        peer = _FlakyPeer(failures=0)
        mgr = _make_manager(peer)
        with pytest.raises(HonchoAuthError):
            mgr.dialectic_query("k", "q")
        assert peer.calls == 0

        _relogin(path)
        assert mgr.dialectic_query("k", "q") == "synthesized answer"
        assert peer.calls == 1
        assert mgr._auth_failure is None

    def test_dead_grant_issues_no_sync_call(self, tmp_path, monkeypatch):
        _kill_grant(tmp_path, monkeypatch)
        flaky = _FlakyHonchoSession(failures=0)
        mgr, session = _make_sync_manager(flaky)
        assert mgr._flush_session(session) is False
        assert flaky.calls == 0
        assert mgr._auth_failure is not None

    def test_relogin_resumes_sync_without_waiting(self, tmp_path, monkeypatch):
        path = _kill_grant(tmp_path, monkeypatch)
        flaky = _FlakyHonchoSession(failures=0)
        mgr, session = _make_sync_manager(flaky)
        assert mgr._flush_session(session) is False
        assert flaky.calls == 0

        _relogin(path)
        assert mgr._flush_session(session) is True
        assert flaky.calls == 1
        assert all(m["_synced"] for m in session.messages)
        assert mgr._auth_failure is None


# ---------------------------------------------------------------------------
# one-time user-facing notice
# ---------------------------------------------------------------------------


class TestAuthNotice:
    def test_manager_emits_notice_exactly_once(self):
        peer = _FlakyPeer(failures=99)
        mgr = _make_manager(peer, reauth_ok=False)
        with pytest.raises(HonchoAuthError):
            mgr.dialectic_query("k", "q")

        first = mgr.pop_auth_notice()
        assert first and "Invalid or expired access token" in first
        assert mgr.pop_auth_notice() is None

        # A second failure inside the same episode does not re-arm the notice.
        with pytest.raises(HonchoAuthError):
            mgr.dialectic_query("k", "q")
        assert mgr.pop_auth_notice() is None

    def test_recorded_failure_and_notice_redact_token_values(self):
        mgr = _make_manager(_FlakyPeer(failures=0))
        mgr._record_auth_failure(Exception("rejected token hch-at-secretvalue99"))
        notice = mgr.pop_auth_notice()
        assert "secretvalue99" not in notice
        assert "hch-at-[redacted]" in notice

    def test_provider_prefetch_injects_notice_once(self):
        class _FakeManager:
            def __init__(self):
                self.notices = ["Invalid or expired access token"]

            def pop_auth_notice(self):
                return self.notices.pop() if self.notices else None

            def pop_context_result(self, session_key):
                return {}

        provider = HonchoMemoryProvider()
        provider._manager = _FakeManager()
        provider._config = SimpleNamespace(timeout=0.01, context_tokens=0)
        provider._session_key = "k"
        provider._session_initialized = True
        provider._recall_mode = "context"
        provider._turn_count = 2
        provider._last_dialectic_turn = 0
        provider._base_context_cache = ""

        first = provider.prefetch("what did we decide about the schema?")
        assert "hermes honcho setup" in first
        assert "paused" in first

        second = provider.prefetch("and the follow-up question?")
        assert second == ""


# ---------------------------------------------------------------------------
# cadence backoff exemption
# ---------------------------------------------------------------------------


class TestBackoffExemption:
    def test_auth_error_does_not_widen_backoff(self):
        provider = HonchoMemoryProvider()
        provider._note_dialectic_failure(HonchoAuthError("still 401 after refresh"))
        assert provider._dialectic_empty_streak == 0

    def test_other_errors_still_widen_backoff(self):
        provider = HonchoMemoryProvider()
        provider._note_dialectic_failure(RuntimeError("timeout"))
        assert provider._dialectic_empty_streak == 1


# ---------------------------------------------------------------------------
# session: context/search 401 recovery through _authed_call
# ---------------------------------------------------------------------------


class _FlakyContextPeer:
    """context() raises an auth error N times, then succeeds."""

    def __init__(self, failures: int, representation: str = "knows Python"):
        self.failures = failures
        self.representation = representation
        self.calls = 0

    def context(self, **kw):
        self.calls += 1
        if self.calls <= self.failures:
            raise Exception("Invalid or expired access token")
        return SimpleNamespace(representation=self.representation, peer_card=["fact one"])


class TestContextAuthRetry:
    def test_401_forces_refresh_and_retries_once(self):
        peer = _FlakyContextPeer(failures=1)
        mgr = _make_manager(peer)
        ctx = mgr.get_session_context("k")
        assert ctx["representation"] == "knows Python"
        assert peer.calls == 2  # original + one retry

    def test_persistent_401_records_failure_and_notices_once(self):
        peer = _FlakyContextPeer(failures=99)
        mgr = _make_manager(peer)
        with pytest.raises(HonchoAuthError):
            mgr.get_session_context("k")
        assert peer.calls == 2  # exactly one retry, no loop
        assert mgr._auth_failure is not None
        assert mgr.pop_auth_notice() is not None
        assert mgr.pop_auth_notice() is None

    def test_peer_card_401_raises_instead_of_reading_empty(self):
        class _FlakyCardPeer:
            calls = 0

            def get_card(self, **kw):
                type(self).calls += 1
                raise Exception("Invalid or expired access token")

        mgr = _make_manager(_FlakyCardPeer(), reauth_ok=False)
        with pytest.raises(HonchoAuthError):
            mgr.get_peer_card("k")
        assert _FlakyCardPeer.calls == 1


class TestDeadGrantSkipsContextAndSearch:
    def test_dead_grant_issues_no_context_call(self, tmp_path, monkeypatch):
        _kill_grant(tmp_path, monkeypatch)
        peer = _FlakyContextPeer(failures=0)
        mgr = _make_manager(peer)
        with pytest.raises(HonchoAuthError):
            mgr.get_session_context("k")
        assert peer.calls == 0

    def test_dead_grant_issues_no_search_call(self, tmp_path, monkeypatch):
        from plugins.memory.honcho import session as session_mod

        _kill_grant(tmp_path, monkeypatch)
        client = MagicMock()
        monkeypatch.setattr(session_mod, "get_honcho_client", lambda *a, **k: client)
        mgr = _make_manager(_FlakyContextPeer(failures=0))
        with pytest.raises(HonchoAuthError):
            mgr.search_context("k", "query")
        client.search.assert_not_called()

    def test_dead_grant_prefetch_returns_empty_and_arms_notice(self, tmp_path, monkeypatch):
        _kill_grant(tmp_path, monkeypatch)
        peer = _FlakyContextPeer(failures=0)
        mgr = _make_manager(peer)
        assert mgr.get_prefetch_context("k") == {}
        assert peer.calls == 0
        assert mgr.pop_auth_notice() is not None


class TestNonAuthFailuresNotRetried:
    def test_context_timeout_fails_open_without_refresh(self):
        class _TimeoutPeer:
            calls = 0

            def _fail(self):
                type(self).calls += 1
                raise TimeoutError("request timed out")

            def context(self, **kw):
                self._fail()

            def representation(self, **kw):
                self._fail()

            def get_card(self, **kw):
                self._fail()

        _TimeoutPeer.calls = 0
        mgr = _make_manager(_TimeoutPeer())
        reauths = []
        mgr._force_reauth = lambda: reauths.append(1) or True

        ctx = mgr.get_session_context("k")
        assert ctx == {"representation": "", "card": []}
        assert _TimeoutPeer.calls == 3  # context, representation, card — no retries
        assert reauths == []

    def test_search_timeout_fails_open_without_refresh(self, monkeypatch):
        from plugins.memory.honcho import session as session_mod

        class _TimeoutSearchPeer:
            calls = 0

            def search(self, *a, **kw):
                type(self).calls += 1
                raise TimeoutError("request timed out")

        _TimeoutSearchPeer.calls = 0
        client = MagicMock()
        client.search.side_effect = TimeoutError("request timed out")
        monkeypatch.setattr(session_mod, "get_honcho_client", lambda *a, **k: client)
        mgr = _make_manager(_TimeoutSearchPeer())
        reauths = []
        mgr._force_reauth = lambda: reauths.append(1) or True

        assert mgr.search_context("k", "q") == ""
        assert client.search.call_count == 1
        assert _TimeoutSearchPeer.calls == 1
        assert reauths == []


# ---------------------------------------------------------------------------
# client rebuild: the retry must use freshly resolved SDK objects
# ---------------------------------------------------------------------------


def _wire_rebuild(tmp_path, monkeypatch, fresh_client):
    """Route _force_reauth down its client-rebuild path, swapping in fresh_client."""
    from plugins.memory.honcho import client as client_mod
    from plugins.memory.honcho import session as session_mod

    clients = {"current": MagicMock()}
    monkeypatch.setattr(session_mod, "get_honcho_client", lambda *a, **k: clients["current"])
    monkeypatch.setattr(client_mod, "resolve_config_path", lambda: tmp_path / "honcho.json")
    monkeypatch.setattr(oauth, "force_refresh_token", lambda p, h: "hch-at-rotated")
    monkeypatch.setattr(oauth, "apply_token_to_client", lambda c, t: False)
    monkeypatch.setattr(
        client_mod, "reset_honcho_client",
        lambda: clients.__setitem__("current", fresh_client),
    )
    return clients


class TestClientRebuildRetry:
    def test_flush_retry_uses_rebuilt_session_not_stale(self, tmp_path, monkeypatch):
        stale_session = MagicMock()
        stale_session.add_messages.side_effect = Exception("Invalid or expired access token")
        stale_peer = MagicMock()
        stale_peer.message.side_effect = lambda content: content

        fresh_session = MagicMock()
        fresh_session.context.return_value = SimpleNamespace(summary=None, messages=[])
        fresh_peer = MagicMock()
        fresh_peer.message.side_effect = lambda content: content
        fresh_client = MagicMock()
        fresh_client.session.return_value = fresh_session
        fresh_client.peer.return_value = fresh_peer

        _wire_rebuild(tmp_path, monkeypatch, fresh_client)

        cfg = HonchoClientConfig(host="hermes", api_key="hch-at-x", enabled=True)
        mgr = HonchoSessionManager(config=cfg)
        mgr._peers_cache.update({"u": stale_peer, "a": stale_peer})
        mgr._sessions_cache["s"] = stale_session
        session = HonchoSession(
            key="k", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s"
        )
        session.add_message("user", "hello")
        session.add_message("assistant", "hi")

        assert mgr._flush_session(session) is True
        # The stale pre-rebuild session must not be retried.
        assert stale_session.add_messages.call_count == 1
        assert fresh_session.add_messages.call_count == 1
        assert all(m["_synced"] for m in session.messages)
        assert mgr._auth_failure is None

    def test_context_retry_uses_rebuilt_peer_not_stale(self, tmp_path, monkeypatch):
        stale_peer = MagicMock()
        stale_peer.context.side_effect = Exception("Invalid or expired access token")

        fresh_peer = MagicMock()
        fresh_peer.context.return_value = SimpleNamespace(
            representation="rep after rebuild", peer_card=["fact"]
        )
        fresh_client = MagicMock()
        fresh_client.peer.return_value = fresh_peer

        _wire_rebuild(tmp_path, monkeypatch, fresh_client)

        cfg = HonchoClientConfig(host="hermes", api_key="hch-at-x", enabled=True)
        mgr = HonchoSessionManager(config=cfg)
        mgr._peers_cache["u"] = stale_peer
        mgr._cache["k"] = HonchoSession(
            key="k", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s"
        )

        ctx = mgr.get_session_context("k")
        assert ctx["representation"] == "rep after rebuild"
        assert stale_peer.context.call_count == 1
        assert fresh_peer.context.call_count == 1


# ---------------------------------------------------------------------------
# tools: auth failures must never read as "no context"
# ---------------------------------------------------------------------------


class TestToolAuthVisibility:
    def _provider(self, manager):
        provider = HonchoMemoryProvider()
        provider._manager = manager
        provider._session_key = "k"
        provider._session_initialized = True
        return provider

    def test_context_tool_reports_auth_failure(self):
        class _Mgr:
            def get_session_context(self, key, peer="user"):
                raise HonchoAuthError("Honcho rejected our credentials")

        out = self._provider(_Mgr()).handle_tool_call("honcho_context", {})
        assert "No context available" not in out
        assert "authentication failed" in out

    def test_search_tool_reports_auth_failure(self):
        class _Mgr:
            def search_context(self, key, query, max_tokens=800, peer="user"):
                raise HonchoAuthError("Honcho rejected our credentials")

        out = self._provider(_Mgr()).handle_tool_call("honcho_search", {"query": "schema"})
        assert "No relevant context found" not in out
        assert "authentication failed" in out

    def test_profile_tool_reports_auth_failure_not_empty_profile(self):
        class _Mgr:
            def get_peer_card(self, key, peer="user"):
                raise HonchoAuthError("Honcho rejected our credentials")

        out = self._provider(_Mgr()).handle_tool_call("honcho_profile", {})
        assert "No profile facts" not in out
        assert "authentication failed" in out


# ---------------------------------------------------------------------------
# initialization-time auth failures: the notice must survive the manager discard
# ---------------------------------------------------------------------------


def _healthy_client():
    """A mock SDK client whose peers and sessions behave like an empty backend."""
    client = MagicMock()
    peer = MagicMock()
    peer.chat.return_value = ""
    peer.get_card.return_value = None
    peer.context.return_value = SimpleNamespace(representation="", peer_card=[])
    client.peer.return_value = peer
    sdk_session = MagicMock()
    sdk_session.context.return_value = SimpleNamespace(summary=None, messages=[])
    client.session.return_value = sdk_session
    return client


def _wire_init(tmp_path, monkeypatch, client, *, recall_mode="hybrid", dead_refresh=True):
    """Route provider initialization through a real manager backed by ``client``."""
    from plugins.memory.honcho import client as client_mod
    from plugins.memory.honcho import session as session_mod

    path = tmp_path / "honcho.json"
    _write(path, {"hosts": {"hermes": _host_block(expires_at=time.time() + 3600)}})
    monkeypatch.setattr(client_mod, "resolve_config_path", lambda: path)
    monkeypatch.setattr(client_mod, "get_honcho_client", lambda *a, **k: client)
    monkeypatch.setattr(session_mod, "get_honcho_client", lambda *a, **k: client)
    if dead_refresh:
        monkeypatch.setattr(oauth, "force_refresh_token", lambda p, h: None)
    cfg = HonchoClientConfig(
        host="hermes", api_key="hch-at-old", enabled=True, recall_mode=recall_mode,
        timeout=0.5, session_strategy="per-session",
    )
    monkeypatch.setattr(
        client_mod.HonchoClientConfig, "from_global_config", lambda *a, **k: cfg
    )
    return path


def _initialized_provider():
    provider = HonchoMemoryProvider()
    provider.initialize(session_id="init-auth-session")
    if provider._init_thread:
        provider._init_thread.join(timeout=5)
    return provider


class TestInitAuthFailureNotice:
    def test_peer_setup_401_in_hybrid_mode_produces_notice(self, tmp_path, monkeypatch):
        client = MagicMock()
        client.peer.side_effect = Exception("HTTP 401 Unauthorized")
        _wire_init(tmp_path, monkeypatch, client)
        provider = _initialized_provider()

        assert provider._manager is None
        notice = provider.prefetch("what did we decide about the schema?")
        assert "hermes honcho setup" in notice
        assert "paused" in notice

    def test_notice_is_emitted_exactly_once(self, tmp_path, monkeypatch):
        client = MagicMock()
        client.peer.side_effect = Exception("HTTP 401 Unauthorized")
        _wire_init(tmp_path, monkeypatch, client)
        provider = _initialized_provider()

        assert "hermes honcho setup" in provider.prefetch("first question")
        # Retries keep failing, but the same episode never re-arms the notice.
        for query in ("second question", "third question"):
            assert provider.prefetch(query) == ""

    def test_dead_grant_during_session_setup_produces_notice(self, tmp_path, monkeypatch):
        client = _healthy_client()
        env = {}

        def _session_dies(*a, **k):
            path = env["path"]
            block = json.loads(path.read_text())["hosts"]["hermes"]
            cred = oauth.OAuthCredential.from_host_block(block)
            oauth._mark_grant_dead((str(path), "hermes"), cred)
            raise Exception("Invalid or expired access token")

        client.session.side_effect = _session_dies
        env["path"] = _wire_init(tmp_path, monkeypatch, client, dead_refresh=False)
        provider = _initialized_provider()

        assert provider._manager is None
        assert client.peer.called  # failure hit session setup, not peer setup
        notice = provider.prefetch("what happened before the grant died?")
        assert "hermes honcho setup" in notice

    def test_tools_lazy_init_reports_auth_error(self, tmp_path, monkeypatch):
        client = MagicMock()
        client.peer.side_effect = Exception("Invalid or expired access token")
        _wire_init(tmp_path, monkeypatch, client, recall_mode="tools")
        provider = _initialized_provider()

        out = provider.handle_tool_call("honcho_profile", {})
        assert "authentication failed" in out
        assert "could not be initialized" not in out

    def test_relogin_resumes_init_and_clears_failure(self, tmp_path, monkeypatch):
        client = _healthy_client()
        client.peer.side_effect = Exception("HTTP 401 Unauthorized")
        path = _wire_init(tmp_path, monkeypatch, client)
        provider = _initialized_provider()
        assert "hermes honcho setup" in provider.prefetch("first question")

        _relogin(path)
        client.peer.side_effect = None
        provider.prefetch("after re-login")
        if provider._init_thread:
            provider._init_thread.join(timeout=5)

        assert provider._session_initialized is True
        assert provider._manager is not None
        assert provider._init_auth_failure is None

    def test_tools_relogin_resumes_without_restart(self, tmp_path, monkeypatch):
        client = _healthy_client()
        client.peer.side_effect = Exception("Invalid or expired access token")
        path = _wire_init(tmp_path, monkeypatch, client, recall_mode="tools")
        provider = _initialized_provider()
        assert "authentication failed" in provider.handle_tool_call("honcho_profile", {})

        _relogin(path)
        client.peer.side_effect = None
        out = provider.handle_tool_call("honcho_profile", {})

        assert "authentication failed" not in out
        assert provider._session_initialized is True
        assert provider._init_auth_failure is None

    def test_non_auth_init_timeout_fails_open_without_notice(self, tmp_path, monkeypatch):
        client = MagicMock()
        client.peer.side_effect = TimeoutError("request timed out")
        _wire_init(tmp_path, monkeypatch, client, dead_refresh=False)
        reauths = []
        monkeypatch.setattr(oauth, "force_refresh_token", lambda p, h: reauths.append(1))
        provider = _initialized_provider()

        assert provider._manager is None
        assert provider._init_auth_failure is None
        assert provider.prefetch("a real question") == ""
        assert reauths == []

    def test_non_auth_tools_init_failure_keeps_generic_error(self, tmp_path, monkeypatch):
        client = MagicMock()
        client.peer.side_effect = TimeoutError("request timed out")
        _wire_init(tmp_path, monkeypatch, client, recall_mode="tools", dead_refresh=False)
        provider = _initialized_provider()

        out = provider.handle_tool_call("honcho_profile", {})
        assert "could not be initialized" in out


# ---------------------------------------------------------------------------
# hardening: exchange budget, failure cooldown, client-generation cache guard
# ---------------------------------------------------------------------------


class TestExchangeBudget:
    def test_timed_out_first_attempt_skips_retry_when_budget_spent(self, tmp_path, monkeypatch):
        """A first attempt that consumed the whole budget must not start a
        second full-timeout exchange while holding the global refresh locks."""
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block()}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)

        calls = []
        clock = {"now": 1000.0}
        monkeypatch.setattr(oauth.time, "monotonic", lambda: clock["now"])

        def slow_timeout(url, data, timeout):
            calls.append(timeout)
            clock["now"] += oauth._REFRESH_TOTAL_BUDGET_SECONDS + 1
            raise TimeoutError("token exchange timed out")

        monkeypatch.setattr(oauth, "_http_post_form_status", slow_timeout)
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)

        assert token == "hch-at-old" and refreshed is False
        assert len(calls) == 1  # no second exchange after the budget is gone

    def test_fast_failure_retry_gets_remaining_budget(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block()}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)

        timeouts = []

        def flaky(url, data, timeout):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise ConnectionError("reset")
            return 200, _rotated_body()

        monkeypatch.setattr(oauth, "_http_post_form_status", flaky)
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)

        assert refreshed is True and token == "hch-at-new1"
        assert len(timeouts) == 2
        # Retry timeout is bounded by both the per-attempt cap and the budget.
        assert 0 < timeouts[1] <= oauth._REFRESH_TIMEOUT_SECONDS


class TestFailureCooldown:
    def test_repeated_calls_within_cooldown_do_not_reexchange(self, tmp_path, monkeypatch):
        """After a transient failure, waiting callers fail open instead of
        serializing their own full exchange cycles (dogpile guard)."""
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block()}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)

        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise ConnectionError("network down")

        monkeypatch.setattr(oauth, "_http_post_form_status", boom)
        oauth.ensure_fresh_token(path, "hermes", now=1000)
        assert len(calls) == 2  # first attempt + its one retry

        # Subsequent callers inside the cooldown window skip the endpoint.
        for _ in range(3):
            token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)
            assert token == "hch-at-old" and refreshed is False
        assert oauth.force_refresh_token(path, "hermes") is None
        assert len(calls) == 2

        # After the cooldown expires the exchange is attempted again.
        key = (str(path), "hermes")
        oauth._refresh_failure_at[key] -= oauth._REFRESH_FAILURE_COOLDOWN_SECONDS + 1
        oauth.ensure_fresh_token(path, "hermes", now=1000)
        assert len(calls) == 4

    def test_relogin_clears_the_cooldown(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block()}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)

        def boom(*a, **k):
            raise ConnectionError("network down")

        monkeypatch.setattr(oauth, "_http_post_form_status", boom)
        oauth.ensure_fresh_token(path, "hermes", now=1000)
        assert oauth._in_failure_cooldown((str(path), "hermes")) is True

        _relogin(path)
        assert oauth._in_failure_cooldown((str(path), "hermes")) is False

    def test_successful_rotation_clears_the_cooldown(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block()}})
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)
        key = (str(path), "hermes")
        oauth._refresh_failure_at[key] = (
            oauth.time.monotonic() - oauth._REFRESH_FAILURE_COOLDOWN_SECONDS - 1
        )
        monkeypatch.setattr(
            oauth, "_http_post_form_status", lambda *a, **k: (200, _rotated_body())
        )
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)
        assert refreshed is True
        assert key not in oauth._refresh_failure_at


class TestClientGenerationGuard:
    def test_stale_object_resolved_across_rebuild_is_not_cached(self, monkeypatch):
        """A resolver that fetched from the OLD client must not store its
        object into the cache after _force_reauth rebuilt the client."""
        from plugins.memory.honcho import session as session_mod

        cfg = HonchoClientConfig(host="hermes", api_key="hch-at-x", enabled=True)
        mgr = HonchoSessionManager(config=cfg)

        stale_session = object()
        fresh_session = object()
        resolutions = []

        class _Client:
            def session(self, sid):
                # First resolve returns the stale object and simulates a
                # concurrent rebuild landing mid-flight; the retry gets fresh.
                if not resolutions:
                    resolutions.append("stale")
                    with mgr._cache_lock:
                        mgr._client_generation += 1
                        mgr._sessions_cache.clear()
                    return stale_session
                resolutions.append("fresh")
                return fresh_session

        client = _Client()
        monkeypatch.setattr(session_mod, "get_honcho_client", lambda *a, **k: client)
        got = mgr._sdk_session("s")
        assert got is fresh_session
        assert mgr._sessions_cache["s"] is fresh_session
        assert resolutions == ["stale", "fresh"]

    def test_dead_grant_check_fast_path_skips_path_resolution(self, monkeypatch):
        """With no dead grants, _reauth_required must not resolve the config
        path at all (it runs before every SDK call)."""
        from plugins.memory.honcho import client as client_mod

        oauth._dead_grants.clear()

        # Recording spy, not a raising stub: _reauth_required swallows all
        # exceptions, so a raise would be silently converted to False and the
        # test would pass even without the fast path.
        calls = []

        def _spy():
            calls.append(1)
            return Path("/nonexistent/honcho.json")

        monkeypatch.setattr(client_mod, "resolve_config_path", _spy)
        cfg = HonchoClientConfig(host="hermes", api_key="hch-at-x", enabled=True)
        mgr = HonchoSessionManager(config=cfg)
        assert mgr._reauth_required() is False
        assert calls == [], "resolve_config_path must not run on the fast path"
