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
    _is_auth_error,
)


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
        redacted = oauth._redact_tokens(
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
