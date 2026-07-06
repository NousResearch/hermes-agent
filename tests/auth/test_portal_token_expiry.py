"""Tests for the Nous Portal token-expiry circuit breaker (#58572).

The gateway used to crash-loop on expired Portal tokens:
  - 401 from inference -> refresh attempt -> refresh itself 401s
    -> return canned error to operator -> repeat on every message.
For headless deployments (Telegram/Discord/Slack/WhatsApp gateway) the
operator had no remote recovery path; the only fix was SSH + `hermes login`.

These tests cover the three behaviors introduced by the fix:

1. **Auto-refresh path** — an expired access_token with a valid refresh_token
   is refreshed transparently and the retry succeeds. The breaker stays
   closed (no spurious open/close cycles for transient expiry).

2. **Graceful error path** — an expired access_token with NO refresh_token
   surfaces a clear, operator-facing error (not a stack trace) and the
   gateway stays alive.

3. **Circuit-breaker path** — when the refresh itself fails with a
   terminal error (revoked / refresh_token_reused / no RT), the breaker
   opens and subsequent resolve calls short-circuit with the same clear
   operator-facing message, without re-hitting the Portal API.

The breaker also closes again after a successful refresh, so a one-off
transient outage self-heals on the next successful resolve.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hermes_cli import auth as auth_mod
from hermes_cli.auth import (
    AuthError,
    _is_nous_token_expiry_breaker_open,
    _record_nous_token_expiry_failure,
    reset_nous_token_expiry_breaker,
    resolve_nous_runtime_credentials,
    get_nous_token_expiry_breaker_state,
)


# ── Per-test breaker reset ─────────────────────────────────────────────────
# The breaker is module-level state. Each test resets it in setup so a
# failure in one test cannot poison the next.
@pytest.fixture(autouse=True)
def _reset_breaker():
    reset_nous_token_expiry_breaker()
    yield
    reset_nous_token_expiry_breaker()


# ── JWT helpers (mirrors test_auth_nous_provider.py conventions) ───────────
def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _make_jwt(*, exp_offset_seconds: int = 3600, scope: str = "inference:invoke") -> str:
    """Build a minimal valid JWT with the given exp offset."""
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = _b64url(
        json.dumps(
            {
                "sub": "test-user",
                "scope": scope,
                "exp": int(time.time() + exp_offset_seconds),
            }
        ).encode()
    )
    return f"{header}.{payload}.sig"


def _setup_nous_auth(
    hermes_home: Path,
    *,
    access_token: str = "",
    refresh_token: str = "",
    expires_at: str | None = None,
    scope: str = "inference:invoke",
) -> None:
    """Write a hermes_home/auth.json with a Nous provider state."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    if not access_token:
        access_token = _make_jwt(exp_offset_seconds=-30, scope=scope)
    if not expires_at:
        # Far in the past — forces the resolver into the refresh path.
        expires_at = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
            seconds=30
        )
    payload = {
        "version": 1,
        "active_provider": "nous",
        "providers": {
            "nous": {
                "portal_base_url": "https://portal.example.com",
                "inference_base_url": "https://inference.example.com/v1",
                "client_id": "hermes-cli",
                "token_type": "Bearer",
                "scope": scope,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "obtained_at": "2026-01-01T00:00:00+00:00",
                "expires_in": 0,
                "expires_at": expires_at.isoformat()
                if isinstance(expires_at, datetime)
                else expires_at,
                "agent_key": None,
                "agent_key_id": None,
                "agent_key_expires_at": None,
            }
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


class _FakeRefreshResponse:
    """Mimics the bits of an httpx.Response that _refresh_access_token reads."""

    def __init__(self, status_code: int, payload: Dict[str, Any] | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeRefreshClient:
    """Captures POST /api/oauth/token calls and returns a canned response."""

    def __init__(
        self,
        *,
        response_status: int = 200,
        response_payload: Dict[str, Any] | None = None,
        capture: List[Dict[str, Any]] | None = None,
    ):
        self.response_status = response_status
        self.response_payload = response_payload or {}
        self.capture = capture if capture is not None else []
        self.closed = False

    def post(self, url: str, **kwargs: Any) -> _FakeRefreshResponse:
        self.capture.append({"url": url, "kwargs": kwargs})
        return _FakeRefreshResponse(self.response_status, self.response_payload)

    def close(self) -> None:  # pragma: no cover - matches httpx contract
        self.closed = True

    def __enter__(self) -> "_FakeRefreshClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# =============================================================================
# 1. Expired token + valid refresh → auto-refresh + retry succeeds
# =============================================================================
class TestExpiredTokenWithValidRefresh:
    """Happy-path: the saved access_token is expired but the refresh_token
    works. The resolver should transparently refresh and return fresh
    credentials. The breaker stays closed.
    """

    def test_auto_refresh_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),  # expired
            refresh_token="refresh-valid",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        fresh_jwt = _make_jwt(exp_offset_seconds=3600)
        capture: List[Dict[str, Any]] = []
        fake = _FakeRefreshClient(
            response_status=200,
            response_payload={
                "access_token": fresh_jwt,
                "refresh_token": "refresh-rotated",
                "token_type": "Bearer",
                "scope": "inference:invoke",
                "expires_in": 3600,
            },
            capture=capture,
        )

        # Patch both the http client constructor and the URL allowlist so the
        # fake response is accepted by the validation gate.
        monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

        creds = resolve_nous_runtime_credentials(timeout_seconds=5.0)

        # The refresh POST was made and used the saved refresh_token.
        assert len(capture) == 1, "refresh endpoint should have been called once"
        assert "refresh_token=refresh-valid" in str(
            capture[0]["kwargs"].get("headers", {})
        ) or capture[0]["kwargs"].get("headers", {}).get(
            "x-nous-refresh-token"
        ) == "refresh-valid"
        # The returned credentials carry the new access_token.
        assert creds["api_key"], "api_key should be populated after refresh"

    def test_breaker_stays_closed_on_successful_refresh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful refresh must NOT open the breaker."""
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="refresh-valid",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        fresh_jwt = _make_jwt(exp_offset_seconds=3600)
        fake = _FakeRefreshClient(
            response_status=200,
            response_payload={
                "access_token": fresh_jwt,
                "refresh_token": "refresh-rotated",
                "expires_in": 3600,
            },
        )
        monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

        # Pre-condition: breaker is closed.
        assert _is_nous_token_expiry_breaker_open() is None

        resolve_nous_runtime_credentials(timeout_seconds=5.0)

        # Post-condition: still closed.
        assert _is_nous_token_expiry_breaker_open() is None

    def test_successful_refresh_after_breaker_open_closes_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the breaker was opened by a previous failure, a subsequent
        successful refresh (with force_refresh=True bypassing the breaker)
        must close it again so normal dispatch resumes.
        """
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="refresh-recovered",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        fresh_jwt = _make_jwt(exp_offset_seconds=3600)
        fake = _FakeRefreshClient(
            response_status=200,
            response_payload={
                "access_token": fresh_jwt,
                "refresh_token": "refresh-rotated",
                "expires_in": 3600,
            },
        )
        monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

        # Open the breaker manually to simulate a prior failed resolve.
        _record_nous_token_expiry_failure(
            code="invalid_grant",
            message="Portal rejected the refresh token.",
            has_refresh_token=True,
        )
        assert _is_nous_token_expiry_breaker_open() is not None

        # force_refresh=True bypasses the breaker for this call.
        resolve_nous_runtime_credentials(timeout_seconds=5.0, force_refresh=True)

        # The successful refresh closes the breaker.
        assert _is_nous_token_expiry_breaker_open() is None


# =============================================================================
# 2. Expired token + no refresh_token → graceful error, gateway stays alive
# =============================================================================
class TestExpiredTokenWithoutRefreshToken:
    """When the saved state has no refresh_token at all, the resolver
    must surface a clear, operator-facing error (no stack trace, no crash).
    The breaker must open so subsequent calls short-circuit without
    re-running the (impossible) refresh dance.
    """

    def test_no_refresh_token_raises_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="",  # no RT at all
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Make sure no HTTP call slips through.
        def _no_http(*a: Any, **kw: Any) -> None:  # pragma: no cover - guard
            raise AssertionError("No HTTP calls expected when no refresh_token exists")

        monkeypatch.setattr(auth_mod.httpx, "Client", _no_http)

        with pytest.raises(AuthError) as excinfo:
            resolve_nous_runtime_credentials(timeout_seconds=5.0)

        # The error is operator-facing: names the cause and the recovery command.
        msg = str(excinfo.value)
        assert "hermes auth add nous" in msg, (
            f"Error must tell the operator how to recover; got: {msg!r}"
        )
        assert excinfo.value.relogin_required is True
        assert excinfo.value.provider == "nous"

    def test_no_refresh_token_opens_breaker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After raising, the breaker is OPEN so subsequent calls don't
        re-do the (impossible) refresh work.
        """
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(
            auth_mod.httpx,
            "Client",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("No HTTP calls expected")
            ),
        )

        with pytest.raises(AuthError):
            resolve_nous_runtime_credentials(timeout_seconds=5.0)

        breaker = _is_nous_token_expiry_breaker_open()
        assert breaker is not None, "breaker should be open after no-RT failure"
        assert breaker["has_refresh_token"] is False

    def test_no_refresh_token_subsequent_calls_short_circuit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The breaker prevents the gateway from re-doing the refresh on
        every incoming message. After the first failure, a second resolve
        call must raise immediately — without touching HTTP — and carry
        the same actionable guidance.
        """
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        http_calls: List[Any] = []

        def _http_should_not_be_called(*a: Any, **kw: Any) -> Any:
            http_calls.append((a, kw))
            raise AssertionError(
                "HTTP client should not be constructed when breaker is open"
            )

        monkeypatch.setattr(auth_mod.httpx, "Client", _http_should_not_be_called)

        # First call: opens the breaker.
        with pytest.raises(AuthError) as first_exc:
            resolve_nous_runtime_credentials(timeout_seconds=5.0)

        # Second call: short-circuits via breaker. NO HTTP calls.
        with pytest.raises(AuthError) as second_exc:
            resolve_nous_runtime_credentials(timeout_seconds=5.0)

        assert http_calls == [], (
            "Breaker must prevent any HTTP work on subsequent calls; "
            f"saw {len(http_calls)} calls"
        )
        # Both errors point the operator at the same recovery command.
        assert "hermes auth add nous" in str(second_exc.value)


# =============================================================================
# 3. Refresh itself returns 401 → circuit breaker opens, operator-facing msg
# =============================================================================
class TestRefreshItselfRejected:
    """The refresh_token itself is rejected by the Portal (revoked /
    refresh_token_reused / invalid_grant). The breaker must open and
    subsequent resolve calls must short-circuit with a clear operator-
    facing message.
    """

    def test_refresh_401_opens_breaker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="refresh-revoked",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Portal rejects the refresh with 401 + invalid_grant.
        fake = _FakeRefreshClient(
            response_status=401,
            response_payload={
                "error": "invalid_grant",
                "error_description": "Refresh token revoked.",
            },
        )
        monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

        with pytest.raises(AuthError):
            resolve_nous_runtime_credentials(timeout_seconds=5.0)

        breaker = _is_nous_token_expiry_breaker_open()
        assert breaker is not None, "breaker should be open after refresh rejection"
        assert breaker["has_refresh_token"] is True
        assert breaker["code"] == "invalid_grant"

    def test_refresh_token_reused_opens_breaker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """refresh_token_reused is a distinct failure mode (token-theft
        signal) — must also open the breaker.
        """
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="refresh-reused",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        fake = _FakeRefreshClient(
            response_status=400,
            response_payload={
                "error": "refresh_token_reused",
                "error_description": "Refresh token reuse detected.",
            },
        )
        monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

        with pytest.raises(AuthError):
            resolve_nous_runtime_credentials(timeout_seconds=5.0)

        breaker = _is_nous_token_expiry_breaker_open()
        assert breaker is not None
        assert breaker["code"] == "refresh_token_reused"

    def test_breaker_short_circuit_after_refresh_rejection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the breaker opens, subsequent resolve calls must NOT
        touch the HTTP client. They must raise an operator-facing message
        that includes the recovery command.
        """
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="refresh-revoked",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        http_calls: List[Any] = []

        def _tracking_http(*a: Any, **kw: Any) -> _FakeRefreshClient:
            http_calls.append((a, kw))
            return _FakeRefreshClient(
                response_status=401,
                response_payload={
                    "error": "invalid_grant",
                    "error_description": "Refresh token revoked.",
                },
            )

        monkeypatch.setattr(auth_mod.httpx, "Client", _tracking_http)

        # 1st call: refresh attempted, fails, breaker opens.
        with pytest.raises(AuthError):
            resolve_nous_runtime_credentials(timeout_seconds=5.0)
        first_call_count = len(http_calls)
        assert first_call_count >= 1, "first call should attempt refresh"

        # 2nd, 3rd, 4th calls: short-circuit, NO new HTTP calls.
        for _ in range(3):
            with pytest.raises(AuthError) as excinfo:
                resolve_nous_runtime_credentials(timeout_seconds=5.0)
            msg = str(excinfo.value)
            assert "hermes auth add nous" in msg, (
                "Breaker error must include the recovery command; "
                f"got: {msg!r}"
            )

        assert len(http_calls) == first_call_count, (
            "Breaker must prevent additional HTTP work; "
            f"expected {first_call_count} calls, saw {len(http_calls)}"
        )

    def test_breaker_error_message_mentions_root_cause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error surfaced to the operator (via format_auth_error, used
        by the gateway's _resolve_runtime_agent_kwargs) must clearly state
        WHAT happened (refresh token rejected) and WHAT to do (re-login).
        """
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="refresh-revoked",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        fake = _FakeRefreshClient(
            response_status=401,
            response_payload={
                "error": "invalid_grant",
                "error_description": "Refresh token revoked by Portal.",
            },
        )
        monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

        # First call: original error carries the Portal's reason.
        with pytest.raises(AuthError) as first_exc:
            resolve_nous_runtime_credentials(timeout_seconds=5.0)
        assert "revoked" in str(first_exc.value).lower()

        # Second call: short-circuit via breaker; message must still be
        # operator-actionable.
        with pytest.raises(AuthError) as second_exc:
            resolve_nous_runtime_credentials(timeout_seconds=5.0)
        msg = str(second_exc.value)
        assert "hermes auth add nous" in msg
        # And must NOT be mangled by format_auth_error (which would append
        # the generic "Run `hermes model` to re-authenticate" suffix).
        # format_auth_error returns the raw message for breaker codes.
        formatted = auth_mod.format_auth_error(second_exc.value)
        assert "hermes auth add nous" in formatted


# =============================================================================
# Breaker lifecycle / state helpers
# =============================================================================
class TestBreakerLifecycle:
    """Direct tests on the breaker state helpers — independent of the
    full resolve flow.
    """

    def test_breaker_initially_closed(self) -> None:
        assert _is_nous_token_expiry_breaker_open() is None
        assert get_nous_token_expiry_breaker_state() is None

    def test_record_then_open_then_reset(self) -> None:
        snap = _record_nous_token_expiry_failure(
            code="invalid_grant",
            message="revoked",
            has_refresh_token=True,
        )
        assert snap["code"] == "invalid_grant"
        assert snap["has_refresh_token"] is True

        opened = _is_nous_token_expiry_breaker_open()
        assert opened is not None
        assert opened["code"] == "invalid_grant"

        reset_nous_token_expiry_breaker()
        assert _is_nous_token_expiry_breaker_open() is None

    def test_breaker_auto_closes_after_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Past the cool-off window the breaker auto-closes so a transient
        Portal outage can recover without an explicit re-login.
        """
        # Use a tiny cooldown so the test runs in milliseconds, not minutes.
        monkeypatch.setattr(
            auth_mod, "_NOUS_TOKEN_EXPIRY_BREAKER_COOLDOWN_SECONDS", 0.05
        )
        _record_nous_token_expiry_failure(
            code="invalid_grant",
            message="revoked",
            has_refresh_token=True,
        )
        assert _is_nous_token_expiry_breaker_open() is not None
        time.sleep(0.1)
        assert _is_nous_token_expiry_breaker_open() is None, (
            "Breaker should auto-close after cool-off elapses"
        )

    def test_force_refresh_bypasses_breaker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``force_refresh=True`` is the explicit "try anyway" escape hatch
        (used by `hermes auth refresh` and by tests); it must reach the
        refresh endpoint even when the breaker is open.
        """
        hermes_home = tmp_path / "hermes"
        _setup_nous_auth(
            hermes_home,
            access_token=_make_jwt(exp_offset_seconds=-30),
            refresh_token="refresh-valid",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        capture: List[Dict[str, Any]] = []
        fresh_jwt = _make_jwt(exp_offset_seconds=3600)
        fake = _FakeRefreshClient(
            response_status=200,
            response_payload={
                "access_token": fresh_jwt,
                "refresh_token": "refresh-rotated",
                "expires_in": 3600,
            },
            capture=capture,
        )
        monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

        # Open the breaker first.
        _record_nous_token_expiry_failure(
            code="invalid_grant", message="x", has_refresh_token=True,
        )
        assert _is_nous_token_expiry_breaker_open() is not None

        # force_refresh=True must reach the HTTP layer.
        resolve_nous_runtime_credentials(timeout_seconds=5.0, force_refresh=True)
        assert len(capture) >= 1, (
            "force_refresh=True must bypass the breaker and call the Portal"
        )
