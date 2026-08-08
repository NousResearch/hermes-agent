"""Regression tests for the shard-s5 wave-1 extraction: ws_auth_gate_mixin.

The dashboard WS-auth gate helpers (``_ws_*``) were moved verbatim from
``hermes_cli/web_server.py`` into ``hermes_cli/ws_auth_gate_mixin.py``
(god-file decomposition, cluster c2).  ``web_server`` re-exports them so
``web_server._ws_auth_ok`` call sites and monkeypatches keep working.

These tests pin the moved pure helpers at their NEW home and assert the
re-export seam, so a regression in the extraction is caught immediately.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import web_server
from hermes_cli import ws_auth_gate_mixin as mixin


def _fake_ws(*, query: dict, client_host: str = "127.0.0.1", path: str = "/api/pty",
             headers: dict | None = None):
    """Stand-in for starlette.WebSocket good enough for the gate helpers."""

    class _QP:
        def __init__(self, q):
            self._q = q

        def get(self, k, default=""):
            return self._q.get(k, default)

    return SimpleNamespace(
        query_params=_QP(query),
        client=SimpleNamespace(host=client_host),
        url=SimpleNamespace(path=path),
        headers=headers or {},
    )


@pytest.fixture
def loopback_state():
    """Pin web_server.app.state to loopback mode and restore afterwards."""
    state = web_server.app.state
    prev = (
        getattr(state, "auth_required", None),
        getattr(state, "bound_host", None),
        getattr(state, "bound_port", None),
    )
    state.auth_required = False
    state.bound_host = "127.0.0.1"
    state.bound_port = 9119
    yield state
    state.auth_required, state.bound_host, state.bound_port = prev


class TestReExportSeam:
    """web_server.<name> must resolve to the mixin's function (identity)."""

    @pytest.mark.parametrize(
        "name",
        [
            "_ws_client_reason",
            "_ws_client_is_allowed",
            "_ws_host_origin_reason",
            "_ws_host_origin_is_allowed",
            "_ws_request_reason",
            "_ws_request_is_allowed",
            "_ws_auth_mode",
            "_ws_auth_reason",
            "_ws_auth_ok",
            "_ws_close_reason",
        ],
    )
    def test_reexported_identity(self, name):
        assert getattr(web_server, name) is getattr(mixin, name)


class TestWsCloseReason:
    """Pure 123-byte close-reason clamp (RFC 6455)."""

    def test_short_passthrough(self):
        assert mixin._ws_close_reason("auth: token_mismatch") == "auth: token_mismatch"

    def test_exactly_123_bytes_passthrough(self):
        text = "x" * 123
        assert mixin._ws_close_reason(text) == text

    def test_long_truncated_to_120_bytes_plus_ellipsis(self):
        out = mixin._ws_close_reason("x" * 200)
        assert out == "x" * 120 + "..."
        assert len(out.encode("utf-8")) == 123

    def test_multibyte_truncation_stays_valid_utf8(self):
        # 100 emoji = 400 bytes; must clamp to <= 123 bytes without
        # splitting a multi-byte sequence mid-character.
        out = mixin._ws_close_reason("\U0001f600" * 100)
        assert len(out.encode("utf-8")) <= 123
        out.encode("utf-8")  # must not raise


class TestWsAuthGate:
    """Auth gate helpers against the pinned loopback app state."""

    def test_auth_ok_accepts_session_token(self, loopback_state):
        ws = _fake_ws(query={"token": web_server._SESSION_TOKEN})
        assert mixin._ws_auth_ok(ws) is True

    def test_auth_reason_rejects_wrong_token(self, loopback_state):
        ws = _fake_ws(query={"token": "not-the-token"})
        assert mixin._ws_auth_reason(ws) == ("token_mismatch", "token")

    def test_auth_reason_no_credential(self, loopback_state):
        ws = _fake_ws(query={})
        assert mixin._ws_auth_reason(ws) == ("no_credential", "none")

    def test_auth_mode_loopback(self, loopback_state):
        assert mixin._ws_auth_mode() == "loopback"

    def test_client_allowed_loopback_peer(self, loopback_state):
        ws = _fake_ws(query={}, client_host="127.0.0.1")
        assert mixin._ws_client_is_allowed(ws) is True
        assert mixin._ws_client_reason(ws) is None

    def test_client_rejected_non_loopback_peer(self, loopback_state):
        ws = _fake_ws(query={}, client_host="203.0.113.7")
        assert mixin._ws_client_is_allowed(ws) is False
        assert mixin._ws_client_reason(ws) == (
            "peer_not_loopback peer=203.0.113.7 bound=127.0.0.1"
        )

    def test_client_rejected_empty_peer_fail_closed(self, loopback_state):
        ws = _fake_ws(query={}, client_host="")
        assert mixin._ws_client_is_allowed(ws) is False
        assert mixin._ws_client_reason(ws) == "missing_or_empty_peer bound=127.0.0.1"

    def test_request_allowed_composes_peer_and_origin(self, loopback_state):
        ok_ws = _fake_ws(
            query={"token": web_server._SESSION_TOKEN},
            client_host="127.0.0.1",
            headers={"host": "127.0.0.1:9119", "origin": "http://127.0.0.1:9119"},
        )
        assert mixin._ws_request_is_allowed(ok_ws) is True

    def test_host_origin_reason_mismatch(self, loopback_state):
        ws = _fake_ws(
            query={},
            client_host="127.0.0.1",
            headers={"host": "evil.example.com", "origin": "http://evil.example.com"},
        )
        reason = mixin._ws_host_origin_reason(ws)
        assert reason is not None and reason.startswith("host_mismatch")

    def test_host_origin_reason_ok(self, loopback_state):
        ws = _fake_ws(
            query={},
            client_host="127.0.0.1",
            headers={"host": "127.0.0.1:9119", "origin": "http://127.0.0.1:9119"},
        )
        assert mixin._ws_host_origin_reason(ws) is None
