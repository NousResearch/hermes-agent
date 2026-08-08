"""Regression tests for the Wave-1 extraction of the WebSocket auth gates and
the Hermes Console websocket out of ``hermes_cli/web_server.py``.

The extraction (shard s5, clusters ``ws_auth_gates`` + ``console_ws``) is
behavior-neutral: the moved functions are re-exported from
``hermes_cli.web_server`` as the *same objects*, the ``/api/console`` route
stays registered on the dashboard app ahead of the SPA catch-all, and the
pure gate logic (close-reason clamping, console frame parsing, auth-mode
labelling, peer-IP gate) is unchanged.  These tests pin that surface and the
pure logic using the same synthetic-WebSocket style as the existing
``tests/hermes_cli/test_dashboard_auth_ws_auth.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import web_server
from hermes_cli import ws_auth_gate_mixin
from hermes_cli import console_ws_mixin


# ---------------------------------------------------------------------------
# Re-export identity — the extraction must not fork the API surface.
# ---------------------------------------------------------------------------

REEXPORTED_AUTH = [
    "_LOOPBACK_HOSTS",
    "_ws_auth_mode",
    "_ws_auth_ok",
    "_ws_auth_reason",
    "_ws_client_is_allowed",
    "_ws_client_reason",
    "_ws_close_reason",
    "_ws_host_origin_is_allowed",
    "_ws_host_origin_reason",
    "_ws_request_is_allowed",
    "_ws_request_reason",
]
REEXPORTED_CONSOLE = [
    "_console_json_payload",
    "_console_profile_from_ws",
    "_console_send",
    "_console_send_result",
    "_execute_console_line",
    "_get_console_executor",
    "console_ws",
]


@pytest.mark.parametrize("name", REEXPORTED_AUTH)
def test_auth_gate_names_reexported_from_web_server(name):
    assert getattr(web_server, name) is getattr(ws_auth_gate_mixin, name)


@pytest.mark.parametrize("name", REEXPORTED_CONSOLE)
def test_console_names_reexported_from_web_server(name):
    assert getattr(web_server, name) is getattr(console_ws_mixin, name)


def test_console_mixin_binds_web_server_logger_by_name():
    # Records emitted from the moved console endpoint must keep the logger
    # name they had when the code lived in web_server.py.
    assert console_ws_mixin._log.name == "hermes_cli.web_server"


def test_console_route_registered_before_spa_catch_all():
    paths = [getattr(r, "path", None) for r in web_server.app.routes]
    assert "/api/console" in paths
    assert paths.index("/api/console") < paths.index("/{full_path:path}")


# ---------------------------------------------------------------------------
# Pure logic: close-reason clamping (RFC 6455 123-byte limit)
# ---------------------------------------------------------------------------


def test_ws_close_reason_passthrough_short():
    reason = "peer_not_loopback peer=10.0.0.5"
    assert ws_auth_gate_mixin._ws_close_reason(reason) == reason


def test_ws_close_reason_passthrough_exactly_123_bytes():
    reason = "x" * 123
    assert ws_auth_gate_mixin._ws_close_reason(reason) == reason


def test_ws_close_reason_truncates_long_ascii():
    clamped = ws_auth_gate_mixin._ws_close_reason("x" * 200)
    assert len(clamped.encode("utf-8")) <= 123
    assert clamped.endswith("...")
    assert clamped.startswith("x" * 117)


def test_ws_close_reason_truncates_multibyte_without_splitting_chars():
    clamped = ws_auth_gate_mixin._ws_close_reason("\u00e9" * 200)
    assert len(clamped.encode("utf-8")) <= 123
    assert clamped.endswith("...")


# ---------------------------------------------------------------------------
# Pure logic: console JSON frame parsing
# ---------------------------------------------------------------------------


def test_console_json_payload_accepts_text_frame():
    payload, err = console_ws_mixin._console_json_payload({"text": '{"type": "ping"}'})
    assert err is None
    assert payload == {"type": "ping"}


def test_console_json_payload_accepts_bytes_frame():
    payload, err = console_ws_mixin._console_json_payload({"bytes": b'{"type": "ping"}'})
    assert err is None
    assert payload == {"type": "ping"}


def test_console_json_payload_rejects_invalid_utf8():
    payload, err = console_ws_mixin._console_json_payload({"bytes": b"\xff\xfe"})
    assert payload is None
    assert err == "Console frames must be UTF-8 JSON."


def test_console_json_payload_rejects_non_json_text():
    payload, err = console_ws_mixin._console_json_payload({"text": "not json"})
    assert payload is None
    assert err == "Console frames must be JSON objects."


def test_console_json_payload_rejects_non_object_json():
    payload, err = console_ws_mixin._console_json_payload({"text": "[1, 2, 3]"})
    assert payload is None
    assert err == "Console frames must be JSON objects."


def test_console_json_payload_empty_message():
    payload, err = console_ws_mixin._console_json_payload({})
    assert payload is None
    assert err is None


def test_console_executor_is_singleton_bounded_pool():
    first = console_ws_mixin._get_console_executor()
    second = console_ws_mixin._get_console_executor()
    assert first is second
    assert first._max_workers == 4


# ---------------------------------------------------------------------------
# Gate logic driven through the real dashboard app object (same object the
# moved functions read via the lazy web_server import).
# ---------------------------------------------------------------------------


def _fake_ws(*, query: dict | None = None, client_host: str = "127.0.0.1"):
    class _QP:
        def __init__(self, q):
            self._q = q

        def get(self, k, default=""):
            return self._q.get(k, default)

    return SimpleNamespace(
        query_params=_QP(query or {}),
        client=SimpleNamespace(host=client_host),
        url=SimpleNamespace(path="/api/ws"),
        headers={},
    )


@pytest.fixture
def ws_gate_state(monkeypatch):
    """Drive gate state on web_server.app.state (restored after each test)."""
    state = web_server.app.state
    monkeypatch.setattr(state, "auth_required", False, raising=False)
    monkeypatch.setattr(state, "bound_host", None, raising=False)
    return state


def test_ws_auth_mode_loopback(ws_gate_state):
    assert ws_auth_gate_mixin._ws_auth_mode() == "loopback"


def test_ws_auth_mode_insecure(ws_gate_state):
    ws_gate_state.bound_host = "0.0.0.0"
    assert ws_auth_gate_mixin._ws_auth_mode() == "insecure"


def test_ws_auth_mode_gated(ws_gate_state):
    ws_gate_state.auth_required = True
    assert ws_auth_gate_mixin._ws_auth_mode() == "gated"


def test_ws_client_reason_loopback_allowed(ws_gate_state):
    assert ws_auth_gate_mixin._ws_client_reason(_fake_ws(client_host="127.0.0.1")) is None


def test_ws_client_reason_peer_not_loopback(ws_gate_state):
    reason = ws_auth_gate_mixin._ws_client_reason(_fake_ws(client_host="203.0.113.7"))
    assert reason is not None
    assert reason.startswith("peer_not_loopback")


def test_ws_client_reason_missing_peer_fails_closed(ws_gate_state):
    reason = ws_auth_gate_mixin._ws_client_reason(SimpleNamespace(client=None))
    assert reason is not None
    assert reason.startswith("missing_or_empty_peer")


def test_ws_client_reason_explicit_bind_allows_any_peer(ws_gate_state):
    ws_gate_state.bound_host = "100.64.0.10"
    assert ws_auth_gate_mixin._ws_client_reason(_fake_ws(client_host="198.51.100.9")) is None


def test_ws_client_is_allowed_loopback_only(ws_gate_state):
    assert ws_auth_gate_mixin._ws_client_is_allowed(_fake_ws(client_host="::1")) is True
    assert ws_auth_gate_mixin._ws_client_is_allowed(_fake_ws(client_host="198.51.100.9")) is False


def test_ws_request_is_allowed_rejects_non_loopback(ws_gate_state):
    ws = _fake_ws(query={"token": "x"}, client_host="198.51.100.9")
    assert ws_auth_gate_mixin._ws_request_is_allowed(ws) is False


def test_ws_auth_ok_loopback_token(ws_gate_state):
    ws = _fake_ws(query={"token": web_server._SESSION_TOKEN})
    assert ws_auth_gate_mixin._ws_auth_ok(ws) is True


def test_ws_auth_reason_missing_credential(ws_gate_state):
    reason, cred = ws_auth_gate_mixin._ws_auth_reason(_fake_ws(query={}))
    assert reason == "no_credential"
    assert cred == "none"


def test_ws_auth_ok_bad_token(ws_gate_state):
    ws = _fake_ws(query={"token": "not-the-token"})
    assert ws_auth_gate_mixin._ws_auth_ok(ws) is False
