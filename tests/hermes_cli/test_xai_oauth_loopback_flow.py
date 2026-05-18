"""Regression coverage for xAI OAuth loopback orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import threading

import pytest

from hermes_cli.auth import (
    AuthError,
    _xai_oauth_loopback_login,
    _xai_wait_for_callback,
)


_TOKEN_PAYLOAD = {
    "access_token": "AT",
    "refresh_token": "RT",
    "id_token": "ID",
    "expires_in": 3600,
    "token_type": "Bearer",
}


def _server_and_thread():
    server = MagicMock()
    thread = MagicMock()
    return server, thread


def test_wait_for_callback_returns_empty_result_after_timeout():
    server, thread = _server_and_thread()
    event = threading.Event()
    result = {"code": None, "state": None, "error": None, "error_description": None}

    out = _xai_wait_for_callback(server, thread, result, event, timeout_seconds=0.001)

    assert out == {"code": None, "state": None, "error": None, "error_description": None}
    server.shutdown.assert_called_once_with()
    server.server_close.assert_called_once_with()
    thread.join.assert_called_once_with(timeout=1.0)


def test_wait_for_callback_returns_callback_payload_when_event_set():
    server, thread = _server_and_thread()
    event = threading.Event()
    event.set()
    result = {
        "code": "AUTHCODE",
        "state": "STATE",
        "error": None,
        "error_description": None,
    }

    out = _xai_wait_for_callback(server, thread, result, event, timeout_seconds=10.0)

    assert out is result
    server.shutdown.assert_called_once_with()
    server.server_close.assert_called_once_with()
    thread.join.assert_called_once_with(timeout=1.0)


def test_wait_for_callback_raises_when_event_has_no_payload():
    server, thread = _server_and_thread()
    event = threading.Event()
    event.set()
    result = {"code": None, "state": None, "error": None, "error_description": None}

    with pytest.raises(AuthError) as exc_info:
        _xai_wait_for_callback(server, thread, result, event, timeout_seconds=10.0)

    assert exc_info.value.code == "xai_callback_empty"
    server.shutdown.assert_called_once_with()
    server.server_close.assert_called_once_with()
    thread.join.assert_called_once_with(timeout=1.0)


def _patch_loopback_common(callback_result, exchange_recorder):
    server, thread = _server_and_thread()
    event = threading.Event()

    def _exchange(**kwargs):
        exchange_recorder.append(kwargs)
        return dict(_TOKEN_PAYLOAD)

    patches = [
        patch(
            "hermes_cli.auth._xai_oauth_discovery",
            return_value={
                "authorization_endpoint": "https://accounts.x.ai/oauth2/auth",
                "token_endpoint": "https://accounts.x.ai/oauth2/token",
            },
        ),
        patch(
            "hermes_cli.auth._xai_start_callback_server",
            return_value=(
                server,
                thread,
                {"code": None, "state": None, "error": None, "error_description": None},
                "http://127.0.0.1:56121/callback",
                event,
            ),
        ),
        patch("hermes_cli.auth._xai_validate_loopback_redirect_uri"),
        patch("hermes_cli.auth._oauth_pkce_code_verifier", return_value="VERIFIER"),
        patch("hermes_cli.auth._oauth_pkce_code_challenge", return_value="CHALLENGE"),
        patch(
            "hermes_cli.auth.uuid.uuid4",
            side_effect=[SimpleNamespace(hex="STATE"), SimpleNamespace(hex="NONCE")],
        ),
        patch("hermes_cli.auth._print_loopback_ssh_hint"),
        patch("hermes_cli.auth._xai_wait_for_callback", return_value=callback_result),
        patch("hermes_cli.auth._xai_oauth_exchange_code_for_tokens", side_effect=_exchange),
    ]
    return patches


def test_loopback_login_passes_code_challenge_to_callback_exchange():
    exchange_calls = []
    callback_result = {
        "code": "AUTHCODE",
        "state": "STATE",
        "error": None,
        "error_description": None,
    }

    patches = _patch_loopback_common(callback_result, exchange_calls)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
         patches[6], patches[7], patches[8]:
        out = _xai_oauth_loopback_login(timeout_seconds=12.0, open_browser=False)

    assert out["tokens"]["access_token"] == "AT"
    assert exchange_calls == [
        {
            "code": "AUTHCODE",
            "code_verifier": "VERIFIER",
            "code_challenge": "CHALLENGE",
            "redirect_uri": "http://127.0.0.1:56121/callback",
            "token_endpoint": "https://accounts.x.ai/oauth2/token",
            "timeout_seconds": 12.0,
        }
    ]


def test_loopback_login_passes_code_challenge_to_manual_paste_exchange():
    exchange_calls = []
    callback_result = {"code": None, "state": None, "error": None, "error_description": None}

    patches = _patch_loopback_common(callback_result, exchange_calls)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
         patches[6], patches[7], patches[8], \
         patch("builtins.input", return_value="PASTEDCODE"):
        out = _xai_oauth_loopback_login(timeout_seconds=12.0, open_browser=False)

    assert out["tokens"]["access_token"] == "AT"
    assert exchange_calls == [
        {
            "code": "PASTEDCODE",
            "code_verifier": "VERIFIER",
            "code_challenge": "CHALLENGE",
            "redirect_uri": "http://127.0.0.1:56121/callback",
            "token_endpoint": "https://accounts.x.ai/oauth2/token",
            "timeout_seconds": 12.0,
        }
    ]
