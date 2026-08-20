"""Tests for bounded CLI urllib response reads."""

from __future__ import annotations

import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _FakeResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.read_calls: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_calls.append(size)
        if size is None or size < 0:
            return self.body
        return self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_read_limited_json_response_rejects_oversized_body():
    from hermes_cli._http_response_limits import read_limited_json_response

    response = _FakeResponse(b"x" * 9)

    with pytest.raises(RuntimeError, match="test response exceeded 8 bytes"):
        read_limited_json_response(response, limit=8, label="test response")

    assert response.read_calls == [9]


def test_dashboard_register_bounds_success_response(monkeypatch):
    from hermes_cli.dashboard_register import _register_self_hosted_client

    monkeypatch.setattr(
        "hermes_cli._http_response_limits.JSON_RESPONSE_BODY_MAX_BYTES",
        8,
    )
    response = _FakeResponse(b"x" * 9)

    with patch(
        "hermes_cli.dashboard_register.urllib.request.urlopen",
        return_value=response,
    ):
        with pytest.raises(
            RuntimeError,
            match="dashboard registration response body exceeded 8 bytes",
        ):
            _register_self_hosted_client(
                access_token="token",
                portal_base_url="https://portal.example",
                name="local",
                custom_redirect_uri=None,
            )

    assert response.read_calls == [9]


def test_gateway_enroll_bounds_success_response(monkeypatch):
    from hermes_cli.gateway_enroll import _post_enroll

    monkeypatch.setattr(
        "hermes_cli._http_response_limits.JSON_RESPONSE_BODY_MAX_BYTES",
        8,
    )
    response = _FakeResponse(b"x" * 9)

    with patch(
        "hermes_cli.gateway_enroll.urllib.request.urlopen",
        return_value=response,
    ):
        with pytest.raises(
            RuntimeError,
            match="gateway enrollment response body exceeded 8 bytes",
        ):
            _post_enroll(
                connector_base_url="https://connector.example",
                access_token="token",
                enrollment_token="enroll",
                gateway_id="gw-test",
            )

    assert response.read_calls == [9]


@pytest.mark.parametrize("status", [401, 403])
def test_dashboard_register_bounds_http_error_body(monkeypatch, status):
    from hermes_cli.dashboard_register import _register_self_hosted_client

    monkeypatch.setattr(
        "hermes_cli._http_response_limits.ERROR_RESPONSE_BODY_MAX_BYTES",
        8,
    )
    response = _FakeResponse(b"x" * 9)
    http_error = urllib.error.HTTPError(
        url="https://portal.example/api/oauth/self-hosted-client",
        code=status,
        msg="Forbidden",
        hdrs={},
        fp=response,
    )

    with patch(
        "hermes_cli.dashboard_register.urllib.request.urlopen",
        side_effect=http_error,
    ):
        with pytest.raises(
            RuntimeError,
            match="dashboard registration error response body exceeded 8 bytes",
        ):
            _register_self_hosted_client(
                access_token="token",
                portal_base_url="https://portal.example",
                name="local",
                custom_redirect_uri=None,
            )

    assert response.read_calls == [9]


def test_gateway_enroll_reports_oversized_unauthorized_body(monkeypatch):
    from hermes_cli.gateway_enroll import _post_enroll

    monkeypatch.setattr(
        "hermes_cli._http_response_limits.ERROR_RESPONSE_BODY_MAX_BYTES",
        8,
    )
    response = _FakeResponse(b"x" * 9)
    http_error = urllib.error.HTTPError(
        url="https://connector.example/relay/enroll",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=response,
    )

    with patch(
        "hermes_cli.gateway_enroll.urllib.request.urlopen",
        side_effect=http_error,
    ):
        with pytest.raises(
            RuntimeError,
            match="gateway enrollment error response body exceeded 8 bytes",
        ):
            _post_enroll(
                connector_base_url="https://connector.example",
                access_token="token",
                enrollment_token="enroll",
                gateway_id="gw-test",
            )

    assert response.read_calls == [9]


def test_dashboard_register_preserves_generic_message_for_bad_error_json():
    from hermes_cli.dashboard_register import _register_self_hosted_client

    response = _FakeResponse(b"not json")
    http_error = urllib.error.HTTPError(
        url="https://portal.example/api/oauth/self-hosted-client",
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=response,
    )

    with patch(
        "hermes_cli.dashboard_register.urllib.request.urlopen",
        side_effect=http_error,
    ):
        with pytest.raises(
            RuntimeError,
            match="Your account is not permitted to register a self-hosted dashboard",
        ):
            _register_self_hosted_client(
                access_token="token",
                portal_base_url="https://portal.example",
                name="local",
                custom_redirect_uri=None,
            )


@pytest.mark.parametrize("command", ["dashboard", "gateway"])
def test_cli_commands_report_oversized_success_responses(monkeypatch, capsys, command):
    from hermes_cli import dashboard_register, gateway_enroll
    from hermes_cli._http_response_limits import ResponseBodyTooLarge

    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    if command == "dashboard":
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_nous_access_token", lambda: "token"
        )
        monkeypatch.setattr("hermes_cli.config.get_env_value", lambda _key: None)
        monkeypatch.setattr(
            dashboard_register,
            "_register_self_hosted_client",
            lambda **_kwargs: (_ for _ in ()).throw(
                ResponseBodyTooLarge("dashboard response exceeded 8 bytes")
            ),
        )
        invoke = lambda: dashboard_register.cmd_dashboard_register(
            SimpleNamespace(portal_url=None, name="local", redirect_uri=None)
        )
        expected = "Registration failed: dashboard response exceeded 8 bytes"
    else:
        monkeypatch.setattr(gateway_enroll, "_resolve_identity_token", lambda: "token")
        monkeypatch.setattr(
            gateway_enroll,
            "_post_enroll",
            lambda **_kwargs: (_ for _ in ()).throw(
                ResponseBodyTooLarge("enrollment response exceeded 8 bytes")
            ),
        )
        invoke = lambda: gateway_enroll.cmd_gateway_enroll(
            SimpleNamespace(
                token="enroll",
                connector_url="https://connector.example",
                gateway_id="gw-test",
                wake_url=None,
            )
        )
        expected = "Enrollment failed: enrollment response exceeded 8 bytes"

    with pytest.raises(SystemExit, match="1"):
        invoke()

    assert expected in capsys.readouterr().out
