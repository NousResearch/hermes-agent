"""Tests for Photon dashboard auth and Spectrum management API helpers."""
from __future__ import annotations

from typing import Any

import pytest

from plugins.platforms.photon import auth as photon_auth


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status
        self._json = json_body if json_body is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


def test_store_and_load_photon_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_cli import config as hermes_config

    env: dict[str, str] = {}

    monkeypatch.setattr(
        hermes_config,
        "save_env_value",
        lambda key, value: env.__setitem__(key, value),
    )
    monkeypatch.setattr(photon_auth, "_get_hermes_env_value", lambda key: env.get(key))

    photon_auth.store_photon_token("dashboard-token")

    assert env["PHOTON_DASHBOARD_TOKEN"] == "dashboard-token"
    assert photon_auth.load_photon_token() == "dashboard-token"


def test_store_and_load_project_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_cli import config as hermes_config

    env: dict[str, str] = {}

    monkeypatch.setattr(
        hermes_config,
        "save_env_value",
        lambda key, value: env.__setitem__(key, value),
    )
    monkeypatch.setattr(photon_auth, "_get_hermes_env_value", lambda key: env.get(key))

    photon_auth.store_project_credentials(
        "project-id",
        "project-secret",
        name="hermes-agent",
        dashboard_project_id="dashboard-id",
    )

    assert photon_auth.load_project_credentials() == ("project-id", "project-secret")
    assert env["PHOTON_PROJECT_NAME"] == "hermes-agent"
    assert env["PHOTON_DASHBOARD_PROJECT_ID"] == "dashboard-id"


def test_request_device_code_posts_compat_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["body"] = kwargs["json"]
        return _FakeResponse(
            json_body={
                "device_code": "device-code",
                "user_code": "USER-CODE",
                "verification_uri": "https://app.photon.codes/device",
                "verification_uri_complete": "https://app.photon.codes/device?code=USER-CODE",
                "expires_in": 600,
                "interval": 5,
            }
        )

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)

    code = photon_auth.request_device_code()

    assert code.device_code == "device-code"
    assert code.user_code == "USER-CODE"
    assert captured["url"].endswith("/api/auth/device/code")
    assert captured["body"]["client_id"] == "photon-cli"
    assert captured["body"]["scope"] == "openid profile email"


def test_poll_for_token_candidates_reads_body_and_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(_url: str, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            json_body={
                "access_token": "body-token",
                "data": {"accessToken": "data-token"},
            },
            headers={"set-auth-token": "Bearer header-token"},
        )

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)

    code = photon_auth.DeviceCode(
        device_code="device",
        user_code="user",
        verification_uri="https://app.photon.codes/device",
        verification_uri_complete=None,
        expires_in=10,
        interval=1,
    )
    candidates = photon_auth.poll_for_token_candidates(code, interval=0, timeout=1)

    assert [(item.source, item.token) for item in candidates] == [
        ("access_token", "body-token"),
        ("data.accessToken", "data-token"),
        ("set-auth-token", "header-token"),
    ]


def test_login_device_flow_persists_first_project_valid_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = photon_auth.DeviceCode(
        device_code="device",
        user_code="user",
        verification_uri="https://app.photon.codes/device",
        verification_uri_complete=None,
        expires_in=10,
        interval=1,
    )
    candidates = [
        photon_auth._DeviceTokenCandidate("access_token", "bad-token"),
        photon_auth._DeviceTokenCandidate("set-auth-token", "good-token"),
    ]
    validated: list[str] = []
    stored: list[str] = []

    def fake_validate(token: str) -> dict[str, Any]:
        validated.append(token)
        if token == "bad-token":
            raise photon_auth.PhotonDashboardAuthError("bad token")
        return {"id": "user-id"}

    monkeypatch.setattr(photon_auth, "request_device_code", lambda **_kwargs: code)
    monkeypatch.setattr(
        photon_auth,
        "poll_for_token_candidates",
        lambda *_args, **_kwargs: candidates,
    )
    monkeypatch.setattr(photon_auth, "validate_photon_token", fake_validate)
    monkeypatch.setattr(photon_auth, "store_photon_token", lambda token: stored.append(token))

    token = photon_auth.login_device_flow(open_browser=False)

    assert token == "good-token"
    assert validated == ["bad-token", "good-token"]
    assert stored == ["good-token"]


def test_create_user_posts_shared_type(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["body"] = kwargs["json"]
        captured["auth"] = kwargs["auth"]
        return _FakeResponse(
            json_body={
                "succeed": True,
                "data": {
                    "id": "user-id",
                    "phoneNumber": "+15550001000",
                    "assignedPhoneNumber": "+15550009999",
                },
            }
        )

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)

    user = photon_auth.create_user(
        "project-id",
        "project-secret",
        phone_number="+15550001000",
    )

    assert user["assignedPhoneNumber"] == "+15550009999"
    assert captured["auth"] == ("project-id", "project-secret")
    assert captured["body"]["type"] == "shared"
    assert captured["body"]["phoneNumber"] == "+15550001000"
    assert captured["url"].endswith("/projects/project-id/users/")


def test_find_project_user_by_phone_matches_assigned_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        photon_auth,
        "list_project_users",
        lambda *_args: [
            {
                "id": "user-id",
                "phoneNumber": "+15550001000",
                "assignedPhoneNumber": "+15550009999",
            }
        ],
    )

    user = photon_auth.find_project_user_by_phone(
        "project-id",
        "project-secret",
        "+15550009999",
    )

    assert user is not None
    assert user["phone_number"] == "+15550001000"
    assert user["assigned_phone_number"] == "+15550009999"


def test_ensure_phone_allowed_adds_to_existing_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import config as hermes_config

    env = {
        "PHOTON_ALLOWED_USERS": "+15550001000",
        "PHOTON_ALLOW_ALL_USERS": "",
    }

    monkeypatch.setattr(photon_auth, "_get_hermes_env_value", lambda key: env.get(key))
    monkeypatch.setattr(
        hermes_config,
        "save_env_value",
        lambda key, value: env.__setitem__(key, value),
    )

    assert photon_auth.ensure_phone_allowed("+15550002000") == "added"
    assert env["PHOTON_ALLOWED_USERS"] == "+15550001000,+15550002000"
