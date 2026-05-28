"""Tests for the Photon auth module (device login + project + user creation)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from plugins.platforms.photon import auth as photon_auth


# ---------------------------------------------------------------------------
# Fake httpx — we don't want to hit the real Photon API in unit tests.

class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        json_body: Any = None,
        headers: Dict[str, str] | None = None,
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
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def tmp_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The auth module memoises by reading get_hermes_home at call time
    # so the env var is what matters.
    return home


def test_store_and_load_photon_token(tmp_hermes_home: Path) -> None:
    photon_auth.store_photon_token("abc123def456")
    assert photon_auth.load_photon_token() == "abc123def456"

    auth_json = json.loads((tmp_hermes_home / "auth.json").read_text())
    assert "credential_pool" in auth_json
    assert auth_json["credential_pool"]["photon"][0]["access_token"] == "abc123def456"


def test_store_and_load_project_credentials(tmp_hermes_home: Path) -> None:
    photon_auth.store_project_credentials(
        "proj-uuid", "secret-key", name="Test Project",
    )
    pid, secret = photon_auth.load_project_credentials()
    assert pid == "proj-uuid"
    assert secret == "secret-key"


def test_load_project_credentials_env_override(
    tmp_hermes_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    photon_auth.store_project_credentials("from-file", "secret-file")
    monkeypatch.setenv("PHOTON_PROJECT_ID", "from-env")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "secret-env")
    pid, secret = photon_auth.load_project_credentials()
    assert pid == "from-env"
    assert secret == "secret-env"


def test_request_device_code(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, *, json: Dict[str, Any], timeout: float) -> _FakeResponse:
        captured["url"] = url
        captured["body"] = json
        return _FakeResponse(json_body={
            "device_code": "dev-code-xyz",
            "user_code": "ABCD-1234",
            "verification_uri": "https://app.photon.codes/device",
            "verification_uri_complete": "https://app.photon.codes/device?code=ABCD-1234",
            "expires_in": 600,
            "interval": 5,
        })

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)

    code = photon_auth.request_device_code()
    assert code.device_code == "dev-code-xyz"
    assert code.user_code == "ABCD-1234"
    assert code.expires_in == 600
    assert "/api/auth/device/code" in captured["url"]
    assert captured["body"]["client_id"] == "photon-cli"
    assert captured["body"]["scope"] == "openid profile email"


def test_poll_for_token_via_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token from set-auth-token header is the documented mechanism."""

    def fake_post(url: str, *, json: Dict[str, Any], timeout: float) -> _FakeResponse:
        return _FakeResponse(
            status=200,
            json_body={"session": {}, "user": {}},
            headers={"set-auth-token": "bearer-xyz"},
        )

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)

    code = photon_auth.DeviceCode(
        device_code="d", user_code="u",
        verification_uri="https://x", verification_uri_complete=None,
        expires_in=10, interval=0,
    )
    token = photon_auth.poll_for_token(code, interval=0, timeout=2)
    assert token == "bearer-xyz"


def test_poll_for_token_via_body_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the header is absent we fall back to session.access_token."""

    def fake_post(url: str, *, json: Dict[str, Any], timeout: float) -> _FakeResponse:
        return _FakeResponse(
            status=200,
            json_body={"session": {"access_token": "from-body"}, "user": {}},
        )

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    code = photon_auth.DeviceCode(
        device_code="d", user_code="u",
        verification_uri="https://x", verification_uri_complete=None,
        expires_in=10, interval=0,
    )
    assert photon_auth.poll_for_token(code, interval=0, timeout=2) == "from-body"


def test_poll_for_token_propagates_access_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, *, json: Dict[str, Any], timeout: float) -> _FakeResponse:
        return _FakeResponse(
            status=400, json_body={"error": "access_denied"},
        )

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    code = photon_auth.DeviceCode(
        device_code="d", user_code="u",
        verification_uri="https://x", verification_uri_complete=None,
        expires_in=10, interval=0,
    )
    with pytest.raises(RuntimeError, match="access_denied"):
        photon_auth.poll_for_token(code, interval=0, timeout=2)


def test_create_user_rejects_invalid_phone() -> None:
    with pytest.raises(ValueError, match="E.164"):
        photon_auth.create_user(
            "proj", "secret", phone_number="not-a-number",
        )


def test_create_user_posts_shared_type(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, *, json: Dict[str, Any], auth: tuple, timeout: float) -> _FakeResponse:
        captured["url"] = url
        captured["body"] = json
        captured["auth"] = auth
        return _FakeResponse(json_body={
            "succeed": True,
            "data": {
                "id": "user-uuid",
                "phoneNumber": "+15551234567",
                "assignedPhoneNumber": "+15559999999",
            },
        })

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    user = photon_auth.create_user(
        "proj-id", "proj-secret",
        phone_number="+15551234567",
    )
    assert user["assignedPhoneNumber"] == "+15559999999"
    assert captured["auth"] == ("proj-id", "proj-secret")
    assert captured["body"]["type"] == "shared"
    assert captured["body"]["phoneNumber"] == "+15551234567"
    assert "/projects/proj-id/users/" in captured["url"]


def test_register_webhook_surfaces_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, *, json: Dict[str, Any], auth: tuple, timeout: float) -> _FakeResponse:
        return _FakeResponse(json_body={
            "succeed": True,
            "data": {
                "id": "wh-uuid",
                "webhookUrl": json["webhookUrl"],
                "signingSecret": "0" * 64,
            },
        })

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    data = photon_auth.register_webhook(
        "proj", "secret", webhook_url="https://x.example.com/hook",
    )
    assert data["signingSecret"] == "0" * 64
    assert data["webhookUrl"] == "https://x.example.com/hook"


def test_persist_webhook_signing_secret_writes_env(
    tmp_hermes_home: Path,
) -> None:
    """The helper hands the secret to save_env_value, never returns it."""
    summary: list = []
    response = {
        "id": "wh-uuid",
        "webhookUrl": "https://x.example.com/hook",
        "signingSecret": "ABCDEF1234567890" * 4,
    }
    ok = photon_auth.persist_webhook_signing_secret(
        response, on_summary=summary.append,
    )

    assert ok is True
    env_path = tmp_hermes_home / ".env"
    assert env_path.exists()
    env_text = env_path.read_text()
    assert "PHOTON_WEBHOOK_SECRET=ABCDEF1234567890" in env_text
    # The on_summary callback gets the redacted response + a saved-to path;
    # none of those strings should leak the raw secret.
    joined = "\n".join(summary)
    assert "<redacted>" in joined
    assert "ABCDEF1234567890" not in joined


def test_persist_webhook_signing_secret_no_secret_no_write(
    tmp_hermes_home: Path,
) -> None:
    summary: list = []
    ok = photon_auth.persist_webhook_signing_secret(
        {"id": "wh-uuid", "webhookUrl": "https://x"},
        on_summary=summary.append,
    )
    assert ok is False
    # No env file written; summary callback still received the redacted
    # response (without a signingSecret key, nothing to redact).
    assert not (tmp_hermes_home / ".env").exists()


def test_credential_summary_returns_only_display_strings(
    tmp_hermes_home: Path,
) -> None:
    """credential_summary must not leak raw token/secret material."""
    photon_auth.store_photon_token("token-aaaaaaaaaaaaaaaa")
    photon_auth.store_project_credentials("proj-uuid", "secret-bbbbbbbbbbb")
    summary = photon_auth.credential_summary()
    blob = "\n".join(summary.values())
    assert "token-aaaa" not in blob
    assert "secret-bbbb" not in blob
    assert summary["device_token"].startswith("✓")
    assert summary["project_key"].startswith("✓")
    assert summary["project_id"] == "proj-uuid"


def test_print_credential_summary_emits_only_display_strings(
    tmp_hermes_home: Path,
) -> None:
    """The emit callback must never receive raw credential bytes."""
    photon_auth.store_photon_token("token-aaaaaaaaaaaaaaaa")
    photon_auth.store_project_credentials("proj-uuid", "secret-bbbbbbbbbbb")
    lines: list = []
    photon_auth.print_credential_summary(lines.append)
    blob = "\n".join(lines)
    assert "token-aaaa" not in blob
    assert "secret-bbbb" not in blob
    assert "✓ stored" in blob   # device token line
    assert "proj-uuid" in blob   # project id is intentionally surfaced
    # Header is always emitted
    assert any("Photon iMessage status" in line for line in lines)
