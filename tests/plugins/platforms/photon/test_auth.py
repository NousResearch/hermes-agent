"""Tests for the Photon auth module (device login + dashboard API)."""
from __future__ import annotations

import json
import os
from base64 import b64encode
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


_PHOTON_ENV = (
    "PHOTON_PROJECT_ID",
    "PHOTON_PROJECT_SECRET",
    "PHOTON_DASHBOARD_PROJECT_ID",
    "PHOTON_SPECTRUM_HOST",
    "PHOTON_ALLOWED_USERS",
    "PHOTON_HOME_CHANNEL",
)


@pytest.fixture
def tmp_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in _PHOTON_ENV:
        monkeypatch.delenv(key, raising=False)
    yield home
    # save_env_value() mutates os.environ directly, so scrub any leakage.
    for key in _PHOTON_ENV:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Credential storage

def test_store_and_load_photon_token(tmp_hermes_home: Path) -> None:
    photon_auth.store_photon_token("abc123def456")
    assert photon_auth.load_photon_token() == "abc123def456"

    auth_json = json.loads((tmp_hermes_home / "auth.json").read_text())
    assert auth_json["credential_pool"]["photon"][0]["access_token"] == "abc123def456"


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits only")
def test_save_auth_never_world_readable(tmp_hermes_home: Path) -> None:
    """auth.json must be created 0o600 — no window at process umask."""
    photon_auth.store_photon_token("secret-token")
    mode = (tmp_hermes_home / "auth.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_save_auth_leaves_no_temp_files(tmp_hermes_home: Path) -> None:
    photon_auth.store_photon_token("secret-token")
    leftovers = [p.name for p in tmp_hermes_home.iterdir() if p.name != "auth.json"]
    assert leftovers == []


def test_store_project_credentials_round_trip(
    tmp_hermes_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Don't touch .env / os.environ here — exercise the auth.json path.
    monkeypatch.setattr(photon_auth, "_persist_runtime_env", lambda *a, **k: None)
    photon_auth.store_project_credentials(
        spectrum_project_id="sp-123",
        project_secret="secret-key",
        dashboard_project_id="dash-456",
        name="Hermes Agent",
    )
    for key in _PHOTON_ENV:
        monkeypatch.delenv(key, raising=False)

    sid, secret = photon_auth.load_project_credentials()
    assert sid == "sp-123"
    assert secret == "secret-key"
    # Post-unification the management id resolves to the Spectrum id, not the
    # stored dashboard id — so a pre-backfill diverged install (whose old
    # dashboard id was rewritten and now 404s) still reaches the live row.
    assert photon_auth.load_dashboard_project_id() == "sp-123"


def test_store_project_credentials_writes_env(tmp_hermes_home: Path) -> None:
    photon_auth.store_project_credentials(
        spectrum_project_id="sp-789",
        project_secret="sek-ret",
        dashboard_project_id="dash-1",
    )
    env_text = (tmp_hermes_home / ".env").read_text()
    assert "PHOTON_PROJECT_ID=sp-789" in env_text
    assert "PHOTON_PROJECT_SECRET=sek-ret" in env_text


def test_store_user_numbers_round_trip(tmp_hermes_home: Path) -> None:
    photon_auth.store_user_numbers(
        phone_number="+15551234567",
        assigned_phone_number="+16282679185",
        user_id="user-uuid",
        dashboard_project_id="dash-uuid",
    )

    phone, assigned = photon_auth.load_user_numbers()
    assert phone == "+15551234567"
    assert assigned == "+16282679185"

    summary = photon_auth.credential_summary()
    assert summary["phone_number"] == "+15551234567"
    assert summary["assigned_phone_number"] == "+16282679185"

    rendered: list[str] = []
    photon_auth.print_credential_summary(rendered.append)
    assert "  my number           : +15551234567" in rendered[0]
    assert "  assigned number     : +16282679185" in rendered[0]


def test_load_user_numbers_falls_back_to_home_channel(
    tmp_hermes_home: Path,
) -> None:
    from hermes_cli.config import save_env_value

    save_env_value("PHOTON_HOME_CHANNEL", "+15551234567")

    phone, assigned = photon_auth.load_user_numbers()
    assert phone == "+15551234567"
    assert assigned is None


def test_refresh_user_numbers_reads_existing_assignment(
    tmp_hermes_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    photon_auth.store_user_numbers(phone_number="+15551234567")

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        assert kwargs.get("headers", {}).get("Authorization") == (
            "Basic " + b64encode(b"sp:secret").decode("ascii")
        )
        assert url.endswith("/projects/sp/users/")
        return _FakeResponse(json_body={"succeed": True, "data": {"users": [{
            "id": "user-uuid",
            "phoneNumber": "+1 (555) 123-4567",
            "assignedPhoneNumber": "+16282679185",
        }]}})

    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)

    phone, assigned = photon_auth.refresh_user_numbers("sp", "secret")
    assert phone == "+15551234567"
    assert assigned == "+16282679185"
    assert photon_auth.load_user_numbers() == ("+15551234567", "+16282679185")


def test_load_project_credentials_env_override(
    tmp_hermes_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(photon_auth, "_persist_runtime_env", lambda *a, **k: None)
    photon_auth.store_project_credentials(
        spectrum_project_id="from-file", project_secret="secret-file",
    )
    monkeypatch.setenv("PHOTON_PROJECT_ID", "from-env")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "secret-env")
    sid, secret = photon_auth.load_project_credentials()
    assert sid == "from-env"
    assert secret == "secret-env"


# ---------------------------------------------------------------------------
# Device login flow

def test_request_device_code_uses_photon_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["body"] = kwargs.get("json")
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
    assert "/api/auth/device/code" in captured["url"]
    # Hosted Photon allowlists registered device clients — an unregistered
    # client_id is rejected with 400 invalid_client. We use Photon's published
    # CLI device client and send the standard scope.
    assert captured["body"]["client_id"] == "photon-cli"
    assert captured["body"]["scope"] == "openid profile email"


def _device_code() -> "photon_auth.DeviceCode":
    return photon_auth.DeviceCode(
        device_code="d", user_code="u",
        verification_uri="https://x", verification_uri_complete=None,
        expires_in=10, interval=0,
    )


def test_poll_for_token_body_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status=200, json_body={"access_token": "tok-body"})

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    assert photon_auth.poll_for_token(_device_code(), interval=0, timeout=2) == "tok-body"


def test_poll_for_token_session_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status=200, json_body={"session": {"access_token": "tok-sess"}})

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    assert photon_auth.poll_for_token(_device_code(), interval=0, timeout=2) == "tok-sess"


def test_poll_for_token_header_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status=200, json_body={}, headers={"set-auth-token": "tok-hdr"})

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    assert photon_auth.poll_for_token(_device_code(), interval=0, timeout=2) == "tok-hdr"


def test_poll_for_token_pending_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(status=400, json_body={"error": "authorization_pending"})
        return _FakeResponse(status=200, json_body={"access_token": "tok-eventual"})

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    assert photon_auth.poll_for_token(_device_code(), interval=0, timeout=5) == "tok-eventual"
    assert calls["n"] == 2


def test_poll_for_token_access_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status=400, json_body={"error": "access_denied"})

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    with pytest.raises(RuntimeError, match="access_denied"):
        photon_auth.poll_for_token(_device_code(), interval=0, timeout=2)


# ---------------------------------------------------------------------------
# Projects

def test_list_projects_unwraps_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(json_body=[{"id": "p1", "name": "Hermes Agent"}])

    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)
    projects = photon_auth.list_projects("tok")
    assert projects[0]["id"] == "p1"


def test_find_project_by_name_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(json_body={"data": [
            {"id": "p1", "name": "Other"},
            {"id": "p2", "name": "hermes agent"},
        ]})

    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)
    proj = photon_auth.find_project_by_name("tok", "Hermes Agent")
    assert proj is not None and proj["id"] == "p2"


def test_create_project_omits_spectrum_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["body"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers")
        return _FakeResponse(json_body={"success": True, "id": "new-proj"})

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    data = photon_auth.create_project("tok", name="Hermes Agent")
    assert data["id"] == "new-proj"
    # Spectrum is always provisioned at create-time; the field was dropped
    # from the API schema, so we must not send it.
    assert "spectrum" not in captured["body"]
    assert captured["body"]["name"] == "Hermes Agent"
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["url"].endswith("/api/projects")


def test_create_project_raises_without_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(json_body={"success": True})

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    with pytest.raises(RuntimeError, match="project id"):
        photon_auth.create_project("tok")


def test_regenerate_project_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        assert url.endswith("/regenerate-secret")
        return _FakeResponse(json_body={"success": True, "projectSecret": "rotated"})

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    assert photon_auth.regenerate_project_secret("tok", "p") == "rotated"


# ---------------------------------------------------------------------------
# Users

def test_create_user_rejects_invalid_phone() -> None:
    with pytest.raises(ValueError, match="E.164"):
        photon_auth.create_user("proj", "secret", phone_number="not-a-number")


def test_create_user_posts_dashboard_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["body"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers")
        return _FakeResponse(json_body={"succeed": True, "data": {
            "id": "user-uuid", "phoneNumber": "+15551234567",
        }})

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    user = photon_auth.create_user("proj-id", "secret", phone_number="+15551234567")
    assert user["id"] == "user-uuid"
    assert captured["body"]["type"] == "shared"
    assert captured["body"]["phoneNumber"] == "+15551234567"
    assert captured["headers"]["Authorization"] == (
        "Basic " + b64encode(b"proj-id:secret").decode("ascii")
    )
    assert captured["url"].endswith("/projects/proj-id/users/")


def test_register_user_if_absent_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    posted = {"n": 0}

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(json_body={"succeed": True, "data": {"users": [{
            "id": "u1",
            "phoneNumber": "+1 (555) 123-4567",
            "assignedPhoneNumber": "+16282679185",
        }]}})

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        posted["n"] += 1
        return _FakeResponse(json_body={"success": True, "user": {}})

    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)
    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    # Same number, different formatting — should match and NOT create.
    user, created = photon_auth.register_user_if_absent(
        "proj", "secret", phone_number="+15551234567",
    )
    assert created is False
    assert user["id"] == "u1"
    assert posted["n"] == 0
    # The reused user carries the assigned iMessage line ("TEXTS ON").
    assert photon_auth.user_assigned_line(user) == "+16282679185"


def test_user_assigned_line() -> None:
    assert (
        photon_auth.user_assigned_line({"assignedPhoneNumber": "+16282679185"})
        == "+16282679185"
    )
    # Own number present but no assignment yet (e.g. freshly created user).
    assert photon_auth.user_assigned_line({"phoneNumber": "+15551234567"}) is None
    assert photon_auth.user_assigned_line({"assignedPhoneNumber": ""}) is None
    assert photon_auth.user_assigned_line({}) is None
    assert photon_auth.user_assigned_line(None) is None


def test_register_user_if_absent_creates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(json_body={"succeed": True, "data": {"users": []}})

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(json_body={"succeed": True, "data": {"id": "u-new"}})

    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)
    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    user, created = photon_auth.register_user_if_absent(
        "proj", "secret", phone_number="+15551234567",
    )
    assert created is True
    assert user["id"] == "u-new"


# ---------------------------------------------------------------------------
# Lines (assigned number)

def test_get_imessage_line_returns_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(json_body=[
            {"id": "l1", "platform": "imessage", "phoneNumber": "+15559999999", "status": "active"},
        ])

    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)
    line = photon_auth.get_imessage_line("tok", "proj")
    assert line is not None and line["phoneNumber"] == "+15559999999"


def test_get_imessage_line_provisions_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    added = {"n": 0}

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(json_body=[])

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        added["n"] += 1
        assert kwargs.get("json", {}).get("platform") == "imessage"
        return _FakeResponse(json_body={"success": True, "line": {
            "id": "l-new", "platform": "imessage", "phoneNumber": "+15558888888",
        }})

    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)
    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    line = photon_auth.get_imessage_line("tok", "proj")
    assert added["n"] == 1
    assert line["phoneNumber"] == "+15558888888"


# ---------------------------------------------------------------------------
# Credential summary (no secret leakage)

def test_credential_summary_no_secret_leak(
    tmp_hermes_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(photon_auth, "_persist_runtime_env", lambda *a, **k: None)
    photon_auth.store_photon_token("token-aaaaaaaaaaaaaaaa")
    photon_auth.store_project_credentials(
        spectrum_project_id="sp-uuid",
        project_secret="secret-bbbbbbbbbbb",
        dashboard_project_id="dash-uuid",
    )
    summary = photon_auth.credential_summary()
    blob = "\n".join(summary.values())
    assert "token-aaaa" not in blob
    assert "secret-bbbb" not in blob
    assert summary["device_token"].startswith("✓")
    assert summary["project_key"].startswith("✓")
    # Unified id: dashboard id == Spectrum id, surfaced as one project id.
    assert summary["project_id"] == "sp-uuid"
    assert summary["phone_number"].startswith("✗ missing")
    assert summary["assigned_phone_number"].startswith("✗ missing")


# ---------------------------------------------------------------------------
# Device-token candidate extraction + dashboard validation.

def test_device_response_candidates_covers_known_shapes() -> None:
    candidates = photon_auth._device_response_token_candidates(
        {
            "access_token": "tok-snake",
            "accessToken": "tok-camel",
            "data": {"access_token": "tok-data"},
        },
        headers={"set-auth-token": "Bearer tok-header"},
    )
    by_source = {c.source: c.token for c in candidates}
    assert by_source["access_token"] == "tok-snake"
    assert by_source["accessToken"] == "tok-camel"
    assert by_source["data.access_token"] == "tok-data"
    # "Bearer " prefix is stripped from the header value.
    assert by_source["set-auth-token"] == "tok-header"


def test_device_response_candidates_dedupes() -> None:
    candidates = photon_auth._device_response_token_candidates(
        {"access_token": "same", "accessToken": "same"},
    )
    assert [c.token for c in candidates] == ["same"]


def test_validate_photon_token_rejects_unrecognized_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, *, headers: Dict[str, str], timeout: float) -> _FakeResponse:
        if url.endswith("/api/auth/get-session"):
            return _FakeResponse(json_body={})  # no "user" key
        return _FakeResponse(json_body=[])

    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)
    with pytest.raises(photon_auth.PhotonDashboardAuthError):
        photon_auth.validate_photon_token("some-token")


def test_validate_photon_token_rejects_project_api_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, *, headers: Dict[str, str], timeout: float) -> _FakeResponse:
        if url.endswith("/api/auth/get-session"):
            return _FakeResponse(json_body={"user": {"id": "u1"}})
        return _FakeResponse(status=403)  # project API rejects

    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)
    with pytest.raises(photon_auth.PhotonDashboardAuthError):
        photon_auth.validate_photon_token("some-token")


def test_login_device_flow_validates_before_persisting(
    tmp_hermes_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, *, json: Dict[str, Any], timeout: float) -> _FakeResponse:
        if url.endswith("/api/auth/device/code"):
            return _FakeResponse(json_body={
                "device_code": "dev", "user_code": "AAAA",
                "verification_uri": "https://app.photon.codes/device",
                "verification_uri_complete": None,
                "expires_in": 600, "interval": 0,
            })
        # device/token approval
        return _FakeResponse(json_body={"access_token": "good-token"})

    def fake_get(url: str, *, headers: Dict[str, str], timeout: float) -> _FakeResponse:
        if url.endswith("/api/auth/get-session"):
            return _FakeResponse(json_body={"user": {"id": "u1"}})
        return _FakeResponse(json_body=[])  # projects OK

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)

    token = photon_auth.login_device_flow(open_browser=False)
    assert token == "good-token"
    assert photon_auth.load_photon_token() == "good-token"


def test_login_device_flow_raises_when_token_invalid(
    tmp_hermes_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, *, json: Dict[str, Any], timeout: float) -> _FakeResponse:
        if url.endswith("/api/auth/device/code"):
            return _FakeResponse(json_body={
                "device_code": "dev", "user_code": "AAAA",
                "verification_uri": "https://app.photon.codes/device",
                "verification_uri_complete": None,
                "expires_in": 600, "interval": 0,
            })
        return _FakeResponse(json_body={"access_token": "bad-token"})

    def fake_get(url: str, *, headers: Dict[str, str], timeout: float) -> _FakeResponse:
        return _FakeResponse(status=401)  # session lookup rejects

    monkeypatch.setattr(photon_auth.httpx, "post", fake_post)
    monkeypatch.setattr(photon_auth.httpx, "get", fake_get)

    with pytest.raises(photon_auth.PhotonDashboardAuthError):
        photon_auth.login_device_flow(open_browser=False)
    # A token that failed validation must never be persisted.
    assert photon_auth.load_photon_token() is None
