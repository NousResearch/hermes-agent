"""Mocked tests for the Square skill's auth and API wrappers."""

import argparse
import importlib.util
import io
import json
import sys
import types
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "skills/productivity/square"
AUTH_PATH = SKILL_DIR / "scripts/square_auth.py"
API_PATH = SKILL_DIR / "scripts/square_api.py"
SETUP_PATH = SKILL_DIR / "scripts/setup.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def square_modules(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.syspath_prepend(str(AUTH_PATH.parent))
    sys.modules.pop("square_auth", None)

    auth = _load_module("square_auth", AUTH_PATH)
    sys.modules["square_auth"] = auth
    api = _load_module("square_api_test", API_PATH)
    setup = _load_module("square_setup_test", SETUP_PATH)
    return auth, api, setup


def _write_auth_files(auth, *, expires_at: str | None, access_token="old-token"):
    token = {
        "access_token": access_token,
        "refresh_token": "refresh-token",
    }
    if expires_at is not None:
        token["expires_at"] = expires_at
    auth.TOKEN_PATH.write_text(json.dumps(token))
    auth.CLIENT_SECRET_PATH.write_text(
        json.dumps({"clientId": "app-id", "clientSecret": "app-secret"})
    )


class _Response:
    def __init__(self, body: dict, status=200):
        self.body = json.dumps(body).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self.body


def test_frontmatter_and_required_sections_follow_skill_standard():
    content = (SKILL_DIR / "SKILL.md").read_text()
    description = next(
        line.removeprefix("description: ").strip('"')
        for line in content.splitlines()
        if line.startswith("description: ")
    )
    assert len(description) <= 60
    assert description.endswith(".")
    assert "author: Jamal Hinton (@Malgsx), Hermes Agent" in content

    sections = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [content.index(section) for section in sections]
    assert positions == sorted(positions)


def test_dependency_has_floor_and_upper_bound():
    setup_source = SETUP_PATH.read_text()
    assert 'SQUAREUP_REQUIREMENT = "squareup>=41.0.0.20250319,<42"' in setup_source
    assert '"--", SQUAREUP_REQUIREMENT' in setup_source


def test_valid_token_is_used_without_refresh(square_modules):
    auth, _api, _setup = square_modules
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    _write_auth_files(auth, expires_at=future, access_token="valid-token")

    with patch.object(auth.urllib.request, "urlopen") as urlopen:
        assert auth.get_valid_access_token() == "valid-token"

    urlopen.assert_not_called()


@pytest.mark.parametrize("expires_at", [None, "not-a-date"])
def test_legacy_or_invalid_expiry_refreshes_before_request(square_modules, expires_at):
    auth, _api, _setup = square_modules
    _write_auth_files(auth, expires_at=expires_at)
    response = _Response(
        {
            "access_token": "new-token",
            "refresh_token": "new-refresh-token",
            "expires_at": "2099-01-01T00:00:00Z",
        }
    )

    with patch.object(auth.urllib.request, "urlopen", return_value=response) as urlopen:
        assert auth.get_valid_access_token() == "new-token"

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data)
    assert payload["grant_type"] == "refresh_token"
    assert payload["refresh_token"] == "refresh-token"
    assert urlopen.call_args.kwargs["timeout"] == auth.REQUEST_TIMEOUT_SECONDS
    assert json.loads(auth.TOKEN_PATH.read_text())["access_token"] == "new-token"


def test_check_auth_rejects_non_200_refresh_response(square_modules, capsys):
    auth, _api, setup = square_modules
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _write_auth_files(auth, expires_at=past)

    with patch.object(
        auth.urllib.request,
        "urlopen",
        return_value=_Response({"error": "invalid_grant"}, status=400),
    ):
        assert setup.check_auth() is False

    output = capsys.readouterr().out
    assert "TOKEN_INVALID" in output
    assert "AUTHENTICATED" not in output
    assert "HTTP 400" in output


def test_sdk_client_uses_refresh_aware_token_loader(square_modules, monkeypatch):
    _auth, api, _setup = square_modules
    fake_square = types.ModuleType("square")
    fake_square_client = types.ModuleType("square.client")
    fake_square_http = types.ModuleType("square.http")
    fake_square_auth = types.ModuleType("square.http.auth")
    fake_oauth = types.ModuleType("square.http.auth.o_auth_2")
    fake_square_client.Client = MagicMock(return_value="client")
    fake_oauth.BearerAuthCredentials = MagicMock(return_value="credentials")
    fake_square.client = fake_square_client
    monkeypatch.setitem(sys.modules, "square", fake_square)
    monkeypatch.setitem(sys.modules, "square.client", fake_square_client)
    monkeypatch.setitem(sys.modules, "square.http", fake_square_http)
    monkeypatch.setitem(sys.modules, "square.http.auth", fake_square_auth)
    monkeypatch.setitem(sys.modules, "square.http.auth.o_auth_2", fake_oauth)

    with patch.object(api, "get_valid_access_token", return_value="fresh-token") as get_token:
        assert api._get_client() == "client"

    get_token.assert_called_once_with()
    fake_oauth.BearerAuthCredentials.assert_called_once_with("fresh-token")
    fake_square_client.Client.assert_called_once_with(
        bearer_auth_credentials="credentials",
        square_version=api.API_VERSION,
    )


def test_inventory_adjustment_uses_unique_keys_and_explicit_retry_key(
    square_modules, capsys
):
    _auth, api, _setup = square_modules
    captured = []
    args = argparse.Namespace(
        catalog_object_id="variation-1",
        location="location-1",
        quantity=2,
        reason="received",
        idempotency_key="",
    )

    def fake_request(_method, _path, body):
        captured.append(body["idempotency_key"])
        return {"ok": True}

    with patch.object(api, "_api_request", side_effect=fake_request):
        api.cmd_inventory_adjust(args)
        api.cmd_inventory_adjust(args)
        args.idempotency_key = "retry-the-same-request"
        api.cmd_inventory_adjust(args)

    assert captured[0] != captured[1]
    assert captured[2] == "retry-the-same-request"
    assert len(captured[0]) == 36
    capsys.readouterr()


def test_pagination_collects_all_pages_and_passes_cursor(square_modules):
    _auth, api, _setup = square_modules
    calls = []

    def page(**params):
        calls.append(params)
        if "cursor" not in params:
            return types.SimpleNamespace(
                body={"customers": [{"id": "a"}], "cursor": "next"},
                is_error=lambda: False,
            )
        return types.SimpleNamespace(
            body={"customers": [{"id": "b"}]},
            is_error=lambda: False,
        )

    result = api._paginate_sdk(page, "customers", limit=100)

    assert result == {"customers": [{"id": "a"}, {"id": "b"}]}
    assert calls == [{"limit": 100}, {"limit": 100, "cursor": "next"}]


def test_body_pagination_injects_cursor_into_each_request(square_modules):
    _auth, api, _setup = square_modules
    bodies = []

    def method(body):
        bodies.append(body)
        if "cursor" not in body:
            return types.SimpleNamespace(
                body={"orders": [{"id": "a"}], "cursor": "next"},
                is_error=lambda: False,
            )
        return types.SimpleNamespace(
            body={"orders": [{"id": "b"}]},
            is_error=lambda: False,
        )

    result = api._paginate_sdk(
        api._body_cursor_call(method, {"location_ids": ["location-1"]}),
        "orders",
    )

    assert result == {"orders": [{"id": "a"}, {"id": "b"}]}
    assert bodies == [
        {"location_ids": ["location-1"]},
        {"location_ids": ["location-1"], "cursor": "next"},
    ]


def test_pagination_stops_at_requested_max(square_modules):
    _auth, api, _setup = square_modules
    page = MagicMock(
        return_value=types.SimpleNamespace(
            body={"customers": [{"id": "a"}, {"id": "b"}], "cursor": "unused"},
            is_error=lambda: False,
        )
    )

    result = api._paginate_sdk(page, "customers", max_items=1)

    assert result == {"customers": [{"id": "a"}]}
    page.assert_called_once_with()


def test_sdk_error_is_reported(square_modules):
    _auth, api, _setup = square_modules
    response = types.SimpleNamespace(
        body={"errors": [{"code": "FORBIDDEN"}]},
        errors=[{"code": "FORBIDDEN"}],
        status_code=403,
        is_error=lambda: True,
    )

    with pytest.raises(api.SquareAPIError, match="403"):
        api._result_body(response)


def test_rest_request_refreshes_once_after_401(square_modules):
    _auth, api, _setup = square_modules
    unauthorized = urllib.error.HTTPError(
        "https://example.test",
        401,
        "Unauthorized",
        {},
        io.BytesIO(b'{"errors":[{"code":"UNAUTHORIZED"}]}'),
    )

    with (
        patch.object(api, "get_valid_access_token", side_effect=["old", "new"]) as get_token,
        patch.object(
            api.urllib.request,
            "urlopen",
            side_effect=[unauthorized, _Response({"ok": True})],
        ),
    ):
        assert api._api_request("GET", "locations") == {"ok": True}

    assert get_token.call_args_list == [
        (( ), {"force_refresh": False}),
        (( ), {"force_refresh": True}),
    ]
