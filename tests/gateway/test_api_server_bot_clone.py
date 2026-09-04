from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "clone-secret"}))


def _request(*, profile: str = "helper", content_type: str = "application/gzip"):
    request = MagicMock()
    request.headers = {"Authorization": "Bearer clone-secret"}
    request.match_info = {"profile_name": profile}
    request.query = {}
    request.content_type = content_type
    return request


def _json(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


def test_bot_clone_routes_are_advertised():
    routes = {(method, path) for method, path, _ in _adapter()._http_route_table()}

    assert ("GET", "/v1/bots/{profile_name}/clone") in routes
    assert ("POST", "/v1/bots/clone") in routes


@pytest.mark.asyncio
async def test_download_hides_profiles_that_did_not_opt_in(monkeypatch):
    request = _request()
    monkeypatch.setattr("hermes_cli.bot_transfer.profile_is_cloneable", lambda _name: False)

    response = await _adapter()._handle_bot_clone_download(request)

    assert response.status == 404
    assert _json(response)["error"]["code"] == "bot_clone_unavailable"


@pytest.mark.asyncio
async def test_download_returns_clone_identity_and_archive(monkeypatch, tmp_path):
    archive = tmp_path / "helper.tar.gz"
    archive.write_bytes(b"archive")
    monkeypatch.setattr("hermes_cli.bot_transfer.profile_is_cloneable", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.bot_transfer.export_bot_profile",
        lambda _name, _output: (archive, "a8c214f7-37ee-4f50-95a4-939a51631283"),
    )

    response = await _adapter()._handle_bot_clone_download(_request())

    assert response.status == 200
    assert response.body == b"archive"
    assert response.headers["X-Hermes-Bot-Id"] == "a8c214f7-37ee-4f50-95a4-939a51631283"
    assert response.headers["X-Hermes-Profile-Name"] == "helper"


@pytest.mark.asyncio
async def test_upload_is_fail_closed_until_gateway_owner_enables_push(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_bot_push_enabled", lambda: False)

    response = await adapter._handle_bot_clone_upload(_request())

    assert response.status == 403
    assert _json(response)["error"]["code"] == "bot_clone_push_disabled"


@pytest.mark.asyncio
async def test_upload_never_overwrites_an_existing_bot(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_bot_push_enabled", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.bot_transfer.import_bot_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileExistsError("already exists")),
    )
    request = _request()
    request.read = AsyncMock(return_value=b"archive")

    response = await adapter._handle_bot_clone_upload(request)

    assert response.status == 409
    assert _json(response)["error"]["code"] == "bot_clone_conflict"


@pytest.mark.asyncio
async def test_upload_installs_under_requested_name(monkeypatch, tmp_path):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_bot_push_enabled", lambda: True)
    profile_dir = tmp_path / "renamed"
    profile_dir.mkdir()
    imported = MagicMock(return_value=(profile_dir, "a8c214f7-37ee-4f50-95a4-939a51631283"))
    monkeypatch.setattr("hermes_cli.bot_transfer.import_bot_profile", imported)
    monkeypatch.setattr("hermes_cli.profiles.check_alias_collision", lambda _name: "collision")
    request = _request()
    request.query = {"name": "renamed"}
    request.read = AsyncMock(return_value=b"archive")

    response = await adapter._handle_bot_clone_upload(request)

    assert response.status == 201
    assert _json(response) == {
        "object": "hermes.bot_clone",
        "name": "renamed",
        "bot_id": "a8c214f7-37ee-4f50-95a4-939a51631283",
    }
    assert imported.call_args.kwargs["name"] == "renamed"
