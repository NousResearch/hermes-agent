"""``GET /v1/profiles`` and ``/v1/profiles/{name}/avatar``: the roster external clients draw from.

Exercises the real route table behind the real profile-prefix middleware with distinct
default and named-profile keys, because the boundary that matters is *who may learn which
profiles exist*: the listener's key sees the roster, a profile's own key sees only itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import secret_scope as ss
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

DEFAULT_KEY = "d" * 32
WORKER_KEY = "w" * 32
OTHER_KEY = "o" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"avatar"


def _profile(root, name, *, key=None, ui_meta=None, avatar=False):
    home = root / name
    home.mkdir(parents=True)
    if key:
        (home / ".env").write_text(f"API_SERVER_KEY={key}\n", encoding="utf-8")
    if ui_meta is not None:
        import hermes_yaml as yaml

        (home / "profile.yaml").write_text(yaml.safe_dump({"ui_meta": ui_meta}), encoding="utf-8")
    if avatar:
        (home / "assets").mkdir()
        (home / "assets" / "avatar.png").write_bytes(PNG)
    return home


def _info(name, is_default):
    return SimpleNamespace(
        name=name, is_default=is_default, model="m", provider="p", description=f"{name} bot",
        display_name=name.title(), skill_count=3, path=f"/secret/{name}",
    )


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Three served profiles; ``default`` and ``worker`` carry a bot title, ``worker`` and ``other`` an avatar."""
    homes = {
        "default": _profile(tmp_path, "default", ui_meta={"title": "Hermes"}),
        "worker": _profile(tmp_path, "worker", key=WORKER_KEY, ui_meta={"title": "Worker"}, avatar=True),
        "other": _profile(tmp_path, "other", key=OTHER_KEY, avatar=True),
    }
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex=False: list(homes.items()))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda lazy_skill_count=False: [_info(n, n == "default") for n in homes],
    )
    return homes


@pytest_asyncio.fixture
async def client(served):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._api_key = DEFAULT_KEY
    adapter.gateway_runner = type("_Runner", (), {"config": GatewayConfig(multiplex_profiles=True)})()
    ss.set_multiplex_active(True)
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        if "/v1/profiles" in path:
            app.router.add_route(method, path, handler)
            app.router.add_route(method, "/p/{profile}" + path, handler)
    async with TestClient(TestServer(app)) as c:
        yield c


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.mark.asyncio
async def test_listener_key_sees_every_served_profile(client):
    response = await client.get("/v1/profiles", headers=_auth(DEFAULT_KEY))
    assert response.status == 200
    payload = await response.json()
    assert payload["object"] == "list"
    rows = {row["name"]: row for row in payload["profiles"]}
    assert set(rows) == {"default", "worker", "other"}
    assert rows["default"]["is_default"] is True
    assert rows["worker"]["has_avatar"] is True and rows["default"]["has_avatar"] is False
    assert rows["worker"]["ui_meta"] == {"title": "Worker"}
    assert "ui_meta" not in rows["other"]


@pytest.mark.asyncio
async def test_roster_never_exposes_paths_or_secrets(client):
    body = await (await client.get("/v1/profiles", headers=_auth(DEFAULT_KEY))).text()
    for leaked in ("/secret/", WORKER_KEY, OTHER_KEY, DEFAULT_KEY, "API_SERVER_KEY"):
        assert leaked not in body


@pytest.mark.asyncio
async def test_profile_key_sees_only_its_own_profile(client):
    response = await client.get("/p/worker/v1/profiles", headers=_auth(WORKER_KEY))
    assert response.status == 200
    assert [row["name"] for row in (await response.json())["profiles"]] == ["worker"]


@pytest.mark.asyncio
async def test_profile_key_cannot_fetch_another_profiles_avatar(client):
    own = await client.get("/p/worker/v1/profiles/worker/avatar", headers=_auth(WORKER_KEY))
    assert own.status == 200 and await own.read() == PNG
    other = await client.get("/p/worker/v1/profiles/other/avatar", headers=_auth(WORKER_KEY))
    assert other.status == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("path,key", [
    ("/v1/profiles", "x" * 32),
    ("/v1/profiles", WORKER_KEY),            # a profile key is not the listener key
    ("/p/worker/v1/profiles", DEFAULT_KEY),  # and the listener key is not a profile key
    ("/p/worker/v1/profiles", OTHER_KEY),
    ("/v1/profiles/worker/avatar", "x" * 32),
])
async def test_wrong_key_is_refused_before_anything_is_read(client, path, key):
    assert (await client.get(path, headers=_auth(key))).status == 401


@pytest.mark.asyncio
async def test_avatar_errors_are_404_not_500(client):
    for name in ("default", "nope", "..%2f..%2fetc"):
        response = await client.get(f"/v1/profiles/{name}/avatar", headers=_auth(DEFAULT_KEY))
        assert response.status == 404


@pytest.mark.asyncio
async def test_oversized_ui_meta_is_dropped_not_served(served, client):
    import hermes_yaml as yaml

    (served["worker"] / "profile.yaml").write_text(
        yaml.safe_dump({"ui_meta": {"blob": "x" * 70_000}}), encoding="utf-8")
    response = await client.get("/v1/profiles", headers=_auth(DEFAULT_KEY))
    rows = {row["name"]: row for row in (await response.json())["profiles"]}
    assert "ui_meta" not in rows["worker"]


def test_capabilities_advertise_the_roster():
    from gateway.platforms import api_server as mod

    assert mod._STATIC_FEATURE_FLAGS["profile_roster"] is True
    names = {name for name, _ in mod._CAPABILITY_ENDPOINTS}
    assert {"profiles", "profile_avatar"} <= names
