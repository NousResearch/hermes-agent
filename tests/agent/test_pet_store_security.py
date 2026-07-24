from __future__ import annotations

import base64

import httpx
import pytest

from agent.pet import manifest, store


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
_UNSAFE_URL = "http://169.254.169.254/latest/meta-data"


@pytest.fixture
def blocked_redirect_transport(monkeypatch):
    real_client = httpx.Client
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "169.254.169.254":
            return httpx.Response(200, content=_PNG_1X1, request=request)
        return httpx.Response(302, headers={"Location": _UNSAFE_URL}, request=request)

    transport = httpx.MockTransport(handler)
    legacy_client = real_client(transport=transport)

    def client_factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr(httpx, "get", legacy_client.get)
    monkeypatch.setattr(httpx, "stream", legacy_client.stream)
    yield requested
    legacy_client.close()


def _patch_client(monkeypatch, handler):
    real_client = httpx.Client
    requested: list[str] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return handler(request)

    transport = httpx.MockTransport(recording_handler)

    def client_factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    return requested


def test_download_blocks_non_petdex_redirect_before_request(
    blocked_redirect_transport,
    tmp_path,
):
    source_url = "https://assets.petdex.dev/sprite.webp"
    dest = tmp_path / "sprite.webp"

    with pytest.raises(store.PetStoreError, match="non-petdex"):
        store._download(source_url, dest, timeout=1)

    assert blocked_redirect_transport == [source_url]
    assert not dest.exists()


def test_download_json_blocks_non_petdex_redirect_before_request(
    blocked_redirect_transport,
):
    source_url = "https://assets.petdex.dev/pet.json"

    with pytest.raises(store.PetStoreError, match="non-petdex"):
        store._download_json(source_url, timeout=1)

    assert blocked_redirect_transport == [source_url]


def test_thumbnail_blocks_non_petdex_redirect_before_request(
    blocked_redirect_transport,
    monkeypatch,
    tmp_path,
):
    source_url = "https://assets.petdex.dev/sprite.png"
    monkeypatch.setattr(store, "get_hermes_home", lambda: tmp_path)

    assert store.thumbnail_png("demo", source_url=source_url, timeout=1) is None
    assert blocked_redirect_transport == [source_url]
    assert not (tmp_path / "pets" / ".thumbs" / "demo.png").exists()


def test_manifest_blocks_non_petdex_redirect_before_request(
    blocked_redirect_transport,
):
    manifest.clear_cache()

    with pytest.raises(manifest.ManifestError, match="non-petdex"):
        manifest.fetch_manifest(timeout=1, force=True)

    assert blocked_redirect_transport == [manifest.MANIFEST_URL]


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_download_json_follows_relative_petdex_redirect(monkeypatch, status_code):
    real_client = httpx.Client
    requested: list[str] = []
    source_url = "https://assets.petdex.dev/pets/demo/pet.json"
    redirected_url = "https://assets.petdex.dev/pets/shared/pet.json"

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if str(request.url) == source_url:
            return httpx.Response(
                status_code,
                headers={"Location": "../shared/pet.json"},
                request=request,
            )
        return httpx.Response(200, json={"id": "demo"}, request=request)

    transport = httpx.MockTransport(handler)
    legacy_client = real_client(transport=transport)

    def client_factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr(httpx, "get", legacy_client.get)
    try:
        assert store._download_json(source_url, timeout=1) == {"id": "demo"}
    finally:
        legacy_client.close()

    assert requested == [source_url, redirected_url]


def test_download_json_stops_after_redirect_limit(monkeypatch):
    real_client = httpx.Client
    requested: list[str] = []
    source_url = "https://assets.petdex.dev/pet.json"

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"Location": source_url}, request=request)

    transport = httpx.MockTransport(handler)

    def client_factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)

    with pytest.raises(store.PetStoreError, match="too many"):
        store._download_json(source_url, timeout=1)

    assert requested == [source_url] * 21


@pytest.mark.parametrize(
    "location",
    [
        "http://assets.petdex.dev/pet.json",
        "https://petdex.dev.evil.example/pet.json",
        "//evil.example/pet.json",
        "https://user@assets.petdex.dev/pet.json",
        "https://@assets.petdex.dev/pet.json",
        "https://assets.petdex.dev:8443/pet.json",
    ],
)
def test_download_json_rejects_unsafe_petdex_url_forms(monkeypatch, location):
    source_url = "https://assets.petdex.dev/pet.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": location}, request=request)

    requested = _patch_client(monkeypatch, handler)

    with pytest.raises(store.PetStoreError, match="non-petdex"):
        store._download_json(source_url, timeout=1)

    assert requested == [source_url]


def test_download_json_rejects_redirect_without_location(monkeypatch):
    source_url = "https://assets.petdex.dev/pet.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, request=request)

    requested = _patch_client(monkeypatch, handler)

    with pytest.raises(store.PetStoreError, match="missing Location"):
        store._download_json(source_url, timeout=1)

    assert requested == [source_url]


def test_download_json_does_not_follow_304(monkeypatch):
    source_url = "https://assets.petdex.dev/pet.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(304, headers={"Location": "/other.json"}, request=request)

    requested = _patch_client(monkeypatch, handler)

    with pytest.raises(httpx.HTTPStatusError):
        store._download_json(source_url, timeout=1)

    assert requested == [source_url]


def test_download_json_allows_twenty_redirects(monkeypatch):
    source_url = "https://assets.petdex.dev/pet.json?hop=0"

    def handler(request: httpx.Request) -> httpx.Response:
        hop = int(request.url.params["hop"])
        if hop < 20:
            return httpx.Response(
                302,
                headers={"Location": f"/pet.json?hop={hop + 1}"},
                request=request,
            )
        return httpx.Response(200, json={"id": "demo"}, request=request)

    requested = _patch_client(monkeypatch, handler)

    assert store._download_json(source_url, timeout=1) == {"id": "demo"}
    assert len(requested) == 21
    assert requested[-1] == "https://assets.petdex.dev/pet.json?hop=20"


@pytest.mark.parametrize(
    "location",
    [
        "https://petdex.dev:bad/pet.json",
        "https://petdex.dev/\x00pet.json",
    ],
)
def test_download_json_normalizes_invalid_redirect_errors(monkeypatch, location):
    source_url = "https://assets.petdex.dev/pet.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": location}, request=request)

    requested = _patch_client(monkeypatch, handler)

    with pytest.raises(store.PetStoreError, match="invalid petdex redirect"):
        store._download_json(source_url, timeout=1)

    assert requested == [source_url]


def test_download_json_rejects_unsafe_initial_url_without_request(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "unexpected"}, request=request)

    requested = _patch_client(monkeypatch, handler)

    with pytest.raises(store.PetStoreError, match="non-petdex"):
        store._download_json("http://assets.petdex.dev/pet.json", timeout=1)

    assert requested == []


def test_download_json_follows_protocol_relative_petdex_redirect(monkeypatch):
    source_url = "https://petdex.dev/pet.json"
    redirected_url = "https://assets.petdex.dev/shared/pet.json"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == source_url:
            return httpx.Response(
                302,
                headers={"Location": "//assets.petdex.dev/shared/pet.json"},
                request=request,
            )
        return httpx.Response(200, json={"id": "demo"}, request=request)

    requested = _patch_client(monkeypatch, handler)

    assert store._download_json(source_url, timeout=1) == {"id": "demo"}
    assert requested == [source_url, redirected_url]
