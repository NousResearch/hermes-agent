"""ElevenLabs dropdown contracts through real profile config and HTTP boundary."""

import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from hermes_cli.config import atomic_config_write
from hermes_cli.web_routers.audio import get_elevenlabs_voices


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    from agent import secret_scope
    from tui_gateway import launch_profile_policy

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    was_active = secret_scope.is_multiplex_active()
    snapshot = launch_profile_policy._snapshot

    def create(name, collection_id):
        target = home if name == "default" else home / "profiles" / name
        target.mkdir(parents=True, exist_ok=True)
        elevenlabs = {"voice_id": "unchanged-voice", "model_id": "unchanged-model"}
        if collection_id is not None:
            elevenlabs["collection_id"] = collection_id
        atomic_config_write(target / "config.yaml", {
            "tts": {"provider": "edge", "elevenlabs": elevenlabs},
        })
        (target / ".env").write_text("ELEVENLABS_API_KEY=unit-test-not-a-real-key\n")
        return target

    yield create
    secret_scope.set_multiplex_active(was_active)
    launch_profile_policy._snapshot = snapshot


def install_http(monkeypatch, respond):
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        assert timeout == 10
        payload = respond(request, len(calls))
        return io.BytesIO(payload if isinstance(payload, bytes) else json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return calls


@pytest.mark.asyncio
async def test_configured_collection_filters_dropdown(profiles, monkeypatch):
    profiles("curated", "test-collection")

    def respond(request, _):
        query = parse_qs(urlparse(request.full_url).query)
        voice_id = "included" if query.get("collection_id") == ["test-collection"] else "outside"
        return {"voices": [{"voice_id": voice_id, "name": "A Voice", "category": "cloned"}]}

    calls = install_http(monkeypatch, respond)
    result = await get_elevenlabs_voices("curated")
    assert result == {"available": True, "voices": [
        {"voice_id": "included", "name": "A Voice", "label": "A Voice (cloned)"},
    ]}
    assert urlparse(calls[0].full_url).path == "/v2/voices"
    assert parse_qs(urlparse(calls[0].full_url).query) == {
        "collection_id": ["test-collection"], "page_size": ["100"],
    }


@pytest.mark.asyncio
async def test_collection_paginates_deduplicates_and_sorts(profiles, monkeypatch):
    profiles("curated", "collection &/?")
    pages = [
        {"voices": [{"voice_id": "z", "name": "Zebra", "category": "cloned"}],
         "has_more": True, "next_page_token": "next &/?"},
        {"voices": [{"voice_id": "z", "name": "Duplicate"},
                    {"voice_id": "a", "name": "alpha"}, None, {"name": "No ID"}],
         "has_more": False},
    ]
    calls = install_http(monkeypatch, lambda req, count: pages[count - 1])
    result = await get_elevenlabs_voices("curated")
    assert result == {"available": True, "voices": [
        {"voice_id": "a", "name": "alpha", "label": "alpha"},
        {"voice_id": "z", "name": "Zebra", "label": "Zebra (cloned)"},
    ]}
    assert len(calls) == 2
    for request in calls:
        query = parse_qs(urlparse(request.full_url).query)
        assert query["collection_id"] == ["collection &/?"]
        assert query["page_size"] == ["100"]
    assert parse_qs(urlparse(calls[1].full_url).query)["next_page_token"] == ["next &/?"]


@pytest.mark.parametrize("token, expected_calls", [(None, 1), ("", 1), ("loop", 2)])
@pytest.mark.asyncio
async def test_invalid_pagination_fails_closed(profiles, monkeypatch, token, expected_calls):
    from fastapi import HTTPException

    profiles("curated", "test-collection")

    def respond(request, count):
        assert count <= 3, "pagination must terminate"
        return {"voices": [], "has_more": True, "next_page_token": token}

    calls = install_http(monkeypatch, respond)
    with pytest.raises(HTTPException) as exc:
        await get_elevenlabs_voices("curated")
    assert exc.value.status_code == 502
    assert len(calls) == expected_calls


@pytest.mark.asyncio
async def test_endless_unique_pagination_is_bounded(profiles, monkeypatch):
    from fastapi import HTTPException

    profiles("curated", "test-collection")

    def respond(request, count):
        assert count <= 101, "pagination must be bounded"
        return {"voices": [], "has_more": True, "next_page_token": str(count)}

    calls = install_http(monkeypatch, respond)
    with pytest.raises(HTTPException) as exc:
        await get_elevenlabs_voices("curated")
    assert exc.value.status_code == 502
    assert len(calls) == 100


def test_collection_default_is_registered(profiles):
    from hermes_cli.config import load_config
    from hermes_cli.web_server_profiles import _config_profile_scope

    profiles("curated", None)
    with _config_profile_scope("curated"):
        assert load_config()["tts"]["elevenlabs"]["collection_id"] == ""


@pytest.mark.parametrize("collection_id", [None, "", "   "])
@pytest.mark.asyncio
async def test_unconfigured_collection_retains_legacy_request(profiles, monkeypatch, collection_id):
    profiles("legacy", collection_id)
    calls = install_http(monkeypatch, lambda req, count: {
        "voices": [{"voice_id": "z", "name": "Zulu", "category": "premade"},
                   {"voice_id": "a"}], "has_more": True,
    })
    result = await get_elevenlabs_voices("legacy")
    assert [v["label"] for v in result["voices"]] == ["a", "Zulu (premade)"]
    assert len(calls) == 1
    assert calls[0].full_url == "https://api.elevenlabs.io/v1/voices"


@pytest.mark.asyncio
async def test_empty_collection_stays_empty(profiles, monkeypatch):
    profiles("curated", "empty-collection")
    calls = install_http(monkeypatch, lambda req, count: {"voices": [], "has_more": False})
    assert await get_elevenlabs_voices("curated") == {"available": True, "voices": []}
    assert len(calls) == 1
    assert urlparse(calls[0].full_url).path == "/v2/voices"


@pytest.mark.parametrize("failure", [401, 403, 404, 429, 500, "timeout", "json"])
@pytest.mark.parametrize("failed_page", [1, 2])
@pytest.mark.asyncio
async def test_upstream_failure_never_falls_back(profiles, monkeypatch, failure, failed_page):
    from email.message import Message
    from fastapi import HTTPException
    from urllib.error import HTTPError

    profiles("curated", "test-collection")

    def respond(request, count):
        if count < failed_page:
            return {"voices": [{"voice_id": "partial"}],
                    "has_more": True, "next_page_token": "next"}
        if failure == "timeout":
            raise TimeoutError("test timeout")
        if failure == "json":
            return b"invalid JSON"
        raise HTTPError(request.full_url, failure, "test failure", Message(), None)

    calls = install_http(monkeypatch, respond)
    if failure in (401, 403):
        assert await get_elevenlabs_voices("curated") == {
            "available": False, "voices": [], "error": "unauthorized",
        }
    else:
        with pytest.raises(HTTPException) as exc:
            await get_elevenlabs_voices("curated")
        assert exc.value.status_code == 502
    assert len(calls) == failed_page
    assert all(urlparse(req.full_url).path == "/v2/voices" for req in calls)


@pytest.mark.asyncio
async def test_collection_is_profile_local_and_read_only(profiles, monkeypatch):
    homes = [profiles("default", "launch-collection"),
             profiles("first", "first-collection"),
             profiles("second", "second-collection"),
             profiles("legacy", None)]
    before = {home: (home / "config.yaml").read_bytes() for home in homes}

    def respond(request, count):
        query = parse_qs(urlparse(request.full_url).query)
        return {"voices": [{"voice_id": query.get("collection_id", ["legacy"])[0]}]}

    calls = install_http(monkeypatch, respond)
    for profile, expected in [("first", "first-collection"), ("second", "second-collection"),
                              ("legacy", "legacy"), ("first", "first-collection"),
                              ("default", "launch-collection")]:
        result = await get_elevenlabs_voices(profile)
        assert [v["voice_id"] for v in result["voices"]] == [expected]
    assert len(calls) == 5
    assert {home: (home / "config.yaml").read_bytes() for home in homes} == before


@pytest.mark.asyncio
async def test_unavailable_config_is_http_503_without_voice_fallback(profiles, monkeypatch):
    from fastapi import HTTPException
    from hermes_cli.config_backend import ConfigBackendUnavailable
    profiles("curated", "collection")
    calls = install_http(monkeypatch, lambda request, count: {"voices": []})
    def unavailable():
        raise ConfigBackendUnavailable("test backend unavailable")
    monkeypatch.setattr("hermes_cli.config.load_config", unavailable)
    with pytest.raises(HTTPException) as raised:
        await get_elevenlabs_voices("curated")
    assert raised.value.status_code == 503
    assert calls == []
