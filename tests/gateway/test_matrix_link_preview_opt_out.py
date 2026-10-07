"""``disable_link_previews`` attaches an empty bundled-preview list (MSC4095) to outbound text so
clients that honor bundled previews render no cards for those events — per room, no client or
room setting involved."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.matrix import adapter as matrix_adapter
from plugins.platforms.matrix.adapter import MatrixAdapter

ROOM = "!digest:example.org"
OTHER = "!chat:example.org"
OPT_OUT = {"m.url_previews": [], "com.beeper.linkpreviews": []}


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("MATRIX_DISABLE_LINK_PREVIEWS", raising=False)


def _adapter(value=None) -> MatrixAdapter:
    extra = {"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}
    if value is not None:
        extra["disable_link_previews"] = value
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra=extra))
    adapter._send_room_message = AsyncMock(return_value="$evt")
    adapter._send_content_event = AsyncMock(return_value=None)
    return adapter


def _sent(adapter) -> dict:
    return adapter._send_room_message.await_args.args[1]


@pytest.mark.parametrize(("raw", "expected"), [
    (None, (False, set())), (False, (False, set())), (True, (True, set())),
    ("true", (True, set())), ("off", (False, set())), ("", (False, set())),
    (f"{ROOM}, {OTHER}", (False, {ROOM, OTHER})), ([ROOM], (False, {ROOM})),
])
def test_parse(raw, expected):
    assert matrix_adapter._parse_link_preview_opt_out(raw) == expected


def test_default_leaves_previews_alone():
    adapter = _adapter()
    asyncio.run(adapter.send(ROOM, "see https://example.org"))
    assert not set(OPT_OUT) & set(_sent(adapter))


def test_listed_room_gets_empty_preview_bundle_other_rooms_do_not():
    adapter = _adapter([ROOM])
    asyncio.run(adapter.send(ROOM, "see https://example.org"))
    content = _sent(adapter)
    assert {k: content[k] for k in OPT_OUT} == OPT_OUT
    assert content["body"] == "see https://example.org"

    asyncio.run(adapter.send(OTHER, "see https://example.org"))
    assert not set(OPT_OUT) & set(_sent(adapter))


def test_true_applies_to_every_room():
    adapter = _adapter(True)
    asyncio.run(adapter.send(OTHER, "https://example.org"))
    assert {k: _sent(adapter)[k] for k in OPT_OUT} == OPT_OUT


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("MATRIX_DISABLE_LINK_PREVIEWS", ROOM)
    adapter = _adapter(False)
    asyncio.run(adapter.send(ROOM, "https://example.org"))
    assert {k: _sent(adapter)[k] for k in OPT_OUT} == OPT_OUT


def test_edit_carries_opt_out_in_new_content():
    adapter = _adapter([ROOM])
    asyncio.run(adapter.edit_message(ROOM, "$orig", "now https://example.org"))
    content = adapter._send_content_event.await_args.args[1]
    assert {k: content[k] for k in OPT_OUT} == OPT_OUT
    assert {k: content["m.new_content"][k] for k in OPT_OUT} == OPT_OUT


def test_yaml_bridge_seeds_extra():
    seeded = matrix_adapter._apply_yaml_config({}, {"disable_link_previews": [ROOM]})
    assert seeded == {"disable_link_previews": [ROOM]}


def test_standalone_send_applies_opt_out(monkeypatch):
    captured = {}

    class _Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {"event_id": "$evt"}

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def put(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

    aiohttp = pytest.importorskip("aiohttp")
    monkeypatch.setattr(aiohttp, "ClientSession", _Session)
    cfg = PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "disable_link_previews": [ROOM]})
    result = asyncio.run(matrix_adapter._standalone_send(cfg, ROOM, "https://example.org"))
    assert result["success"] is True
    assert {k: captured["payload"][k] for k in OPT_OUT} == OPT_OUT

    asyncio.run(matrix_adapter._standalone_send(cfg, OTHER, "https://example.org"))
    assert not set(OPT_OUT) & set(captured["payload"])
