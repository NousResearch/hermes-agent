"""Adapter-side tests for the Telegram sticker collection hooks.

Covers ``_handle_sticker`` recording (plan §2): every inbound sticker —
static (vision path and cache-hit path), animated, video — is upserted into
the persistent collection, and entries new to the collection append a
mid-session "saved to your collection" note to the injected ``event.text``.
Also covers the config-driven seed refresh (plan §6):
``telegram.sticker_sets`` → ``PlatformConfig.extra`` →
``refresh_from_sets`` during post-connect housekeeping.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram import sticker_collection
import plugins.platforms.telegram.adapter as telegram_adapter_mod
from plugins.platforms.telegram.adapter import TelegramAdapter, _apply_yaml_config


@pytest.fixture(autouse=True)
def _sticker_cache_tmp(monkeypatch: pytest.MonkeyPatch):
    """Re-point the vision-description cache at the per-test HERMES_HOME.

    ``gateway.sticker_cache.CACHE_PATH`` is a module-level constant computed
    at import time, so patch it explicitly to keep the real cache functions
    hermetic (conftest already isolates HERMES_HOME per test).
    """
    import gateway.sticker_cache as sticker_cache
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(
        sticker_cache, "CACHE_PATH", get_hermes_home() / "sticker_cache.json"
    )
    return sticker_cache


def _make_adapter(extra: Dict[str, Any] | None = None) -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra or {})
    adapter._bot = None
    adapter._post_connect_task = None
    return adapter


def _make_sticker(
    uid: str = "uid-1",
    fid: str = "fid-1",
    *,
    emoji: str = "😀",
    set_name: str = "MyPack",
    animated: bool = False,
    video: bool = False,
    with_file: bool = False,
) -> Any:
    sticker = SimpleNamespace(
        file_unique_id=uid,
        file_id=fid,
        emoji=emoji,
        set_name=set_name,
        is_animated=animated,
        is_video=video,
    )
    if with_file:
        file_obj = SimpleNamespace(
            download_as_bytearray=AsyncMock(return_value=bytearray(b"webp-bytes"))
        )
        sticker.get_file = AsyncMock(return_value=file_obj)
    return sticker


def _event() -> Any:
    return SimpleNamespace(text=None)


# ---------------------------------------------------------------------------
# _handle_sticker: animated / video early-return branch


@pytest.mark.asyncio
async def test_animated_sticker_recorded_with_note() -> None:
    adapter = _make_adapter()
    sticker = _make_sticker("uid-anim", "fid-anim", animated=True)
    event = _event()

    await adapter._handle_sticker(SimpleNamespace(sticker=sticker), event)

    # Existing injection behavior is intact...
    assert event.text.startswith("[The user sent an animated sticker 😀~")
    # ...and the new sticker gained the mid-session collection note.
    assert "saved to your collection" in event.text
    assert "tg_send_sticker" in event.text
    assert "emoji 😀" in event.text
    assert 'set "MyPack"' in event.text

    entry = sticker_collection.resolve("uid-anim")
    assert entry is not None
    assert entry["file_id"] == "fid-anim"
    assert entry["kind"] == "animated"
    assert entry["set_name"] == "MyPack"


@pytest.mark.asyncio
async def test_video_sticker_recorded_with_note() -> None:
    adapter = _make_adapter()
    sticker = _make_sticker("uid-vid", "fid-vid", video=True, emoji="🚀")
    event = _event()

    await adapter._handle_sticker(SimpleNamespace(sticker=sticker), event)

    assert "saved to your collection" in event.text
    entry = sticker_collection.resolve("uid-vid")
    assert entry is not None
    assert entry["kind"] == "video"


# ---------------------------------------------------------------------------
# _handle_sticker: static cache-hit branch


@pytest.mark.asyncio
async def test_static_cache_hit_records_file_id_and_note(
    _sticker_cache_tmp,
) -> None:
    # Seed the vision cache so the static branch hits it (no vision call).
    _sticker_cache_tmp.cache_sticker_description(
        "uid-static", "a cat waving its paw", "😀", "MyPack"
    )
    adapter = _make_adapter()
    # No get_file on the sticker: a vision download would blow up.
    sticker = _make_sticker("uid-static", "fid-static")
    event = _event()

    await adapter._handle_sticker(SimpleNamespace(sticker=sticker), event)

    assert event.text.startswith('[The user sent a sticker 😀 from "MyPack"~')
    assert "a cat waving its paw" in event.text
    assert "saved to your collection" in event.text

    entry = sticker_collection.resolve("uid-static")
    assert entry is not None
    assert entry["file_id"] == "fid-static"  # file_id recorded on cache hit
    assert entry["kind"] == "static"
    # Description backfilled from the vision cache.
    assert entry["description"] == "a cat waving its paw"


@pytest.mark.asyncio
async def test_known_sticker_gets_no_note_and_refreshes_file_id(
    _sticker_cache_tmp,
) -> None:
    # The collection already knows this sticker (with a stale file_id).
    assert sticker_collection.record_sticker(
        "uid-known", "fid-old", emoji="😀", set_name="MyPack", kind="static"
    ) is True
    _sticker_cache_tmp.cache_sticker_description(
        "uid-known", "a cat waving its paw", "😀", "MyPack"
    )
    adapter = _make_adapter()
    sticker = _make_sticker("uid-known", "fid-new")
    event = _event()

    await adapter._handle_sticker(SimpleNamespace(sticker=sticker), event)

    # Injection still happens, but no mid-session note for known stickers.
    assert "a cat waving its paw" in event.text
    assert "saved to your collection" not in event.text
    # file_id refreshes silently (per-bot validity).
    assert sticker_collection.resolve("uid-known")["file_id"] == "fid-new"


# ---------------------------------------------------------------------------
# _handle_sticker: static vision path


@pytest.mark.asyncio
async def test_static_vision_path_records_after_cache_write(
    _sticker_cache_tmp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        telegram_adapter_mod,
        "cache_image_from_bytes",
        lambda data, ext=".webp": "/tmp/fake-sticker.webp",
    )
    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        AsyncMock(
            return_value=json.dumps(
                {"success": True, "analysis": "a cat waving its paw"}
            )
        ),
    )
    adapter = _make_adapter()
    sticker = _make_sticker("uid-vision", "fid-vision", with_file=True)
    event = _event()

    await adapter._handle_sticker(SimpleNamespace(sticker=sticker), event)

    assert "a cat waving its paw" in event.text
    assert "saved to your collection" in event.text

    entry = sticker_collection.resolve("uid-vision")
    assert entry is not None
    assert entry["file_id"] == "fid-vision"
    # The record ran AFTER cache_sticker_description(), so the collection's
    # best-effort cache backfill picked up the fresh vision description.
    assert entry["description"] == "a cat waving its paw"


@pytest.mark.asyncio
async def test_static_vision_failure_still_records_with_fallback(
    _sticker_cache_tmp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        telegram_adapter_mod,
        "cache_image_from_bytes",
        lambda data, ext=".webp": "/tmp/fake-sticker.webp",
    )
    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        AsyncMock(return_value=json.dumps({"success": False, "error": "boom"})),
    )
    adapter = _make_adapter()
    sticker = _make_sticker("uid-fail", "fid-fail")
    event = _event()

    await adapter._handle_sticker(SimpleNamespace(sticker=sticker), event)

    assert "a sticker with emoji 😀" in event.text  # existing fallback text
    assert "saved to your collection" in event.text
    entry = sticker_collection.resolve("uid-fail")
    assert entry is not None
    assert entry["description"] == ""  # nothing in the vision cache to backfill


@pytest.mark.asyncio
async def test_collection_failure_does_not_break_sticker_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(sticker_collection, "record_sticker", _explode)
    adapter = _make_adapter()
    sticker = _make_sticker("uid-anim", "fid-anim", animated=True)
    event = _event()

    await adapter._handle_sticker(SimpleNamespace(sticker=sticker), event)

    # Injection still happened; the collection error was swallowed.
    assert event.text.startswith("[The user sent an animated sticker 😀~")
    assert "saved to your collection" not in event.text


# ---------------------------------------------------------------------------
# Seed refresh (plan §6)


class _FakeBot:
    """Duck-typed stand-in for telegram.Bot (only get_sticker_set is used)."""

    def __init__(self, sets: Dict[str, List[Any]], failing: List[str] | None = None):
        self._sets = sets
        self._failing = set(failing or [])
        self.requested: List[str] = []

    async def get_sticker_set(self, name: str) -> Any:
        self.requested.append(name)
        if name in self._failing:
            raise RuntimeError(f"telegram exploded for {name}")
        return SimpleNamespace(name=name, stickers=self._sets[name])


def _pack_sticker(uid: str, emoji: str) -> Any:
    return SimpleNamespace(
        file_id=f"fid-{uid}", file_unique_id=uid, emoji=emoji,
        is_animated=False, is_video=False, set_name="",
    )


@pytest.mark.asyncio
async def test_seed_imports_configured_sets_and_swallows_failures() -> None:
    bot = _FakeBot(
        {"PackOne": [_pack_sticker("uid1", "😀"), _pack_sticker("uid2", "😂")]},
        failing=["BadPack"],
    )
    adapter = _make_adapter(extra={"sticker_sets": ["PackOne", "BadPack"]})
    adapter._bot = bot

    await adapter._seed_sticker_collection_from_config()

    assert bot.requested == ["PackOne", "BadPack"]
    entries = {e["file_unique_id"] for e in sticker_collection.list_stickers()}
    assert entries == {"uid1", "uid2"}


@pytest.mark.asyncio
async def test_seed_accepts_comma_separated_string() -> None:
    bot = _FakeBot({"PackOne": [], "PackTwo": []})
    adapter = _make_adapter(extra={"sticker_sets": "PackOne, PackTwo"})
    adapter._bot = bot

    await adapter._seed_sticker_collection_from_config()

    assert bot.requested == ["PackOne", "PackTwo"]


@pytest.mark.asyncio
async def test_seed_noop_when_unconfigured() -> None:
    bot = _FakeBot({})
    for extra in ({}, {"sticker_sets": []}, {"sticker_sets": None}, {"sticker_sets": 42}):
        adapter = _make_adapter(extra=extra)
        adapter._bot = bot
        await adapter._seed_sticker_collection_from_config()
    assert bot.requested == []


@pytest.mark.asyncio
async def test_post_connect_housekeeping_invokes_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    monkeypatch.setattr(adapter, "_setup_dm_topics", AsyncMock())
    seed = AsyncMock()
    monkeypatch.setattr(adapter, "_seed_sticker_collection_from_config", seed)

    await adapter._run_post_connect_housekeeping()

    seed.assert_awaited_once()


# ---------------------------------------------------------------------------
# Config flow: telegram.sticker_sets → PlatformConfig.extra


def test_apply_yaml_config_bridges_sticker_sets() -> None:
    extras = _apply_yaml_config({}, {"sticker_sets": ["HotCherry", "CatsPack"]})
    assert extras is not None
    assert extras["sticker_sets"] == ["HotCherry", "CatsPack"]


def test_apply_yaml_config_without_sticker_sets_returns_none() -> None:
    assert _apply_yaml_config({}, {}) is None


def test_sticker_sets_flow_from_config_yaml_to_platform_extra(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  enabled: true\n"
        "  token: test-token\n"
        "  sticker_sets:\n"
        "    - HotCherry\n"
        "    - CatsPack\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway.config import load_gateway_config

    config = load_gateway_config()

    telegram = config.platforms[Platform.TELEGRAM]
    assert telegram.extra["sticker_sets"] == ["HotCherry", "CatsPack"]
