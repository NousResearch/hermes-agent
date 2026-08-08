"""Behavior tests for the Telegram sticker collection store.

Covers record/dedup/last_seen, description merge rules (vision-cache
backfill vs agent curation), eviction at capacity, resolve priority,
prompt formatting, refresh_from_sets with a mock bot, and the
first-turn collection note renderer.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from plugins.platforms.telegram import sticker_collection


@pytest.fixture(autouse=True)
def _no_vision_cache(monkeypatch: pytest.MonkeyPatch):
    """Default: the vision-description cache misses for every lookup.

    Individual tests re-patch ``get_cached_description`` when they want a
    hit. Keeping this autouse means no test can accidentally read the
    developer's real ~/.hermes/sticker_cache.json.
    """
    import gateway.sticker_cache as sticker_cache

    monkeypatch.setattr(
        sticker_cache, "get_cached_description", lambda file_unique_id: None
    )
    return sticker_cache


def _collection_file() -> Any:
    from hermes_cli.config import get_hermes_home

    return get_hermes_home() / "telegram_stickers.json"


def _read_raw() -> Dict[str, Any]:
    return json.loads(_collection_file().read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# record_sticker: dedup, last_seen refresh, new-vs-known return


def test_record_new_sticker_returns_true_and_persists() -> None:
    assert sticker_collection.record_sticker(
        "uid1", "fid1", emoji="😀", set_name="MyPack", kind="static",
        description="a cat waving",
    ) is True

    raw = _read_raw()
    assert raw["version"] == 1
    entry = raw["stickers"]["uid1"]
    assert entry["file_id"] == "fid1"
    assert entry["emoji"] == "😀"
    assert entry["set_name"] == "MyPack"
    assert entry["kind"] == "static"
    assert entry["description"] == "a cat waving"
    assert entry["first_seen"] == entry["last_seen"]


def test_record_known_sticker_returns_false_and_refreshes_last_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([1000.0, 2000.0])
    monkeypatch.setattr(sticker_collection, "_now", lambda: next(times))

    assert sticker_collection.record_sticker("uid1", "fid1") is True
    assert sticker_collection.record_sticker("uid1", "fid1-new") is False

    entries = sticker_collection.list_stickers()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["file_unique_id"] == "uid1"
    assert entry["file_id"] == "fid1-new"  # file_id refreshes (per-bot validity)
    assert entry["first_seen"] == 1000.0
    assert entry["last_seen"] == 2000.0


def test_record_without_identity_is_a_noop() -> None:
    assert sticker_collection.record_sticker("", "fid1") is False
    assert sticker_collection.record_sticker("uid1", "") is False
    assert sticker_collection.list_stickers() == []


# ---------------------------------------------------------------------------
# Description merge rules


def test_record_backfills_description_from_vision_cache_when_empty(
    _no_vision_cache, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _no_vision_cache,
        "get_cached_description",
        lambda fuid: {"description": "a cat waving its paw", "emoji": "😀"},
    )
    assert sticker_collection.record_sticker("uid1", "fid1", emoji="😀") is True
    entry = sticker_collection.resolve("uid1")
    assert entry["description"] == "a cat waving its paw"


def test_record_never_clobbers_existing_description(
    _no_vision_cache, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sticker_collection.record_sticker("uid1", "fid1", description="agent note")

    # Caller-supplied non-empty description does not overwrite either.
    assert sticker_collection.record_sticker("uid1", "fid1", description="new attempt") is False
    # ... and neither does a cache backfill.
    monkeypatch.setattr(
        _no_vision_cache,
        "get_cached_description",
        lambda fuid: {"description": "cache attempt"},
    )
    assert sticker_collection.record_sticker("uid1", "fid1") is False

    assert sticker_collection.resolve("uid1")["description"] == "agent note"


def test_update_description_overwrites_and_clears() -> None:
    sticker_collection.record_sticker("uid1", "fid1", description="first")

    assert sticker_collection.update_description("uid1", "second") is True
    assert sticker_collection.resolve("uid1")["description"] == "second"

    # "" clears the annotation.
    assert sticker_collection.update_description("uid1", "") is True
    assert sticker_collection.resolve("uid1")["description"] == ""

    # Unknown id -> False, no crash.
    assert sticker_collection.update_description("nope", "x") is False


def test_update_description_beats_later_cache_backfill(
    _no_vision_cache, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sticker_collection.record_sticker("uid1", "fid1")
    assert sticker_collection.update_description("uid1", "use when user is sarcastic") is True

    monkeypatch.setattr(
        _no_vision_cache,
        "get_cached_description",
        lambda fuid: {"description": "cache attempt"},
    )
    sticker_collection.record_sticker("uid1", "fid1")
    assert sticker_collection.resolve("uid1")["description"] == "use when user is sarcastic"


# ---------------------------------------------------------------------------
# Capacity / eviction


def test_eviction_at_capacity_removes_oldest_by_last_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sticker_collection, "MAX_STICKERS", 3)
    clock = {"t": 0.0}

    def tick() -> float:
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(sticker_collection, "_now", tick)

    for i in range(4):
        sticker_collection.record_sticker(f"uid{i}", f"fid{i}")

    entries = sticker_collection.list_stickers()
    assert len(entries) == 3
    surviving = {e["file_unique_id"] for e in entries}
    assert surviving == {"uid1", "uid2", "uid3"}  # uid0 (oldest) evicted
    # Newest first.
    assert [e["file_unique_id"] for e in entries] == ["uid3", "uid2", "uid1"]


def test_re_record_refreshes_last_seen_and_avoids_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sticker_collection, "MAX_STICKERS", 2)
    clock = {"t": 0.0}
    monkeypatch.setattr(
        sticker_collection, "_now", lambda: clock.__setitem__("t", clock["t"] + 1.0) or clock["t"]
    )

    sticker_collection.record_sticker("old", "fid-old")
    sticker_collection.record_sticker("keep", "fid-keep")
    sticker_collection.record_sticker("old", "fid-old")  # refresh: now newest
    sticker_collection.record_sticker("new", "fid-new")  # evicts "keep"

    surviving = {e["file_unique_id"] for e in sticker_collection.list_stickers()}
    assert surviving == {"old", "new"}


# ---------------------------------------------------------------------------
# resolve priority


@pytest.fixture()
def _packed() -> None:
    """Two same-emoji stickers in different packs + one distinct sticker."""
    sticker_collection.record_sticker(
        "uid-a", "fid-a", emoji="😀", set_name="PackOne", kind="static",
        description="first cat",
    )
    sticker_collection.record_sticker(
        "uid-b", "fid-b", emoji="😀", set_name="PackTwo", kind="animated",
        description="second cat",
    )
    sticker_collection.record_sticker(
        "uid-c", "CAAC-x9", emoji="🚀", set_name="PackTwo", kind="video",
    )


def test_resolve_file_id_passthrough_wins(_packed) -> None:
    entry = sticker_collection.resolve("CAAC-x9")
    assert entry["file_unique_id"] == "uid-c"
    assert entry["kind"] == "video"


def test_resolve_file_id_beats_file_unique_id_collision(_packed) -> None:
    # A query that is both some entry's file_id and another's file_unique_id
    # resolves as the file_id (priority order).
    sticker_collection.record_sticker("fid-a", "fid-other", emoji="🐶")
    entry = sticker_collection.resolve("fid-a")
    assert entry["file_unique_id"] == "uid-a"


def test_resolve_file_unique_id_exact(_packed) -> None:
    assert sticker_collection.resolve("uid-a")["file_id"] == "fid-a"


def test_resolve_set_name_emoji_exact(_packed) -> None:
    entry = sticker_collection.resolve("PackOne:😀")
    assert entry["file_unique_id"] == "uid-a"


def test_resolve_bare_emoji_picks_most_recent(_packed) -> None:
    entry = sticker_collection.resolve("😀")
    assert entry["file_unique_id"] == "uid-b"  # recorded after uid-a


def test_resolve_bare_emoji_with_set_name_disambiguation(_packed) -> None:
    entry = sticker_collection.resolve("😀", set_name="PackOne")
    assert entry["file_unique_id"] == "uid-a"


def test_resolve_unknown_returns_none(_packed) -> None:
    assert sticker_collection.resolve("🦄") is None
    assert sticker_collection.resolve("NoSuchPack:😀") is None
    assert sticker_collection.resolve("") is None


# ---------------------------------------------------------------------------
# list / remove


def test_list_stickers_filter_limit_and_shape() -> None:
    sticker_collection.record_sticker("uid1", "fid1", emoji="😀", set_name="A")
    sticker_collection.record_sticker("uid2", "fid2", emoji="😂", set_name="B")
    sticker_collection.record_sticker("uid3", "fid3", emoji="🚀", set_name="B")

    all_entries = sticker_collection.list_stickers()
    assert [e["file_unique_id"] for e in all_entries] == ["uid3", "uid2", "uid1"]

    only_b = sticker_collection.list_stickers(set_name="B")
    assert {e["file_unique_id"] for e in only_b} == {"uid2", "uid3"}

    capped = sticker_collection.list_stickers(limit=2)
    assert len(capped) == 2

    entry = all_entries[0]
    for key in ("file_unique_id", "file_id", "emoji", "set_name", "kind",
                "description", "first_seen", "last_seen"):
        assert key in entry


def test_remove_sticker() -> None:
    sticker_collection.record_sticker("uid1", "fid1")
    assert sticker_collection.remove_sticker("uid1") is True
    assert sticker_collection.remove_sticker("uid1") is False
    assert sticker_collection.list_stickers() == []
    # Deletion is persisted, not just in-memory.
    assert "uid1" not in _read_raw()["stickers"]


# ---------------------------------------------------------------------------
# Corrupt / dirty data tolerance


def test_corrupt_json_reads_as_empty_collection() -> None:
    _collection_file().write_text("not json {{{", encoding="utf-8")
    assert sticker_collection.list_stickers() == []
    assert sticker_collection.resolve("😀") is None
    assert sticker_collection.format_for_prompt() == ""
    # ... and the store still works afterwards.
    assert sticker_collection.record_sticker("uid1", "fid1") is True
    assert sticker_collection.resolve("uid1") is not None


def test_dirty_entries_skipped_on_read_and_self_heal_on_record() -> None:
    _collection_file().write_text(json.dumps({
        "version": 1,
        "stickers": {
            "dirty-no-file-id": {"emoji": "😀"},
            "dirty-wrong-types": {"file_id": "fid", "emoji": 3},
            "good": {
                "file_id": "fid-good", "emoji": "😀", "set_name": "Pack",
                "kind": "static", "description": "",
                "first_seen": 1.0, "last_seen": 1.0,
            },
        },
    }), encoding="utf-8")

    # Read paths skip dirty entries.
    assert [e["file_unique_id"] for e in sticker_collection.list_stickers()] == ["good"]
    assert sticker_collection.update_description("dirty-no-file-id", "x") is False

    # record_sticker drops them from the persisted file (self-heal).
    sticker_collection.record_sticker("uid-new", "fid-new")
    raw = _read_raw()
    assert set(raw["stickers"]) == {"good", "uid-new"}


# ---------------------------------------------------------------------------
# format_for_prompt


def test_format_for_prompt_empty_collection() -> None:
    assert sticker_collection.format_for_prompt() == ""


def test_format_for_prompt_lines_sorted_and_truncated() -> None:
    sticker_collection.record_sticker(
        "uid1", "fid1", emoji="😀", set_name="MyPack", kind="static",
        description="a cat waving",
    )
    sticker_collection.record_sticker(
        "uid2", "fid2", emoji="🚀", set_name="MyPack", kind="video",
    )

    listing = sticker_collection.format_for_prompt()
    lines = listing.split("\n")
    # Newest (uid2) first; empty description renders without quotes.
    assert lines[0] == '- 🚀 (set: MyPack, kind: video)'
    assert lines[1] == '- 😀 "a cat waving" (set: MyPack, kind: static)'

    capped = sticker_collection.format_for_prompt(limit=1)
    assert capped.split("\n") == [lines[0]]


# ---------------------------------------------------------------------------
# refresh_from_sets


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


def _sticker(uid: str, emoji: str, *, animated: bool = False, video: bool = False,
             set_name: str = "") -> Any:
    return SimpleNamespace(
        file_id=f"fid-{uid}", file_unique_id=uid, emoji=emoji,
        is_animated=animated, is_video=video, set_name=set_name,
    )


@pytest.mark.asyncio
async def test_refresh_from_sets_records_and_summarizes() -> None:
    bot = _FakeBot(
        {
            "PackOne": [_sticker("uid1", "😀"), _sticker("uid2", "😂", animated=True)],
            "PackTwo": [_sticker("uid3", "🚀", video=True)],
        },
        failing=["BadPack"],
    )

    summary = await sticker_collection.refresh_from_sets(
        bot, ["PackOne", "BadPack", "PackTwo", "  "]
    )

    assert summary == {"sets": 2, "sets_failed": 1, "stickers": 3, "new": 3}
    assert bot.requested == ["PackOne", "BadPack", "PackTwo"]  # blanks skipped

    entries = {e["file_unique_id"]: e for e in sticker_collection.list_stickers()}
    assert set(entries) == {"uid1", "uid2", "uid3"}
    assert entries["uid1"]["kind"] == "static"
    assert entries["uid2"]["kind"] == "animated"
    assert entries["uid3"]["kind"] == "video"
    # Set name comes from the pack being imported when the sticker lacks one.
    assert entries["uid1"]["set_name"] == "PackOne"


@pytest.mark.asyncio
async def test_refresh_from_sets_is_idempotent_and_backfills(
    _no_vision_cache, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _no_vision_cache,
        "get_cached_description",
        lambda fuid: {"description": "previously visioned"} if fuid == "uid1" else None,
    )
    bot = _FakeBot({"Pack": [_sticker("uid1", "😀"), _sticker("uid2", "😂")]})

    first = await sticker_collection.refresh_from_sets(bot, ["Pack"])
    assert first["new"] == 2
    assert sticker_collection.resolve("uid1")["description"] == "previously visioned"

    second = await sticker_collection.refresh_from_sets(bot, ["Pack"])
    assert second == {"sets": 1, "sets_failed": 0, "stickers": 2, "new": 0}


# ---------------------------------------------------------------------------
# build_sticker_collection_note


def test_build_note_empty_collection() -> None:
    assert sticker_collection.build_sticker_collection_note() == ""


def test_build_note_renders_header_listing_and_guidance() -> None:
    sticker_collection.record_sticker(
        "uid1", "fid1", emoji="😀", set_name="MyPack", kind="static",
        description="a cat waving",
    )

    note = sticker_collection.build_sticker_collection_note()
    assert note.startswith("## Your Telegram Sticker Collection\n")
    assert '- 😀 "a cat waving" (set: MyPack, kind: static)' in note
    assert "tg_send_sticker" in note
    assert "tg_manage_stickers" in note
    assert "PNG" in note  # "never draw sticker PNGs" guidance
