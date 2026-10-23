"""A damaged optional cache must behave like a cold sticker cache."""
import json

import pytest

from gateway.sticker_cache import cache_sticker_description, get_cached_description


@pytest.mark.parametrize("damaged", [b"[]", b"null", b"123", b'"cache"', b"\xff"])
def test_invalid_cache_recovers_on_next_description(tmp_path, monkeypatch, damaged):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "sticker_cache.json"
    path.write_bytes(damaged)
    assert get_cached_description("sticker1") is None
    cache_sticker_description("sticker1", "A waving cat", emoji="🐈")
    assert get_cached_description("sticker1")["description"] == "A waving cat"
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


@pytest.mark.parametrize("entry", ["description", 1, ["cat"], {"emoji": "🐈"}, {"description": None}, {"description": 7}])
def test_invalid_entry_is_a_miss_without_discarding_valid_neighbors(tmp_path, monkeypatch, entry):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    good = {"description": "A happy dog", "emoji": "🐕", "future_field": "keep"}
    (tmp_path / "sticker_cache.json").write_text(json.dumps({"broken": entry, "good": good}), encoding="utf-8")
    assert get_cached_description("broken") is None
    assert get_cached_description("good") == good
    cache_sticker_description("broken", "A repaired description")
    assert get_cached_description("good") == good
    assert get_cached_description("broken")["description"] == "A repaired description"
