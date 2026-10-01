"""DeepInfra: one canonical, profile-isolated catalogue across all media surfaces."""
from __future__ import annotations

import pytest

from models.catalog_deepinfra import (
    _cache_key,
    catalog,
    models_by_tag,
    reset_catalog_cache,
)


@pytest.fixture(autouse=True)
def clean_catalogue():
    reset_catalog_cache()
    yield
    reset_catalog_cache()


def test_one_fetch_supplies_all_surfaces_and_preserves_legacy_chat_fallback():
    calls = []
    rows = [
        {"id": "chat-tagged", "metadata": {"tags": ["chat", "vision"]}},
        {"id": "image-tagged", "metadata": {"tags": ["image-gen"]}},
        {"id": "video-tagged", "metadata": {"tags": ["video-gen"]}},
        {"id": "unlabelled-chat", "metadata": {"tags": ["vision"]}},
        {"id": "whisper-audio", "metadata": {"tags": ["vision"]}},
        {"id": "stub", "metadata": None},
    ]

    def fetch(**kwargs):
        calls.append(kwargs)
        return rows

    kw = {"base_url": "https://one.example/v1", "api_key": "one", "fetch_catalog": fetch}
    assert [r["id"] for r in models_by_tag("chat", **kw)] == [
        "chat-tagged", "unlabelled-chat"
    ]
    assert [r["id"] for r in models_by_tag("image-gen", **kw)] == ["image-tagged"]
    assert [r["id"] for r in models_by_tag("video-gen", **kw)] == ["video-tagged"]
    assert [r["id"] for r in models_by_tag("stt", **kw)] == []
    assert len(calls) == 1


def test_endpoint_and_key_scope_are_isolated_without_retaining_raw_keys():
    calls = []

    def fetch(**kwargs):
        calls.append((kwargs["base_url"], kwargs["api_key"]))
        return [{"id": kwargs["api_key"], "metadata": {"tags": ["chat"]}}]

    def select(url, key):
        return models_by_tag(
            "chat", base_url=url, api_key=key, fetch_catalog=fetch
        )[0]["id"]

    assert select("https://one.example", "credential-A") == "credential-A"
    assert select("https://one.example", "credential-B") == "credential-B"
    assert select("https://two.example", "credential-A") == "credential-A"
    assert select("https://one.example", "credential-A") == "credential-A"
    assert len(calls) == 3
    assert "credential-A" not in _cache_key("https://one.example", "credential-A")


def test_negative_ttl_and_force_refresh(monkeypatch):
    import models.catalog_deepinfra as source

    now = [100.0]
    monkeypatch.setattr(source.time, "monotonic", lambda: now[0])
    calls = []

    def fetch(**_kwargs):
        calls.append(1)
        return None if len(calls) == 1 else []

    kwargs = {"base_url": "https://offline.example", "api_key": "key", "fetch_catalog": fetch}
    assert catalog(**kwargs) is None
    assert catalog(**kwargs) is None
    assert len(calls) == 1
    assert catalog(force_refresh=True, **kwargs) == []
    assert len(calls) == 2
    assert catalog(**kwargs) == []
    assert len(calls) == 2


def test_cache_only_never_fetches_and_empty_result_is_not_failure():
    calls = []

    def fetch(**_kwargs):
        calls.append(1)
        return []

    kwargs = {"base_url": "https://quiet.example", "fetch_catalog": fetch}
    assert catalog(cached_only=True, **kwargs) is None
    assert calls == []
    assert catalog(**kwargs) == []
    assert catalog(cached_only=True, **kwargs) == []
    assert len(calls) == 1


def test_invalid_payload_cannot_poison_authoritative_cache():
    calls = []

    def fetch(**_kwargs):
        calls.append(1)
        return ["invalid"] if len(calls) == 1 else []

    kwargs = {"base_url": "https://invalid.example", "fetch_catalog": fetch}
    assert catalog(**kwargs) is None
    assert catalog(cached_only=True, **kwargs) is None
    assert catalog(force_refresh=True, **kwargs) == []
