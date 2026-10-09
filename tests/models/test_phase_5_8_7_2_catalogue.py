"""Phase 5.8.7.2: DeepInfra provider source and single catalogue ownership."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from models import catalog_deepinfra as catalog
from providers import get_provider_profile


@pytest.fixture(autouse=True)
def _isolated_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    catalog.reset_catalog_cache()
    yield
    catalog.reset_catalog_cache()


def test_provider_fetch_uses_guarded_transport(monkeypatch):
    profile = get_provider_profile("deepinfra")
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({"data": []}).encode()

    def safe_open(req, *, timeout):
        seen.update(url=req.full_url, key=req.get_header("Authorization"), timeout=timeout)
        return Response()

    monkeypatch.setitem(profile.fetch_catalog.__func__.__globals__, "open_credentialed_url", safe_open)
    assert profile.fetch_catalog(api_key="private", timeout=5.0) == []
    assert seen["key"] == "Bearer private"
    assert seen["url"].endswith("/models?filter=true&sort_by=hermes")
    assert seen["timeout"] == 5.0


def test_one_fetch_serves_chat_image_video_and_audio():
    rows = [
        {"id": "vendor/chat", "metadata": {"tags": ["chat"]}},
        {"id": "vendor/image", "metadata": {"tags": ["image-gen"]}},
        {"id": "vendor/video", "metadata": {"tags": ["video-gen"]}},
        {"id": "vendor/tts", "metadata": {"tags": ["tts"]}},
        {"id": "vendor/stt", "metadata": {"tags": ["stt"]}},
        {"id": "vendor/embed", "metadata": {"tags": ["embed"]}},
        {"id": "Qwen/Qwen3", "metadata": {"tags": ["reasoning", "vision"]}},
        {"id": "openai/whisper-large", "metadata": {"tags": ["reasoning"]}},
        {"id": "whisper-chat", "metadata": {"tags": ["chat"]}},
        {"id": "stub", "metadata": None},
    ]
    requests = []

    def source(**kwargs):
        requests.append(kwargs)
        return rows

    base_url = "https://example.invalid/v1/openai"
    for surface in ("chat", "image-gen", "video-gen", "tts", "stt", "embed"):
        result = catalog.models_by_tag(
            surface, api_key="private", base_url=base_url, fetch_catalog=source,
        )
        assert result is not None
        ids = {row["id"] for row in result}
        assert "stub" not in ids
        if surface == "chat":
            assert {"vendor/chat", "Qwen/Qwen3", "whisper-chat"} <= ids
            assert "openai/whisper-large" not in ids
        else:
            assert all(surface in row["metadata"]["tags"] for row in result)
    assert len(requests) == 1


def test_missing_empty_negative_cache_and_forced_recovery(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(catalog.time, "monotonic", lambda: clock[0])
    hits = []

    def failed(**kwargs):
        hits.append(1)
        return None

    url = "https://negative.example/v1/openai"
    args = dict(api_key="secret", base_url=url)
    assert catalog.catalog(**args, fetch_catalog=failed) is None
    assert catalog.catalog(**args, fetch_catalog=failed) is None
    assert catalog.catalog(**args, cached_only=True, fetch_catalog=failed) is None
    assert len(hits) == 1
    assert catalog.catalog(**args, force_refresh=True, fetch_catalog=lambda **kw: []) == []
    assert catalog.is_catalog_cached(**args)
    assert catalog.catalog(**args, fetch_catalog=failed) == []


def test_catalogue_isolated_by_endpoint_and_credential():
    hits = []

    def source(**kwargs):
        hits.append(kwargs)
        return [{"id": kwargs["api_key"], "metadata": {"tags": ["chat"]}}]

    for token in ("A", "B"):
        result = catalog.models_by_tag("chat", api_key=token, fetch_catalog=source)
        assert result[0]["id"] == token
    result = catalog.models_by_tag(
        "chat", api_key="A", base_url="https://custom.invalid/v1", fetch_catalog=source
    )
    assert result[0]["id"] == "A"
    assert len(hits) == 3


def test_pricing_cached_only_never_fetches(monkeypatch):
    from application_model_pricing import get_pricing_for_provider

    hits = []
    payload = [
        {"id": "vendor/chat", "metadata": {"tags": ["chat"], "pricing": {
            "input_tokens": 0.1, "output_tokens": 0.3, "cache_read_tokens": 0.02}}},
        {"id": "vendor/image", "metadata": {"tags": ["image-gen"], "pricing": {
            "per_image_unit": 0.5}}},
    ]

    def source(**kwargs):
        hits.append(kwargs)
        return payload

    monkeypatch.setattr(get_provider_profile("deepinfra"), "fetch_catalog", source)
    assert get_pricing_for_provider("deepinfra", cached_only=True) == {}
    result = get_pricing_for_provider("deepinfra")
    assert list(result) == ["vendor/chat"]
    assert float(result["vendor/chat"]["prompt"]) == pytest.approx(0.1 / 1_000_000)
    assert "input_cache_read" in result["vendor/chat"]
    assert get_pricing_for_provider("deepinfra", cached_only=True) == result
    assert len(hits) == 1


def test_profile_vision_rejects_image_generation_surface(monkeypatch):
    import agent.secret_scope as secret_scope

    monkeypatch.setattr(secret_scope, "get_secret", lambda *args, **kw: "test-key")
    monkeypatch.setattr(get_provider_profile("deepinfra"), "fetch_catalog", lambda **kw: [
        {"id": "image-only", "metadata": {"tags": ["vision", "image-gen"]}},
        {"id": "vision-chat", "metadata": {"tags": ["vision", "chat"]}},
    ])
    assert get_provider_profile("deepinfra").default_vision_model() == "vision-chat"


def test_plugin_and_pricing_do_not_import_old_cli_catalogue():
    root = Path(__file__).resolve().parents[2]
    for relpath in (
        "plugins/model-providers/anthropic/__init__.py",
        "plugins/model-providers/deepinfra/__init__.py",
        "plugins/image_gen/deepinfra/__init__.py",
        "plugins/video_gen/deepinfra/__init__.py",
        "hermes_cli/models_pricing.py",
    ):
        text = (root / relpath).read_text(encoding="utf-8")
        assert "from hermes_cli.models import _fetch_deepinfra_models_by_tag" not in text
        assert "from hermes_cli.models import _ANTHROPIC_MODELS_MAX_PAGES" not in text
    legacy = (root / "hermes_cli/models.py").read_text(encoding="utf-8")
    assert "def _fetch_deepinfra_catalog(" not in legacy
    assert "def _anthropic_models_url(" not in legacy
