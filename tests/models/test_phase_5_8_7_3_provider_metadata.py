"""Phase 5.8.7.3: Nous ownership and existing canonical reasoning metadata."""
from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from models import catalog_nous_recommendations as cache
from providers import get_provider_profile


def test_one_nous_cache_owner_is_scoped_by_portal_and_does_not_renew_stale_data(
    monkeypatch, tmp_path
):
    path = tmp_path / "nous_recommended_cache.json"
    monkeypatch.setattr(cache, "_disk_path", lambda: path)
    monkeypatch.setattr(cache, "hermes_home_key", lambda: str(tmp_path))
    cache.reset_cache()
    calls = []

    def source(*, base_url, timeout):
        calls.append(base_url)
        return {"paidRecommendedModels": [{"modelName": base_url}]}

    try:
        a = cache.fetch_recommended_models("https://one.example", fetch_source=source)
        assert cache.fetch_recommended_models(
            "https://one.example", fetch_source=source
        ) == a
        b = cache.fetch_recommended_models("https://two.example", fetch_source=source)
        assert a != b
        assert calls == ["https://one.example", "https://two.example"]
        disk = json.loads(path.read_text(encoding="utf-8"))
        first_ts = disk["https://one.example"]["ts"]
        assert cache.fetch_recommended_models(
            "https://one.example", force_refresh=True, fetch_source=lambda **_: None
        ) == a
        assert json.loads(path.read_text(encoding="utf-8"))["https://one.example"]["ts"] == first_ts
    finally:
        cache.reset_cache()


def test_nous_plugin_owns_public_http_and_handles_gzip(monkeypatch):
    from hermes_cli import urllib_security

    profile = get_provider_profile("nous")
    assert profile is not None
    seen = {}

    class Response:
        headers = {"Content-Encoding": "gzip"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return gzip.compress(json.dumps({"freeRecommendedModels": []}).encode("utf-8"))

    def open_request(request, *, timeout):
        seen["url"] = request.full_url
        seen["token"] = request.get_header("Authorization")
        seen["encoding"] = request.get_header("Accept-encoding")
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib_security, "open_credentialed_url", open_request)
    assert profile.fetch_recommended_models(
        base_url="https://portal.example", timeout=1.5
    ) == {"freeRecommendedModels": []}
    assert seen == {
        "url": "https://portal.example/api/nous/recommended-models",
        "token": None,
        "encoding": "gzip",
        "timeout": 1.5,
    }


def test_nous_recommendations_respect_tier_and_media(monkeypatch):
    import application_nous_recommendations as app
    from hermes_cli import models as old_app_tier

    payload = {
        "paidRecommendedCompactionModel": {"modelName": "paid/text"},
        "freeRecommendedCompactionModel": {"modelName": "free/text"},
        "paidRecommendedVisionModel": {"modelName": "paid/vision"},
        "freeRecommendedVisionModel": {"modelName": "free/vision"},
    }
    monkeypatch.setattr(app, "fetch_recommended_models", lambda **_kw: payload)
    profile = get_provider_profile("nous")
    assert profile is not None
    monkeypatch.setattr(old_app_tier, "check_nous_free_tier", lambda: False)
    assert profile.resolve_aux_model() == "paid/text"
    assert profile.default_vision_model() == "paid/vision"
    monkeypatch.setattr(old_app_tier, "check_nous_free_tier", lambda: True)
    assert profile.resolve_aux_model() == "free/text"
    assert profile.default_vision_model() == "free/vision"


def test_copilot_reasoning_is_sourced_from_canonical_metadata(monkeypatch):
    import models.metadata.github as metadata

    calls = []
    monkeypatch.setattr(
        metadata, "github_model_reasoning_efforts",
        lambda model: calls.append(model) or ("low", "medium", "high"),
    )
    monkeypatch.setattr(
        metadata, "clamp_github_reasoning_effort",
        lambda effort, supported: "high",
    )
    profile = get_provider_profile("copilot")
    assert profile is not None
    assert profile.build_api_kwargs_extras(
        model="gpt-5.4", supports_reasoning=True, reasoning_config={"effort": "ultra"}
    ) == ({"reasoning": {"effort": "high"}}, {})
    assert calls == ["gpt-5.4"]


def test_openrouter_reasoning_metadata_unknown_and_mandatory(monkeypatch):
    import models.metadata.reasoning as metadata

    profile = get_provider_profile("openrouter")
    assert profile is not None
    monkeypatch.setattr(
        metadata, "openrouter_model_reasoning_capabilities",
        lambda _model: SimpleNamespace(
            supported=True, mandatory=True, supported_efforts=("low", "medium", "high")
        ),
    )
    assert profile._clamp_reasoning_to_catalog(
        {"enabled": False, "effort": "none"}, "provider/mandatory"
    ) is None
    monkeypatch.setattr(
        metadata, "openrouter_model_reasoning_capabilities", lambda _model: None
    )
    fallback = {"effort": "ultra"}
    assert profile._clamp_reasoning_to_catalog(fallback, "provider/unknown") == fallback


def test_plugin_dependency_boundary_has_no_cli_model_semantics():
    root = Path(__file__).resolve().parents[2]
    for relpath in (
        "plugins/model-providers/nous/__init__.py",
        "plugins/model-providers/copilot/__init__.py",
        "plugins/model-providers/openrouter/__init__.py",
    ):
        tree = ast.parse((root / relpath).read_text(encoding="utf-8"))
        modules = {
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any(name.startswith("hermes_cli.models") for name in modules)
    tree = ast.parse(
        (root / "plugins/model-providers/actual/__init__.py").read_text(encoding="utf-8")
    )
    imported = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "hermes_cli.auth"
    ]
    assert all(
        {item.name for item in node.names} == {"resolve_api_key_provider_credentials"}
        for node in imported
    )
