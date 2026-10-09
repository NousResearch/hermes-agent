from hermes_cli import model_selection_picker as picker


def test_picker_model_ids_uses_endpoint_whitespace_fact(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.models_validate.provider_allows_model_whitespace",
        lambda provider, base_url=None: provider == "ollama",
    )
    assert picker.picker_model_ids("openai", ["good", "bad model"]) == ["good"]
    assert picker.picker_model_ids("ollama", ["good", "model with spaces"]) == [
        "good",
        "model with spaces",
    ]


def test_providerless_setup_preserves_custom_spaced_model_ids():
    assert picker.picker_model_ids("", ["custom model name"]) == ["custom model name"]


def test_provider_data_uses_cached_catalog_only_when_row_is_empty(monkeypatch):
    seen = []

    def cached(provider):
        seen.append(provider)
        return ["cached-b", "cached-a"]

    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", cached)
    monkeypatch.setattr(
        "hermes_cli.models_validate.provider_allows_model_whitespace",
        lambda *_a, **_k: False,
    )

    assert picker.picker_model_ids_for_provider_data(
        {"slug": "openai", "models": []},
        current_model="cached-a",
    ) == ["cached-a", "cached-b"]
    assert seen == ["openai"]

    seen.clear()
    assert picker.picker_model_ids_for_provider_data(
        {"slug": "openai", "models": ["curated"]},
    ) == ["curated"]
    assert seen == []


def test_project_picker_rows_is_non_mutating_and_projects_featured(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.models_validate.provider_allows_model_whitespace",
        lambda *_a, **_k: False,
    )
    original = [{
        "slug": "openai",
        "models": ["good", "bad model", "good"],
        "featured_models": ["bad model", "good"],
        "total_models": 3,
    }]

    projected = picker.project_picker_rows(original)

    assert projected[0]["models"] == ["good"]
    assert projected[0]["featured_models"] == ["good"]
    assert projected[0]["total_models"] == 1
    assert original[0]["models"] == ["good", "bad model", "good"]
