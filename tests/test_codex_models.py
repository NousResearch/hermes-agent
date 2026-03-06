import json

from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, get_codex_model_ids


def test_get_codex_model_ids_prioritizes_default_and_cache(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text('model = "gpt-5.2-codex"\n')
    (codex_home / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "gpt-5.3-codex", "priority": 20, "supported_in_api": True},
                    {"slug": "gpt-5.1-codex", "priority": 5, "supported_in_api": True},
                    {"slug": "gpt-4o", "priority": 1, "supported_in_api": True},
                    {"slug": "gpt-5-hidden-codex", "priority": 2, "visibility": "hidden"},
                ]
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    models = get_codex_model_ids()

    assert models[0] == "gpt-5.2-codex"
    assert "gpt-5.1-codex" in models
    assert "gpt-5.3-codex" in models
    assert "gpt-4o" not in models
    assert "gpt-5-hidden-codex" not in models


def test_get_codex_model_ids_falls_back_to_curated_defaults(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    models = get_codex_model_ids()

    assert models[: len(DEFAULT_CODEX_MODELS)] == DEFAULT_CODEX_MODELS
