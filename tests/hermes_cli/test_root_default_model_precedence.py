"""A root-level default_model is a fallback, not an override of legacy nested model IDs."""

from hermes_cli.config_model_section import _normalize_root_model_keys


def test_root_default_model_only_fills_missing_model_id():
    for nested_key in ("model", "name"):
        normalized = _normalize_root_model_keys({
            "default_model": "root-model",
            "model": {nested_key: "nested-model"},
        })
        assert normalized["model"]["default"] == "nested-model"
        assert "default_model" not in normalized
        assert nested_key not in normalized["model"]

    fallback = _normalize_root_model_keys({"default_model": "root-model"})
    assert fallback["model"]["default"] == "root-model"
    assert "default_model" not in fallback
