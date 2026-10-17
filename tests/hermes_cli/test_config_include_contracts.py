import os
from unittest.mock import patch

from hermes_cli import config, config_includes


def _clear_caches():
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    config_includes.INCLUDE_SIGNATURES.clear()


def test_multiple_include_forms_merge_in_declaration_order(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        (tmp_path / "first.yaml").write_text("model:\n  default: first/model\n  max_tokens: 100\n")
        (tmp_path / "second.yaml").write_text("model:\n  default: second/model\n  temperature: 0.2\n")
        (tmp_path / "config.yaml").write_text(
            "include:\n  - first.yaml\nincludes: second.yaml\nmodel:\n  max_tokens: 200\n"
        )
        _clear_caches()

        loaded = config.load_config()

        assert loaded["model"]["default"] == "second/model"
        assert loaded["model"]["max_tokens"] == 200
        assert loaded["model"]["temperature"] == 0.2


def test_missing_include_is_reloaded_when_dependency_is_created(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        (tmp_path / "config.yaml").write_text("include: later.yaml\n")
        _clear_caches()

        assert config.load_config()["model"] != "created/model"
        (tmp_path / "later.yaml").write_text("model:\n  default: created/model\n")

        assert config.load_config()["model"]["default"] == "created/model"


def test_include_path_environment_change_invalidates_dependency_cache(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "HERMES_INCLUDE_FILE": "one.yaml"}):
        (tmp_path / "one.yaml").write_text("model:\n  default: one/model\n")
        (tmp_path / "two.yaml").write_text("model:\n  default: two/model\n")
        (tmp_path / "config.yaml").write_text("include: ${HERMES_INCLUDE_FILE}\n")
        _clear_caches()

        assert config.load_config()["model"]["default"] == "one/model"
        os.environ["HERMES_INCLUDE_FILE"] = "two.yaml"

        assert config.load_config()["model"]["default"] == "two/model"


def test_invalid_include_does_not_replace_existing_main_config(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        root = tmp_path / "config.yaml"
        included = tmp_path / "included.yaml"
        root.write_text("include: included.yaml\ndisplay:\n  language: fr\n")
        included.write_text("model:\n  default: valid/model\n")
        _clear_caches()
        loaded = config.load_config()
        original = root.read_text()
        included.write_text("model: [broken\n")

        try:
            config.save_config(loaded)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid include must refuse save")

        assert root.read_text() == original


def test_explicit_equal_value_stays_pinned_after_include_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = tmp_path / "config.yaml"
    included = tmp_path / "included.yaml"
    root.write_text("include: included.yaml\nmodel:\n  default: pinned/model\n", encoding="utf-8")
    included.write_text("model:\n  default: pinned/model\n", encoding="utf-8")
    _clear_caches()

    config.save_config(config.load_config())
    included.write_text("model:\n  default: changed/model\n", encoding="utf-8")

    assert config.load_config()["model"]["default"] == "pinned/model"
