import os
from unittest.mock import patch
import hermes_yaml as yaml
from hermes_cli.config import load_config, save_config


class TestConfigIncludes:
    def _clear_caches(self):
        import hermes_cli.config as config_mod

        config_mod._LOAD_CONFIG_CACHE.clear()
        config_mod._RAW_CONFIG_CACHE.clear()

    def test_load_config_merges_include_before_main_config(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            (tmp_path / "included.yaml").write_text(
                yaml.safe_dump({
                    "model": {
                        "default": "example/included-model",
                        "base_url": "http://localhost:8000/v1",
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    },
                }),
                encoding="utf-8",
            )
            (tmp_path / "config.yaml").write_text(
                "include: included.yaml\nmodel:\n  max_tokens: 8192\n",
                encoding="utf-8",
            )
            self._clear_caches()

            config = load_config()

            assert "include" not in config
            assert config["model"]["default"] == "example/included-model"
            assert config["model"]["base_url"] == "http://localhost:8000/v1"
            assert config["model"]["max_tokens"] == 8192
            assert config["model"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_load_config_cache_invalidates_when_include_changes(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            include_path = tmp_path / "included.yaml"
            include_path.write_text("model:\n  default: old/model\n", encoding="utf-8")
            (tmp_path / "config.yaml").write_text("include: included.yaml\n", encoding="utf-8")
            self._clear_caches()

            assert load_config()["model"]["default"] == "old/model"

            include_path.write_text("model:\n  default: newer/model-with-longer-name\n", encoding="utf-8")
            assert load_config()["model"]["default"] == "newer/model-with-longer-name"

    def test_save_config_preserves_include_without_materializing_included_values(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            (tmp_path / "included.yaml").write_text(
                yaml.safe_dump({
                    "model": {
                        "default": "example/included-model",
                        "base_url": "http://localhost:8000/v1",
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    },
                }),
                encoding="utf-8",
            )
            (tmp_path / "config.yaml").write_text(
                "include: included.yaml\ndisplay:\n  language: fr\n",
                encoding="utf-8",
            )
            self._clear_caches()

            config = load_config()
            config["display"]["skin"] = "cyberpunk"
            save_config(config)

            raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
            assert raw["include"] == "included.yaml"
            assert "model" not in raw
            assert raw["display"]["language"] == "fr"
            assert raw["display"]["skin"] == "cyberpunk"
