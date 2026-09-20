"""Regression tests: ``model.provider`` must survive config load as a string.

An unquoted YAML scalar (``provider: 2``) loads as ``int``, and every downstream
reader does ``(provider or "").strip()`` — a gateway turn dies before the agent
runs (#117345). The load-path chokepoint (``_normalize_root_model_keys``) is where
the value is canonicalized, so it must emit a string for every carrier: a plain
``model.provider``, the legacy root-level ``provider`` alias, and the nested
``{provider: <p>, model: <m>}`` flattening.
"""

import os
from unittest.mock import patch

import yaml

from hermes_cli.config import _normalize_root_model_keys, load_config


class TestNormalizeProviderIdIsString:
    """model.provider reaches readers as ``str`` regardless of YAML scalar type."""

    def test_int_model_provider_is_stringified(self):
        config = {"model": {"default": "m", "provider": 2}}
        out = _normalize_root_model_keys(dict(config))
        assert out["model"]["provider"] == "2"
        assert isinstance(out["model"]["provider"], str)

    def test_str_model_provider_is_not_perturbed(self):
        config = {"model": {"default": "m", "provider": "glm-flash"}}
        out = _normalize_root_model_keys(dict(config))
        assert out["model"]["provider"] == "glm-flash"

    def test_legacy_root_provider_alias_is_stringified(self):
        config = {"provider": 2, "base_url": "https://api.example.com/v1",
                  "model": {"default": "m"}}
        out = _normalize_root_model_keys(dict(config))
        assert out["model"]["provider"] == "2"
        assert isinstance(out["model"]["provider"], str)

    def test_nested_provider_flatten_is_stringified(self):
        config = {"model": {"default": {"provider": 2, "model": "m"}}}
        out = _normalize_root_model_keys(dict(config))
        provider = out["model"]["provider"]
        assert provider == "2"
        assert isinstance(provider, str)


class TestLoadConfigIntProviderEndToEnd:
    """E2E through the real config.yaml path — the gateway's exact entry."""

    def test_unquoted_yaml_provider_loads_as_string(self, tmp_path):
        # Unquoted 2 is what `hermes config set model.provider 2` writes and
        # what PyYAML loads as int (the #117345 write-side bug shape).
        (tmp_path / "config.yaml").write_text(
            "model:\n  default: deepseek-flash\n  provider: 2\n", encoding="utf-8"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
        provider = config["model"]["provider"]
        assert provider == "2"
        assert isinstance(provider, str)

    def test_provider_zero_and_negative_are_preserved_as_strings(self, tmp_path):
        # 0 is falsy: the normalize loop uses `root_val and ...`, and readers use
        # `(provider or "")` — a stringified "0" must not be dropped or blanked.
        (tmp_path / "config.yaml").write_text(
            "model:\n  default: m\n  provider: 0\n", encoding="utf-8"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
        assert config["model"]["provider"] == "0"

    def test_absent_provider_key_is_not_injected(self, tmp_path):
        # A provider-less model section must not gain an empty provider key:
        # injecting one would rewrite config.yaml on the next save.
        (tmp_path / "config.yaml").write_text(
            "model:\n  default: deepseek-flash\n", encoding="utf-8"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
        assert "provider" not in config["model"]

    def test_float_provider_becomes_its_literal_string(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "model:\n  default: m\n  provider: 2.0\n", encoding="utf-8"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
        assert config["model"]["provider"] == "2.0"
