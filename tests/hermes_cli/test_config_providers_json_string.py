"""Regression for #88451: a JSON-string-typed ``providers`` must be decoded.

``providers`` is an open-dict top-level key, so config validation accepts any
shape under it. A serialization round-trip (dashboard/config write-back through
``json.dumps``) can persist the whole section as a single JSON-string scalar:

    providers: '{"custom": {"models": {"M": {"supports_vision": true}}}}'

Every consumer guards with ``isinstance(providers, dict)`` and silently
degrades to "no custom providers configured" — per-model ``supports_vision``
overrides and custom-provider identity lookups never resolve. ``load_config``
now normalizes the string into a dict at the shared normalization chokepoint,
so downstream readers see the shape they expect.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

from agent.image_routing import _supports_vision_override
from hermes_cli.config import _normalize_providers_string, load_config


class TestNormalizeProvidersString:
    def test_decodes_json_string_to_dict(self):
        raw = '{"custom": {"base_url": "http://x", "models": {"M": {"supports_vision": true}}}}'
        out = _normalize_providers_string({"providers": raw})
        assert out["providers"] == {
            "custom": {"base_url": "http://x", "models": {"M": {"supports_vision": True}}}
        }

    def test_dict_passthrough_untouched(self):
        cfg = {"providers": {"custom": {"models": {}}}}
        out = _normalize_providers_string(cfg)
        assert out["providers"] is cfg["providers"]

    def test_absent_providers_is_noop(self):
        cfg = {"model": {"default": "x"}}
        assert _normalize_providers_string(cfg) is cfg

    def test_json_string_decoding_to_non_dict_falls_back_to_empty(self, caplog):
        # A JSON array is valid JSON but the wrong shape — must not survive.
        out = _normalize_providers_string({"providers": "[1, 2, 3]"})
        assert out["providers"] == {}

    def test_malformed_json_string_falls_back_to_empty(self):
        out = _normalize_providers_string({"providers": "{not json"})
        assert out["providers"] == {}

    def test_original_config_not_mutated(self):
        cfg = {"providers": "{}"}
        _normalize_providers_string(cfg)
        assert cfg["providers"] == "{}"  # caller's dict untouched


class TestLoadConfigDecodesProvidersString:
    def test_load_config_decodes_and_vision_override_resolves(self, tmp_path):
        """The reporter's end-to-end repro, driven through real ``load_config``."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            providers = {
                "custom": {
                    "base_url": "http://127.0.0.1:16520/v1",
                    "api_key": "vllm-local",
                    "models": {
                        "Qwen3.8-27B": {
                            "context_length": 126976,
                            "supports_vision": True,
                            "capabilities": ["text", "image", "tool_use"],
                        }
                    },
                }
            }
            # Persisted as a single-quoted YAML scalar holding a JSON string.
            (tmp_path / "config.yaml").write_text(
                "providers: " + json.dumps(json.dumps(providers)) + "\n",
                encoding="utf-8",
            )

            config = load_config()

            assert isinstance(config["providers"], dict)
            assert (
                _supports_vision_override(config, "custom", "Qwen3.8-27B") is True
            )

    def test_env_refs_inside_decoded_providers_still_expand(self, tmp_path):
        """Decoding runs before env-expansion, so ``${VAR}`` inside the JSON
        string is expanded normally once it lands in the dict."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "VLLM_KEY": "secret-123"}):
            providers = {"custom": {"api_key": "${VLLM_KEY}", "models": {}}}
            (tmp_path / "config.yaml").write_text(
                "providers: " + json.dumps(json.dumps(providers)) + "\n",
                encoding="utf-8",
            )

            config = load_config()

            assert config["providers"]["custom"]["api_key"] == "secret-123"

    def test_malformed_providers_string_loads_as_empty(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            (tmp_path / "config.yaml").write_text(
                "providers: '{not valid json'\n", encoding="utf-8"
            )

            config = load_config()

            assert config["providers"] == {}
