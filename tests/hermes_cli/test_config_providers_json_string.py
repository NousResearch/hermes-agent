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

import logging

from agent.image_routing import _supports_vision_override
from hermes_cli import config as config_mod
from hermes_cli.config import _normalize_providers_string, load_config
from hermes_cli.managed_scope import apply_managed_overlay


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

    def test_whitespace_only_is_empty_without_warning(self, caplog):
        # ``providers: '   '`` is an empty value, not corruption — no warning.
        config_mod._PROVIDERS_STRING_WARNED.clear()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            out = _normalize_providers_string({"providers": "   "})
        assert out["providers"] == {}
        assert not caplog.records

    def test_malformed_payload_warns_once_per_payload(self, caplog):
        # ``_load_config_impl`` re-runs on every cache-signature change; a
        # persistently-broken payload must log once, not on every load.
        config_mod._PROVIDERS_STRING_WARNED.clear()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            for _ in range(3):
                _normalize_providers_string({"providers": "{not json"})
        assert sum("malformed 'providers'" in r.message for r in caplog.records) == 1
        # A *different* broken payload still surfaces after the first is seen.
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            _normalize_providers_string({"providers": "[1, 2, 3]"})
        assert sum("malformed 'providers'" in r.message for r in caplog.records) == 2

    def test_warning_names_which_kind_of_malformed(self, caplog):
        # The two malformed branches describe different problems and must not
        # share wording: a non-JSON scalar vs. valid JSON of the wrong shape.
        config_mod._PROVIDERS_STRING_WARNED.clear()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            _normalize_providers_string({"providers": "{not json"})
        assert "not valid JSON" in caplog.records[-1].getMessage()

        caplog.clear()
        config_mod._PROVIDERS_STRING_WARNED.clear()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            _normalize_providers_string({"providers": "[1, 2, 3]"})
        assert "valid JSON that is not an object" in caplog.records[-1].getMessage()

    def test_json_null_is_labeled_valid_json_not_undecodable(self, caplog):
        # ``json.loads("null")`` decodes to None: it parsed fine, it's just the
        # wrong shape. It must not be mislabeled as un-decodable JSON.
        config_mod._PROVIDERS_STRING_WARNED.clear()
        out = _normalize_providers_string({"providers": "null"})
        assert out["providers"] == {}
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            config_mod._PROVIDERS_STRING_WARNED.clear()
            _normalize_providers_string({"providers": "null"})
        msg = caplog.records[-1].getMessage()
        assert "valid JSON that is not an object" in msg
        assert "not valid JSON" not in msg


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


class TestManagedOverlayDecodesProvidersString:
    """``managed_scope.apply_managed_overlay`` is a third config path whose
    docstring pledges to "Mirror _load_config_impl's managed merge exactly";
    it must decode a JSON-string ``providers`` too, or a managed string would
    _deep_merge over — and drop — a valid user ``providers`` mapping."""

    def test_managed_overlay_decodes_json_string_providers(self):
        managed = {"providers": '{"custom": {"base_url": "http://managed:1/v1"}}'}
        with patch(
            "hermes_cli.managed_scope.load_managed_config", return_value=managed
        ):
            merged = apply_managed_overlay({"providers": {"other": {}}})
        assert merged["providers"]["custom"]["base_url"] == "http://managed:1/v1"

    def test_managed_string_providers_does_not_clobber_user_mapping(self):
        # Managed wins at the leaf, so an *undecoded* string would replace the
        # user's whole ``providers`` dict. Decoded, it merges per-leaf instead.
        managed = {"providers": '{"custom": {"base_url": "http://managed:1/v1"}}'}
        user = {"providers": {"mine": {"base_url": "http://user:2/v1"}}}
        with patch(
            "hermes_cli.managed_scope.load_managed_config", return_value=managed
        ):
            merged = apply_managed_overlay(user)
        assert merged["providers"]["mine"]["base_url"] == "http://user:2/v1"
        assert merged["providers"]["custom"]["base_url"] == "http://managed:1/v1"

    def test_env_refs_inside_managed_providers_string_expand(self):
        managed = {"providers": '{"custom": {"api_key": "${MANAGED_KEY}"}}'}
        with patch(
            "hermes_cli.managed_scope.load_managed_config", return_value=managed
        ), patch.dict(os.environ, {"MANAGED_KEY": "managed-secret"}):
            merged = apply_managed_overlay({"providers": {}})
        assert merged["providers"]["custom"]["api_key"] == "managed-secret"
