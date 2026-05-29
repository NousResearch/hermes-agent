"""Config-surface tests for the top-level ``model:`` block (cpf-zkw.5, #8919).

Covers:
  - ``model.api_base`` accepted as a permanent alias, normalized to
    ``model.base_url`` at load (gentle one-time INFO, not a deprecation).
  - Unknown ``model.*`` keys warn once at load (warning, not hard error —
    preserves forward-compat).
"""

from __future__ import annotations

import logging

import pytest

import hermes_cli.config as cfg_mod
from hermes_cli.config import load_config, save_config


@pytest.fixture(autouse=True)
def _reset_model_surface_guards():
    """Reset the one-time process guards so warn-once tests are order-independent.

    Module-level state persists within a test file (each file runs in its own
    subprocess); clear the guards before every test here.
    """
    cfg_mod._MODEL_API_BASE_INFO_EMITTED = False
    cfg_mod._UNKNOWN_MODEL_KEYS_WARNED.clear()
    yield


def _write_model(model_section):
    cfg = load_config()
    cfg["model"] = model_section
    save_config(cfg)


# ── api_base → base_url normalization ─────────────────────────────────────


class TestApiBaseAlias:
    def test_api_base_normalized_to_base_url(self):
        """model.api_base is folded into model.base_url at load."""
        _write_model({"provider": "custom", "api_base": "http://localhost:1234/v1"})

        loaded = load_config()
        model = loaded["model"]
        assert model["base_url"] == "http://localhost:1234/v1"
        # The alias key is consumed — downstream reads a single field.
        assert "api_base" not in model

    def test_explicit_base_url_wins_over_api_base(self):
        """When both are set, explicit base_url is canonical; api_base is dropped."""
        _write_model({
            "provider": "custom",
            "base_url": "http://canonical/v1",
            "api_base": "http://alias/v1",
        })

        model = load_config()["model"]
        assert model["base_url"] == "http://canonical/v1"
        assert "api_base" not in model

    def test_api_base_emits_one_time_info(self, caplog):
        """A gentle INFO line is emitted (once) when api_base is normalized."""
        _write_model({"provider": "custom", "api_base": "http://localhost:1234/v1"})

        with caplog.at_level(logging.INFO, logger="hermes_cli.config"):
            cfg_mod._MODEL_API_BASE_INFO_EMITTED = False
            cfg_mod._normalize_model_api_base({
                "model": {"provider": "custom", "api_base": "http://localhost:1234/v1"}
            })
            # Second call must NOT re-emit.
            before = len([r for r in caplog.records if "api_base" in r.message])
            cfg_mod._normalize_model_api_base({
                "model": {"provider": "custom", "api_base": "http://localhost:1234/v1"}
            })
            after = len([r for r in caplog.records if "api_base" in r.message])

        assert before == 1, "api_base INFO should be emitted exactly once"
        assert after == before, "api_base INFO must not re-emit on subsequent loads"

    def test_string_model_section_untouched(self):
        """A bare-string model section (just a model name) is left alone."""
        out = cfg_mod._normalize_model_api_base({"model": "gpt-4o"})
        assert out["model"] == "gpt-4o"


# ── unknown model.* key warning ───────────────────────────────────────────


class TestUnknownModelKeys:
    def test_unknown_key_warns_once(self, caplog):
        """An unrecognized model.* key warns once (forward-compat, not an error)."""
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            cfg_mod._normalize_model_api_base({
                "model": {"provider": "custom", "totally_bogus_key": 1}
            })
            # Repeat load — must not re-warn for the same key.
            cfg_mod._normalize_model_api_base({
                "model": {"provider": "custom", "totally_bogus_key": 1}
            })

        warnings = [r for r in caplog.records
                    if "unknown" in r.message.lower() and "totally_bogus_key" in r.message]
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"

    def test_known_keys_do_not_warn(self, caplog):
        """A normal model dict produces no unknown-key warning."""
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            cfg_mod._normalize_model_api_base({
                "model": {
                    "provider": "custom",
                    "default": "my-model",
                    "base_url": "http://x/v1",
                    "api_mode": "chat_completions",
                    "context_length": 8192,
                }
            })

        assert not [r for r in caplog.records if "unknown" in r.message.lower()], \
            "known model keys must not trigger an unknown-key warning"

    def test_api_base_is_a_known_key(self, caplog):
        """api_base must NOT be reported as unknown — it's a permanent alias."""
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            cfg_mod._normalize_model_api_base({
                "model": {"provider": "custom", "api_base": "http://x/v1"}
            })

        assert not [r for r in caplog.records
                    if "unknown" in r.message.lower() and "api_base" in r.message]
