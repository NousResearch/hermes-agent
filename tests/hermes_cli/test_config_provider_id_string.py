"""Regression for #117345: ``model.provider`` must survive config load as a string.

An unquoted YAML scalar (``provider: 2``) loads as ``int``, and downstream readers
do ``(provider or "").strip()`` — a gateway turn dies before the agent runs. The
load-path chokepoint (``_normalize_root_model_keys``) canonicalizes the value, so
``load_config()`` — the gateway's exact entry — is the surface under test.
"""

import os
from unittest.mock import patch

from hermes_cli.config import load_config


def _load(tmp_path, model_section: str):
    (tmp_path / "config.yaml").write_text(f"model:\n{model_section}", encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        return load_config()


def test_unquoted_numeric_provider_loads_as_string(tmp_path):
    # ``0`` is also falsy: a stringified "0" must not be dropped or blanked by the
    # ``root_val and ...`` / ``(provider or "")`` guards on the way through.
    for scalar in ("2", "0", "2.0"):
        config = _load(tmp_path, f"  default: deepseek-flash\n  provider: {scalar}\n")
        assert config["model"]["provider"] == scalar, scalar


def test_absent_provider_key_is_not_injected(tmp_path):
    # Coercing ``None`` would yield ""; an injected empty key rewrites config.yaml on
    # the next save, so a provider-less model section must come back without one.
    config = _load(tmp_path, "  default: deepseek-flash\n")
    assert "provider" not in config["model"]
