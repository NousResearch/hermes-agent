"""Migration 53→54: prompt_caching.cache_ttl defaults to "auto".

Contract: only a ``cache_ttl: "5m"`` Hermes provably wrote is dropped — the line the config
template seeded, its trailing comment intact. A bare ``"5m"`` may be the user's pin and survives,
as do ``1h``, ``auto`` and an off value. Driven through ``run_migrations`` against a temp home.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml
import pytest

#: The block cli-config.yaml.example shipped before "auto" (9e6f34a76e24).
TEMPLATE_SEEDED = """\
# Hand comment the migration must keep
_config_version: 53
prompt_caching:
  cache_ttl: "5m" # use "1h" for long sessions with pauses between turns
model:
  default: some-model  # pinned for evals
"""


def _run(home, text):
    from hermes_cli.config_migrations import run_migrations

    home.mkdir()
    (home / "config.yaml").write_text(text, encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}, clear=False):
        run_migrations(53, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
    return (home / "config.yaml").read_text(encoding="utf-8")


def _ttl_section(text):
    return yaml.safe_load(text).get("prompt_caching") or {}


def test_template_seeded_5m_is_dropped_and_follows_auto(tmp_path):
    from hermes_cli.config import load_config

    home = tmp_path / "seeded"
    text = _run(home, TEMPLATE_SEEDED)
    assert "cache_ttl" not in _ttl_section(text), "the template's line is Hermes' copy, not a choice"
    assert "# Hand comment the migration must keep" in text and "# pinned for evals" in text
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        assert load_config()["prompt_caching"]["cache_ttl"] == "auto"


def test_bare_5m_is_a_user_pin_and_survives(tmp_path):
    text = yaml.safe_dump({"_config_version": 53, "prompt_caching": {"cache_ttl": "5m"}})
    assert _ttl_section(_run(tmp_path / "pinned", text))["cache_ttl"] == "5m"


@pytest.mark.parametrize("ttl", ["1h", "auto", "off", False])
def test_other_values_survive(tmp_path, ttl):
    text = yaml.safe_dump({"_config_version": 53, "prompt_caching": {"cache_ttl": ttl}})
    assert _ttl_section(_run(tmp_path / "pinned", text))["cache_ttl"] == ttl
