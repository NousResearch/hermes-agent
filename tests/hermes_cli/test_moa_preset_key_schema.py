"""``moa.presets.<name>`` key schema: supported preset keys must validate as known.

``_normalize_preset`` reads exactly the keys in ``_FLAT_PRESET_KEYS``, and preset names are
user-chosen. The config-key validator used to walk ``DEFAULT_CONFIG['moa']['presets']`` instead —
a single ``default`` template seeded with only ``reference_models`` / ``aggregator`` / ``enabled``.
Consequences (#60388):

* ``moa.presets.<any-other-name>.*`` was refused as an unknown key even for a valid preset;
* supported keys DEFAULT_CONFIG does not seed (``reference_temperature``, ``fanout``,
  ``reference_timeout``, ``degraded_reference_policy``, ``aggregator_temperature``) were flagged
  "not a recognized config key — it was saved anyway", sometimes with a did-you-mean that pointed
  at ``reference_models``.

So the CLI warned about settings the runtime honors while staying quiet about settings it drops.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import _validate_config_key, set_config_value


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    (tmp_path / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _read_config(tmp_path) -> str:
    path = tmp_path / "config.yaml"
    return path.read_text() if path.exists() else ""


# The authoritative preset schema: what _normalize_preset returns, i.e. what it reads.
SUPPORTED_PRESET_KEYS = (
    "reference_models",
    "aggregator",
    "reference_temperature",
    "aggregator_temperature",
    "reference_timeout",
    "degraded_reference_policy",
    "fanout",
    "enabled",
)


class TestSupportedPresetKeysValidate:
    @pytest.mark.parametrize("preset", ["default", "fast", "cheap-advisors"])
    @pytest.mark.parametrize("leaf", SUPPORTED_PRESET_KEYS)
    def test_supported_key_on_any_preset_name_is_known(self, preset, leaf):
        assert _validate_config_key(f"moa.presets.{preset}.{leaf}") == (True, None)

    @pytest.mark.parametrize("key", [
        "moa.presets",            # container itself
        "moa.presets.fast",       # a whole preset (user-chosen name)
        "moa.presets.fast.aggregator.provider",  # slot payload, validated at read time
    ])
    def test_preset_container_and_namespaces_are_known(self, key):
        assert _validate_config_key(key) == (True, None)

    def test_unknown_leaf_inside_a_preset_is_still_flagged(self):
        """The schema fix must not turn the preset namespace into an open container."""
        known, suggestion = _validate_config_key("moa.presets.default.reference_model")
        assert known is False
        assert suggestion == "moa.presets.default.reference_models"

    def test_exported_key_set_is_exactly_what_the_normalizer_reads(self):
        from hermes_cli.moa_config import SUPPORTED_PRESET_KEYS as EXPORTED
        from hermes_cli.moa_config import _FLAT_PRESET_KEYS

        assert isinstance(EXPORTED, frozenset)
        assert EXPORTED == frozenset(_FLAT_PRESET_KEYS)


class TestPresetKeySetConfigNoNotice:
    def test_supported_key_on_a_named_preset_writes_without_a_notice(
        self, _isolated_hermes_home, capsys
    ):
        set_config_value("moa.presets.fast.fanout", "per_iteration")
        captured = capsys.readouterr()

        assert "not a recognized config key" not in captured.out
        assert "not a recognized config key" not in captured.err
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert saved["moa"]["presets"]["fast"]["fanout"] == "per_iteration"

    def test_seeded_supported_key_writes_without_a_notice(
        self, _isolated_hermes_home, capsys
    ):
        set_config_value("moa.presets.default.reference_temperature", "0.7")
        captured = capsys.readouterr()

        assert "not a recognized config key" not in captured.out
        assert "not a recognized config key" not in captured.err
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert saved["moa"]["presets"]["default"]["reference_temperature"] == 0.7

    def test_a_real_typo_still_warns(self, _isolated_hermes_home, capsys):
        set_config_value("moa.presets.default.reference_model", "x")
        captured = capsys.readouterr()

        assert "not a recognized config key" in captured.out + captured.err
        assert "reference_models" in captured.out + captured.err
