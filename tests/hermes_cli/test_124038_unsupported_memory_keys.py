"""#124038: config.yaml ``memory.<extra>`` keys are silently ignored — warn once.

A user who sets ``memory.hindsight.mode`` in config.yaml gets zero effect
and zero feedback: Hermes' memory schema declares a fixed key set, and
memory-provider plugins (e.g. hindsight via the catalog) read their OWN
configuration (``$HERMES_HOME/hindsight/config.json``), never config.yaml.
The silent ignore makes the trap expensive to diagnose.
"""

import textwrap

import pytest

# Local-env workaround (#124038 tests): the checkout's venv python3 is a
# symlink into <home>/hermes-agent/.hermes-runtime/..., so stdlib zoneinfo's
# first import (sysconfig._safe_realpath(sys.executable)) trips the real-home
# IO guard after it installs. Importing here — during collection, before any
# autouse guard fixture — caches the module. CI pythons are not symlinked into
# the home, so this is a no-op there.
import zoneinfo  # noqa: F401  (cached for the moa-loop fixture)

import hermes_cli.config as cfg

SCHEMA_KEYS = frozenset({
    "memory_enabled", "user_profile_enabled", "write_approval",
    "memory_char_limit", "user_char_limit", "nudge_interval", "provider",
})


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    return home


def _write(home, body):
    (home / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()


class TestUnsupportedMemoryKeys:
    def test_unknown_memory_segment_is_reported(self, homes):
        _write(homes, """
        memory:
          memory_enabled: true
          hindsight:
            mode: local_external
        """)
        assert cfg._unsupported_memory_keys({"memory": {
            "memory_enabled": True, "hindsight": {"mode": "local_external"}}}) == ["memory.hindsight"]

    def test_known_schema_keys_are_not_reported(self):
        keys = {k: True for k in SCHEMA_KEYS}
        assert cfg._unsupported_memory_keys({"memory": keys}) == []

    def test_missing_memory_section_is_not_reported(self):
        assert cfg._unsupported_memory_keys({}) == []
        assert cfg._unsupported_memory_keys({"agent": {"max_turns": 3}}) == []

    def test_non_dict_memory_is_not_reported(self):
        assert cfg._unsupported_memory_keys({"memory": "on"}) == []

    def test_schema_keys_derived_from_default_config(self):
        """The whitelist must never drift from DEFAULT_CONFIG's memory section."""
        defaults = dict(cfg.DEFAULT_CONFIG["memory"])
        assert set(defaults) == SCHEMA_KEYS


class TestLoadConfigWarnsOnce:
    def test_load_config_warns_for_unknown_memory_key(self, homes, caplog):
        _write(homes, """
memory:
  memory_enabled: true
  hindsight:
    mode: local_external
""")
        with caplog.at_level("WARNING", logger="hermes_cli.config"):
            cfg.load_config()
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "memory.hindsight" in joined

    def test_load_config_silent_for_schema_keys(self, homes, caplog):
        _write(homes, """
memory:
  memory_enabled: true
  provider: ""
""")
        with caplog.at_level("WARNING", logger="hermes_cli.config"):
            cfg.load_config()
        assert not any("memory." in r.getMessage() for r in caplog.records)

    def test_load_config_silent_without_memory_section(self, homes, caplog):
        _write(homes, "agent:\n  max_turns: 1\n")
        with caplog.at_level("WARNING", logger="hermes_cli.config"):
            cfg.load_config()
        assert not any("memory." in r.getMessage() for r in caplog.records)