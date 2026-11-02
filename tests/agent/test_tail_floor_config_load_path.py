"""compression.max_tail_message_floor — real config load-path regression.

The companion tests in ``test_context_compressor.py`` construct ``ContextCompressor``
directly, which proves the ctor honours the knob but not that a value written into
``config.yaml`` reaches the compressor the agent actually uses.  Reviewer request
(https://github.com/NousResearch/hermes-agent/pull/60662): exercise the full
``config.yaml`` → ``load_config_readonly`` → ``_parse_compression_config`` →
``ContextCompressor`` path with a temporary ``HERMES_HOME``, i.e. the loader the
agent-construction path actually takes.

The assertions use the built-in compressor's resolved floor (not the raw ctor
attribute) so they also cover the ``0 → module default`` fallback in
``_effective_max_tail_message_floor``.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

import hermes_yaml as yaml

from hermes_state import SessionDB
from run_agent import AIAgent


def _write_config(home: Path, compression: dict) -> None:
    """Install ``config.yaml`` under a temp ``HERMES_HOME`` exactly as a user would."""
    home.mkdir(parents=True, exist_ok=True)
    config = {
        "model": {"default": "gpt-5.5"},
        "compression": {
            "enabled": True,
            "threshold": 0.50,
            "target_ratio": 0.20,
            "protect_first_n": 3,
            "protect_last_n": 20,
            **compression,
        },
        "prompt_caching": {"cache_ttl": "5m"},
        "sessions": {},
        "bedrock": {},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _flush_config_cache() -> None:
    """Drop the process-wide config caches so each test re-reads its own temp config.yaml."""
    from hermes_cli import config as config_mod

    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()


def _make_agent(db_dir: Path) -> AIAgent:
    """Build an agent the way ``init_agent`` does — it loads config itself via the real loader."""
    db_dir.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=db_dir / "state.db")
    with contextlib.redirect_stdout(io.StringIO()):
        return AIAgent(
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-key",
            provider="openai-codex",
            model="gpt-5.5",
            enabled_toolsets=[],
            disabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            session_db=db,
            session_id="tail-floor-config-load-test",
        )


@pytest.fixture(autouse=True)
def _uncached_config(monkeypatch, tmp_path):
    """Every test gets its own HERMES_HOME and a cold config cache."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _flush_config_cache()
    yield
    _flush_config_cache()


class TestMaxTailMessageFloorLoadPath:
    """The value in config.yaml must survive the loader and reach the compressor."""

    def test_non_default_value_reaches_compressor(self, monkeypatch, tmp_path):
        _write_config(tmp_path / "home", {"max_tail_message_floor": 20})

        cc = _make_agent(tmp_path / "db-a").context_compressor
        assert cc.max_tail_message_floor == 20
        assert cc._effective_max_tail_message_floor == 20

    def test_default_falls_back_to_module_default(self, tmp_path):
        # Config omitted entirely → DEFAULT_CONFIG's 0 → module default of 8.
        _write_config(tmp_path / "home", {})

        cc = _make_agent(tmp_path / "db-a").context_compressor
        assert cc.max_tail_message_floor == 0
        assert cc._effective_max_tail_message_floor == 8

    def test_string_value_is_coerced(self, tmp_path):
        # YAML strings are accepted (int() coercion), matching every other int knob
        # in the section; a quoted "16" must not silently become the default.
        _write_config(tmp_path / "home", {"max_tail_message_floor": "16"})

        assert _make_agent(tmp_path / "db-a").context_compressor.max_tail_message_floor == 16

    def test_negative_is_clamped_to_disabled(self, tmp_path):
        # The ctor clamps negatives to 0 (disabled → module default), so a stray
        # ``-1`` degrades to the default rather than a nonsensical floor.
        _write_config(tmp_path / "home", {"max_tail_message_floor": -5})

        cc = _make_agent(tmp_path / "db-a").context_compressor
        assert cc.max_tail_message_floor == 0
        assert cc._effective_max_tail_message_floor == 8

    def test_change_is_picked_up_after_reload(self, monkeypatch, tmp_path):
        """Two temp homes with different floors must not share a cached parse."""
        home_a = tmp_path / "home-a"
        _write_config(home_a, {"max_tail_message_floor": 12})
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert _make_agent(tmp_path / "db-a").context_compressor.max_tail_message_floor == 12

        home_b = tmp_path / "home-b"
        _write_config(home_b, {"max_tail_message_floor": 30})
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        assert _make_agent(tmp_path / "db-b").context_compressor.max_tail_message_floor == 30
