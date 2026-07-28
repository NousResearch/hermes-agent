"""compression.proactive_prune_* — config parse seam for the proactive prune.

Mirrors ``test_compression_max_attempts_config.py``: the three knobs are
parsed in ``agent_init`` with the same hardened semantics (booleans rejected,
fractional floats rejected — not truncated, integral floats and numeric
strings accepted) and attached to the built-in compressor.  Default is
0 / 8000 / 4096, i.e. the feature is OFF and behavior-neutral unless
``proactive_prune_tokens`` is set above 0.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from hermes_state import SessionDB
from run_agent import AIAgent


def _config(**prune_keys) -> dict:
    compression = {
        "enabled": True,
        "threshold": 0.50,
        "target_ratio": 0.20,
        "protect_first_n": 3,
        "protect_last_n": 20,
    }
    compression.update(prune_keys)
    return {
        "compression": compression,
        "prompt_caching": {"cache_ttl": "5m"},
        "sessions": {},
        "bedrock": {},
    }


def _make_agent(monkeypatch, tmp_path: Path, **prune_keys):
    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: _config(**prune_keys))
    db = SessionDB(db_path=tmp_path / "state.db")
    with contextlib.redirect_stdout(io.StringIO()):
        agent = AIAgent(
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-key",
            provider="openai-codex",
            model="gpt-5.5",
            enabled_toolsets=[],
            disabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            session_db=db,
            session_id="proactive-prune-config-test",
        )
    return agent


class TestProactivePruneConfig:
    def test_default_is_disabled_when_unset(self, monkeypatch, tmp_path):
        agent = _make_agent(monkeypatch, tmp_path)
        cc = agent.context_compressor
        assert cc.proactive_prune_tokens == 0
        assert cc.proactive_prune_min_result_chars == 8000
        assert cc.proactive_prune_min_reclaim_tokens == 4096

    def test_custom_values_are_honored(self, monkeypatch, tmp_path):
        agent = _make_agent(
            monkeypatch,
            tmp_path,
            proactive_prune_tokens=48_000,
            proactive_prune_min_result_chars=12_000,
            proactive_prune_min_reclaim_tokens=8_192,
        )
        cc = agent.context_compressor
        assert cc.proactive_prune_tokens == 48_000
        assert cc.proactive_prune_min_result_chars == 12_000
        assert cc.proactive_prune_min_reclaim_tokens == 8_192

    def test_boolean_is_rejected_not_coerced(self, monkeypatch, tmp_path):
        # bool subclasses int: YAML `proactive_prune_tokens: true` must fall
        # back to disabled, never coerce to 1 token.
        agent = _make_agent(monkeypatch, tmp_path, proactive_prune_tokens=True)
        assert agent.context_compressor.proactive_prune_tokens == 0

    def test_fractional_float_is_rejected_not_truncated(self, monkeypatch, tmp_path):
        agent = _make_agent(monkeypatch, tmp_path, proactive_prune_tokens=48_000.7)
        assert agent.context_compressor.proactive_prune_tokens == 0

    def test_integral_float_and_numeric_string_accepted(self, monkeypatch, tmp_path):
        agent = _make_agent(monkeypatch, tmp_path, proactive_prune_tokens=48_000.0)
        assert agent.context_compressor.proactive_prune_tokens == 48_000
        agent = _make_agent(monkeypatch, tmp_path, proactive_prune_tokens="32000")
        assert agent.context_compressor.proactive_prune_tokens == 32_000

    def test_negative_trigger_treated_as_disabled(self, monkeypatch, tmp_path):
        agent = _make_agent(monkeypatch, tmp_path, proactive_prune_tokens=-100)
        assert agent.context_compressor.proactive_prune_tokens == 0

    def test_garbage_falls_back_to_defaults(self, monkeypatch, tmp_path):
        agent = _make_agent(
            monkeypatch,
            tmp_path,
            proactive_prune_tokens="lots",
            proactive_prune_min_result_chars=None,
            proactive_prune_min_reclaim_tokens="???",
        )
        cc = agent.context_compressor
        assert cc.proactive_prune_tokens == 0
        assert cc.proactive_prune_min_result_chars == 8000
        assert cc.proactive_prune_min_reclaim_tokens == 4096
