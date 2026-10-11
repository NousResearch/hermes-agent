"""compression.proactive_prune_* — config parse seam for the proactive prune.

Mirrors ``test_compression_max_attempts_config.py``: the three knobs are
parsed in ``agent_init`` with the same hardened semantics (booleans rejected,
fractional floats rejected — not truncated, integral floats and numeric
strings accepted) and attached to the built-in compressor.  Default is
64000 / 1000 / 64000 (on); ``proactive_prune_tokens: 0`` or ``false`` turns it off.
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

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: _config(**prune_keys))
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

    def test_unset_ships_the_measured_defaults(self, monkeypatch, tmp_path):
        cc = _make_agent(monkeypatch, tmp_path).context_compressor
        assert (cc.proactive_prune_tokens, cc.proactive_prune_min_result_chars, cc.proactive_prune_min_reclaim_tokens) == (
            64_000, 1_000, 64_000,
        )

    def test_boolean_is_rejected_not_coerced(self, monkeypatch, tmp_path):
        # bool subclasses int: YAML `true` must never coerce to a 1-token trigger;
        # it keeps the default. `false` is an explicit off switch.
        agent = _make_agent(monkeypatch, tmp_path, proactive_prune_tokens=True)
        assert agent.context_compressor.proactive_prune_tokens == 64_000
        agent = _make_agent(monkeypatch, tmp_path / "off", proactive_prune_tokens=False)
        assert agent.context_compressor.proactive_prune_tokens == 0




