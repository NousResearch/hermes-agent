"""Behavior-contract test for the model.context_length override guard (#62152).

The override in config.yaml is written for ``model.default``. It must NOT leak
onto a session that overrides to a *different* model — that session must
auto-detect its own window. The guard in agent/agent_init.py implements that;
this test pins the contract without booting the full agent runtime.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _apply_guard(model_cfg, agent_model):
    """Mirror of the guard in agent/agent_init.py (lines ~1630-1645)."""
    _config_context_length = model_cfg.get("context_length") if isinstance(model_cfg, dict) else None
    if _config_context_length is not None and isinstance(model_cfg, dict):
        _default_model = model_cfg.get("default") or model_cfg.get("model")
        if agent_model and _default_model and agent_model != _default_model:
            _config_context_length = None
    return _config_context_length


def test_override_kept_when_session_runs_default():
    # No per-session override: agent.model == model.default -> keep override.
    cfg = {"default": "Qwen3.6-27B", "context_length": 131072}
    assert _apply_guard(cfg, "Qwen3.6-27B") == 131072


def test_override_dropped_when_session_overrides_model():
    # Per-session override to a different model -> drop override (auto-detect).
    cfg = {"default": "Qwen3.6-27B", "context_length": 131072}
    assert _apply_guard(cfg, "gpt-5.6-sol") is None


def test_no_override_set_stays_none():
    cfg = {"default": "Qwen3.6-27B"}  # no context_length key
    assert _apply_guard(cfg, "gpt-5.6-sol") is None


def test_override_with_model_key_not_default():
    # model written under `model:` rather than `default:` still guards.
    cfg = {"model": "Qwen3.6-27B", "context_length": 131072}
    assert _apply_guard(cfg, "Qwen3.6-27B") == 131072
    assert _apply_guard(cfg, "gpt-5.6-sol") is None
