"""Regression tests for #69737: hermes -z must honor ``checkpoints:`` config.

``cli.py`` parses the ``checkpoints:`` section into instance attributes and
the interactive chat path forwards them to ``AIAgent``
(``hermes_cli/cli_agent_setup_mixin.py``), but oneshot mode bypasses cli.py
entirely — ``hermes_cli/oneshot.py::_run_agent`` builds its own agent — so
``checkpoints.enabled: true`` silently produced zero snapshots for every
``hermes -z`` run. Mirrors tests/gateway/test_checkpoint_config.py for the
CLI oneshot construction path.
"""

import hermes_cli.config as config_mod
import hermes_cli.runtime_provider as runtime_provider_mod
import run_agent as run_agent_mod
from hermes_cli import oneshot


class _CapturingAgent:
    """Stands in for AIAgent and records the constructor kwargs."""

    captured: dict = {}

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)
        self.suppress_status_output = False
        self.stream_delta_callback = None
        self.tool_gen_callback = None

    def run_conversation(self, prompt):
        return {"final_response": "done"}

    def shutdown_memory_provider(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _wire_oneshot_stubs(monkeypatch, cfg):
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **_kw: {
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": None,
            "command": None,
            "args": None,
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda _cfg: None)
    _CapturingAgent.captured = {}
    monkeypatch.setattr(run_agent_mod, "AIAgent", _CapturingAgent)


def test_69737_oneshot_forwards_checkpoints_config_to_agent(monkeypatch):
    """The four checkpoints.* settings must reach the AIAgent constructor."""
    cfg = {
        "model": {"default": "test/model", "provider": "openrouter"},
        "checkpoints": {
            "enabled": True,
            "max_snapshots": 11,
            "max_total_size_mb": 345,
            "max_file_size_mb": 6,
        },
    }
    _wire_oneshot_stubs(monkeypatch, cfg)

    response, _result = oneshot._run_agent("hello", use_config_toolsets=False)

    assert response == "done"
    captured = _CapturingAgent.captured
    assert captured["checkpoints_enabled"] is True
    assert captured["checkpoint_max_snapshots"] == 11
    assert captured["checkpoint_max_total_size_mb"] == 345
    assert captured["checkpoint_max_file_size_mb"] == 6


def test_69737_oneshot_tolerates_legacy_boolean_checkpoints_config(monkeypatch):
    """Legacy ``checkpoints: true`` enables snapshots with default limits,
    matching the gateway's ``_checkpoint_agent_kwargs`` semantics."""
    cfg = {
        "model": {"default": "test/model", "provider": "openrouter"},
        "checkpoints": True,
    }
    _wire_oneshot_stubs(monkeypatch, cfg)

    oneshot._run_agent("hello", use_config_toolsets=False)

    captured = _CapturingAgent.captured
    assert captured["checkpoints_enabled"] is True
    assert captured["checkpoint_max_snapshots"] == 20
    assert captured["checkpoint_max_total_size_mb"] == 500
    assert captured["checkpoint_max_file_size_mb"] == 10
