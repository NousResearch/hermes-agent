"""Regression tests: hermes -z must honor ``agent.reasoning_effort``.

The interactive chat path resolves reasoning config at startup (cli.py, via
``hermes_constants.resolve_reasoning_config``) and forwards it to ``AIAgent``
(``hermes_cli/cli_agent_setup_mixin.py``); the gateway resolves through the
same chokepoint (``_load_reasoning_config``). Oneshot mode bypasses both --
``hermes_cli/oneshot.py::_run_agent`` builds its own agent -- so
``agent.reasoning_effort`` was silently ignored and every ``hermes -z`` run
executed with ``reasoning_config=None`` (observable as
``model_config.reasoning_config: null`` in the session record).
"""

import hermes_cli.config as config_mod
import hermes_cli.runtime_provider as runtime_provider_mod
import run_agent as run_agent_mod
from hermes_cli import oneshot
from hermes_constants import resolve_reasoning_config


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


def test_oneshot_forwards_global_reasoning_effort(monkeypatch):
    """``agent.reasoning_effort`` must reach the AIAgent constructor, resolved
    through the shared chokepoint (same result the chat path would get)."""
    cfg = {
        "model": {"default": "test/model", "provider": "openrouter"},
        "agent": {"reasoning_effort": "high"},
    }
    _wire_oneshot_stubs(monkeypatch, cfg)

    response, _result = oneshot._run_agent("hello", use_config_toolsets=False)

    assert response == "done"
    captured = _CapturingAgent.captured
    expected = resolve_reasoning_config(cfg, "test/model")
    assert expected is not None  # guard: the fixture config must resolve
    assert captured["reasoning_config"] == expected


def test_oneshot_applies_per_model_reasoning_override(monkeypatch):
    """Per-model ``agent.reasoning_overrides`` beat the global effort, using
    the FINAL effective model -- same priority the chokepoint documents."""
    cfg = {
        "model": {"default": "test/model", "provider": "openrouter"},
        "agent": {
            "reasoning_effort": "low",
            "reasoning_overrides": {"test/model": "high"},
        },
    }
    _wire_oneshot_stubs(monkeypatch, cfg)

    oneshot._run_agent("hello", use_config_toolsets=False)

    captured = _CapturingAgent.captured
    expected = resolve_reasoning_config(cfg, "test/model")
    assert captured["reasoning_config"] == expected
    assert captured["reasoning_config"] is not None


def test_oneshot_resolves_reasoning_for_remapped_model(monkeypatch):
    """When provider auto-detection remaps an explicitly requested model
    (oneshot.py's ``detect_provider_for_model`` branch), reasoning must be
    resolved against the REMAPPED model, not the name the caller typed --
    otherwise a per-model override keyed by the real model id is missed."""
    import hermes_cli.models as models_mod
    from hermes_cli import model_switch as ms_mod

    cfg = {
        "model": {"default": "cfg/unused"},
        "agent": {
            "reasoning_effort": "low",
            "reasoning_overrides": {"real/remapped-model": "high"},
        },
    }
    _wire_oneshot_stubs(monkeypatch, cfg)
    # Force the catalog-detection branch: no direct alias, and detection
    # remaps the requested name to the catalog's real model id.
    monkeypatch.setattr(ms_mod, "_ensure_direct_aliases", lambda: None)
    monkeypatch.setattr(ms_mod, "DIRECT_ALIASES", {})
    monkeypatch.setattr(
        models_mod,
        "detect_provider_for_model",
        lambda _model, _provider: ("openrouter", "real/remapped-model"),
    )

    oneshot._run_agent("hello", model="requested-alias", use_config_toolsets=False)

    captured = _CapturingAgent.captured
    assert captured["model"] == "real/remapped-model"
    expected = resolve_reasoning_config(cfg, "real/remapped-model")
    assert captured["reasoning_config"] == expected
    # Guard that the assertion bites: resolving with the raw requested name
    # would yield a different (global-effort) config.
    assert expected != resolve_reasoning_config(cfg, "requested-alias")


def test_oneshot_resolves_reasoning_for_direct_alias_model(monkeypatch):
    """The other remapping branch: a config.yaml ``model_aliases:`` hit
    (``DIRECT_ALIASES``) also rewrites effective_model before construction,
    so reasoning must likewise resolve against the alias TARGET."""
    from hermes_cli import model_switch as ms_mod

    cfg = {
        "model": {"default": "cfg/unused"},
        "agent": {
            "reasoning_effort": "low",
            "reasoning_overrides": {"real/alias-target": "high"},
        },
    }
    _wire_oneshot_stubs(monkeypatch, cfg)
    monkeypatch.setattr(ms_mod, "_ensure_direct_aliases", lambda: None)
    monkeypatch.setattr(
        ms_mod,
        "DIRECT_ALIASES",
        {
            "my-alias": ms_mod.DirectAlias(
                model="real/alias-target", provider="openrouter", base_url=""
            )
        },
    )

    # Mixed case exercises the .lower() lookup normalization too.
    oneshot._run_agent("hello", model="My-Alias", use_config_toolsets=False)

    captured = _CapturingAgent.captured
    assert captured["model"] == "real/alias-target"
    expected = resolve_reasoning_config(cfg, "real/alias-target")
    assert captured["reasoning_config"] == expected
    assert expected != resolve_reasoning_config(cfg, "my-alias")


def test_oneshot_reasoning_absent_config_passes_none(monkeypatch):
    """No reasoning config anywhere -> the constructor receives None
    (unchanged default behavior, pinned so the wiring can't invent one)."""
    cfg = {"model": {"default": "test/model", "provider": "openrouter"}}
    _wire_oneshot_stubs(monkeypatch, cfg)

    oneshot._run_agent("hello", use_config_toolsets=False)

    captured = _CapturingAgent.captured
    assert "reasoning_config" in captured
    assert captured["reasoning_config"] == resolve_reasoning_config(cfg, "test/model")
