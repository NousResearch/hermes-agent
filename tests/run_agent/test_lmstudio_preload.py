"""Tests for local-model preload gating (``model.preload``).

Opening or switching a session must never force-load a model on a local
server (it would evict / OOM the model another session is using). Loading is
lazy — it happens only on a real request — and can be disabled entirely with
``model.preload: false`` so the server's own JIT/lazy loading manages the model.

The OpenAI client and tool loading are mocked so no network calls are made.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _build_agent(model_cfg):
    """Construct an AIAgent with a patched config ``model`` section."""
    cfg = {"agent": {}, "model": model_cfg}
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("terminal"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
    ):
        a = AIAgent(
            model="openai/gpt-4o-mini",
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def _fake_agent(provider="lmstudio", preload=True, **overrides):
    """Minimal object exposing only what _ensure_lmstudio_runtime_loaded reads."""
    base = dict(
        provider=provider,
        _model_preload=preload,
        model="local/test-model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="",
        _config_context_length=None,
        context_compressor=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── Agent build / config parsing ──────────────────────────────────────────


def test_build_invokes_preload_hook():
    """Eager mode (default): agent build invokes the preload hook.

    The hook itself is a no-op off-provider or in lazy mode (see the gating
    tests below); this just asserts the eager call site is wired on build.
    """
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("terminal"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value={"agent": {}, "model": {}}),
        patch.object(AIAgent, "_ensure_lmstudio_runtime_loaded") as spy,
    ):
        AIAgent(
            model="openai/gpt-4o-mini",
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        spy.assert_called()


def test_model_preload_defaults_true():
    agent = _build_agent(model_cfg={})
    assert agent._model_preload is True


def test_model_preload_reads_false_from_config():
    agent = _build_agent(model_cfg={"preload": False})
    assert agent._model_preload is False


# ── Lazy load gating (the chat-completion path) ───────────────────────────


def test_preload_disabled_skips_load():
    with patch("hermes_cli.models.ensure_lmstudio_model_loaded") as load:
        AIAgent._ensure_lmstudio_runtime_loaded(_fake_agent(preload=False))
        load.assert_not_called()


def test_preload_enabled_loads_lmstudio():
    with patch(
        "hermes_cli.models.ensure_lmstudio_model_loaded", return_value=64000
    ) as load:
        AIAgent._ensure_lmstudio_runtime_loaded(_fake_agent(preload=True))
        load.assert_called_once()


def test_non_lmstudio_provider_never_loads():
    with patch("hermes_cli.models.ensure_lmstudio_model_loaded") as load:
        AIAgent._ensure_lmstudio_runtime_loaded(
            _fake_agent(provider="openai", preload=True)
        )
        load.assert_not_called()


def test_missing_flag_defaults_to_load():
    """Agents predating the flag (no _model_preload attribute) still load."""
    fake = _fake_agent(preload=True)
    del fake._model_preload
    with patch(
        "hermes_cli.models.ensure_lmstudio_model_loaded", return_value=64000
    ) as load:
        AIAgent._ensure_lmstudio_runtime_loaded(fake)
        load.assert_called_once()
