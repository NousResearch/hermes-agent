"""Smoke tests for the user-installed Langfuse plugin (when present).

The canonical plugin lives under ``~/.hermes/plugins/langfuse_tracing/``.
These tests are skipped in CI unless that directory exists locally.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PLUGIN_INIT = Path.home() / ".hermes" / "plugins" / "langfuse_tracing" / "__init__.py"

needs_user_plugin = pytest.mark.skipif(
    not PLUGIN_INIT.is_file(),
    reason="langfuse_tracing plugin not installed at ~/.hermes/plugins/langfuse_tracing/",
)


def _load_user_plugin():
    name = "langfuse_tracing_user_plugin"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PLUGIN_INIT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load langfuse plugin")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@needs_user_plugin
def test_langfuse_plugin_registers_api_request_hooks():
    mod = _load_user_plugin()
    ctx = MagicMock()
    ctx.manifest.name = "langfuse_tracing"
    mod.register(ctx)
    registered = [c[0][0] for c in ctx.register_hook.call_args_list]
    assert "pre_api_request" in registered
    assert "post_api_request" in registered
    assert "pre_llm_call" in registered


@needs_user_plugin
def test_pre_post_api_request_smoke_with_mock_langfuse():
    mod = _load_user_plugin()
    mod._TRACE_STATE.clear()

    gen_obs = MagicMock()
    root_obs = MagicMock()
    root_obs.start_observation.return_value = gen_obs

    client = MagicMock()
    client.create_trace_id.return_value = "trace-smoke-test"
    client.start_observation.return_value = root_obs

    with patch.object(mod, "_get_langfuse", return_value=client):
        mod.on_pre_api_request(
            task_id="t1",
            session_id="s1",
            platform="cli",
            model="test/model",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            api_call_count=1,
            message_count=3,
            tool_count=5,
            approx_input_tokens=100,
            request_char_count=400,
            max_tokens=4096,
        )
        mod.on_post_api_request(
            task_id="t1",
            session_id="s1",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            model="test/model",
            api_call_count=1,
            api_duration=0.05,
            finish_reason="stop",
            usage={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
            assistant_content_chars=42,
            assistant_tool_call_count=0,
            response_model="test/model",
        )

    gen_obs.update.assert_called()
    gen_obs.end.assert_called()
