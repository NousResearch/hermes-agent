"""Compose real installed output plugins before persistence or messaging previews."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import plugins


@pytest.fixture
def output_plugins(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin = home / "plugins" / "output-pipeline"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: output-pipeline\nversion: 1.0.0\n")
    (plugin / "__init__.py").write_text('''
def add_footer(response_text):
    return response_text + " secret-footer"

async def redact(response_text, turn_id="", **kwargs):
    return response_text.replace("secret", "REDACTED")

def register(ctx):
    ctx.register_hook("transform_llm_output", add_footer)
    ctx.register_hook("transform_llm_output", redact)
''')
    (home / "config.yaml").write_text("plugins:\n  enabled: [output-pipeline]\n")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: bundled)
    manager = plugins.PluginManager()
    manager.discover_and_load()
    assert len(manager.iter_hook_callbacks("transform_llm_output")) == 2
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("intermediate", ["none", "invalid", "timeout"])
async def test_installed_plugins_compose_in_registration_order(
    output_plugins, asynchronous, intermediate, monkeypatch,
):
    from hermes_cli import lifecycle

    def broken(**kwargs):
        raise SystemExit("broken plugin")

    release = threading.Event()
    finished = threading.Event()

    def stalled(response_text):
        try:
            release.wait(10)
            return "late replacement"
        finally:
            finished.set()

    async def stalled_async(response_text):
        await asyncio.sleep(10)
        return "late replacement"

    hooks = output_plugins._hooks["transform_llm_output"]
    if intermediate == "invalid":
        hooks[1:1] = [lambda **kw: None, lambda **kw: "", lambda **kw: {"bad": True}, broken]
    elif intermediate == "timeout":
        hooks.insert(1, stalled_async if asynchronous else stalled)
        monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 2)
    kwargs = dict(response_text="secret body", turn_id="turn-1")
    try:
        result = (await lifecycle.ainvoke_hook("transform_llm_output", **kwargs)
                  if asynchronous else lifecycle.invoke_hook("transform_llm_output", **kwargs))
        assert result == ["REDACTED body REDACTED-footer"]
    finally:
        release.set()
        if intermediate == "timeout" and not asynchronous:
            assert finished.wait(2)


def test_composed_reply_is_persisted_once_and_replayed(output_plugins, tmp_path):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        with (patch("model_tools.get_tool_definitions", return_value=[]),
              patch("model_tools.check_toolset_requirements", return_value={}),
              patch("agent.process_bootstrap.OpenAI")):
            agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                            model="fake/model", quiet_mode=True, skip_context_files=True,
                            skip_memory=True, platform="cli", session_id="composition", session_db=db)
        agent.client = MagicMock()
        msg = SimpleNamespace(content="secret body", tool_calls=None, reasoning=None)
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15), model="fake/model")
        result = agent.run_conversation("hello")
        expected = "REDACTED body REDACTED-footer"
        assert result["final_response"] == expected
        assert result["pre_transform_response"] == "secret body"
        assert result["response_transformed"] is True
        assert result["messages"][-1]["content"] == expected
        stored = [r["content"] for r in db.get_messages("composition") if r["role"] == "assistant"]
        assert stored == [expected]
        assert "secret" not in json.dumps(stored)
    finally:
        db.close()


@pytest.mark.parametrize("text_streaming", [False, True])
def test_gateway_does_not_expose_raw_preview_interim_or_tts(output_plugins, text_streaming):
    from gateway.config import StreamingConfig
    from gateway.turn_context import TurnContext
    from gateway.run_turn_runner import TurnRunner

    voice = MagicMock()
    ctx = TurnContext(
        streaming_tts_consumer_holder=[voice], user_config={},
        resolve_display_setting=lambda *args: text_streaming,
        interim_assistant_messages_enabled=True,
        source=SimpleNamespace(platform=SimpleNamespace(value="telegram"), chat_id="chat"),
        _run_still_current=lambda: True, _status_adapter=MagicMock(),
    )
    runner = SimpleNamespace(config=SimpleNamespace(streaming=StreamingConfig()),
                             _delivery_adapter_for=MagicMock())
    consumer, delta, interim, want_interim = TurnRunner(runner, ctx)._setup_stream_consumer("telegram")
    if delta:
        delta("secret preview")
    if interim:
        interim("secret commentary")
    assert consumer is None
    assert delta is None
    assert not want_interim
    voice.on_delta.assert_not_called()
    runner._delivery_adapter_for.assert_not_called()
