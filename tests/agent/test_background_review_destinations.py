"""Plugin review destinations in the background review: prompt placement, tool admission, the
action summary, fail-open behaviour, and an E2E run from a real plugin in a temp HERMES_HOME."""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_yaml as yaml
from agent import background_review as bg
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

DESTINATION = "fixture-matters.updates"
TOOL = "fixture_matters_update"
INSTRUCTIONS = "If this session changed an ongoing matter, call fixture_matters_update once."


def _fake_worker(captured: dict):
    # Same shape as the real worker without destinations, so an unconditional new kwarg breaks it.
    def fake_worker(agent, messages_snapshot, prompt, task_cfg=None, review_run=None,
                    review_memory=False, explicit=False, **destination_kwargs):
        captured.update(prompt=prompt, kwargs=destination_kwargs)
    return fake_worker


def _strict_worker(captured: dict):
    def fake_worker(agent, messages_snapshot, prompt, task_cfg=None, review_run=None,
                    review_memory=False, explicit=False):
        captured["prompt"] = prompt
    return fake_worker


@pytest.fixture
def plugin_manager(monkeypatch):
    manager = PluginManager()
    manager._discovered = True
    ctx = PluginContext(PluginManifest(name="fixture_matters", key="fixture_matters", source="user"),
                        manager)
    ctx.register_tool(name=TOOL, toolset="fixture_matters",
                      schema={"name": TOOL, "parameters": {"type": "object", "properties": {}}},
                      handler=lambda args, **kwargs: "{}")
    ctx.register_review_destination(DESTINATION, INSTRUCTIONS, tools=(TOOL,))
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    yield manager
    manager.unload()


def _session_agent(**overrides):
    fields = dict(session_id="sess-1", platform="cli", model="m", provider="p",
                  valid_tool_names={TOOL, "memory", "skill_manage"})
    return SimpleNamespace(**{**fields, **overrides})


def test_without_destinations_the_review_prompt_and_worker_call_are_unchanged(monkeypatch):
    def must_not_render(*_args, **_kwargs):
        raise AssertionError("no destination configured, nothing to render")

    monkeypatch.setattr(plugins_mod, "render_review_destinations", must_not_render)
    captured: dict = {}
    with patch.object(bg, "_run_review_in_thread", _strict_worker(captured)):
        target, prompt = bg.spawn_background_review_thread(
            _session_agent(), [], review_memory=True, task_cfg={"destinations": []})
        target()

    assert prompt == captured["prompt"] == bg._MEMORY_REVIEW_PROMPT


def test_enabled_destination_sits_between_the_prompt_and_the_focus(plugin_manager):
    captured: dict = {}
    with patch.object(bg, "_run_review_in_thread", _fake_worker(captured)):
        target, prompt = bg.spawn_background_review_thread(
            _session_agent(), [], review_memory=True, focus="remember my timezone",
            task_cfg={"destinations": [DESTINATION]})
        target()

    review_prompt = captured["prompt"]
    assert review_prompt.startswith(bg._MEMORY_REVIEW_PROMPT)
    assert (review_prompt.index(bg._MEMORY_REVIEW_PROMPT) < review_prompt.index(INSTRUCTIONS)
            < review_prompt.index("remember my timezone"))
    assert DESTINATION in review_prompt and TOOL in review_prompt
    assert captured["kwargs"] == {"destination_tools": frozenset({TOOL})}
    # The returned prompt is what core built; plugin text is only rendered in the review thread.
    assert INSTRUCTIONS not in prompt


def test_destination_is_skipped_when_its_trigger_or_tool_is_absent(plugin_manager):
    for agent, review_memory in ((_session_agent(), False), (_session_agent(valid_tool_names={"memory"}), True)):
        captured: dict = {}
        with patch.object(bg, "_run_review_in_thread", _fake_worker(captured)):
            target, prompt = bg.spawn_background_review_thread(
                agent, [], review_memory=review_memory, review_skills=not review_memory,
                task_cfg={"destinations": [DESTINATION]})
            target()
        assert captured == {"prompt": prompt, "kwargs": {}}


def test_a_failing_destination_render_reviews_without_it(plugin_manager, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("renderer broke")

    monkeypatch.setattr(plugins_mod, "render_review_destinations", broken)
    captured: dict = {}
    with patch.object(bg, "_run_review_in_thread", _fake_worker(captured)):
        target, prompt = bg.spawn_background_review_thread(
            _session_agent(), [], review_memory=True, task_cfg={"destinations": [DESTINATION]})
        target()

    assert captured == {"prompt": prompt, "kwargs": {}}


def _destination_transcript() -> list[dict]:
    return [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": TOOL, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": json.dumps({"success": True, "message": "Matter updated: reunion dinner"})},
        {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"success": True, "message": "Read 3 lines"})},
    ]


def test_summary_reports_the_destination_tool_message_and_nothing_else():
    for mode in ("on", "verbose"):
        assert bg.summarize_background_review_actions(
            _destination_transcript(), [], notification_mode=mode, extra_notify_tools=frozenset({TOOL}),
        ) == ["Matter updated: reunion dinner"]
    assert bg.summarize_background_review_actions(
        _destination_transcript(), [], notification_mode="off", extra_notify_tools=frozenset({TOOL})) == []
    # Without the destination enabled the same tool result stays out of the summary.
    assert bg.summarize_background_review_actions(_destination_transcript(), []) == []


class _SyncThread:
    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def _write_matters_plugin(hermes_home) -> None:
    plugin_dir = hermes_home / "plugins" / "fixture_matters"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(
        {"name": "fixture_matters", "version": "0.1.0", "description": "review destination fixture"}))
    (plugin_dir / "__init__.py").write_text(
        "import json\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_tool(\n"
        f"        name={TOOL!r},\n"
        "        toolset='fixture_matters',\n"
        f"        schema={{'name': {TOOL!r}, 'parameters': {{'type': 'object', 'properties': {{}}}}}},\n"
        "        handler=lambda args, **kwargs: json.dumps({'success': True, 'message': 'Matter updated'}),\n"
        "    )\n"
        f"    ctx.register_review_destination({DESTINATION!r}, {INSTRUCTIONS!r}, tools=({TOOL!r},))\n"
    )
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["fixture_matters"]},
        "auxiliary": {"background_review": {"destinations": [DESTINATION]}},
    }))


def _parent_agent(agent_cls):
    """The minimal AIAgent surface ``_spawn_background_review`` touches (see the toolset test)."""
    agent = object.__new__(agent_cls)
    agent.model, agent.platform, agent.provider, agent.session_id = "test-model", "test", "openai", "sess-e2e"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled, agent._user_profile_enabled = True, False
    agent._memory_nudge_interval = agent._skill_nudge_interval = 5
    agent.status_callback = None
    agent._cached_system_prompt = None
    agent.session_start = dt.datetime(2026, 1, 1, 12, 0, 0)
    agent.enabled_toolsets, agent.disabled_toolsets = ["memory", "skills", "fixture_matters"], []
    agent.tools = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
                   for name in ("memory", TOOL, "terminal")]
    agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools}
    return agent


def test_e2e_real_plugin_destination_reaches_the_review_fork(tmp_path, monkeypatch):
    import run_agent
    from hermes_cli import config as config_module

    hermes_home = tmp_path / "hermes"
    _write_matters_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "empty-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    assert DESTINATION in manager._review_destinations, manager._plugins["fixture_matters"].error

    captured: dict = {"summaries": []}

    def capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)

    def fork_run(self, *, user_message, **kwargs):
        captured["prompt"] = user_message
        self._session_messages = _destination_transcript()
        return {"final_response": "done"}

    agent = _parent_agent(run_agent.AIAgent)
    agent.background_review_callback = captured["summaries"].append
    agent._safe_print = lambda *args, **kwargs: None
    try:
        with patch.object(run_agent.AIAgent, "__init__", lambda self, *a, **k: None), \
             patch.object(run_agent.AIAgent, "run_conversation", fork_run), \
             patch.object(plugins_mod, "set_thread_tool_whitelist", capture_whitelist), \
             patch("threading.Thread", _SyncThread):
            agent._spawn_background_review(messages_snapshot=[], review_memory=True, review_skills=False)
    finally:
        manager.unload("fixture_matters")

    assert TOOL in captured["whitelist"]
    assert "terminal" not in captured["whitelist"]
    prompt = captured["prompt"]
    assert prompt.index(bg._MEMORY_REVIEW_PROMPT) < prompt.index(INSTRUCTIONS) < prompt.index("You can only call")
    assert f"also allowed: {TOOL}" in prompt
    assert any("Matter updated: reunion dinner" in line for line in captured["summaries"])
    assert not any("Read 3 lines" in line for line in captured["summaries"])
