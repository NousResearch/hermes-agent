"""Fail-closed delivery-gate contract and turn-finalizer wiring."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

import hermes_cli.plugins as plugins_mod
from agent.turn_finalizer import _apply_output_hooks
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def _make_enabled_plugin(hermes_home: Path, body: str) -> PluginManager:
    plugin_dir = hermes_home / "plugins" / "delivery_gate"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "delivery_gate", "version": "0.1.0"}), encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n" + body, encoding="utf-8"
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["delivery_gate"]}}), encoding="utf-8"
    )
    manager = PluginManager()
    manager.discover_and_load()
    return manager


def _agent():
    return SimpleNamespace(
        session_id="session-1",
        model="test/model",
        platform="telegram",
        _persist_disabled=False,
    )


def _apply(agent, response="unsafe answer"):
    return _apply_output_hooks(
        agent,
        response,
        logging.getLogger(__name__),
        platform="telegram",
        effective_task_id="task-1",
        turn_id="turn-1",
        original_user_message="clinical question",
        messages=[{"role": "user", "content": "clinical question"}],
    )


def test_pre_delivery_is_a_supported_hook():
    assert "pre_delivery" in VALID_HOOKS


def test_pre_delivery_explicit_allow_preserves_response(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = _make_enabled_plugin(
        home,
        '    ctx.register_hook("pre_delivery", lambda **kw: {"action": "allow"})\n',
    )

    assert manager.invoke_hook("pre_delivery", response_text="answer") == [{"action": "allow"}]


def test_pre_delivery_exception_fails_closed(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = _make_enabled_plugin(
        home,
        "    def gate(**kw):\n"
        "        raise RuntimeError('boom')\n"
        '    ctx.register_hook("pre_delivery", gate)\n',
    )

    results = manager.invoke_hook("pre_delivery", response_text="unsafe")

    assert results == [{
        "action": "block",
        "message": plugins_mod.PRE_DELIVERY_FAIL_CLOSED_MESSAGE,
        "reason": "callback_error",
    }]


def test_pre_delivery_none_result_fails_closed(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = _make_enabled_plugin(
        home,
        '    ctx.register_hook("pre_delivery", lambda **kw: None)\n',
    )

    assert manager.invoke_hook("pre_delivery", response_text="unsafe") == [{
        "action": "block",
        "message": plugins_mod.PRE_DELIVERY_FAIL_CLOSED_MESSAGE,
        "reason": "invalid_result",
    }]


def test_turn_finalizer_applies_replacement_before_post_llm_call():
    calls = []

    def invoke(name, _logger, **kwargs):
        calls.append((name, kwargs))
        if name == "pre_delivery":
            return [{"action": "replace", "response_text": "certified answer"}]
        return []

    with patch("agent.turn_finalizer._invoke_hook_safely", side_effect=invoke):
        response, transformed, original = _apply(_agent())

    assert response == "certified answer"
    assert transformed is True
    assert original == "unsafe answer"
    assert [name for name, _ in calls] == [
        "transform_llm_output",
        "pre_delivery",
        "post_llm_call",
    ]
    assert calls[-1][1]["assistant_response"] == "certified answer"


def test_turn_finalizer_block_wins_over_replacement():
    def invoke(name, _logger, **kwargs):
        if name == "pre_delivery":
            return [
                {"action": "replace", "response_text": "candidate"},
                {"action": "block", "message": "safe degradation"},
            ]
        return []

    with patch("agent.turn_finalizer._invoke_hook_safely", side_effect=invoke):
        response, transformed, original = _apply(_agent())

    assert response == ""
    assert transformed is True
    assert original == "unsafe answer"


def test_turn_finalizer_dispatch_exception_fails_closed():
    agent = _agent()

    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        side_effect=RuntimeError("registry unavailable"),
    ):
        response, transformed, original = _apply(agent)

    assert response == ""
    assert transformed is False
    assert original is None


def test_pre_delivery_block_without_message_fails_closed(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = _make_enabled_plugin(
        home,
        '    ctx.register_hook("pre_delivery", lambda **kw: {"action": "block"})\n',
    )

    assert manager.invoke_hook("pre_delivery", response_text="unsafe") == [{
        "action": "block",
        "message": plugins_mod.PRE_DELIVERY_FAIL_CLOSED_MESSAGE,
        "reason": "invalid_result",
    }]


def test_pre_delivery_callback_exception_does_not_log_response(tmp_path, monkeypatch, caplog):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    marker = "PHI_SENTINEL_SHOULD_NOT_BE_LOGGED"
    manager = _make_enabled_plugin(
        home,
        "    def gate(response_text='', **kw):\n"
        "        raise RuntimeError(response_text)\n"
        '    ctx.register_hook("pre_delivery", gate)\n',
    )

    with caplog.at_level(logging.WARNING):
        result = manager.invoke_hook("pre_delivery", response_text=marker)

    assert result[0]["action"] == "block"
    assert marker not in caplog.text


def test_changed_cached_candidate_is_regated_without_retransforming():
    calls = []
    agent = _agent()
    agent._pre_delivery_output_cache = {
        "response_text": "certified answer",
        "transformed": True,
        "pre_transform": "original answer",
    }

    def invoke(name, _logger, **kwargs):
        calls.append((name, kwargs))
        if name == "pre_delivery":
            return [{"action": "replace", "response_text": "safe final answer"}]
        return []

    with patch("agent.turn_finalizer._invoke_hook_safely", side_effect=invoke):
        response, transformed, original = _apply(
            agent, response="certified answer\n\nPHI_SENTINEL_FOOTER"
        )

    assert response == "safe final answer"
    assert transformed is True
    assert original == "original answer"
    assert [name for name, _ in calls] == ["pre_delivery", "post_llm_call"]
