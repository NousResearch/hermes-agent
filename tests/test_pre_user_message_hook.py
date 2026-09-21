"""Tests for the ``pre_user_message`` plugin hook.

The hook fires inside ``run_conversation`` (agent/conversation_loop.py),
after the user message arrives and before it is appended to the message
history. It mirrors ``pre_gateway_dispatch`` for the CLI/TUI surfaces,
supporting the same action dict shape:

  * ``{"action": "rewrite", "text": "..."}`` — replace user_message
  * ``{"action": "skip", "reason": "..."}`` — drop the turn, no LLM call
  * ``{"action": "allow"}`` / ``None`` — pass through

Driving the full conversation loop from a unit test is heavy, so these
tests exercise the dispatch semantics that the call site in
``agent/conversation_loop.py`` depends on. The action-walking logic is
re-implemented inline to keep the contract pinned to a single place.
"""

from pathlib import Path

import yaml

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def _make_enabled_plugin(hermes_home: Path, name: str, register_body: str) -> Path:
    plugin_dir = hermes_home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "0.1.0"}), encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    {register_body}\n",
        encoding="utf-8",
    )
    cfg_path = hermes_home / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg.setdefault("plugins", {}).setdefault("enabled", []).append(name)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return plugin_dir


def _walk(results, user_message):
    """Mirror the action-walking loop in agent/conversation_loop.py."""
    for r in results:
        if not isinstance(r, dict):
            continue
        action = r.get("action")
        if action == "skip":
            return {"skipped": True, "reason": r.get("reason", "")}
        if action == "rewrite":
            new_text = r.get("text", "")
            if isinstance(new_text, str) and new_text.strip():
                return {"skipped": False, "user_message": new_text}
    return {"skipped": False, "user_message": user_message}


def test_pre_user_message_in_valid_hooks():
    assert "pre_user_message" in VALID_HOOKS


def test_hook_receives_expected_kwargs(tmp_path, monkeypatch):
    """Hook callback sees message, session_id, platform, model."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "capture_pum",
        register_body=(
            'ctx.register_hook("pre_user_message", '
            'lambda **kw: {"action": "rewrite", '
            '"text": f"{kw[\'message\']}|{kw[\'session_id\']}|'
            '{kw[\'platform\']}|{kw[\'model\']}"})'
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "pre_user_message",
        message="hello",
        session_id="s1",
        platform="cli",
        model="anthropic/claude-sonnet-4.6",
    )
    assert results == [{
        "action": "rewrite",
        "text": "hello|s1|cli|anthropic/claude-sonnet-4.6",
    }]


def test_rewrite_replaces_user_message():
    """First rewrite action wins; subsequent results ignored."""
    hook_returns = [
        None,
        {"action": "allow"},
        {"action": "rewrite", "text": "tightened prompt"},
        {"action": "rewrite", "text": "second-loser"},
    ]
    out = _walk(hook_returns, "original verbose prompt")
    assert out == {"skipped": False, "user_message": "tightened prompt"}


def test_skip_drops_the_turn():
    hook_returns = [{"action": "skip", "reason": "duplicate"}]
    out = _walk(hook_returns, "anything")
    assert out == {"skipped": True, "reason": "duplicate"}


def test_skip_wins_over_later_rewrite():
    """First non-None action wins — skip stops the walk before rewrite."""
    hook_returns = [
        {"action": "skip", "reason": "blocked"},
        {"action": "rewrite", "text": "should not apply"},
    ]
    out = _walk(hook_returns, "anything")
    assert out == {"skipped": True, "reason": "blocked"}


def test_none_and_allow_pass_through():
    hook_returns = [None, {"action": "allow"}, "garbage", 42]
    out = _walk(hook_returns, "kept as-is")
    assert out == {"skipped": False, "user_message": "kept as-is"}


def test_empty_rewrite_text_passes_through():
    """A rewrite with empty/whitespace text must not blank the prompt."""
    hook_returns = [{"action": "rewrite", "text": "   "}]
    out = _walk(hook_returns, "kept")
    assert out == {"skipped": False, "user_message": "kept"}


def test_hook_exception_does_not_crash_dispatch(tmp_path, monkeypatch):
    """A raising plugin contributes nothing; the walk finds no rewrite."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "raising_pum",
        register_body=(
            'def _boom(**kw):\n'
            '        raise RuntimeError("boom")\n'
            '    ctx.register_hook("pre_user_message", _boom)'
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "pre_user_message",
        message="survives",
        session_id="s1",
        platform="cli",
        model="m",
    )
    out = _walk(results, "survives")
    assert out == {"skipped": False, "user_message": "survives"}


def test_no_plugins_returns_empty_results(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_empty"))
    plugins_mod._plugin_manager = PluginManager()

    mgr = plugins_mod._plugin_manager
    results = mgr.invoke_hook(
        "pre_user_message",
        message="m", session_id="", platform="cli", model="",
    )
    assert results == []
