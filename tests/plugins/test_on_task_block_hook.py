"""Tests for the on_task_block plugin hook."""
from __future__ import annotations
from unittest.mock import MagicMock


def _make_board(tmp_path):
    from hermes_cli import kanban_db as kb
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path=db_path)
    with conn:
        tid = kb.create_task(conn, title="test task", assignee="worker")
        kb.claim_task(conn, tid, claimer="worker")
    kb.claim_task(conn, tid, claimer="worker")
    return str(db_path), tid


class TestOnTaskBlockToolHook:
    def test_hook_fires_with_correct_kwargs(self, tmp_path, monkeypatch):
        db_path, tid = _make_board(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_DB", db_path)
        captured = []
        from hermes_cli import plugins as _plugins
        mock_manager = MagicMock()
        mock_manager.invoke_hook.side_effect = lambda name, **kw: captured.append((name, kw))
        monkeypatch.setattr(_plugins, "_plugin_manager", mock_manager)
        from tools import kanban_tools
        kanban_tools._handle_block({"task_id": tid, "reason": "blocked for review"})
        assert any(name == "on_task_block" for name, _ in captured)
        kw = next(kw for name, kw in captured if name == "on_task_block")
        assert kw["task_id"] == tid
        assert kw["reason"] == "blocked for review"
        assert kw["source"] == "tool"

    def test_hook_failure_does_not_propagate(self, tmp_path, monkeypatch):
        db_path, tid = _make_board(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_DB", db_path)
        from hermes_cli import plugins as _plugins
        mock_manager = MagicMock()
        mock_manager.invoke_hook.side_effect = RuntimeError("hook exploded")
        monkeypatch.setattr(_plugins, "_plugin_manager", mock_manager)
        from tools import kanban_tools
        kanban_tools._handle_block({"task_id": tid, "reason": "test"})


class TestOnTaskBlockInValidHooks:
    def test_on_task_block_in_valid_hooks(self):
        from hermes_cli.plugins import VALID_HOOKS
        assert "on_task_block" in VALID_HOOKS

    def test_register_hook_no_warning(self, caplog):
        import logging
        from hermes_cli.plugins import PluginManager, PluginManifest, PluginContext
        manager = PluginManager()
        ctx = PluginContext(PluginManifest(name="test-plugin"), manager)
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            ctx.register_hook("on_task_block", lambda **kw: None)
        assert "unknown hook" not in caplog.text
