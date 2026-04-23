"""Tests for the subagent_stop hook event.

Covers wire-up from tools.delegate_tool.delegate_task:
  * fires once per child in both single-task and batch modes
  * runs on the parent thread (no re-entrancy for hook authors)
  * carries child_role when the agent exposes _delegate_role
  * carries child_role=None when _delegate_role is not set (pre-M3)
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import delegate_task
from hermes_cli import plugins


def _make_parent(depth: int = 0, session_id: str = "parent-1"):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._memory_manager = None
    parent.session_id = session_id
    return parent


@pytest.fixture(autouse=True)
def _fresh_plugin_manager():
    """Each test gets a fresh PluginManager so hook callbacks don't
    leak between tests."""
    original = plugins._plugin_manager
    plugins._plugin_manager = plugins.PluginManager()
    yield
    plugins._plugin_manager = original


@pytest.fixture(autouse=True)
def _stub_child_builder(monkeypatch):
    """Replace _build_child_agent with a MagicMock factory so delegate_task
    never transitively imports run_agent / openai.  Keeps the test runnable
    in environments without heavyweight runtime deps installed."""
    def _fake_build_child(task_index, **kwargs):
        child = MagicMock()
        child._delegate_saved_tool_names = []
        child._credential_pool = None
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent", _fake_build_child,
    )


def _register_capturing_hook():
    captured = []

    def _cb(**kwargs):
        kwargs["_thread"] = threading.current_thread()
        captured.append(kwargs)

    mgr = plugins.get_plugin_manager()
    mgr._hooks.setdefault("subagent_stop", []).append(_cb)
    return captured


# ── single-task mode ──────────────────────────────────────────────────────


class TestSingleTask:
    def test_fires_once(self):
        captured = _register_capturing_hook()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = {
                "task_index": 0,
                "status": "completed",
                "summary": "Done!",
                "api_calls": 3,
                "duration_seconds": 5.0,
                "_child_role": "analyst",
            }
            delegate_task(goal="do X", parent_agent=_make_parent())

        assert len(captured) == 1
        payload = captured[0]
        assert payload["child_role"] == "analyst"
        assert payload["child_status"] == "completed"
        assert payload["child_summary"] == "Done!"
        assert payload["duration_ms"] == 5000

    def test_fires_on_parent_thread(self):
        captured = _register_capturing_hook()
        main_thread = threading.current_thread()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "x", "api_calls": 1, "duration_seconds": 0.1,
                "_child_role": None,
            }
            delegate_task(goal="go", parent_agent=_make_parent())

        assert captured[0]["_thread"] is main_thread

    def test_payload_includes_parent_session_id(self):
        captured = _register_capturing_hook()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "x", "api_calls": 1, "duration_seconds": 0.1,
                "_child_role": None,
            }
            delegate_task(
                goal="go",
                parent_agent=_make_parent(session_id="sess-xyz"),
            )

        assert captured[0]["parent_session_id"] == "sess-xyz"


# ── batch mode ────────────────────────────────────────────────────────────


class TestBatchMode:
    def test_fires_per_child(self):
        captured = _register_capturing_hook()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                {"task_index": 0, "status": "completed",
                 "summary": "A", "api_calls": 1, "duration_seconds": 1.0,
                 "_child_role": "role-a"},
                {"task_index": 1, "status": "completed",
                 "summary": "B", "api_calls": 2, "duration_seconds": 2.0,
                 "_child_role": "role-b"},
                {"task_index": 2, "status": "completed",
                 "summary": "C", "api_calls": 3, "duration_seconds": 3.0,
                 "_child_role": "role-c"},
            ]
            delegate_task(
                tasks=[
                    {"goal": "A"}, {"goal": "B"}, {"goal": "C"},
                ],
                parent_agent=_make_parent(),
            )

        assert len(captured) == 3
        roles = sorted(c["child_role"] for c in captured)
        assert roles == ["role-a", "role-b", "role-c"]

    def test_all_fires_on_parent_thread(self):
        captured = _register_capturing_hook()
        main_thread = threading.current_thread()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                {"task_index": 0, "status": "completed",
                 "summary": "A", "api_calls": 1, "duration_seconds": 1.0,
                 "_child_role": None},
                {"task_index": 1, "status": "completed",
                 "summary": "B", "api_calls": 2, "duration_seconds": 2.0,
                 "_child_role": None},
            ]
            delegate_task(
                tasks=[{"goal": "A"}, {"goal": "B"}],
                parent_agent=_make_parent(),
            )

        for payload in captured:
            assert payload["_thread"] is main_thread


# ── payload shape ─────────────────────────────────────────────────────────


class TestPayloadShape:
    def test_role_absent_becomes_none(self):
        captured = _register_capturing_hook()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "x", "api_calls": 1, "duration_seconds": 0.1,
                # Deliberately omit _child_role — pre-M3 shape.
            }
            delegate_task(goal="do X", parent_agent=_make_parent())

        assert captured[0]["child_role"] is None

    def test_result_does_not_leak_child_role_field(self):
        """The internal _child_role key must be stripped before the
        result dict is serialised to JSON."""
        _register_capturing_hook()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "x", "api_calls": 1, "duration_seconds": 0.1,
                "_child_role": "leaf",
            }
            raw = delegate_task(goal="do X", parent_agent=_make_parent())

        parsed = json.loads(raw)
        assert "results" in parsed
        assert "_child_role" not in parsed["results"][0]
