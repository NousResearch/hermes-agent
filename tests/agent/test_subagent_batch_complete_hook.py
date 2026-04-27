"""Tests for the subagent_batch_complete hook event.

Covers aggregate firing from tools.delegate_tool.delegate_task:
  * fires once per delegate_task invocation, single mode AND batch mode
  * five counters partition the full status keyspace and sum to child_count
    (completed / failed / errored / interrupted / timeout)
  * fires after every per-child subagent_stop
  * fires on the parent thread
  * carries role, task_index, status, duration, summary for each child
  * total_duration_ms matches the returned total_duration_seconds
  * the helper field _child_role is stripped from the tool result
  * shell hooks registered on the event receive the extra.* nested payload
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from agent import shell_hooks
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
    original = plugins._plugin_manager
    plugins._plugin_manager = plugins.PluginManager()
    yield
    plugins._plugin_manager = original


@pytest.fixture(autouse=True)
def _stub_child_builder(monkeypatch):
    def _fake_build_child(task_index, **kwargs):
        child = MagicMock()
        child._delegate_saved_tool_names = []
        child._credential_pool = None
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent", _fake_build_child,
    )


def _register_capturing_hook(event: str):
    captured = []

    def _cb(**kwargs):
        kwargs["_thread"] = threading.current_thread()
        captured.append(kwargs)

    mgr = plugins.get_plugin_manager()
    mgr._hooks.setdefault(event, []).append(_cb)
    return captured


# ── single-task vs batch firing ──────────────────────────────────────────


class TestFiringShape:
    def test_fires_once_per_delegate_task_single_mode(self):
        captured = _register_capturing_hook("subagent_batch_complete")

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = {
                "task_index": 0,
                "status": "completed",
                "summary": "Single task result",
                "api_calls": 2,
                "duration_seconds": 1.25,
                "_child_role": None,
            }
            delegate_task(goal="one-shot task", parent_agent=_make_parent())

        assert len(captured) == 1
        payload = captured[0]
        assert payload["child_count"] == 1
        assert payload["completed_count"] == 1
        assert payload["failed_count"] == 0
        assert payload["errored_count"] == 0
        assert payload["interrupted_count"] == 0
        assert payload["timeout_count"] == 0
        assert payload["parent_session_id"] == "parent-1"
        assert len(payload["children"]) == 1
        assert payload["children"][0]["status"] == "completed"

    def test_fires_once_per_delegate_task_batch_mode(self):
        captured = _register_capturing_hook("subagent_batch_complete")

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                {"task_index": 0, "status": "completed", "summary": "A",
                 "api_calls": 1, "duration_seconds": 1.0, "_child_role": None},
                {"task_index": 1, "status": "completed", "summary": "B",
                 "api_calls": 2, "duration_seconds": 2.0, "_child_role": None},
                {"task_index": 2, "status": "completed", "summary": "C",
                 "api_calls": 3, "duration_seconds": 3.0, "_child_role": None},
            ]
            delegate_task(
                tasks=[{"goal": "A"}, {"goal": "B"}, {"goal": "C"}],
                parent_agent=_make_parent(),
            )

        assert len(captured) == 1
        payload = captured[0]
        assert payload["child_count"] == 3
        assert payload["completed_count"] == 3
        assert payload["failed_count"] == 0
        assert payload["errored_count"] == 0
        assert payload["interrupted_count"] == 0
        assert payload["timeout_count"] == 0
        assert len(payload["children"]) == 3


# ── counter partitioning ─────────────────────────────────────────────────


class TestCounters:
    def test_counters_reflect_mixed_outcomes(self, monkeypatch):
        """Batch with one child per known status.  The five counters must
        partition the set exactly (no 'unknown' bucket).  Raises the
        per-batch cap since the default max_concurrent_children (3) is
        lower than the four distinct statuses we need to exercise."""
        monkeypatch.setattr(
            "tools.delegate_tool._get_max_concurrent_children", lambda: 10,
        )

        captured = _register_capturing_hook("subagent_batch_complete")

        side_effects = [
            {"task_index": 0, "status": "completed", "summary": "done",
             "api_calls": 1, "duration_seconds": 1.0, "_child_role": None},
            {"task_index": 1, "status": "failed", "summary": None,
             "api_calls": 5, "duration_seconds": 5.0, "_child_role": None},
            {"task_index": 2, "status": "error", "summary": None,
             "api_calls": 0, "duration_seconds": 0.1, "_child_role": None},
            {"task_index": 3, "status": "interrupted", "summary": None,
             "api_calls": 1, "duration_seconds": 0.5, "_child_role": None},
        ]

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = side_effects
            delegate_task(
                tasks=[{"goal": f"T{i}"} for i in range(4)],
                parent_agent=_make_parent(),
            )

        assert len(captured) == 1
        payload = captured[0]
        assert payload["completed_count"] == 1
        assert payload["failed_count"] == 1
        assert payload["errored_count"] == 1
        assert payload["interrupted_count"] == 1
        assert payload["timeout_count"] == 0
        assert payload["child_count"] == 4
        # Invariant: the five counters partition the keyspace exactly.
        assert (
            payload["completed_count"]
            + payload["failed_count"]
            + payload["errored_count"]
            + payload["interrupted_count"]
            + payload["timeout_count"]
        ) == payload["child_count"]

    def test_timeout_counter_pinned(self):
        """Counts a 'timeout' status emitted by _run_single_child's
        FuturesTimeoutError branch, ensuring the fifth counter is wired."""
        captured = _register_capturing_hook("subagent_batch_complete")

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                {"task_index": 0, "status": "timeout", "summary": None,
                 "api_calls": 0, "duration_seconds": 60.0, "_child_role": None},
                {"task_index": 1, "status": "completed", "summary": "ok",
                 "api_calls": 1, "duration_seconds": 1.0, "_child_role": None},
            ]
            delegate_task(
                tasks=[{"goal": "slow"}, {"goal": "fast"}],
                parent_agent=_make_parent(),
            )

        payload = captured[0]
        assert payload["timeout_count"] == 1
        assert payload["completed_count"] == 1
        assert payload["child_count"] == 2


# ── ordering and threading ───────────────────────────────────────────────


class TestOrdering:
    def test_fires_after_all_subagent_stop_firings(self):
        order = []

        def _stop_cb(**kwargs):
            order.append(("stop", kwargs.get("child_summary")))

        def _batch_cb(**kwargs):
            order.append(("batch", kwargs.get("child_count")))

        mgr = plugins.get_plugin_manager()
        mgr._hooks.setdefault("subagent_stop", []).append(_stop_cb)
        mgr._hooks.setdefault("subagent_batch_complete", []).append(_batch_cb)

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                {"task_index": 0, "status": "completed", "summary": "A",
                 "api_calls": 1, "duration_seconds": 1.0, "_child_role": None},
                {"task_index": 1, "status": "completed", "summary": "B",
                 "api_calls": 2, "duration_seconds": 2.0, "_child_role": None},
            ]
            delegate_task(
                tasks=[{"goal": "A"}, {"goal": "B"}],
                parent_agent=_make_parent(),
            )

        batch_indices = [i for i, (kind, _) in enumerate(order) if kind == "batch"]
        stop_indices = [i for i, (kind, _) in enumerate(order) if kind == "stop"]
        assert len(batch_indices) == 1
        assert len(stop_indices) == 2
        assert max(stop_indices) < batch_indices[0]

    def test_fires_on_parent_thread(self):
        captured = _register_capturing_hook("subagent_batch_complete")
        caller_thread = threading.current_thread()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                {"task_index": 0, "status": "completed", "summary": "A",
                 "api_calls": 1, "duration_seconds": 1.0, "_child_role": None},
                {"task_index": 1, "status": "completed", "summary": "B",
                 "api_calls": 2, "duration_seconds": 2.0, "_child_role": None},
            ]
            delegate_task(
                tasks=[{"goal": "A"}, {"goal": "B"}],
                parent_agent=_make_parent(),
            )

        assert captured[0]["_thread"] is caller_thread


# ── payload shape ────────────────────────────────────────────────────────


class TestPayload:
    def test_role_preserved_per_child(self):
        captured = _register_capturing_hook("subagent_batch_complete")

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                {"task_index": 0, "status": "completed", "summary": "A",
                 "api_calls": 1, "duration_seconds": 1.0,
                 "_child_role": "orchestrator"},
                {"task_index": 1, "status": "completed", "summary": "B",
                 "api_calls": 2, "duration_seconds": 2.0,
                 "_child_role": "leaf"},
            ]
            delegate_task(
                tasks=[{"goal": "A"}, {"goal": "B"}],
                parent_agent=_make_parent(),
            )

        children = captured[0]["children"]
        children_by_idx = {c["task_index"]: c for c in children}
        assert children_by_idx[0]["role"] == "orchestrator"
        assert children_by_idx[1]["role"] == "leaf"

    def test_child_entries_have_documented_keys(self):
        captured = _register_capturing_hook("subagent_batch_complete")

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "result", "api_calls": 1,
                "duration_seconds": 1.25, "_child_role": None,
            }
            delegate_task(goal="solo", parent_agent=_make_parent())

        child = captured[0]["children"][0]
        assert set(child.keys()) == {
            "task_index", "role", "status", "duration_ms", "summary",
        }
        assert child["task_index"] == 0
        assert child["status"] == "completed"
        assert child["duration_ms"] == 1250
        assert child["summary"] == "result"

    def test_payload_strips_helper_field_from_tool_result(self):
        """The internal ``_child_role`` helper must be stripped from
        every result entry before ``delegate_task`` serialises the
        tool result — otherwise it leaks into the JSON returned to the
        model."""
        _register_capturing_hook("subagent_batch_complete")

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                {"task_index": 0, "status": "completed", "summary": "A",
                 "api_calls": 1, "duration_seconds": 1.0,
                 "_child_role": "leaf"},
                {"task_index": 1, "status": "completed", "summary": "B",
                 "api_calls": 2, "duration_seconds": 2.0,
                 "_child_role": "leaf"},
            ]
            raw = delegate_task(
                tasks=[{"goal": "A"}, {"goal": "B"}],
                parent_agent=_make_parent(),
            )

        parsed = json.loads(raw)
        assert "_child_role" not in json.dumps(parsed), (
            "_child_role leaked into the serialised tool result; the "
            "strip pass after subagent_batch_complete is broken."
        )

    def test_total_duration_matches_returned_json(self):
        captured = _register_capturing_hook("subagent_batch_complete")

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "done", "api_calls": 1,
                "duration_seconds": 0.5, "_child_role": None,
            }
            raw = delegate_task(goal="go", parent_agent=_make_parent())

        returned = json.loads(raw)
        expected_ms = int(returned["total_duration_seconds"] * 1000)
        assert captured[0]["total_duration_ms"] == expected_ms


# ── shell-hook payload shape ────────────────────────────────────────────


class TestShellHookPayload:
    def test_shell_hook_receives_batch_payload_with_extra_nesting(
        self, tmp_path, monkeypatch,
    ):
        """End-to-end: a shell hook registered on subagent_batch_complete
        gets a stdin JSON whose extra.* fields match the documented shape."""
        capture = tmp_path / "payload.json"
        script = tmp_path / "cap.sh"
        script.write_text(
            f"#!/usr/bin/env bash\ncat - > {capture}\nprintf '{{}}\\n'\n",
        )
        script.chmod(0o755)

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        # Register the shell hook directly with the plugin manager,
        # bypassing the allowlist flow — the registration path is covered
        # by test_shell_hooks.py already.  Here we just need the bridge
        # to serialize the real payload.
        spec = shell_hooks.ShellHookSpec(
            event="subagent_batch_complete",
            command=str(script),
            timeout=5,
        )
        mgr = plugins.get_plugin_manager()
        mgr._hooks.setdefault(
            "subagent_batch_complete", [],
        ).append(shell_hooks._make_callback(spec))

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                {"task_index": 0, "status": "completed", "summary": "A",
                 "api_calls": 1, "duration_seconds": 1.0, "_child_role": None},
                {"task_index": 1, "status": "failed", "summary": None,
                 "api_calls": 0, "duration_seconds": 0.5, "_child_role": None},
            ]
            delegate_task(
                tasks=[{"goal": "A"}, {"goal": "B"}],
                parent_agent=_make_parent(session_id="sess-1234"),
            )

        payload = json.loads(capture.read_text())
        assert payload["hook_event_name"] == "subagent_batch_complete"
        assert payload["tool_name"] is None
        assert payload["tool_input"] is None
        assert payload["session_id"] == "sess-1234"
        extra = payload["extra"]
        assert extra["child_count"] == 2
        assert extra["completed_count"] == 1
        assert extra["failed_count"] == 1
        assert extra["errored_count"] == 0
        assert extra["interrupted_count"] == 0
        assert extra["timeout_count"] == 0
        # All counters sum to child_count.
        assert (
            extra["completed_count"] + extra["failed_count"]
            + extra["errored_count"] + extra["interrupted_count"]
            + extra["timeout_count"]
        ) == extra["child_count"]
        assert isinstance(extra["total_duration_ms"], int)
        assert extra["total_duration_ms"] >= 0
        assert isinstance(extra["children"], list) and len(extra["children"]) == 2
        for child in extra["children"]:
            assert set(child.keys()) == {
                "task_index", "role", "status", "duration_ms", "summary",
            }


# ── default payload coverage ────────────────────────────────────────────


class TestDefaultPayloadsEntry:
    def test_entry_present(self):
        """hermes hooks test subagent_batch_complete needs a synthetic
        payload that matches the runtime shape exactly."""
        from hermes_cli.hooks import _DEFAULT_PAYLOADS

        assert "subagent_batch_complete" in _DEFAULT_PAYLOADS
        payload = _DEFAULT_PAYLOADS["subagent_batch_complete"]
        # Same five-counter invariant as the live payload.
        total = (
            payload["completed_count"]
            + payload["failed_count"]
            + payload["errored_count"]
            + payload["interrupted_count"]
            + payload["timeout_count"]
        )
        assert total == payload["child_count"]
        assert isinstance(payload["children"], list)
        assert len(payload["children"]) == payload["child_count"]
