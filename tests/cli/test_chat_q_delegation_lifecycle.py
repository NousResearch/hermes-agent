"""Process-level regression coverage for chat -q delegation lifecycle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_single_query_delegation_probe(tmp_path: Path) -> dict:
    """Run cli.main(query=...) in a fresh interpreter with a fake CLI/agent.

    The subprocess boundary is intentional: this catches regressions where the
    one-shot CLI process would dispatch a background delegation handle and then
    exit before any later completion could be drained.
    """
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    script = textwrap.dedent(
        r'''
        import json
        import threading
        from types import SimpleNamespace

        import cli as cli_mod
        import tools.delegate_tool as delegate_tool
        from gateway.session_context import reset_session_vars
        from tools import async_delegation

        observed = {}

        class FakeChild:
            _delegate_role = "leaf"
            _subagent_id = "child-1"
            session_id = "child-session"
            model = "test-model"
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_reasoning_tokens = 0
            session_estimated_cost_usd = 0.0
            tool_progress_callback = None

            def close(self):
                pass

        class FakeAgent:
            def __init__(self):
                self._delegate_depth = 0
                self._interrupt_requested = False
                self._active_children = []
                self._active_children_lock = threading.Lock()
                self._memory_manager = None
                self.context_compressor = None
                self.enabled_toolsets = []
                self.valid_tool_names = []
                self.model = "test-model"
                self.provider = "test-provider"
                self.session_id = "single-query-parent"
                self.session_prompt_tokens = 0
                self.session_estimated_cost_usd = 0.0
                self.session_cost_source = "none"
                self.session_cost_status = "unknown"

        class FakeCLI:
            def __init__(self, **_kwargs):
                self.console = SimpleNamespace(print=lambda *_a, **_kw: None)
                self.session_id = "single-query-parent"
                self.agent = FakeAgent()

            def _claim_active_session(self, surface, *, stderr=False):
                return True

            def _show_security_advisories(self):
                pass

            def _print_exit_summary(self, clear_screen=True):
                pass

            def chat(self, query, images=None):
                payload = delegate_tool.delegate_task(
                    goal="child work that the one-shot process cannot deliver later",
                    context="process-level regression probe",
                    background=True,
                    parent_agent=self.agent,
                )
                observed["payload"] = json.loads(payload)
                return "done"

        def fake_build_child_agent(**_kwargs):
            return FakeChild()

        def fake_run_single_child(task_index, goal, child=None, parent_agent=None, **_kwargs):
            return {
                "task_index": task_index,
                "status": "completed",
                "summary": "inline child result",
                "api_calls": 0,
                "duration_seconds": 0.0,
                "model": "test-model",
                "exit_reason": "completed",
            }

        delegate_tool._build_child_agent = fake_build_child_agent
        delegate_tool._run_single_child = fake_run_single_child
        delegate_tool._resolve_delegation_credentials = lambda _cfg, _parent: {
            "model": "test-model",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": None,
        }
        cli_mod.HermesCLI = FakeCLI
        cli_mod.atexit.register = lambda *_a, **_kw: None
        cli_mod._finalize_single_query = lambda _cli: None

        reset_session_vars()
        cli_mod.main(query="delegate in one-shot mode", quiet=False, toolsets="delegation")

        with async_delegation._DB_LOCK, async_delegation._connect() as conn:
            durable_rows = conn.execute(
                "SELECT delegation_id, state, delivery_state, delivery_attempts "
                "FROM async_delegations ORDER BY delegation_id"
            ).fetchall()
        print(json.dumps({
            "delegate_payload": observed["payload"],
            "durable_rows": durable_rows,
            "active_async_delegations": async_delegation.active_count(),
        }, sort_keys=True))
        '''
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HERMES_KANBAN")
    }
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(
            "single-query delegation probe failed "
            f"(rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    pytest.fail(f"probe did not print JSON payload\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def test_chat_q_process_runs_background_delegation_inline(tmp_path):
    """A one-shot chat -q process must not dispatch an undeliverable child.

    Top-level model delegation normally forces background dispatch. In the
    single-query CLI process, that would hand back a handle and then terminate
    before the completion queue can be drained. The live process path must bind
    stateless delivery before chat starts, making delegate_task run inline.
    """
    probe = _run_single_query_delegation_probe(tmp_path)

    payload = probe["delegate_payload"]
    assert payload.get("status") != "dispatched"
    assert payload["results"][0]["summary"] == "inline child result"
    assert "ran SYNCHRONOUSLY" in payload["note"]
    assert probe["durable_rows"] == []
    assert probe["active_async_delegations"] == 0
