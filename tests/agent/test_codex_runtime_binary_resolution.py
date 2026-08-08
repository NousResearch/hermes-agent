"""Regression test for the Codex-binary PATH-shadowing bug.

``run_codex_app_server_turn`` always spawned the bare string "codex" for the
app-server subprocess (CodexAppServerSession/CodexAppServerClient both default
``codex_bin`` to the literal "codex"). On a machine with multiple Codex CLI
installs, OS PATH resolution can land on an incomplete install missing the
Windows sandbox helper (codex-windows-sandbox-setup.exe), so workspace-write
turns fail with orchestrator_helper_launch_failed even though a known-good
install exists and CODEX_CLI_PATH already names it.

Fix: the spawn site must call resolve_codex_binary() and pass its result as
codex_bin instead of relying on the bare-string default.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.codex_runtime import run_codex_app_server_turn


def _make_turn():
    return SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[],
        tool_iterations=0,
        final_text="ok",
        should_retire=False,
    )


def _make_spawning_agent():
    """Mirrors tests/agent/test_codex_app_server_persist.py::_make_agent,
    but leaves _codex_session unset so the spawn branch (codex_runtime.py,
    ``if not hasattr(agent, "_codex_session") or agent._codex_session is
    None``) actually runs instead of being skipped."""
    agent = MagicMock()
    agent._codex_session = None
    agent.session_cwd = "/tmp/fake-cwd"
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._session_db_created = True
    agent.session_id = "sess-binary-resolution"
    return agent


class FakeCodexAppServerSession:
    """Captures the kwargs CodexAppServerSession is constructed with, without
    spawning a real subprocess."""

    captured: dict = {}

    def __init__(self, **kwargs):
        type(self).captured = kwargs

    def run_turn(self, user_input):
        return _make_turn()


@pytest.fixture(autouse=True)
def _patch_codex_app_server_session(monkeypatch):
    import agent.transports.codex_app_server_session as session_module

    FakeCodexAppServerSession.captured = {}
    monkeypatch.setattr(
        session_module, "CodexAppServerSession", FakeCodexAppServerSession
    )
    yield


def test_spawn_passes_resolved_codex_binary_from_codex_cli_path(
    monkeypatch, tmp_path
):
    """The exact bug: codex_bin must come from resolve_codex_binary(), not
    default to the bare "codex" PATH lookup, when CODEX_CLI_PATH is set."""
    resolved = tmp_path / "known-good-codex.exe"
    resolved.write_text("")
    monkeypatch.setenv("CODEX_CLI_PATH", str(resolved))

    agent = _make_spawning_agent()
    run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="task-binary-resolution",
    )

    codex_bin = FakeCodexAppServerSession.captured.get("codex_bin")
    assert codex_bin == str(resolved), (
        f"expected codex_bin={str(resolved)!r} (from CODEX_CLI_PATH), got "
        f"{codex_bin!r} — spawn must not fall back to the bare 'codex' PATH "
        "lookup when CODEX_CLI_PATH is set"
    )
    assert codex_bin != "codex"
