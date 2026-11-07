"""Workspace prefetch preserves prompt bytes and the session-start snapshot pin."""

import subprocess
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import coding_context, runtime_cwd, workspace_prefetch
from agent.system_prompt import _coding_parts, _workspace_pin_key


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    repo = tmp_path / "workspace"
    repo.mkdir()
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "Test"],
        ["config", "core.autocrlf", "false"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True)
    (repo / "main.py").write_text("print(1)\n")
    subprocess.run(["git", "add", "main.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial workspace"], cwd=repo, check=True)
    config = {"agent": {"coding_context": "auto", "coding_instructions": "Keep changes focused."}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    token = runtime_cwd.set_session_cwd(str(repo))
    try:
        yield repo
    finally:
        runtime_cwd.reset_session_cwd(token)


def _agent(prefetch=None, built_ago=0.0):
    agent = SimpleNamespace(
        platform="tui", model="gpt-4o", valid_tool_names={"terminal", "file_write"},
        _frozen_workspace_snapshot=None, _workspace_snapshot_prefetch=None,
    )
    if prefetch is not None:
        workspace_prefetch.attach_workspace_snapshot_prefetch(agent, prefetch)
        key, future, built_at = agent._workspace_snapshot_prefetch
        agent._workspace_snapshot_prefetch = (key, future, built_at - built_ago)
    return agent


def test_prefetched_workspace_and_coding_parts_are_byte_identical(workspace):
    prefetch = workspace_prefetch.start_workspace_snapshot_prefetch(platform="tui")
    key, future = prefetch
    block = future.result(timeout=15)
    assert key == _workspace_pin_key()
    assert block == coding_context.build_coding_workspace_block(workspace)
    assert "initial workspace" in block
    assert _coding_parts(_agent(prefetch)) == _coding_parts(_agent())


def test_prefetch_is_consumed_without_reprobing_and_pinned(workspace, monkeypatch):
    prefetch = workspace_prefetch.start_workspace_snapshot_prefetch(platform="tui")
    key, future = prefetch
    block = future.result(timeout=15)
    probe = Mock(side_effect=AssertionError("turn 1 must replay the prefetch"))
    monkeypatch.setattr(coding_context, "build_coding_workspace_block", probe)
    agent = _agent(prefetch)

    assert _coding_parts(agent)[1] == [block]
    assert getattr(agent, "_workspace_snapshot_prefetch", None) is None
    assert agent._frozen_workspace_snapshot == (key, block)
    assert _coding_parts(agent)[1] == [block]
    probe.assert_not_called()


def _fail(future):
    future.set_exception(RuntimeError("probe failed"))


def _finish_with_open_time_block(future):
    future.set_result("block from when the session opened")


# failure -> (pin-key suffix, future setup, seconds since the build ended)
_UNUSABLE = {
    "mismatch": ("/other", None, 0.0),  # an unfinished future for another cwd must never be awaited
    "exception": ("", _fail, 0.0),
    "cancelled": ("", Future.cancel, 0.0),
    "queued": ("", None, 0.0),  # still waiting for a pool worker: cancelled, never awaited
    # Opened, then typed into after the build finished: the session-open snapshot may be stale.
    "not_waited_for": ("", _finish_with_open_time_block, workspace_prefetch.WAITING_TURN_WINDOW_S + 1),
}


@pytest.mark.parametrize("failure", list(_UNUSABLE))
def test_unusable_prefetch_is_popped_and_falls_back(workspace, monkeypatch, failure):
    key_suffix, prepare, built_ago = _UNUSABLE[failure]
    future = Future()
    if prepare is not None:
        prepare(future)
    probe = Mock(return_value="fresh workspace block")
    monkeypatch.setattr(coding_context, "build_coding_workspace_block", probe)
    agent = _agent((_workspace_pin_key() + key_suffix, future), built_ago=built_ago)

    assert _coding_parts(agent)[1] == ["fresh workspace block"]
    assert getattr(agent, "_workspace_snapshot_prefetch", None) is None
    assert agent._frozen_workspace_snapshot == (_workspace_pin_key(), "fresh workspace block")
    probe.assert_called_once_with(workspace)


def test_session_without_tools_drops_prefetch(workspace):
    """No snapshot is taken without tools, so a later rebuild must probe fresh, not replay the build-time block."""
    future = Future()
    agent = _agent((_workspace_pin_key(), future))
    agent.valid_tool_names = set()

    assert _coding_parts(agent) == ([], [], [])
    assert getattr(agent, "_workspace_snapshot_prefetch", None) is None
    assert agent._frozen_workspace_snapshot is None


@pytest.mark.parametrize("platform,config", [
    ("telegram", {"agent": {"coding_context": "auto"}}),
    ("tui", {"agent": {"coding_context": "off"}}),
])
def test_non_coding_starts_no_work(workspace, monkeypatch, platform, config):
    submit = Mock(side_effect=AssertionError("non-coding session must not submit work"))
    probe = Mock(side_effect=AssertionError("non-coding session must not probe"))
    monkeypatch.setattr(workspace_prefetch._executor, "submit", submit)
    monkeypatch.setattr(workspace_prefetch, "build_coding_workspace_block", probe)

    assert workspace_prefetch.start_workspace_snapshot_prefetch(platform=platform, config=config) is None
    submit.assert_not_called()
    probe.assert_not_called()


def test_reset_clears_prefetch():
    from run_agent import AIAgent

    agent = _agent(("old-workspace", Future()))
    agent._transition_context_engine_session = lambda **kwargs: None
    AIAgent.reset_session_state(agent)
    assert agent._workspace_snapshot_prefetch is None
    assert agent._frozen_workspace_snapshot is None
