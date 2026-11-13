"""A gateway-owned session opened in a linked git worktree gets the isolated-worktree guidance.

Regression for #134889 (CLI-4): the classic CLI appended its ``hermes -w`` worktree note to its
own system prompt, which the gateway never reads. The gateway now derives the note from the
session's frozen cwd, so every local surface opened in a worktree carries it.
"""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_NOSYSTEM": "1"}


def _git(cwd, *args):
    env = {**_GIT_ENV, "PATH": os.environ.get("PATH", ""), "HOME": str(cwd)}
    subprocess.run(["git", "-C", str(cwd), *args], check=True, env=env, capture_output=True)


@pytest.fixture
def worktree(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "main.py").write_text("print('hi')\n")
    _git(primary, "init", "-q", "-b", "main")
    _git(primary, "config", "core.autocrlf", "false")
    _git(primary, "add", "-A")
    _git(primary, "commit", "-q", "-m", "init")
    linked = tmp_path / "linked"
    _git(primary, "worktree", "add", "-q", "-b", "hermes/wt-feature", str(linked))
    return primary, linked


def _gateway_prompt(source, cwd):
    """System prompt a gateway turn builds for a local session frozen at *cwd*."""
    from agent.system_prompt import build_system_prompt
    from gateway.session_policy import build_policy, policy_scope

    policy = build_policy({"source": source, "cwd": str(cwd), "model": "m"}, {})
    agent = SimpleNamespace(
        _cached_system_prompt=None, _cached_system_prompt_static=None, _memory_store=None,
        _memory_manager=None, provider="", model="gpt-4o", platform=policy.platform,
        load_soul_identity=False, skip_context_files=True, valid_tool_names={"terminal"},
        _memory_enabled=False, _user_profile_enabled=False, _task_completion_guidance=False,
        _parallel_tool_call_guidance=False, _tool_use_enforcement=False, _execution_guidance=False,
        _environment_probe=False, _bot_mode_protocol=False, _kanban_worker_guidance="",
        pass_session_id=False, session_id="s1", _emit_status=lambda *a, **k: None,
    )
    with patch("agent.prompt_builder.load_soul_md", return_value=""), \
            patch("agent.prompt_builder.build_environment_hints", return_value=""), \
            policy_scope(policy):
        return build_system_prompt(agent)


@pytest.mark.parametrize("source", ["cli", "tui", "gui", "acp"])
def test_gateway_session_in_linked_worktree_carries_isolation_guidance(worktree, source):
    primary, linked = worktree
    prompt = _gateway_prompt(source, linked)

    note = next(line for line in prompt.splitlines() if line.startswith("- Worktree: linked"))
    assert "isolated on branch hermes/wt-feature" in note
    assert "do not affect the main working tree or other agents" in note
    assert "commit and push" in note and "PR" in note
    # The primary tree path stays out of the prompt: a second absolute path makes the model
    # run commands in the wrong directory.
    assert str(primary) not in prompt and str(primary.resolve()) not in prompt


def test_gateway_session_in_primary_tree_gets_no_worktree_note(worktree):
    primary, _ = worktree
    prompt = _gateway_prompt("cli", primary)
    assert "Workspace (snapshot at session start" in prompt
    assert "- Worktree:" not in prompt
