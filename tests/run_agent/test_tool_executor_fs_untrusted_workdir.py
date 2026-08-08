"""Regression test for PR #57712 review feedback (teknium1 / hermes-sweeper).

``terminal`` accepts a per-call ``workdir`` that overrides the session cwd
for that command (``tools.terminal_tool._resolve_command_cwd``: "workdir=
must still override everything"). The filesystem-provenance tracking added
in #57712 (``AIAgent._is_fs_tool_result_untrusted`` /
``_extract_fetch_roots``) must resolve relative clone/download destinations
against that SAME effective cwd — otherwise a ``git clone ... repo`` issued
with an explicit ``workdir`` records its untrusted root relative to the
wrong directory, and a later read of that same clone (via ``terminal`` or
``read_file``) never gets the ``<untrusted_tool_result>`` wrapper even
though its content is exactly as attacker-controllable as a
``web_extract`` result.

This exercises the real sequential dispatch surface
(``agent._execute_tool_calls_sequential`` -> ``agent/tool_executor.py``),
mocking only ``run_agent.handle_function_call`` for determinism (no real
subprocess), matching the pattern in
``test_tool_call_incremental_persistence.py``.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("terminal"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_tool_call(name, arguments, call_id):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_terminal_provenance_uses_effective_workdir_not_session_cwd():
    """A clone issued with an explicit `workdir=` must record its fetch root
    relative to that workdir, not the session's `env.cwd` -- otherwise a
    later `read_file` on the clone's real (absolute) location never matches.

    Before the fix, ``tool_executor.py`` computed the fs-provenance cwd from
    ``get_active_env(task_id).cwd`` only, ignoring ``args["workdir"]``. With
    no active env registered for the fake task id (the common case in these
    unit tests, and equally possible for real short-lived tool calls), that
    fell back to ``os.getcwd()`` -- the wrong directory whenever a real
    workdir override was supplied. The recorded root (``os.getcwd()/repo``)
    then never matched the clone's real absolute location
    (``<workdir>/repo``), so `read_file` on the cloned README silently
    skipped the untrusted-content wrapper.

    A `cat repo/README` read-back via `terminal` would NOT have caught this:
    ``terminal``'s own cwd resolution was equally (and consistently) wrong
    on both the extraction and the lookup side, so the two wrongs matched
    each other. `read_file` takes an absolute path independent of cwd, so
    it isolates the root-extraction bug on its own.
    """
    agent = _make_agent()
    workdir = tempfile.mkdtemp(prefix="hermes-test-workdir-")

    import json as _json

    clone_call = _mock_tool_call(
        "terminal",
        _json.dumps({
            "command": "git clone https://example.com/attacker/repo.git repo",
            "workdir": workdir,
        }),
        "c1",
    )
    read_back_call = _mock_tool_call(
        "read_file",
        _json.dumps({
            "path": str(Path(workdir) / "repo" / "README.md"),
        }),
        "c2",
    )
    messages: list = []
    assistant_message = SimpleNamespace(
        content="", tool_calls=[clone_call, read_back_call]
    )

    # Long enough to clear _UNTRUSTED_WRAP_MIN_CHARS (32 chars) so wrapping
    # is observable in the message content.
    dummy_result = "cloned repo readme contents " * 3

    with (
        patch("run_agent.handle_function_call", return_value=dummy_result),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    read_back_content = messages[1]["content"]
    assert '<untrusted_tool_result source="read_file">' in read_back_content, (
        "read_file on the cloned repo's real (workdir-relative) location "
        "was not wrapped as untrusted. The fs-provenance cwd resolution in "
        "agent/tool_executor.py is ignoring `args['workdir']` for the "
        "`terminal` clone call and recording the fetch root relative to the "
        "(wrong) session cwd -- regression of the fix for teknium1's "
        "review on PR #57712."
    )
