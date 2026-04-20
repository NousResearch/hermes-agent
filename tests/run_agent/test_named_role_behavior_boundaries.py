import json
from importlib import import_module

import agent.archetypes as archetypes

if not hasattr(archetypes, "resolve_specialist_mapping"):
    archetypes.resolve_specialist_mapping = lambda value: None
if not hasattr(archetypes, "resolve_specialist_defaults"):
    archetypes.resolve_specialist_defaults = lambda value: {}

run_agent = import_module("run_agent")
AIAgent = run_agent.AIAgent


SAMPLE_CONTRACT = {
    "task": "Verify the implementation",
    "expected_outcome": "Verification evidence is reported with bounded context precedence.",
    "required_skills": ["testing"],
    "required_tools": ["read_file", "terminal"],
    "must_do": ["inspect the repo before concluding"],
    "must_not_do": ["do not mutate files while reviewing"],
    "context": {"ticket": "swarm-d"},
}


def _make_bare_agent():
    agent = AIAgent.__new__(AIAgent)
    agent.skip_context_files = False
    agent.valid_tool_names = set()
    agent.tools = []
    agent._tool_use_enforcement = False
    agent.model = "gpt-5.4"
    agent.provider = "openai"
    agent.pass_session_id = False
    agent.session_id = None
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_manager = None
    agent._session_db = None
    agent._todo_store = None
    agent.clarify_callback = None
    agent.platform = "cli"
    agent.runtime_activation_state = {}
    agent._runtime_activation_state = {}
    agent.get_runtime_activation_state = lambda: agent.runtime_activation_state
    return agent


def test_build_system_prompt_passes_runtime_task_contract_to_context_prompt(monkeypatch):
    agent = _make_bare_agent()
    agent.runtime_activation_state = {"task_contract": SAMPLE_CONTRACT}
    captured = {}

    monkeypatch.setattr(run_agent, "load_soul_md", lambda: "SOUL")
    monkeypatch.setattr(run_agent, "build_nous_subscription_prompt", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(run_agent, "build_skills_system_prompt", lambda **_kwargs: "")
    monkeypatch.setattr(run_agent, "build_environment_hints", lambda: "")

    def _fake_context_prompt(*, cwd=None, skip_soul=False, task_contract=None, max_hermes_hierarchy_layers=3):
        captured["cwd"] = cwd
        captured["skip_soul"] = skip_soul
        captured["task_contract"] = task_contract
        captured["layers"] = max_hermes_hierarchy_layers
        return "CONTEXT"

    monkeypatch.setattr(run_agent, "build_context_files_prompt", _fake_context_prompt)

    prompt = agent._build_system_prompt()

    assert captured["task_contract"] == SAMPLE_CONTRACT
    assert captured["skip_soul"] is True
    assert "CONTEXT" in prompt


def test_named_role_completion_gate_requires_verification_evidence():
    agent = _make_bare_agent()
    agent.runtime_activation_state = {
        "specialist": "code_reviewer",
        "archetype": "verifier",
        "delegation_profile": "verification",
    }

    gate_error = agent._evaluate_named_role_completion_gate(
        [{"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]}]
    )

    assert gate_error == (
        "Reviewer/verifier completion gate blocked success: no verification evidence tool was used in this run."
    )


def test_named_role_completion_gate_allows_evidence_tool_usage():
    agent = _make_bare_agent()
    agent.runtime_activation_state = {
        "specialist": "code_reviewer",
        "archetype": "verifier",
        "delegation_profile": "verification",
    }

    gate_error = agent._evaluate_named_role_completion_gate(
        [{"role": "assistant", "tool_calls": [{"function": {"name": "terminal"}}]}]
    )

    assert gate_error is None


def test_named_role_completion_gate_keeps_legacy_reviewer_alias_compatible():
    agent = _make_bare_agent()
    agent.runtime_activation_state = {
        "specialist": "reviewer",
        "archetype": "generalist",
        "delegation_profile": "verification",
    }

    gate_error = agent._evaluate_named_role_completion_gate(
        [{"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]}]
    )

    assert gate_error == (
        "Reviewer/verifier completion gate blocked success: no verification evidence tool was used in this run."
    )


def test_named_role_completion_gate_does_not_treat_archetype_only_verifier_as_code_reviewer():
    agent = _make_bare_agent()
    agent.runtime_activation_state = {
        "specialist": None,
        "archetype": "verifier",
        "delegation_profile": "verification",
    }

    gate_error = agent._evaluate_named_role_completion_gate(
        [{"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]}]
    )

    assert gate_error is None


def test_get_runtime_scoped_tools_filters_mutating_tools_for_reviewer_runtime():
    agent = _make_bare_agent()
    agent.valid_tool_names = {"read_file", "terminal", "patch", "write_file", "memory"}
    agent.tools = [
        {"type": "function", "function": {"name": "patch"}},
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "terminal"}},
        {"type": "function", "function": {"name": "write_file"}},
        {"type": "function", "function": {"name": "memory"}},
    ]
    agent.runtime_activation_state = {
        "specialist": "code_reviewer",
        "archetype": "verifier",
        "delegation_profile": "verification",
    }

    scoped_names = [tool["function"]["name"] for tool in agent._get_runtime_scoped_tools()]

    assert scoped_names == ["read_file", "terminal"]


def test_maybe_block_named_role_tool_call_blocks_mutating_tools_and_destructive_terminal():
    agent = _make_bare_agent()
    agent.runtime_activation_state = {
        "specialist": "code_reviewer",
        "archetype": "verifier",
        "delegation_profile": "verification",
    }

    assert "cannot call 'patch'" in agent._maybe_block_named_role_tool_call("patch", {})
    assert agent._maybe_block_named_role_tool_call("terminal", {"command": "git status"}) is None
    assert "destructive terminal commands are blocked" in agent._maybe_block_named_role_tool_call(
        "terminal", {"command": "rm -rf tmp"}
    )


def test_live_tool_invoke_blocks_disallowed_mutating_tool_for_reviewer_runtime(monkeypatch):
    agent = _make_bare_agent()
    agent.valid_tool_names = {"patch", "read_file", "terminal"}
    agent.runtime_activation_state = {
        "specialist": "qa_guard",
        "archetype": "verifier",
        "delegation_profile": "verification",
    }

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("blocked mutating tool should not reach handle_function_call")

    monkeypatch.setattr(run_agent, "handle_function_call", _unexpected_call)

    result = json.loads(agent._invoke_tool("patch", {"path": "x", "old_string": "a", "new_string": "b"}, "task-1"))

    assert result == {
        "error": (
            "Reviewer/verifier runtime boundary: this role is read-only and cannot call 'patch'. "
            "Use read-only inspection, testing, or delegation instead."
        )
    }


def test_live_tool_invoke_allows_evidence_tools_and_non_destructive_terminal_for_reviewer_runtime(monkeypatch):
    agent = _make_bare_agent()
    agent.valid_tool_names = {"read_file", "terminal", "patch"}
    agent.runtime_activation_state = {
        "specialist": "code_reviewer",
        "archetype": "verifier",
        "delegation_profile": "verification",
    }
    calls = []

    def _fake_handle(function_name, function_args, effective_task_id, **kwargs):
        calls.append((function_name, function_args, effective_task_id, kwargs))
        return json.dumps({"ok": function_name}, ensure_ascii=False)

    monkeypatch.setattr(run_agent, "handle_function_call", _fake_handle)

    read_result = json.loads(agent._invoke_tool("read_file", {"path": "run_agent.py"}, "task-2"))
    terminal_result = json.loads(agent._invoke_tool("terminal", {"command": "git status --short"}, "task-2"))

    assert read_result == {"ok": "read_file"}
    assert terminal_result == {"ok": "terminal"}
    assert calls == [
        (
            "read_file",
            {"path": "run_agent.py"},
            "task-2",
            {
                "tool_call_id": None,
                "session_id": "",
                "enabled_tools": ["read_file", "terminal"],
                "skip_pre_tool_call_hook": True,
            },
        ),
        (
            "terminal",
            {"command": "git status --short"},
            "task-2",
            {
                "tool_call_id": None,
                "session_id": "",
                "enabled_tools": ["read_file", "terminal"],
                "skip_pre_tool_call_hook": True,
            },
        ),
    ]
