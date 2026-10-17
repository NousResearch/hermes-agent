"""Skill discovery converges and never abandons an executable parent task."""

import json
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.agent.test_tool_call_guardrail_runtime import (
    _hard_stop_config,
    _make_agent,
    _mock_response,
    _mock_tool_call,
)


SKILL = "systematic-debugging"
VIEW = {"name": SKILL}


@pytest.fixture
def skill_file(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    path = home / "skills" / SKILL / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nname: systematic-debugging\ndescription: Diagnose a reproducible failure\n---\n"
        "Reproduce the failure, identify its cause, repair it, and verify the result.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return path


def _response(name, args=None):
    return _mock_response(content="", finish_reason="tool_calls", tool_calls=[
        _mock_tool_call(name, json.dumps(args or {})),
    ])


def _dispatcher(calls, skill_file):
    def dispatch(name, args, task_id=None, **kwargs):
        calls[name] += 1
        if name == "skills_list":
            return json.dumps({"success": True, "skills": [{"name": SKILL}], "count": 1})
        if name == "skill_view":
            return json.dumps({"success": True, "name": SKILL,
                               "skill_path": str(skill_file), "content": skill_file.read_text()})
        return json.dumps({"output": args.get("command", "completed"), "exit_code": 0})
    return dispatch


def _run(agent, responses, dispatch):
    agent.skip_background_review = True
    agent.client.chat.completions.create.side_effect = responses
    with ExitStack() as stack:
        stack.enter_context(patch("model_tools.handle_function_call", side_effect=dispatch))
        for method in ("_persist_session", "_save_trajectory", "_cleanup_task_resources"):
            stack.enter_context(patch.object(agent, method))
        return agent.run_conversation("Diagnose and repair the failure using systematic-debugging.")


def _assert_completed(result):
    assert result["final_response"] == "Repair verified."
    assert result["turn_exit_reason"].startswith("text_response")
    assert "Tell me how you'd like to proceed" not in result["final_response"]


def test_normal_skill_use_reuses_catalog_and_loads_once(skill_file):
    agent = _make_agent("skills_list", "skill_view", "terminal", max_iterations=25,
                        config=_hard_stop_config())
    calls = Counter()
    responses = [_response("skills_list"), _response("skill_view", VIEW)]
    responses += [_response("skills_list"), _response("skill_view", VIEW)] * 6
    responses += [_response("terminal", {"command": "verify repair"}),
                  _mock_response(content="Repair verified.")]

    result = _run(agent, responses, _dispatcher(calls, skill_file))

    _assert_completed(result)
    assert calls == Counter(skills_list=1, skill_view=1, terminal=1)
    skill_results = [m["content"] for m in result["messages"] if m.get("role") == "tool"
                     and "skill_resolution" in m.get("content", "")]
    assert skill_results
    assert any("execute_parent_task" in result for result in skill_results)


def test_known_skill_needs_no_further_discovery_or_read(skill_file):
    agent = _make_agent("skills_list", "skill_view", "terminal", config=_hard_stop_config())
    calls = Counter()
    responses = [_response("skill_view", VIEW), _response("terminal", {"command": "reproduce"}),
                 _response("skill_view", VIEW), _response("terminal", {"command": "verify"}),
                 _mock_response(content="Repair verified.")]

    result = _run(agent, responses, _dispatcher(calls, skill_file))

    _assert_completed(result)
    assert calls["skills_list"] == 0
    assert calls["skill_view"] == 1
    assert calls["terminal"] == 2


def test_full_catalog_answers_filtered_discovery_without_dispatch(skill_file):
    agent = _make_agent("skills_list", "terminal", config=_hard_stop_config())
    calls = Counter()

    def dispatch(name, args, task_id=None, **kwargs):
        calls[name] += 1
        return json.dumps({"success": True, "skills": [{"name": SKILL, "category": "debugging"}]})

    result = _run(agent, [_response("skills_list"),
                          _response("skills_list", {"category": "debugging"}),
                          _mock_response(content="Repair verified.")], dispatch)

    _assert_completed(result)
    assert calls == Counter(skills_list=1)
    tools = [json.loads(m["content"]) for m in result["messages"] if m.get("role") == "tool"]
    assert tools[-1]["skills"] == [{"name": SKILL, "category": "debugging"}]


@pytest.mark.parametrize("tool_name,args", [("skills_list", {}), ("skill_view", VIEW)])
def test_lookup_guardrail_blocks_only_lookup_and_continues_parent(skill_file, tool_name, args):
    agent = _make_agent("skills_list", "skill_view", "terminal", config=_hard_stop_config())
    calls = Counter()
    seeded = False
    responses = iter([_response(tool_name, args), _response(tool_name, args),
                      _response("terminal", {"command": "use ordinary tools"}),
                      _mock_response(content="Repair verified.")])

    def model_call(**kwargs):
        nonlocal seeded
        if not seeded:
            for _ in range(5):
                agent._tool_guardrails.after_call(tool_name, args, '{"success":true}', failed=False)
            seeded = True
        return next(responses)

    result = _run(agent, model_call, _dispatcher(calls, skill_file))

    _assert_completed(result)
    assert calls[tool_name] == 0
    assert calls["terminal"] == 1
    tool_results = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("idempotent_no_progress_block" in content for content in tool_results)
    assert agent._tool_guardrail_halt_decision is None


def test_missing_indispensable_instructions_report_actual_dependency(skill_file):
    agent = _make_agent("skill_view", "terminal", config=_hard_stop_config())
    calls = Counter()

    def missing(name, args, task_id=None, **kwargs):
        calls[name] += 1
        return json.dumps({"success": False, "error": "Required private deployment instructions unavailable"})

    dependency = "Deployment requires the unavailable private deployment instructions."
    result = _run(agent, [_response("skill_view", {"name": "private-deployment"}),
                          _mock_response(content=dependency)], missing)

    assert result["final_response"] == dependency
    assert result["turn_exit_reason"].startswith("text_response")
    assert calls == Counter(skill_view=1)
    assert agent._tool_guardrail_halt_decision is None


def test_long_parent_task_keeps_loaded_skill_state(skill_file):
    agent = _make_agent("skills_list", "skill_view", "terminal", max_iterations=40,
                        config=_hard_stop_config())
    calls = Counter()
    responses = [_response("skills_list"), _response("skill_view", VIEW)]
    responses += [_response("terminal", {"command": f"inspect file {index}"}) for index in range(24)]
    responses += [_response("skill_view", VIEW), _response("skills_list"),
                  _mock_response(content="Repair verified.")]

    result = _run(agent, responses, _dispatcher(calls, skill_file))

    _assert_completed(result)
    assert calls == Counter(terminal=24, skills_list=1, skill_view=1)


def test_real_skill_registry_loads_distinct_reference_and_changed_instructions(skill_file):
    from model_tools import handle_function_call
    from tools import skills_tool

    reference = skill_file.parent / "references" / "verification.md"
    reference.parent.mkdir()
    reference.write_text("Verify the repaired behavior and its sibling paths.", encoding="utf-8")
    skills_tool.clear_skills_cache()
    agent = _make_agent("skills_list", "skill_view", "terminal", config=_hard_stop_config())
    calls = Counter()

    def dispatch(name, args, task_id=None, **kwargs):
        calls[name] += 1
        if name == "terminal":
            skill_file.write_text(skill_file.read_text() + "\nCheck the changed instructions.\n")
            return json.dumps({"exit_code": 0, "output": "Instructions updated"})
        return handle_function_call(name, args, task_id=task_id, **kwargs)

    responses = [_response("skills_list"), _response("skill_view", VIEW),
                 _response("skill_view", {"name": SKILL, "file_path": "references/verification.md"}),
                 _response("skill_view", VIEW),
                 _response("terminal", {"command": "update skill instructions"}),
                 _response("skill_view", VIEW), _mock_response(content="Repair verified.")]

    result = _run(agent, responses, dispatch)

    _assert_completed(result)
    assert calls == Counter(skills_list=1, skill_view=3, terminal=1)
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("Verify the repaired behavior and its sibling paths" in content for content in tool_contents)
    assert any("Check the changed instructions" in content for content in tool_contents)


def test_compacted_skill_restores_retained_instructions_without_skill_dispatch(skill_file):
    from agent.skill_resolution import preserve_resolution_summary
    from tests.agent.test_proactive_tool_result_pruning import _compressor

    skill_file.write_text(skill_file.read_text() + "\nPreserved operational instruction.\n" * 1500)
    agent = _make_agent("skills_list", "skill_view", "terminal", max_iterations=25,
                        config=_hard_stop_config())
    calls = Counter()
    dispatch = _dispatcher(calls, skill_file)
    responses = [_response("skills_list"), _response("skill_view", VIEW)]
    responses += [_response("terminal", {"command": f"work item {index}"}) for index in range(12)]
    responses.append(_mock_response(content="Repair verified."))
    result = _run(agent, responses, dispatch)
    resolution = agent._skill_resolution

    compressor = _compressor(proactive_prune_tokens=48_000,
                             proactive_prune_min_result_chars=8000)
    compressed, count = compressor.prune_tool_results_only(result["messages"], current_tokens=120_000)
    assert count > 0
    assert any("SKILL_PRUNED:" in message.get("content", "") for message in compressed)
    compressed = preserve_resolution_summary(agent, compressed)
    assert any("<skill_resolution>" in message.get("content", "") for message in compressed)
    assert agent._skill_resolution is resolution

    request = SimpleNamespace(content="", tool_calls=[_mock_tool_call("skill_view", json.dumps(VIEW))])
    with patch("model_tools.handle_function_call", side_effect=dispatch):
        agent._execute_tool_calls_sequential(request, compressed, "compacted-task")

    assert calls["skill_view"] == 1
    assert "Preserved operational instruction" in compressed[-1]["content"]
    assert "execute_parent_task" in compressed[-1]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_failed_lookup_requires_changed_reason_before_dispatch_retry(skill_file):
    agent = _make_agent("skill_view", "terminal", config=_hard_stop_config())
    calls = Counter()

    def dispatch(name, args, task_id=None, **kwargs):
        calls[name] += 1
        if args.get("reload_reason"):
            return json.dumps({"success": True, "name": SKILL, "content": skill_file.read_text()})
        return json.dumps({"success": False, "error": "Temporary skill source failure"})

    responses = [_response("skill_view", VIEW), _response("skill_view", VIEW),
                 _response("skill_view", {**VIEW, "reload_reason": "previous_read_failed"}),
                 _mock_response(content="Repair verified.")]
    result = _run(agent, responses, dispatch)

    _assert_completed(result)
    assert calls["skill_view"] == 2


def test_consecutive_retained_skill_read_keeps_content_for_compaction_recovery(skill_file):
    skill_file.write_text(skill_file.read_text() + "\nRecover this instruction.\n" * 100)
    agent = _make_agent("skill_view", config=_hard_stop_config())
    calls = Counter()
    result = _run(agent, [_response("skill_view", VIEW), _response("skill_view", VIEW),
                          _mock_response(content="Repair verified.")], _dispatcher(calls, skill_file))

    _assert_completed(result)
    tools = [m for m in result["messages"] if m.get("role") == "tool"]
    assert "Recover this instruction." in tools[-1]["content"]
    assert calls["skill_view"] == 1


def test_reference_edit_during_dispatch_is_not_mistaken_for_served_version(skill_file):
    agent = _make_agent("skill_view", config=_hard_stop_config())
    reference = skill_file.parent / "verification.md"
    reference.write_text("Old instructions")
    calls = Counter()

    def dispatch(name, args, task_id=None, **kwargs):
        calls[name] += 1
        content = reference.read_text()
        reference.write_text("New instructions")
        return json.dumps({"success": True, "name": SKILL, "skill_dir": str(skill_file.parent),
                           "file": "verification.md", "content": content})

    args = {**VIEW, "file_path": "verification.md"}
    result = _run(agent, [_response("skill_view", args), _response("skill_view", args),
                          _response("skill_view", args), _mock_response(content="Repair verified.")], dispatch)

    _assert_completed(result)
    assert calls["skill_view"] == 2
    tools = [m for m in result["messages"] if m.get("role") == "tool"]
    assert "New instructions" in tools[-1]["content"]
