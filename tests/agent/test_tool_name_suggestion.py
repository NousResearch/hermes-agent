"""A tool name that is only *close* to a real tool is suggested, never executed.

``repair_tool_call`` still normalizes spelling variants of the same name (case, separators,
CamelCase, ``_tool`` suffixes; see ``test_repair_tool_call_name.py``). A merely similar name
(a typo, a sibling, an inverse operation) used to be fuzzy-matched and run under the repaired
name. It now takes the ordinary unknown-tool path: the model receives the error with a
"Did you mean" hint and must issue the call itself. A name the registry knows but this session
withholds is reported as unavailable, and never steered to a sibling that does something else.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import repair_tool_call
from agent.conversation_loop import _invalid_tool_name_error_content
from agent.turn_tool_validation import validate_tool_calls
from tools.registry import registry

VALID = {"read_file", "terminal", "web_search", "kanban_link"}


def _registered(monkeypatch, names):
    """Pin the tools the registry knows (some of which a session may withhold)."""
    monkeypatch.setattr(registry, "get_all_tool_names", lambda: sorted(names))


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    import hermes_cli.observability.shared_metrics_model as metrics

    monkeypatch.setattr(metrics, "record_tool_call_quality", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _registry_is_the_available_tools(monkeypatch):
    _registered(monkeypatch, VALID)


def _tool_call(call_id, name, arguments="{}"):
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments),
    )


def _agent(valid=VALID):
    agent = SimpleNamespace(
        valid_tool_names=set(valid), log_prefix="", _invalid_tool_retries=0, _invalid_json_retries=0,
        _uniquify_tool_call_ids=lambda calls, taken=(): None,
        _vprint=lambda *a, **k: None, _buffer_vprint=lambda *a, **k: None, _flush_status_buffer=lambda: None,
        _build_assistant_message=lambda message, finish_reason: {"role": "assistant", "content": ""},
        _persist_session=lambda *a: None, _cleanup_task_resources=lambda *a: None,
    )
    agent._repair_tool_call = lambda name: repair_tool_call(agent, name)  # the real repair step
    return agent


def _validate(agent, tool_calls, messages):
    return validate_tool_calls(
        agent, SimpleNamespace(content="", tool_calls=tool_calls), "tool_calls",
        messages=messages, conversation_history=[], api_call_count=1, effective_task_id="t1",
    )


class TestErrorContent:
    def test_close_name_gets_a_hint_and_keeps_the_catalog(self):
        content = _invalid_tool_name_error_content("terminl", VALID)
        assert "Tool 'terminl' does not exist." in content
        assert "Did you mean `terminal`?" in content
        assert "Available tools: kanban_link, read_file, terminal, web_search" in content

    def test_unrelated_name_gets_no_hint(self):
        content = _invalid_tool_name_error_content("quantum_flux_capacitor", VALID)
        assert "Did you mean" not in content
        assert "Available tools:" in content

    def test_only_available_tools_are_suggested(self):
        assert "Did you mean `read_file`?" in _invalid_tool_name_error_content("Read_Fle", ["read_file"])
        assert "Did you mean" not in _invalid_tool_name_error_content("read_fle", [])

    def test_xml_polluted_name_is_trimmed_before_matching(self):
        content = _invalid_tool_name_error_content('terminl" parameter="command" string="true', VALID)
        assert "Did you mean `terminal`?" in content

    def test_a_withheld_tool_is_reported_unavailable_with_no_hint(self, monkeypatch):
        _registered(monkeypatch, {"kanban_list", "kanban_link"})
        content = _invalid_tool_name_error_content("kanban_list", {"kanban_link"})
        assert "Tool 'kanban_list' is not available in this session." in content
        assert "does not exist" not in content and "Did you mean" not in content
        assert "Available tools: kanban_link" in content

    @pytest.mark.parametrize("emitted", ["Kanban-List", "kanban list", 'kanban_list" parameter="x'])
    def test_a_withheld_tool_is_recognised_through_the_spelling_repair_folds(self, monkeypatch, emitted):
        _registered(monkeypatch, {"kanban_list", "kanban_link"})
        content = _invalid_tool_name_error_content(emitted, {"kanban_link"})
        assert f"Tool '{emitted}' is not available in this session." in content and "Did you mean" not in content

    def test_swapped_word_order_suggests_the_available_tool_not_a_lookalike(self, monkeypatch):
        _registered(monkeypatch, {"web_search", "search_files"})
        content = _invalid_tool_name_error_content("search_web", {"web_search", "search_files"})
        assert "Did you mean `web_search`?" in content and "search_files`?" not in content

    def test_a_typo_of_a_withheld_tool_gets_no_hint_instead_of_a_sibling(self, monkeypatch):
        _registered(monkeypatch, {"kanban_unblock", "kanban_block"})
        content = _invalid_tool_name_error_content("kanban_unblok", {"kanban_block"})
        assert "Tool 'kanban_unblok' does not exist." in content
        assert "Did you mean" not in content

    @pytest.mark.parametrize("name", ["", "   ", None])
    def test_blank_name_stays_terse_with_no_catalog_and_no_hint(self, name):
        content = _invalid_tool_name_error_content(name, VALID)
        assert "Did you mean" not in content and "Available tools" not in content


class TestValidateToolCalls:
    def test_close_name_is_not_executed_and_the_model_is_told_which_tool_to_use(self):
        agent, messages = _agent(), []
        call = _tool_call("c1", "read_fle", '{"path": "a"}')

        verdict = _validate(agent, [call], messages)

        assert verdict.action == "continue"               # error returned to the model, nothing dispatched
        assert call.function.name == "read_fle"           # the emitted name is NOT rewritten
        assert agent._invalid_tool_retries == 1
        result = messages[-1]
        assert result["role"] == "tool" and result["name"] == "read_fle"
        assert "Did you mean `read_file`?" in result["content"]

    def test_sibling_of_an_unavailable_tool_is_not_executed(self, monkeypatch):
        _registered(monkeypatch, {"kanban_list", "kanban_link", "read_file"})
        agent, messages = _agent({"kanban_link", "read_file"}), []
        call = _tool_call("c1", "kanban_list")

        verdict = _validate(agent, [call], messages)

        assert verdict.action == "continue"
        assert call.function.name == "kanban_list"        # never dispatched as kanban_link
        assert "not available in this session" in messages[-1]["content"]
        assert "Did you mean" not in messages[-1]["content"]

    @pytest.mark.parametrize("emitted, repaired", [
        ("Read_File", "read_file"), ("read-file", "read_file"), ("ReadFile", "read_file"),
        ("TerminalTool_tool", "terminal"),
    ])
    def test_spelling_variants_are_still_repaired_and_executed(self, emitted, repaired):
        agent, messages = _agent(), []
        call = _tool_call("c1", emitted)

        verdict = _validate(agent, [call], messages)

        assert verdict.action == "ok" and call.function.name == repaired
        assert agent._invalid_tool_retries == 0

    def test_mixed_batch_runs_the_valid_call_and_leaves_the_close_name_as_emitted(self):
        agent, messages = _agent(), []
        good, near = _tool_call("c1", "read_file", '{"path": "a"}'), _tool_call("c2", "web_serch")

        verdict = _validate(agent, [good, near], messages)

        assert verdict.action == "ok" and verdict.mixed_invalid_batch is True
        assert near.function.name == "web_serch"          # run_tool_round answers it with the hint
        assert "Did you mean `web_search`?" in _invalid_tool_name_error_content(
            near.function.name, agent.valid_tool_names)

    def test_a_model_that_keeps_repeating_the_near_miss_still_halts_at_three(self):
        agent, messages = _agent(), []
        actions = [_validate(agent, [_tool_call(f"c{i}", "read_fle")], messages).action for i in range(3)]
        assert actions == ["continue", "continue", "return"]
