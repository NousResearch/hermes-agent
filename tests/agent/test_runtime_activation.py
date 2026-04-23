from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.intent_preclassifier import IntentPreclassification, preclassify_intent
from hermes_cli.command_templates import build_command_invocation
from hermes_cli.work_command_adapter import prepare_work_command
from hermes_state import SessionDB
from run_agent import AIAgent


def _sample_task_contract() -> dict:
    return {
        "task": "Implement the delegated change",
        "expected_outcome": "A passing implementation with verification evidence",
        "required_skills": ["python", "testing"],
        "required_tools": ["read_file", "patch", "terminal"],
        "must_do": [
            "inspect repo patterns before editing",
            {"verification": ["run targeted pytest", "report exact command output"]},
        ],
        "must_not_do": {
            "forbidden_files": ["tools/delegate_tool.py"],
            "notes": ["do not flatten structured contract sections into prose"],
        },
        "context": {
            "ticket": "W2-2 / PR-W2-2",
            "paths": {"repo": "/root/.hermes/hermes-agent"},
        },
    }


class TestRuntimeActivationPreclassifier:
    def test_ultrawork_and_ulw_alias_to_same_runtime_target(self):
        ultrawork = preclassify_intent("Use ultrawork mode to implement the patch and run tests.")
        ulw = preclassify_intent("Use ulw mode to implement the patch and run tests.")

        assert ultrawork.inferred_runtime_mode == "ultrawork"
        assert ulw.inferred_runtime_mode == "ultrawork"
        assert ultrawork.inferred_specialist == ulw.inferred_specialist == "builder"
        assert ultrawork.inferred_archetype == ulw.inferred_archetype == "implementer"
        assert ultrawork.inferred_route_category == ulw.inferred_route_category == "deep"
        assert "ultrawork" in ultrawork.activation_reason
        assert "ulw" in ulw.activation_reason

    def test_ralph_keyword_resolves_to_distinct_runtime_target(self):
        result = preclassify_intent("Use ralph mode to implement the patch and keep working until the task is closed.")

        assert result.inferred_runtime_mode == "ralph"
        assert result.inferred_specialist == "builder"
        assert result.inferred_archetype == "implementer"
        assert result.inferred_route_category == "deep"
        assert "ralph" in result.activation_reason

    def test_loop_commands_preserve_distinct_runtime_modes_via_structured_contracts(self):
        ralph = build_command_invocation(
            "ralph-loop",
            raw_args="Implement the requested change",
            session_id="sess-1",
            cwd="/tmp",
        )
        ulw = build_command_invocation(
            "ulw-loop",
            raw_args="Implement the requested change",
            session_id="sess-1",
            cwd="/tmp",
        )

        ralph_result = preclassify_intent({"message": "Implement the requested change", "task_contract": ralph.task_contract})
        ulw_result = preclassify_intent({"message": "Implement the requested change", "task_contract": ulw.task_contract})

        ralph_runtime = ralph.task_contract["context"]["command_runtime"]
        ulw_runtime = ulw.task_contract["context"]["command_runtime"]

        assert ralph_runtime["command_name"] == "ralph-loop"
        assert ralph_runtime["runtime_mode"] == "ralph"
        assert ralph_runtime["continuation_semantics"] == {
            "retry_on_failed_or_interrupted": True,
            "stop_requires_explicit_exit": True,
        }
        assert ulw_runtime["command_name"] == "ulw-loop"
        assert ulw_runtime["runtime_mode"] == "ultrawork"
        assert ulw_runtime["continuation_semantics"] == {
            "completion_gate": "open_todos_block_done",
            "require_open_work_closure": True,
        }
        assert ralph_result.inferred_runtime_mode == "ralph"
        assert ulw_result.inferred_runtime_mode == "ultrawork"
        assert ralph_result.task_contract is not None
        assert ulw_result.task_contract is not None
        assert ralph_result.task_contract.model_dump() == ralph.task_contract
        assert ulw_result.task_contract.model_dump() == ulw.task_contract

    def test_preclassifier_is_deterministic_for_same_input(self):
        message = "Research the API behavior, compare sources, and summarize the findings."

        first = preclassify_intent(message)
        second = preclassify_intent(message)

        assert first == second
        assert first.inferred_specialist == "analyst"
        assert first.inferred_archetype == "researcher"
        assert first.inferred_route_category == "deep"
        assert first.inferred_runtime_mode == "default"
        assert first.inferred_delegation_profile == "research"

    def test_unknown_or_underspecified_input_falls_back_safely(self):
        result = preclassify_intent("help")

        assert result.inferred_specialist is None
        assert result.inferred_archetype == "generalist"
        assert result.inferred_route_category == "unspecified_low"
        assert result.inferred_runtime_mode == "default"
        assert result.inferred_delegation_profile == "general"
        assert "fallback" in result.activation_reason.lower()

    def test_specialist_runtime_route_and_delegation_layers_remain_distinct(self):
        result = preclassify_intent(
            "Research the architecture, but activate ultrawork while keeping the work in the deep lane."
        )

        assert result.inferred_specialist == "investigator"
        assert result.inferred_archetype == "researcher"
        assert result.inferred_delegation_profile == "research"
        assert result.inferred_route_category == "deep"
        assert result.inferred_runtime_mode == "ultrawork"
        assert result.inferred_specialist != result.inferred_archetype
        assert result.inferred_specialist != result.inferred_route_category
        assert result.inferred_specialist != result.inferred_runtime_mode
        assert result.inferred_archetype != result.inferred_route_category
        assert result.inferred_archetype != result.inferred_runtime_mode
        assert result.inferred_delegation_profile != result.inferred_route_category

    def test_task_contract_remains_structured_and_is_not_collapsed_into_prose(self):
        task_contract = _sample_task_contract()

        result = preclassify_intent(
            {"message": "Implement the change in ultrawork mode.", "task_contract": task_contract}
        )

        assert isinstance(result, IntentPreclassification)
        assert result.inferred_specialist == "builder"
        assert result.task_contract is not None
        assert result.task_contract.model_dump() == task_contract
        assert isinstance(result.task_contract.must_do, list)
        assert isinstance(result.task_contract.must_not_do, dict)
        assert isinstance(result.task_contract.context, dict)

    def test_ambiguous_review_prompt_exposes_specialist_without_overloading_archetype(self):
        result = preclassify_intent("Review this patch, verify the risky changes, and call out regressions.")

        assert result.inferred_specialist == "code_reviewer"
        assert result.inferred_archetype == "verifier"
        assert result.inferred_delegation_profile == "verification"
        assert result.inferred_route_category == "quick"
        assert "specialist=code_reviewer" in result.activation_reason
        assert "archetype=verifier" in result.activation_reason

    def test_planner_specialist_applies_overlay_defaults_without_collapsing_into_archetype(self):
        result = preclassify_intent("Plan the rollout, decompose the work, and sequence the execution plan.")

        assert result.inferred_specialist == "planner"
        assert result.inferred_archetype == "generalist"
        assert result.inferred_route_category == "deep"
        assert result.inferred_delegation_profile == "general"
        assert result.inferred_specialist != result.inferred_archetype
        assert "specialist=planner" in result.activation_reason

    @pytest.mark.parametrize(
        ("prompt", "specialist", "archetype", "route_category", "delegation_profile"),
        [
            (
                "implement the delegated change",
                "builder",
                "implementer",
                "deep",
                "implementation",
            ),
            (
                "Review this patch, verify the risky changes, and call out regressions.",
                "code_reviewer",
                "verifier",
                "quick",
                "verification",
            ),
            (
                "qa regression validate the fix",
                "qa_guard",
                "verifier",
                "quick",
                "verification",
            ),
            (
                "Plan the rollout, decompose the work, and sequence the execution plan.",
                "planner",
                "generalist",
                "deep",
                "general",
            ),
            (
                "Reproduce the bug, trace the failure, and identify the root cause.",
                "bug_hunter",
                "implementer",
                "deep",
                "implementation",
            ),
            (
                "Analyze the API behavior, compare sources, and synthesize the findings.",
                "analyst",
                "researcher",
                "deep",
                "research",
            ),
            (
                "Investigate the architecture, gather evidence, and triage the issue.",
                "investigator",
                "researcher",
                "deep",
                "research",
            ),
            (
                "Inspect this PDF diagram and explain the visual flow.",
                "multimodal_specialist",
                "researcher",
                "visual",
                "research",
            ),
        ],
        ids=["builder", "code_reviewer", "qa_guard", "planner", "bug_hunter", "analyst", "investigator", "multimodal_specialist"],
    )
    def test_preclassifier_specialist_matrix_keeps_taxonomy_layers_explicit(
        self,
        prompt: str,
        specialist: str,
        archetype: str,
        route_category: str,
        delegation_profile: str,
    ):
        result = preclassify_intent(prompt)

        assert result.inferred_specialist == specialist
        assert result.inferred_archetype == archetype
        assert result.inferred_route_category == route_category
        assert result.inferred_delegation_profile == delegation_profile
        assert result.inferred_runtime_mode == "default"
        assert f"specialist={specialist}" in result.activation_reason
        assert f"archetype={archetype}" in result.activation_reason
        assert f"route_category={route_category}" in result.activation_reason
        assert f"delegation_profile={delegation_profile}" in result.activation_reason
        assert "runtime_mode=default" in result.activation_reason

        if specialist != archetype:
            assert result.inferred_specialist != result.inferred_archetype
        assert result.inferred_specialist != result.inferred_route_category
        assert result.inferred_specialist != result.inferred_runtime_mode
        assert result.inferred_archetype != result.inferred_route_category
        assert result.inferred_archetype != result.inferred_runtime_mode
        assert result.inferred_delegation_profile != result.inferred_route_category

    def test_runtime_activation_normalizes_legacy_reviewer_alias_to_canonical_specialist(self):
        agent = _make_agent()
        legacy_alias = SimpleNamespace(
            inferred_specialist="reviewer",
            inferred_archetype="verifier",
            inferred_route_category="quick",
            inferred_runtime_mode="default",
            inferred_delegation_profile="verification",
            activation_reason="legacy specialist alias",
            task_contract=None,
            inference_source="test_preclassifier",
        )

        with patch("run_agent.preclassify_intent", return_value=legacy_alias):
            state = agent._resolve_runtime_activation_state("Review this patch.")

        assert state["specialist"] == "code_reviewer"
        assert state["archetype"] == "verifier"
        assert state["route_category"] == "quick"
        assert state["delegation_profile"] == "verification"
        assert state["runtime_mode"] == "default"
        assert state["inference_source"] == "test_preclassifier"
        assert "specialist: code_reviewer" in state["activation_note"]
        assert "## Archetype\nname: verifier" in state["wave1_overlay_prompt"]
        assert "## Delegation Profile\nname: verification" in state["wave1_overlay_prompt"]

    def test_runtime_activation_extracts_structured_contract_from_rendered_handoff_prompt(self):
        agent = _make_agent()
        invocation = build_command_invocation(
            "handoff",
            raw_args='{"task":"Resume the delegated task","expected_outcome":"Done","required_skills":["python"],"required_tools":["terminal"],"must_do":["inspect state"],"must_not_do":["discard context"],"context":{"ticket":"p1"}}',
            session_id="sess-1",
            cwd="/tmp",
        )

        state = agent._resolve_runtime_activation_state(invocation.prompt_text)

        assert state["task_contract"] == invocation.task_contract
        assert state["runtime_mode"] == "default"
        assert state["delegation_profile"] == "general"
        assert state["named_workflow"] is None
        assert state["activation_applied"] is True
        assert "Resume the delegated task" in (state["activation_note"] or "")

    def test_preclassifier_mixed_review_and_bug_prompt_keeps_keyword_collision_behavior_explicit(self):
        result = preclassify_intent(
            "Review this patch, reproduce the bug, and call out regressions with verification evidence."
        )

        assert result.inferred_specialist == "bug_hunter"
        assert result.inferred_archetype == "implementer"
        assert result.inferred_route_category == "deep"
        assert result.inferred_delegation_profile == "implementation"
        assert result.inferred_runtime_mode == "default"
        assert "specialist=bug_hunter" in result.activation_reason
        assert "specialist=code_reviewer" not in result.activation_reason

    def test_preclassifier_routes_pdf_and_diagram_work_to_multimodal_specialist(self):
        result = preclassify_intent("Inspect this PDF diagram and explain the visual flow.")

        assert result.inferred_specialist == "multimodal_specialist"
        assert result.inferred_archetype == "researcher"
        assert result.inferred_route_category == "visual"
        assert result.inferred_delegation_profile == "research"
        assert result.inferred_runtime_mode == "default"
        assert "specialist=multimodal_specialist" in result.activation_reason
        assert "route_category=visual" in result.activation_reason


def _make_tool_defs(*names: str) -> list[dict]:
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


class _FakeClient(MagicMock):
    pass


def _make_agent(*tool_names: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url="https://example.test/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = _FakeClient()
    return agent


class TestRuntimeActivationRunAgentIntegration:
    def test_multimodal_runtime_activation_applies_when_multimodal_tools_are_loaded(self):
        agent = _make_agent("vision_analyze")
        captured_api_kwargs = {}

        def fake_api_call(api_kwargs):
            captured_api_kwargs.update(api_kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Visual analysis ready.", tool_calls=[]),
                    )
                ],
                usage=None,
            )

        with patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call):
            result = agent.run_conversation("Inspect this PDF diagram and explain the visual flow.")

        assert result["final_response"] == "Visual analysis ready."
        state = agent.get_runtime_activation_state()
        assert state["specialist"] == "multimodal_specialist"
        assert state["archetype"] == "researcher"
        assert state["route_category"] == "visual"
        assert state["delegation_profile"] == "research"
        assert state["runtime_mode"] == "default"
        assert state["activation_applied"] is True
        assert "specialist: multimodal_specialist" in state["activation_note"]
        assert "## Route Category\nname: visual" in state["wave1_overlay_prompt"]

        injected_user_message = captured_api_kwargs["messages"][-1]["content"]
        assert "<wave2-runtime-activation>" in injected_user_message
        assert "specialist: multimodal_specialist" in injected_user_message

    @pytest.mark.parametrize("tool_name", ["browser_navigate", "browser_snapshot", None])
    def test_multimodal_runtime_activation_falls_back_safely_when_multimodal_tools_are_unavailable(self, tool_name):
        agent = _make_agent(*( [tool_name] if tool_name else [] ))
        captured_api_kwargs = {}

        def fake_api_call(api_kwargs):
            captured_api_kwargs.update(api_kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Safe fallback.", tool_calls=[]),
                    )
                ],
                usage=None,
            )

        with patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call):
            result = agent.run_conversation("Inspect this PDF diagram and explain the visual flow.")

        assert result["final_response"] == "Safe fallback."
        state = agent.get_runtime_activation_state()
        assert state["specialist"] is None
        assert state["archetype"] == "generalist"
        assert state["route_category"] == "unspecified_low"
        assert state["delegation_profile"] == "general"
        assert state["runtime_mode"] == "default"
        assert state["activation_applied"] is False
        assert "multimodal tools unavailable" in state["activation_reason"]
        assert captured_api_kwargs["messages"][-1]["content"] == "Inspect this PDF diagram and explain the visual flow."

    def test_run_conversation_stores_and_applies_top_level_runtime_activation(self):
        agent = _make_agent("delegate_task")
        captured_api_kwargs = {}

        def fake_api_call(api_kwargs):
            captured_api_kwargs.update(api_kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Done.", tool_calls=[]),
                    )
                ],
                usage=None,
            )

        with patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call):
            result = agent.run_conversation("Use ultrawork mode to implement the patch and run tests.")

        assert result["final_response"] == "Done."
        state = agent.get_runtime_activation_state()
        assert state["specialist"] == "builder"
        assert state["archetype"] == "implementer"
        assert state["route_category"] == "deep"
        assert state["delegation_profile"] == "implementation"
        assert state["runtime_mode"] == "ultrawork"
        assert state["activation_applied"] is True
        assert state["task_contract"] is None
        assert "specialist: builder" in state["activation_note"]
        assert "# Wave 1 Prompt Overlays" in state["wave1_overlay_prompt"]

        injected_user_message = captured_api_kwargs["messages"][-1]["content"]
        assert injected_user_message.startswith("Use ultrawork mode to implement the patch and run tests.")
        assert "<wave2-runtime-activation>" in injected_user_message
        assert "specialist: builder" in injected_user_message
        assert "## Archetype" in injected_user_message
        assert "## Route Category" in injected_user_message
        assert "## Delegation Profile" in injected_user_message
        assert "## Runtime Mode" in injected_user_message

    def test_runtime_activation_snapshot_is_persisted_without_polluting_transcript(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-12345678",
                base_url="https://example.test/v1",
                provider="custom",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id="snapshot-session",
            )
        agent.client = _FakeClient()

        def fake_api_call(_api_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Done.", tool_calls=[]),
                    )
                ],
                usage=None,
            )

        classification = preclassify_intent(
            {
                "message": "Implement the delegated change in ultrawork mode.",
                "task_contract": _sample_task_contract(),
            }
        )

        with (
            patch("run_agent.preclassify_intent", return_value=classification),
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        ):
            result = agent.run_conversation("Implement the delegated change in ultrawork mode.")

        assert result["final_response"].startswith(
            "Task contract completion gate blocked success: required tool(s) unavailable in this runtime:"
        )
        snapshot = agent.get_runtime_activation_snapshot()
        assert snapshot["current"]["archetype"] == "implementer"
        assert snapshot["current"]["runtime_mode"] == "ultrawork"
        assert snapshot["current"]["task_contract_present"] is True
        assert "Implement the delegated change" in (snapshot["current"]["task_contract_summary"] or "")
        assert snapshot["last"] is None

        session_row = db.get_session("snapshot-session")
        assert session_row["runtime_activation_snapshot"] == snapshot

        transcript = db.get_messages_as_conversation("snapshot-session")
        assert transcript[0]["role"] == "user"
        assert "<wave2-runtime-activation>" not in (transcript[0]["content"] or "")
        db.close()

    def test_run_conversation_accepts_prepared_handoff_command_objects(self):
        agent = _make_agent()
        prepared = prepare_work_command(
            "handoff",
            raw_args='{"task":"Resume the delegated task","expected_outcome":"Done","required_skills":["python"],"required_tools":["terminal"],"must_do":["inspect state"],"must_not_do":["discard context"],"context":{"ticket":"p1"}}',
            session_id="sess-1",
            cwd="/tmp",
        )
        captured_api_kwargs = {}

        def fake_api_call(api_kwargs):
            captured_api_kwargs.update(api_kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Done.", tool_calls=[]),
                    )
                ],
                usage=None,
            )

        with patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call):
            result = agent.run_conversation(prepared)

        assert result["final_response"].startswith(
            "Task contract completion gate blocked success: required tool(s) unavailable in this runtime:"
        )
        state = agent.get_runtime_activation_state()
        assert state["task_contract"] == prepared.task_contract
        assert state["named_workflow"] is None
        assert captured_api_kwargs["messages"][-1]["content"].startswith("[OMO command handoff]")

    def test_run_conversation_supports_leading_named_agent_invocation_for_oracle(self):
        agent = _make_agent("read_file", "search_files", "web_search", "terminal", "patch")
        captured_api_kwargs = {}

        def fake_api_call(api_kwargs):
            captured_api_kwargs.update(api_kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Done.", tool_calls=[]),
                    )
                ],
                usage=None,
            )

        with patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call):
            result = agent.run_conversation("@oracle Compare the docs and summarize the answer.")

        assert result["final_response"] == "Done."
        state = agent.get_runtime_activation_state()
        assert state["named_agent"] == "oracle"
        assert state["specialist"] == "consultant"
        assert state["archetype"] == "researcher"
        assert state["route_category"] == "deep"
        assert state["delegation_profile"] == "research"
        assert state["runtime_mode"] == "default"
        assert state["activation_applied"] is True
        assert "named_agent: oracle" in state["activation_note"]
        assert "named-agent invocation" in state["activation_reason"]

        injected_user_message = captured_api_kwargs["messages"][-1]["content"]
        assert injected_user_message.startswith("Compare the docs and summarize the answer.")
        assert "@oracle" not in injected_user_message
        assert "<wave2-runtime-activation>" in injected_user_message

        assert agent._maybe_block_named_role_tool_call("terminal", {"command": "git status"}) == (
            "Named role runtime boundary (oracle): tool 'terminal' is blocked for this role. "
            "Use the role's permitted tools instead."
        )

    def test_run_conversation_canonicalizes_leading_named_agent_alias_invocation(self):
        agent = _make_agent("read_file", "search_files", "web_search", "terminal", "patch")
        captured_api_kwargs = {}

        def fake_api_call(api_kwargs):
            captured_api_kwargs.update(api_kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Explored.", tool_calls=[]),
                    )
                ],
                usage=None,
            )

        with patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call):
            result = agent.run_conversation("@explorer Trace the relevant files.")

        assert result["final_response"] == "Explored."
        state = agent.get_runtime_activation_state()
        assert state["named_agent"] == "explore"
        assert state["specialist"] == "investigator"
        assert state["archetype"] == "researcher"
        assert state["route_category"] == "deep"
        assert state["delegation_profile"] == "research"
        assert captured_api_kwargs["messages"][-1]["content"].startswith("Trace the relevant files.")
        assert "@explorer" not in captured_api_kwargs["messages"][-1]["content"]

    def test_prepared_work_command_uses_display_text_for_hooks_and_memory_sync(self):
        agent = _make_agent()
        prepared = prepare_work_command(
            "handoff",
            raw_args='{"task":"Resume the delegated task","expected_outcome":"Done","required_skills":["python"],"required_tools":["terminal"],"must_do":["inspect state"],"must_not_do":["discard context"],"context":{"ticket":"p1"}}',
            session_id="sess-1",
            cwd="/tmp",
        )
        sync_calls = []
        prefetch_calls = []
        hook_calls = []

        class _FakeMemoryManager:
            def sync_all(self, user_message, assistant_response):
                sync_calls.append((user_message, assistant_response))

            def queue_prefetch_all(self, user_message):
                prefetch_calls.append(user_message)

        def fake_invoke_hook(name, **kwargs):
            if name == "post_llm_call":
                hook_calls.append(kwargs)

        agent._memory_manager = _FakeMemoryManager()

        with (
            patch.object(agent, "_interruptible_api_call", return_value=SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Done.", tool_calls=[]))],
                usage=None,
            )),
            patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook),
        ):
            result = agent.run_conversation(prepared)

        assert result["final_response"].startswith(
            "Task contract completion gate blocked success: required tool(s) unavailable in this runtime:"
        )
        assert sync_calls == [(prepared.display_text, result["final_response"])]
        assert prefetch_calls == [prepared.display_text]
        assert hook_calls[-1]["user_message"] == prepared.display_text
        assert isinstance(hook_calls[-1]["user_message"], str)

    def test_runtime_activation_snapshot_tracks_current_and_last_across_session_reload(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-12345678",
                base_url="https://example.test/v1",
                provider="custom",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id="reload-session",
            )
        agent.client = _FakeClient()

        def fake_api_call(_api_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Done.", tool_calls=[]),
                    )
                ],
                usage=None,
            )

        with patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call):
            agent.run_conversation("Use ultrawork mode to implement the patch and run tests.")
            agent.run_conversation("hello", conversation_history=db.get_messages_as_conversation("reload-session"))

        snapshot = db.get_session("reload-session")["runtime_activation_snapshot"]
        assert snapshot["current"]["runtime_mode"] == "default"
        assert snapshot["last"]["runtime_mode"] == "ultrawork"
        assert snapshot["last"]["archetype"] == "implementer"

        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            reloaded = AIAgent(
                api_key="test-key-12345678",
                base_url="https://example.test/v1",
                provider="custom",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id="reload-session",
            )

        assert reloaded.get_runtime_activation_snapshot() == snapshot
        db.close()

    def test_unknown_preclassifier_outputs_degrade_safely_without_activation_overlay(self):
        agent = _make_agent()
        captured_api_kwargs = {}

        bogus = SimpleNamespace(
            inferred_specialist="ghost_specialist",
            inferred_archetype="ghost_archetype",
            inferred_route_category="ghost_lane",
            inferred_runtime_mode="ghost_runtime",
            inferred_delegation_profile="",
            activation_reason="classifier returned unknown values",
            task_contract=None,
            inference_source="test_preclassifier",
        )

        def fake_api_call(api_kwargs):
            captured_api_kwargs.update(api_kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Safe fallback.", tool_calls=[]),
                    )
                ],
                usage=None,
            )

        with (
            patch("run_agent.preclassify_intent", return_value=bogus),
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        ):
            result = agent.run_conversation("hello")

        assert result["final_response"] == "Safe fallback."
        state = agent.get_runtime_activation_state()
        assert state["specialist"] is None
        assert state["archetype"] == "generalist"
        assert state["route_category"] == "unspecified_low"
        assert state["delegation_profile"] == "general"
        assert state["runtime_mode"] == "default"
        assert state["activation_applied"] is False
        assert captured_api_kwargs["messages"][-1]["content"] == "hello"
