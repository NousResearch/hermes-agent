import json
import queue
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.continuation_engine import should_use_continuation_engine
from agent.intent_preclassifier import preclassify_intent
from agent.task_contracts import ORCHESTRATION_HINTS_SCHEMA, ORCHESTRATION_HINTS_VERSION
from hermes_cli.command_templates import WORK_STARTING_COMMANDS, build_command_invocation
from hermes_cli.commands import resolve_command
from hermes_cli.work_command_adapter import PreparedWorkCommand, prepare_work_command


def _sample_contract_payload() -> dict:
    return {
        "task": "Resume the delegated task",
        "expected_outcome": "A resumed implementation with fresh verification evidence",
        "required_skills": ["python", "testing"],
        "required_tools": ["read_file", "patch", "terminal"],
        "must_do": ["inspect current state before acting"],
        "must_not_do": ["do not discard the handed-off contract"],
        "context": {"ticket": "wave5-smoke"},
    }


def _normalized_cwd(cwd: str) -> str:
    return str(Path(cwd).resolve())


def _active_snapshot() -> dict:
    return {
        "outcomeStatus": "interrupted",
        "activeTodos": [{"id": "todo-1", "content": "Finish the delegated task", "status": "in_progress"}],
    }


class TestOMOWave5CommandTemplates:
    def test_registry_contains_wave5_commands(self):
        for name in {"handoff", "stop-continuation", *WORK_STARTING_COMMANDS}:
            assert resolve_command(name) is not None

    @pytest.mark.parametrize("command_name", sorted(WORK_STARTING_COMMANDS))
    def test_work_commands_build_structured_task_contracts(self, command_name):
        invocation = build_command_invocation(
            command_name,
            raw_args="Implement the requested change",
            session_id="sess-1",
            cwd="/tmp",
        )

        assert invocation.task_contract["context"]["command"] == command_name
        assert invocation.orchestration_hints["schema"] == ORCHESTRATION_HINTS_SCHEMA
        assert invocation.orchestration_hints["schema_version"] == ORCHESTRATION_HINTS_VERSION
        assert invocation.orchestration_hints["bounded_context"]["enabled"] is True
        assert invocation.orchestration_hints["bounded_context"]["task_contract_precedence"] == "preserve_existing_fields"
        assert "TASK_CONTRACT_JSON:" in invocation.prompt_text
        assert json.loads(invocation.prompt_text.split("TASK_CONTRACT_JSON:\n", 1)[1].split("\nORCHESTRATION_HINTS_JSON:", 1)[0]) == invocation.task_contract

    def test_start_work_emits_named_workflow_machine_readably(self):
        invocation = build_command_invocation(
            "start-work",
            raw_args="Implement the requested change",
            session_id="sess-1",
            cwd="/tmp",
        )

        assert "NAMED_WORKFLOW_JSON:" in invocation.prompt_text
        named_workflow = json.loads(
            invocation.prompt_text.split("NAMED_WORKFLOW_JSON:\n", 1)[1].split("\nUSER_REQUEST:", 1)[0]
        )
        assert named_workflow["schema"] == "hermes/named-workflow"
        assert named_workflow["workflow_name"] == "deep_worker"
        assert named_workflow["execution_task_contract"] == invocation.task_contract

    def test_handoff_does_not_emit_named_workflow_by_default(self):
        invocation = build_command_invocation(
            "handoff",
            raw_args="Resume the delegated task",
            session_id="sess-1",
            cwd="/tmp",
        )

        assert invocation.named_workflow is None
        assert "NAMED_WORKFLOW_JSON:" not in invocation.prompt_text

    def test_loop_commands_emit_distinct_command_runtime_markers(self):
        cases = [
            ("ralph-loop", "ralph"),
            ("ulw-loop", "ultrawork"),
        ]

        for command_name, expected_runtime_mode in cases:
            invocation = build_command_invocation(
                command_name,
                raw_args="Implement the requested change",
                session_id="sess-1",
                cwd="/tmp",
            )

            assert invocation.task_contract["context"]["command"] == command_name
            assert invocation.task_contract["context"]["loop_family"] == command_name
            assert invocation.task_contract["context"]["command_runtime"] == {
                "command_name": command_name,
                "runtime_mode": expected_runtime_mode,
                "continuation_semantics": (
                    {
                        "retry_on_failed_or_interrupted": True,
                        "stop_requires_explicit_exit": True,
                    }
                    if command_name == "ralph-loop"
                    else {
                        "completion_gate": "open_todos_block_done",
                        "require_open_work_closure": True,
                    }
                ),
            }
            assert invocation.orchestration_hints["command"] == command_name
            assert invocation.orchestration_hints["loop"] == {
                "family": command_name,
                "bounded": True,
                "continuation_semantics": invocation.task_contract["context"]["command_runtime"]["continuation_semantics"],
            }

    def test_handoff_consumes_structured_contract_json(self):
        payload = _sample_contract_payload()
        invocation = build_command_invocation("handoff", raw_args=json.dumps(payload), session_id="sess-1", cwd="/tmp")

        assert invocation.task_contract == payload
        assert invocation.orchestration_hints["handoff"]["consume_existing_contract"] is True
        assert invocation.orchestration_hints["request"] == payload["task"]
        assert invocation.orchestration_hints["invocation_metadata"] == {
            "command": "handoff",
            "session_id": "sess-1",
            "cwd": _normalized_cwd("/tmp"),
            "input_mode": "explicit_json_contract",
            "preserve_exact_task_contract": True,
            "handoff_mode": "resume_or_consume_contract",
        }
        assert "USER_REQUEST: Resume the delegated task" in invocation.prompt_text

    def test_start_work_preserves_explicit_contract_json_exactly(self):
        payload = _sample_contract_payload()
        payload["context"]["command_runtime"] = {
            "command_name": "handoff",
            "runtime_mode": "ralph",
        }

        invocation = build_command_invocation("start-work", raw_args=json.dumps(payload), session_id="sess-2", cwd="/tmp")

        assert invocation.task_contract == payload
        assert invocation.named_workflow is None
        assert "NAMED_WORKFLOW_JSON:" not in invocation.prompt_text
        assert invocation.orchestration_hints["request"] == payload["task"]
        assert invocation.orchestration_hints["invocation_metadata"] == {
            "command": "start-work",
            "session_id": "sess-2",
            "cwd": _normalized_cwd("/tmp"),
            "input_mode": "explicit_json_contract",
            "preserve_exact_task_contract": True,
        }
        result = preclassify_intent({"message": payload["task"], "task_contract": invocation.task_contract})
        assert result.inferred_runtime_mode == "ralph"
        assert should_use_continuation_engine(result.inferred_runtime_mode, _active_snapshot()) is True

    @pytest.mark.parametrize(
        ("command_name", "expected_runtime_mode"),
        [("ralph-loop", "ralph"), ("ulw-loop", "ultrawork")],
    )
    def test_loop_commands_enable_continuation_from_structured_contracts(self, command_name, expected_runtime_mode):
        invocation = build_command_invocation(
            command_name,
            raw_args="Finish the delegated task",
            session_id="sess-3",
            cwd="/tmp",
        )

        result = preclassify_intent({"message": "Finish the delegated task", "task_contract": invocation.task_contract})

        assert result.inferred_runtime_mode == expected_runtime_mode
        assert should_use_continuation_engine(result.inferred_runtime_mode, _active_snapshot()) is True

    def test_prepare_work_command_reports_malformed_handoff_json(self):
        with pytest.raises(ValueError, match=r"Malformed /handoff work command JSON"):
            prepare_work_command("handoff", raw_args="{not valid", session_id="sess-1", cwd="/tmp")

    def test_prepare_work_command_reports_invalid_contract_shape(self):
        with pytest.raises(ValueError, match=r"Invalid /handoff task contract: required_tools"):
            prepare_work_command(
                "handoff",
                raw_args=json.dumps({
                    "task": "Resume work",
                    "expected_outcome": "done",
                    "required_skills": ["python"],
                    "must_do": ["inspect"],
                    "must_not_do": ["skip verification"],
                    "context": {"ticket": "w2"},
                }),
                session_id="sess-1",
                cwd="/tmp",
            )


class TestCLICommandSmoke:
    def _make_cli(self):
        import agent.archetypes as archetypes

        if not hasattr(archetypes, "resolve_specialist_mapping"):
            archetypes.resolve_specialist_mapping = lambda *args, **kwargs: None
        if not hasattr(archetypes, "resolve_specialist_defaults"):
            archetypes.resolve_specialist_defaults = lambda *args, **kwargs: {}

        from cli import HermesCLI

        cli = object.__new__(HermesCLI)
        cli.session_id = "sess-cli"
        cli._pending_input = queue.Queue()
        return cli

    def test_handoff_command_queues_structured_prompt(self):
        cli = self._make_cli()

        assert cli.process_command("/handoff Continue the login fix") is True
        queued = cli._pending_input.get_nowait()

        assert isinstance(queued, PreparedWorkCommand)
        assert queued.command_name == "handoff"
        assert queued.task_contract["context"]["command"] == "handoff"
        assert queued.orchestration_hints["schema"] == ORCHESTRATION_HINTS_SCHEMA
        assert "[OMO command handoff]" in queued.agent_message
        assert "Continue the login fix" in queued.agent_message

    def test_init_deep_command_queues_structured_prompt(self):
        cli = self._make_cli()

        assert cli.process_command("/init-deep Investigate the migration surface") is True
        queued = cli._pending_input.get_nowait()

        assert isinstance(queued, PreparedWorkCommand)
        assert queued.command_name == "init-deep"
        assert queued.orchestration_hints["schema_version"] == ORCHESTRATION_HINTS_VERSION
        assert "[OMO command init-deep]" in queued.agent_message
        assert '"initialization": {' in queued.agent_message
        assert "Investigate the migration surface" in queued.agent_message

    @pytest.mark.parametrize(
        ("command_text", "expected_runtime_mode"),
        [
            ("/start-work ship it", "default"),
            ("/ralph-loop close it", "ralph"),
            ("/ulw-loop finish it", "ultrawork"),
        ],
    )
    def test_additional_work_commands_queue_structured_prompts_with_expected_continuation_modes(
        self, command_text, expected_runtime_mode
    ):
        cli = self._make_cli()

        assert cli.process_command(command_text) is True
        queued = cli._pending_input.get_nowait()

        assert isinstance(queued, PreparedWorkCommand)
        result = preclassify_intent({"message": queued.task_contract["task"], "task_contract": queued.task_contract})
        assert result.inferred_runtime_mode == expected_runtime_mode
        assert should_use_continuation_engine(result.inferred_runtime_mode, _active_snapshot()) is (
            expected_runtime_mode != "default"
        )
        if expected_runtime_mode == "default":
            assert "NAMED_WORKFLOW_JSON:" in queued.agent_message
        else:
            assert f'"runtime_mode": "{expected_runtime_mode}"' in queued.agent_message


class TestGatewayCommandSmoke:
    def _make_runner(self):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import GatewayRunner
        from gateway.session import SessionEntry

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
        runner.adapters = {}
        runner._voice_mode = {}
        runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
        runner.session_store = MagicMock()
        runner.session_store.get_or_create_session.return_value = SessionEntry(
            session_key="agent:main:telegram:dm:c1:u1",
            session_id="sess-gw",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        runner.session_store.load_transcript.return_value = []
        runner.session_store.has_any_sessions.return_value = True
        runner.session_store.append_to_transcript = MagicMock()
        runner.session_store.rewrite_transcript = MagicMock()
        runner._running_agents = {}
        runner._running_agents_ts = {}
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._session_db = None
        runner._reasoning_config = None
        runner._provider_routing = {}
        runner._fallback_model = None
        runner._show_reasoning = False
        runner._is_user_authorized = lambda _source: True
        runner._set_session_env = lambda _context: []
        runner._clear_session_env = lambda _tokens: None
        runner._should_send_voice_reply = lambda *_args, **_kwargs: False
        runner._send_voice_reply = AsyncMock()
        runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
        runner._emit_gateway_run_progress = AsyncMock()
        runner._prepare_inbound_message_text = AsyncMock(
            side_effect=lambda *, event, source, history: event.text
        )
        runner._run_agent = AsyncMock(
            return_value={
                "final_response": "ok",
                "messages": [],
                "tools": [],
                "history_offset": 0,
                "last_prompt_tokens": 0,
            }
        )
        runner._draining = False
        runner._restart_requested = False
        runner._background_tasks = set()
        runner._agent_cache = {}
        runner._agent_cache_lock = None
        runner._session_model_overrides = {}
        runner._update_prompt_pending = {}
        runner._failed_platforms = {}
        runner._busy_input_mode = "interrupt"
        return runner

    def _make_event(self, text="/handoff Continue the repo work"):
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        from gateway.session import SessionSource

        return MessageEvent(
            text=text,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                user_id="u1",
                chat_id="c1",
                user_name="tester",
                chat_type="dm",
            ),
            message_id="m1",
        )

    @pytest.mark.asyncio
    async def test_gateway_handoff_smoke_routes_structured_prompt_to_agent(self, monkeypatch):
        import gateway.run as gateway_run

        runner = self._make_runner()
        event = self._make_event()
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
        monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_args, **_kwargs: 100_000)

        result = await runner._handle_message(event)

        assert result == "ok"
        forwarded = runner._run_agent.call_args.kwargs["message"]
        assert "[OMO command handoff]" in forwarded
        assert '"command": "handoff"' in forwarded
        assert "Continue the repo work" in forwarded

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("command_text", "expected_runtime_mode", "should_continue"),
        [
            ("/handoff Continue the repo work", "default", False),
            ("/init-deep Investigate the repo work", "default", False),
            ("/start-work Continue the repo work", "default", False),
            ("/ralph-loop Close the repo work", "ralph", True),
            ("/ulw-loop Finish the repo work", "ultrawork", True),
        ],
    )
    async def test_gateway_work_commands_preserve_continuation_semantics(self, monkeypatch, command_text, expected_runtime_mode, should_continue):
        import gateway.run as gateway_run

        runner = self._make_runner()
        event = self._make_event(text=command_text)
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
        monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_args, **_kwargs: 100_000)

        result = await runner._handle_message(event)

        assert result == "ok"
        forwarded = runner._run_agent.call_args.kwargs["message"]
        prepared = getattr(event, "_prepared_work_command")
        classification = preclassify_intent({"message": prepared.task_contract["task"], "task_contract": prepared.task_contract})
        assert classification.inferred_runtime_mode == expected_runtime_mode
        assert should_use_continuation_engine(classification.inferred_runtime_mode, _active_snapshot()) is should_continue
        assert forwarded == prepared.agent_message

    @pytest.mark.asyncio
    async def test_gateway_handoff_returns_clean_error_for_malformed_json(self, monkeypatch):
        import gateway.run as gateway_run

        runner = self._make_runner()
        event = self._make_event(text="/handoff {not valid")
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
        monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_args, **_kwargs: 100_000)

        result = await runner._handle_message(event)

        assert result == "Malformed /handoff work command JSON: expected a valid JSON object."
        runner._run_agent.assert_not_called()
