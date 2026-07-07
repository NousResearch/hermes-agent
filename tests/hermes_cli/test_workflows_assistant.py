import json

import pytest

from hermes_cli.workflows_assistant import (
    AssistantValidationError,
    WorkflowDraftResult,
    build_draft_prompt,
    draft_workflow,
    parse_assistant_payload,
    refine_workflow,
)


def _valid_payload():
    return {
        "spec": {
            "id": "code_review_flow",
            "name": "Code Review Flow",
            "version": 1,
            "triggers": [{"type": "manual", "id": "manual"}],
            "nodes": {
                "implement": {
                    "type": "agent_task",
                    "profile": "implementer",
                    "title": "Implement change",
                    "prompt": "Implement ${ input.request } and return JSON.",
                },
                "done": {"type": "pass", "output": {"status": "ok"}},
            },
            "edges": [{"from": "implement", "to": "done"}],
        },
        "summary": "Implements a change then marks it done.",
        "assumptions": ["Manual trigger for first version."],
        "questions": [],
        "warnings": [],
        "unsupported_requests": [],
    }


def test_parse_assistant_payload_returns_validated_draft_result():
    result = parse_assistant_payload(_valid_payload())

    assert isinstance(result, WorkflowDraftResult)
    assert result.spec.id == "code_review_flow"
    assert result.valid is True
    assert result.validation_errors == []
    assert result.summary.startswith("Implements")
    assert result.spec.nodes["implement"].type == "agent_task"


def test_parse_assistant_payload_rejects_unsupported_runtime_primitives():
    payload = _valid_payload()
    payload["spec"]["nodes"]["notify"] = {"type": "send_message", "output": {}}
    payload["spec"]["edges"].append({"from": "done", "to": "notify"})

    with pytest.raises(AssistantValidationError) as exc:
        parse_assistant_payload(payload)

    assert "unsupported node type" in str(exc.value)
    assert "send_message" in str(exc.value)


def test_parse_assistant_payload_returns_clear_validation_errors():
    payload = _valid_payload()
    del payload["spec"]["nodes"]["implement"]["profile"]

    with pytest.raises(AssistantValidationError) as exc:
        parse_assistant_payload(payload)

    assert "agent_task node implement requires a non-blank profile" in str(exc.value)


def test_draft_workflow_calls_runner_with_plain_goal_and_returns_valid_spec():
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(_valid_payload())

    from hermes_cli.workflows_assistant import draft_workflow

    result = draft_workflow(
        "When I ask for a code change, have an implementer do it.",
        runner=fake_runner,
    )

    assert result.spec.id == "code_review_flow"
    assert "When I ask for a code change" in calls[0]
    assert "Return JSON only" in calls[0]
    assert "send_message" in calls[0]  # listed as unsupported, not allowed


def test_refine_workflow_includes_current_spec_and_instruction():
    payload = _valid_payload()
    current = parse_assistant_payload(payload).spec
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        payload["summary"] = "Added reviewer step."
        return json.dumps(payload)

    from hermes_cli.workflows_assistant import refine_workflow

    result = refine_workflow(current, "Add a reviewer after implement", runner=fake_runner)

    assert result.summary == "Added reviewer step."
    assert "Add a reviewer after implement" in calls[0]
    assert '"nodes"' in calls[0]


def test_draft_workflow_repairs_once_after_invalid_first_response():
    responses = [
        json.dumps({"summary": "bad", "spec": {"id": "bad", "name": "Bad", "version": 1, "nodes": {}}}),
        json.dumps(_valid_payload()),
    ]

    def fake_runner(prompt: str) -> str:
        return responses.pop(0)

    from hermes_cli.workflows_assistant import draft_workflow

    result = draft_workflow("Build a valid workflow", runner=fake_runner, repair_attempts=1)
    assert result.spec.id == "code_review_flow"


def test_draft_workflow_rejects_blank_goal_without_calling_runner():
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(_valid_payload())

    with pytest.raises(AssistantValidationError, match="workflow goal is required"):
        draft_workflow(" \t\n", runner=fake_runner)

    assert calls == []


def test_refine_workflow_rejects_blank_instruction_without_calling_runner():
    current = parse_assistant_payload(_valid_payload()).spec
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(_valid_payload())

    with pytest.raises(AssistantValidationError, match="refine instruction is required"):
        refine_workflow(current, " \t\n", runner=fake_runner)

    assert calls == []


def test_draft_workflow_honors_multiple_repair_attempts():
    responses = [
        json.dumps({"summary": "bad", "spec": {"id": "bad", "name": "Bad", "version": 1, "nodes": {}}}),
        json.dumps({"summary": "bad", "spec": {"id": "also_bad", "name": "Also Bad", "version": 1, "nodes": {}}}),
        json.dumps(_valid_payload()),
    ]
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        return responses.pop(0)

    result = draft_workflow("Build a valid workflow", runner=fake_runner, repair_attempts=2)

    assert result.spec.id == "code_review_flow"
    assert len(calls) == 3


def test_draft_prompt_uses_valid_empty_edges_example():
    prompt = build_draft_prompt("Build a valid workflow")

    assert '"edges": []' in prompt
    assert "next_node_id" not in prompt


def test_default_runner_uses_agent_runtime_adapter(monkeypatch):
    from hermes_cli import workflows_assistant as wa

    calls = []
    agents = []
    pool = object()

    class FakeAgent:
        def __init__(
            self,
            *,
            model,
            api_key,
            base_url,
            provider,
            api_mode,
            acp_command,
            acp_args,
            credential_pool,
            enabled_toolsets,
            quiet_mode,
            skip_context_files,
            skip_memory,
            platform,
            max_iterations,
        ):
            kwargs = locals().copy()
            kwargs.pop("self")
            calls.append(("init", kwargs))
            agents.append(self)

        def run_conversation(self, prompt):
            calls.append(("run", prompt))
            return json.dumps(_valid_payload())

    monkeypatch.setattr(wa, "AIAgent", FakeAgent)
    monkeypatch.setattr(
        wa,
        "_resolve_assistant_runtime",
        lambda: {"model": "fake", "provider": "fake", "credential_pool": pool},
    )

    result = wa.draft_workflow_with_default_runner("Build a demo workflow")

    assert result.spec.id == "code_review_flow"
    assert calls[0][0] == "init"
    assert calls[0][1]["credential_pool"] is pool
    assert calls[0][1]["enabled_toolsets"] == []
    assert calls[0][1]["skip_memory"] is True
    assert getattr(agents[0], "suppress_status_output", None) is True
    assert calls[1][0] == "run"


def test_resolve_assistant_runtime_does_not_load_missing_credential_pool(monkeypatch):
    from hermes_cli import workflows_assistant as wa

    calls = []

    def fake_resolve_runtime_provider(*, requested, target_model):
        calls.append(("resolve", requested, target_model))
        return {
            "model": "runtime-model",
            "provider": "openai",
            "api_key": "runtime-key",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
        }

    def fake_load_pool(provider):
        calls.append(("load_pool", provider))
        raise AssertionError(f"load_pool should not be called for {provider}")

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "cfg-model"}})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve_runtime_provider)
    monkeypatch.setattr("agent.credential_pool.load_pool", fake_load_pool)

    runtime = wa._resolve_assistant_runtime()

    assert runtime["model"] == "runtime-model"
    assert runtime["provider"] == "openai"
    assert "credential_pool" not in runtime
    assert calls == [("resolve", None, "cfg-model")]


def test_resolve_assistant_runtime_preserves_resolver_credential_pool(monkeypatch):
    from hermes_cli import workflows_assistant as wa

    pool = object()
    pool_calls = []

    def fake_resolve_runtime_provider(*, requested, target_model):
        assert requested is None
        assert target_model == "cfg-model"
        return {"model": "runtime-model", "provider": "openai", "credential_pool": pool}

    def fake_load_pool(provider):
        pool_calls.append(provider)
        raise AssertionError(f"load_pool should not be called for {provider}")

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "cfg-model"}})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve_runtime_provider)
    monkeypatch.setattr("agent.credential_pool.load_pool", fake_load_pool)

    runtime = wa._resolve_assistant_runtime()

    assert runtime["credential_pool"] is pool
    assert runtime["model"] == "runtime-model"
    assert pool_calls == []
