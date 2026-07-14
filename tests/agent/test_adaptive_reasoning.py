import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from agent.adaptive_reasoning import (
    apply_model_reasoning_directive,
    attach_reasoning_receipt,
    configure_adaptive_reasoning,
    effective_reasoning_config,
    preserve_verified_reasoning_payload,
    refresh_adaptive_reasoning_policy,
    reset_adaptive_reasoning_turn,
)
from agent.transports.codex import ResponsesApiTransport
from hermes_cli.config import DEFAULT_CONFIG
from tools.todo_tool import TODO_SCHEMA


def _agent(*, reasoning_config=None, cap="xhigh"):
    agent = SimpleNamespace(
        api_mode="codex_responses",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        reasoning_config=reasoning_config,
        _current_turn_id="turn-1",
    )
    configure_adaptive_reasoning(
        agent,
        {
            "adaptive_reasoning": {
                "enabled": True,
                "max_effort": cap,
            }
        },
    )
    reset_adaptive_reasoning_turn(agent, "turn-1")
    return agent


def _apply(agent, directive, *, turn_id=None):
    return apply_model_reasoning_directive(
        agent,
        directive,
        originating_turn_id=(
            str(getattr(agent, "_current_turn_id", "") or "")
            if turn_id is None
            else turn_id
        ),
    )


def test_quality_first_baseline_and_model_authored_escalation_are_turn_scoped():
    agent = _agent()
    assert effective_reasoning_config(agent) == {"enabled": True, "effort": "high"}

    receipt = _apply(agent, {"effort": "xhigh"})
    assert receipt["status"] == "applied"
    assert effective_reasoning_config(agent) == {
        "enabled": True,
        "effort": "xhigh",
    }

    agent._current_turn_id = "turn-2"
    assert effective_reasoning_config(agent) == {"enabled": True, "effort": "high"}
    reset_adaptive_reasoning_turn(agent, "turn-2")
    assert effective_reasoning_config(agent) == {"enabled": True, "effort": "high"}


def test_verified_route_has_static_high_floor_without_task_classification():
    for configured in ("minimal", "low", "medium", "high", "", "unknown"):
        agent = _agent(reasoning_config={"enabled": True, "effort": configured})
        assert effective_reasoning_config(agent) == {
            "enabled": True,
            "effort": "high",
        }

    # A deeper explicit operator baseline remains authoritative. The adaptive
    # action only ever raises from that baseline and never classifies task text.
    agent = _agent(reasoning_config={"enabled": True, "effort": "xhigh"})
    assert effective_reasoning_config(agent) == {
        "enabled": True,
        "effort": "xhigh",
    }
    receipt = _apply(agent, {"effort": "high"})
    assert receipt["status"] == "rejected"
    assert receipt["reason"] == "below_user_baseline"
    assert receipt["baseline_effort"] == "xhigh"


def test_static_defaults_are_model_gated_bounded_and_pro_absent():
    assert DEFAULT_CONFIG["agent"]["reasoning_effort"] == ""
    assert DEFAULT_CONFIG["agent"]["adaptive_reasoning"] == {
        "enabled": True,
        "max_effort": "xhigh",
    }

    agent = _agent()
    configure_adaptive_reasoning(agent, {"adaptive_reasoning": False})
    receipt = _apply(agent, {"effort": "xhigh"})
    assert receipt["reason"] == "adaptive_reasoning_disabled"

    agent = _agent()
    configure_adaptive_reasoning(
        agent, {"adaptive_reasoning": {"enabled": "true"}}
    )
    assert _apply(agent, {"effort": "xhigh"})["reason"] == "adaptive_reasoning_disabled"

    agent = _agent()
    receipt = _apply(agent, {"effort": "xhigh", "mode": "pro"})
    assert receipt["reason"] == "reasoning_mode_unverified"
    assert "mode" not in TODO_SCHEMA["parameters"]["properties"]["reasoning"]["properties"]


def test_malformed_adaptive_policy_sections_and_caps_fail_closed():
    malformed_sections = ("true", ["enabled"], 1)
    for section in malformed_sections:
        agent = _agent()
        configure_adaptive_reasoning(agent, {"adaptive_reasoning": section})
        receipt = _apply(agent, {"effort": "xhigh"})
        assert receipt["reason"] == "adaptive_reasoning_disabled"

    # Authority policy is exact-shape.  In particular, adding a purported
    # external selector cannot coexist with model-authored adaptive effort.
    for extra in ("effort_router", "classifier", "semantic_dispatch"):
        agent = _agent()
        configure_adaptive_reasoning(
            agent,
            {
                "adaptive_reasoning": {
                    "enabled": True,
                    "max_effort": "xhigh",
                    extra: "external",
                }
            },
        )
        receipt = _apply(agent, {"effort": "xhigh"})
        assert receipt["reason"] == "adaptive_reasoning_disabled"

    for cap in ("ultra", 7, ["xhigh"]):
        agent = _agent()
        configure_adaptive_reasoning(
            agent,
            {"adaptive_reasoning": {"enabled": True, "max_effort": cap}},
        )
        receipt = _apply(agent, {"effort": "xhigh"})
        assert receipt["reason"] == "adaptive_reasoning_disabled"


def test_policy_is_upward_only_bounded_and_idempotent():
    agent = _agent(reasoning_config={"enabled": True, "effort": "high"})

    below = _apply(agent, {"effort": "medium"})
    assert below == {
        "status": "rejected",
        "scope": "current_turn",
        "expires": "end_of_current_turn",
        "reason": "below_user_baseline",
        "baseline_effort": "high",
    }

    capped = _apply(agent, {"effort": "max"})
    assert capped["status"] == "rejected"
    assert capped["reason"] == "above_policy_cap"
    assert capped["max_effort"] == "xhigh"

    first = _apply(agent, {"effort": "xhigh"})
    second = _apply(agent, {"effort": "xhigh"})
    assert first["status"] == "applied"
    assert second["status"] == "unchanged"
    assert first["change_count"] == second["change_count"] == 1


def test_directive_runtime_boundary_rejects_non_schema_fields_without_mutation():
    """Provider-side schema enforcement is useful but not an authority rail."""
    agent = _agent(reasoning_config={"enabled": True, "effort": "high"})

    for directive in (
        {},
        {"effort": "xhigh", "router": "external"},
        {"effort": "xhigh", "task_class": "complex"},
        {"effort": "xhigh", "next_model": "other"},
    ):
        receipt = _apply(agent, directive)
        assert receipt["status"] == "rejected"
        assert receipt["reason"] == "reasoning_shape_invalid"
        assert effective_reasoning_config(agent) == {
            "enabled": True,
            "effort": "high",
        }

    assert agent._turn_reasoning_changes == 0
    assert agent._turn_reasoning_override is None


def test_turn_override_is_monotonic_across_concurrent_completion_order():
    for first, second in (("high", "xhigh"), ("xhigh", "high")):
        agent = _agent()
        _apply(agent, {"effort": first})
        _apply(agent, {"effort": second})
        assert effective_reasoning_config(agent)["effort"] == "xhigh"

    agent = _agent()
    barrier = threading.Barrier(2)

    def apply(effort):
        barrier.wait()
        return _apply(agent, {"effort": effort})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(apply, "high"), pool.submit(apply, "xhigh")]
        for future in futures:
            future.result()

    assert effective_reasoning_config(agent) == {
        "enabled": True,
        "effort": "xhigh",
    }


def test_late_prior_turn_worker_cannot_mutate_new_turn():
    agent = _agent()
    worker_ready = threading.Event()
    allow_completion = threading.Event()

    def late_worker():
        worker_ready.set()
        allow_completion.wait(timeout=2)
        return _apply(agent, {"effort": "xhigh"}, turn_id="turn-1")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(late_worker)
        assert worker_ready.wait(timeout=2)
        agent._current_turn_id = "turn-2"
        reset_adaptive_reasoning_turn(agent, "turn-2")
        allow_completion.set()
        receipt = future.result(timeout=2)

    assert receipt["status"] == "rejected"
    assert receipt["reason"] == "originating_turn_expired"
    assert effective_reasoning_config(agent) == {"enabled": True, "effort": "high"}


def test_verified_backend_gate_rejects_copilot_github_and_custom_routes():
    cases = [
        ("copilot", "https://api.githubcopilot.com", "gpt-5.6-sol"),
        ("openai-codex", "https://models.github.ai/inference", "gpt-5.6-sol"),
        ("openai-codex", "https://example.com/backend-api/codex", "gpt-5.6-sol"),
        ("openai-codex", "http://chatgpt.com/backend-api/codex", "gpt-5.6-sol"),
        (
            "openai-codex",
            "https://chatgpt.com/backend-api/codex",
            "gpt-5.6",
        ),
        (
            "openai-codex",
            "https://chatgpt.com/backend-api/codex",
            "gpt-5.6-sol-pro",
        ),
        (
            "openai-codex",
            "https://chatgpt.com/backend-api/codex",
            "openai/gpt-5.6-sol",
        ),
        (
            "openai-codex",
            "https://chatgpt.com/backend-api/codex?proxy=1",
            "gpt-5.6-sol",
        ),
    ]
    for provider, base_url, model in cases:
        agent = _agent()
        agent.provider = provider
        agent.base_url = base_url
        agent.model = model
        receipt = _apply(agent, {"effort": "xhigh"})
        assert receipt["reason"] == "unsupported_model_or_transport"
        assert effective_reasoning_config(agent) is None


def test_nonverified_routes_preserve_historical_reasoning_config_exactly():
    agent = _agent()
    original = {
        "enabled": "legacy-value",
        "effort": "minimal",
        "provider_extension": {"keep": True},
    }
    agent.reasoning_config = original
    agent.provider = "copilot"
    agent.base_url = "https://api.githubcopilot.com"
    assert effective_reasoning_config(agent) is original


def test_verified_reasoning_payload_cannot_be_rewritten_by_middleware():
    agent = _agent()
    authoritative = {
        "model": "gpt-5.6-sol",
        "reasoning": {"effort": "xhigh"},
    }
    rewritten = {
        "model": "gpt-5.5",
        "reasoning": {"effort": "low"},
        "metadata": {"plugin": True},
    }
    preserved = preserve_verified_reasoning_payload(
        agent,
        authoritative,
        rewritten,
    )
    assert preserved["model"] == "gpt-5.6-sol"
    assert preserved["reasoning"] == {"effort": "xhigh"}
    assert preserved["metadata"] == {"plugin": True}

    agent.provider = "copilot"
    assert (
        preserve_verified_reasoning_payload(agent, authoritative, rewritten)
        is rewritten
    )


def test_live_policy_refresh_revokes_cap_and_disable_without_prompt_mutation():
    agent = _agent()
    agent._cached_system_prompt = "stable prompt"
    agent.tools = [{"type": "function", "name": "todo"}]
    prompt = agent._cached_system_prompt
    tools = agent.tools
    assert _apply(agent, {"effort": "xhigh"})["status"] == "applied"

    refresh_adaptive_reasoning_policy(
        agent,
        {"adaptive_reasoning": {"enabled": True, "max_effort": "high"}},
    )
    assert effective_reasoning_config(agent) == {"enabled": True, "effort": "high"}
    assert _apply(agent, {"effort": "xhigh"})["reason"] == "above_policy_cap"

    refresh_adaptive_reasoning_policy(agent, {"adaptive_reasoning": False})
    assert _apply(agent, {"effort": "high"})["reason"] == "adaptive_reasoning_disabled"
    assert agent._cached_system_prompt is prompt
    assert agent.tools is tools


def test_user_disable_is_authoritative():
    disabled = _agent(reasoning_config={"enabled": False})
    receipt = _apply(disabled, {"effort": "xhigh"})
    assert receipt["reason"] == "reasoning_disabled_by_user"
    assert effective_reasoning_config(disabled) == {"enabled": False}


def test_todo_receipt_applies_only_after_successful_tool_result():
    agent = _agent()
    result = attach_reasoning_receipt(
        agent,
        json.dumps({"todos": [], "summary": {"total": 0}}),
        {"effort": "xhigh"},
        originating_turn_id="turn-1",
    )
    payload = json.loads(result)
    assert payload["reasoning_control"]["status"] == "applied"
    assert effective_reasoning_config(agent)["effort"] == "xhigh"

    other = _agent()
    error = '{"error":"todo failed"}'
    assert (
        attach_reasoning_receipt(
            other,
            error,
            {"effort": "xhigh"},
            originating_turn_id="turn-1",
        )
        == error
    )
    assert effective_reasoning_config(other)["effort"] == "high"


def test_todo_schema_exposes_static_model_authored_effort_directive():
    reasoning = TODO_SCHEMA["parameters"]["properties"]["reasoning"]
    assert reasoning["properties"]["effort"]["enum"] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert set(reasoning["properties"]) == {"effort"}


def test_next_call_changes_effort_without_changing_cache_prefix_or_roles():
    agent = _agent()
    transport = ResponsesApiTransport()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "todo",
                "description": "plan",
                "parameters": TODO_SCHEMA["parameters"],
            },
        }
    ]
    system = "Stable system prompt"
    first_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Solve the complex task"},
    ]
    first = transport.build_kwargs(
        model=agent.model,
        messages=first_messages,
        tools=tools,
        reasoning_config=effective_reasoning_config(agent),
        session_id="session-1",
        is_codex_backend=True,
    )

    tool_arguments = json.dumps(
        {
            "todos": [{"id": "1", "content": "solve", "status": "in_progress"}],
            "reasoning": {"effort": "xhigh"},
        },
        separators=(",", ":"),
    )
    receipt = attach_reasoning_receipt(
        agent,
        json.dumps({"todos": [], "summary": {"total": 0}}),
        {"effort": "xhigh"},
        originating_turn_id="turn-1",
    )
    second_messages = first_messages + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-plan",
                    "type": "function",
                    "function": {"name": "todo", "arguments": tool_arguments},
                }
            ],
        },
        {
            "role": "tool",
            "name": "todo",
            "tool_call_id": "call-plan",
            "content": receipt,
        },
    ]
    second = transport.build_kwargs(
        model=agent.model,
        messages=second_messages,
        tools=tools,
        reasoning_config=effective_reasoning_config(agent),
        session_id="session-1",
        is_codex_backend=True,
    )

    assert first["reasoning"]["effort"] == "high"
    assert second["reasoning"]["effort"] == "xhigh"
    assert first["instructions"] == second["instructions"] == system
    assert first["tools"] == second["tools"]
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert not any(
        item.get("role") == "user" and "reasoning" in str(item.get("content"))
        for item in second["input"]
        if isinstance(item, dict)
    )
    assert {item.get("type") for item in second["input"]} >= {
        "function_call",
        "function_call_output",
    }
