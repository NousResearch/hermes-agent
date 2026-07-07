from types import SimpleNamespace

from agent.smart_model_router import evaluate_turn_route
from agent.smart_router_prompts import parse_router_json
from agent.task_features import extract_task_features


def test_task_features_detect_code_analysis_request():
    features = extract_task_features(
        "\u5206\u6790\u8fd9\u4e2a\u9879\u76ee\u7684 "
        "agent/conversation_loop.py "
        "\u5e76\u8bbe\u8ba1\u4fee\u6539\u65b9\u6848"
    )

    assert features.architecture_or_analysis is True
    assert features.likely_needs_file_access is True
    assert features.has_explicit_file_reference is True
    assert features.complexity_score >= 4


def test_router_off_returns_none():
    agent = SimpleNamespace(provider="openai", model="gpt-test", api_mode="chat_completions")

    decision = evaluate_turn_route(
        agent,
        "hello",
        config={"mode": "off"},
    )

    assert decision is None


def test_router_observe_heuristic_does_not_mutate_agent_model():
    agent = SimpleNamespace(provider="openai", model="gpt-test", api_mode="chat_completions")

    decision = evaluate_turn_route(
        agent,
        "fix the failing pytest in agent/router.py",
        messages=[],
        effective_task_id="task-1",
        config={
            "mode": "observe",
            "routes": {
                "strong": {"provider": "openrouter", "model": "strong-model"},
            },
        },
    )

    assert decision is not None
    assert decision.mode == "observe"
    assert decision.route in {"default", "strong", "moa"}
    assert agent.provider == "openai"
    assert agent.model == "gpt-test"


def test_parse_router_json_tolerates_fenced_json():
    parsed = parse_router_json(
        "```json\n"
        '{"route":"strong","confidence":0.8,"risk":"medium",'
        '"expected_tool_calls":4,"reason":"x","should_use_moa":false}'
        "\n```"
    )

    assert parsed["route"] == "strong"
    assert parsed["confidence"] == 0.8
