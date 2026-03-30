import json

from agent.auto_learning import (
    build_auto_learning_review_prompt,
    normalize_candidate,
    parse_auto_learning_review,
    should_promote_candidate,
)


def test_build_auto_learning_review_prompt_mentions_thresholds_and_json_contract():
    prompt = build_auto_learning_review_prompt(
        allow_memory=True,
        allow_skills=False,
        min_tool_iterations=4,
        promotion_threshold=0.8,
    )

    assert "strict JSON" in prompt
    assert '"candidates"' in prompt
    assert '"memory"' in prompt
    assert '"skill"' not in prompt
    assert "4" in prompt
    assert "0.8" in prompt



def test_normalize_candidate_clamps_confidence_and_validates_category():
    candidate = normalize_candidate(
        {
            "category": "not-real",
            "summary": "  User prefers concise responses  ",
            "confidence": 1.7,
            "reason": "Repeated explicit correction",
            "payload": {"action": "add", "content": "User prefers concise responses."},
        }
    )

    assert candidate["category"] == "unknown"
    assert candidate["summary"] == "User prefers concise responses"
    assert candidate["confidence"] == 1.0
    assert candidate["payload"]["action"] == "add"



def test_parse_auto_learning_review_returns_memory_candidate_from_valid_json():
    text = json.dumps(
        {
            "candidates": [
                {
                    "category": "memory",
                    "summary": "User prefers concise responses",
                    "confidence": 0.93,
                    "reason": "Repeated explicit correction from user",
                    "target": "user",
                    "payload": {"action": "add", "content": "User prefers concise responses."},
                }
            ]
        }
    )

    candidates = parse_auto_learning_review(text)

    assert len(candidates) == 1
    assert candidates[0]["category"] == "memory"
    assert candidates[0]["target"] == "user"
    assert candidates[0]["confidence"] == 0.93



def test_parse_auto_learning_review_returns_skill_candidate_from_valid_json():
    text = json.dumps(
        {
            "candidates": [
                {
                    "category": "skill",
                    "summary": "Patch outdated OpenVINO skill steps",
                    "confidence": 0.88,
                    "reason": "Workflow required iterative fixes",
                    "target": "openvino-qwen-no-think",
                    "payload": {
                        "action": "patch",
                        "old_string": "old step",
                        "new_string": "new step",
                    },
                }
            ]
        }
    )

    candidates = parse_auto_learning_review(text)

    assert len(candidates) == 1
    assert candidates[0]["category"] == "skill"
    assert candidates[0]["payload"]["action"] == "patch"



def test_parse_auto_learning_review_returns_empty_list_on_invalid_json():
    assert parse_auto_learning_review("not valid json") == []



def test_should_promote_candidate_uses_threshold():
    candidate = normalize_candidate(
        {
            "category": "memory",
            "summary": "User prefers concise responses",
            "confidence": 0.81,
            "payload": {"action": "add", "content": "User prefers concise responses."},
        }
    )

    assert should_promote_candidate(candidate, 0.8) is True
    assert should_promote_candidate(candidate, 0.9) is False
