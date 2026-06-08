from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "mlops"
    / "training"
    / "trl-fine-tuning"
    / "examples"
    / "reward_functions_library.py"
)


def load_library():
    spec = importlib.util.spec_from_file_location("reward_functions_library", LIBRARY_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_xml_format_rewards_handle_chat_completion_shape():
    rewards = load_library()
    completions = [
        [{"role": "assistant", "content": "<reasoning>2+2</reasoning><answer>4</answer>"}],
        [{"role": "assistant", "content": "The answer is 4"}],
    ]

    assert rewards.xml_format_reward(completions) == [1.0, 0.0]
    assert rewards.incremental_xml_format_reward(completions) == [1.0, 0.0]
    assert rewards.answer_tag_reward(completions) == [1.0, 0.0]


def test_numeric_rewards_match_reference_columns():
    rewards = load_library()
    completions = [
        "<answer>42</answer>",
        "<answer>40</answer>",
        "no number here",
    ]

    assert rewards.numeric_match_reward(completions, answer=["42", "42", "42"]) == [1.0, 0.0, 0.0]
    close_scores = rewards.numeric_close_reward(completions, answer=["42", "42", "42"])
    assert close_scores[0] == 1.0
    assert 0.9 < close_scores[1] < 1.0
    assert close_scores[2] == 0.0


def test_reward_factories_create_task_specific_rewards():
    rewards = load_library()

    keyword_reward = rewards.make_keyword_coverage_reward(["alpha", "beta"])
    assert keyword_reward(["alpha beta", "alpha", "gamma"]) == [1.0, 0.5, 0.0]

    json_keys_reward = rewards.make_json_keys_reward(["answer", "confidence"])
    assert json_keys_reward(['{"answer": "yes", "confidence": 0.9}', '{"answer": "yes"}', 'nope']) == [
        1.0,
        0.0,
        0.0,
    ]


def test_combined_math_xml_reward_scores_correct_formatted_answer_higher():
    rewards = load_library()
    completions = [
        "<reasoning>2 + 2 = 4</reasoning><answer>4</answer>",
        "four",
    ]

    scores = rewards.math_xml_reward(completions, answer=["4", "4"])

    assert scores[0] > scores[1]
    assert scores[0] > 2.0
