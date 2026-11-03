"""
DeepEval test suite for the Hermes Agent.

Evaluates the agent's CLI health and basic response quality.
Run with:
    PYTHONPATH= .venv/Scripts/deepeval.exe test run tests/evals/test_agent.py
"""
import pytest

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from tests.evals.metrics import AGENT_EVAL_METRICS


def test_agent_cli_health():
    """Verify hermes CLI is callable and functional."""
    test_case = LLMTestCase(
        input="hermes --help",
        actual_output="",  # Measured internally by the metric
    )
    assert_test(test_case=test_case, metrics=[AGENT_EVAL_METRICS[0]])


def test_agent_basic_factual():
    """Verify hermes can answer a simple factual question."""
    test_case = LLMTestCase(
        input="What is the capital of France? Answer in one word.",
        actual_output="",  # Filled by the metric running hermes chat -q
    )
    assert_test(test_case=test_case, metrics=[AGENT_EVAL_METRICS[1]])


def test_agent_code_reasoning():
    """Verify hermes can reason about code."""
    test_case = LLMTestCase(
        input="What does the Python expression '3 + 4 * 2' evaluate to? Answer with just the number.",
        actual_output="",
    )
    assert_test(test_case=test_case, metrics=[AGENT_EVAL_METRICS[1]])
