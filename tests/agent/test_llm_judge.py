"""Tests for agent.llm_judge — the LLM-judge side of Phase 3."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from agent.llm_judge import (
    DEFAULT_JUDGE_PROMPT_TEMPLATE,
    DefaultLLMJudge,
    JudgeScore,
    build_judge_prompt,
    parse_score_text,
)


class TestBuildJudgePrompt:
    def test_includes_prompt_and_response(self):
        out = build_judge_prompt("hello?", "hi there")
        assert "hello?" in out
        assert "hi there" in out

    def test_default_template_has_placeholders(self):
        # Sanity: the shipped template actually has the placeholders.
        assert "{prompt}" in DEFAULT_JUDGE_PROMPT_TEMPLATE
        assert "{response}" in DEFAULT_JUDGE_PROMPT_TEMPLATE


class TestParseScoreText:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ('{"score": 75, "reasoning": "good"}', (75, "good")),
            ('{"score": 0, "reasoning": ""}', (0, "")),
            ('{"score": 100, "reasoning": "perfect"}', (100, "perfect")),
        ],
    )
    def test_parses_clean_json(self, text, expected):
        assert parse_score_text(text) == expected

    def test_parses_json_with_surrounding_prose(self):
        text = 'Here is my verdict:\n{"score": 60, "reasoning": "ok"}\nDone.'
        assert parse_score_text(text) == (60, "ok")

    def test_parses_fenced_json_block(self):
        text = '```json\n{"score": 88, "reasoning": "great"}\n```'
        assert parse_score_text(text) == (88, "great")

    def test_parses_score_field_only(self):
        text = 'My answer: "score": 45'
        assert parse_score_text(text) == (45, "")

    def test_returns_none_for_unparseable(self):
        assert parse_score_text("nothing here") == (None, "")
        assert parse_score_text("") == (None, "")
        assert parse_score_text('{"reasoning": "missing score"}') == (None, "")

    def test_rejects_out_of_range(self):
        # Score must be in [0, 100]; a raw 150 should not be returned.
        text = '{"score": 150, "reasoning": "out of range"}'
        result = parse_score_text(text)
        assert result[0] is None

    def test_falls_back_to_standalone_integer(self):
        # Last-ditch: bare integer in [0, 100].
        assert parse_score_text("I'd give this a 42 overall.") == (42, "")

    def test_handles_nested_braces(self):
        text = '{"score": 55, "reasoning": "has {curly} brace"}'
        assert parse_score_text(text) == (55, "has {curly} brace")


class TestJudge:
    def test_returns_failure_when_aux_client_missing(self):
        judge = DefaultLLMJudge()
        with patch.dict(sys.modules, {"agent.auxiliary_client": None}):
            result = judge.judge("p", "r")
        assert result.success is False
        assert "auxiliary_client unavailable" in (result.error or "")

    def test_returns_failure_when_call_llm_raises(self):
        judge = DefaultLLMJudge()
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("provider down"),
        ):
            result = judge.judge("p", "r")
        assert result.success is False
        assert "provider down" in (result.error or "")

    def test_parses_clean_json_response(self):
        class _Usage:
            prompt_tokens = 0
            completion_tokens = 0

        class _Choice:
            class _Msg:
                content = '{"score": 82, "reasoning": "good"}'

            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()
            model = "judge-model"

        judge = DefaultLLMJudge()
        with patch("agent.auxiliary_client.call_llm", return_value=_Resp()):
            result = judge.judge("hello?", "hi")
        assert result.success
        assert result.score == 82
        assert result.reasoning == "good"
        assert result.model == "judge-model"

    def test_unparseable_response_returns_failure(self):
        class _Choice:
            class _Msg:
                content = "I refuse to score"

            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            model = "judge"

        judge = DefaultLLMJudge()
        with patch("agent.auxiliary_client.call_llm", return_value=_Resp()):
            result = judge.judge("p", "r")
        assert result.success is False
        assert "parseable" in (result.error or "")

    def test_score_below_50_marks_failure_in_caller(self):
        """The judge returns success=True even for low scores; the
        harness is responsible for marking ``success=False`` based on
        its own threshold. This documents the contract.
        """

        class _Choice:
            class _Msg:
                content = '{"score": 10, "reasoning": "bad"}'

            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            model = "judge"

        judge = DefaultLLMJudge()
        with patch("agent.auxiliary_client.call_llm", return_value=_Resp()):
            result = judge.judge("p", "r")
        assert result.success
        assert result.score == 10
