"""Tests for the ``llm.oneshot`` ``reasoning_effort`` param."""

from unittest.mock import MagicMock, patch

import pytest

from agent.oneshot import run_oneshot


class TestRunOneshotReasoningConfig:

    def _mock_response(self, content):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.choices[0].message.reasoning = None
        resp.choices[0].message.reasoning_content = None
        resp.choices[0].message.reasoning_details = None
        return resp

    def test_reasoning_config_forwards_to_call_llm(self):
        with patch(
            "agent.oneshot.call_llm",
            return_value=self._mock_response("hello"),
        ) as llm:
            out = run_oneshot(
                instructions="be brief", user_input="say hi",
                reasoning_config={"enabled": True, "effort": "high"},
            )

        assert out == "hello"
        assert llm.call_args.kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}

    def test_no_reasoning_config_passes_none(self):
        with patch(
            "agent.oneshot.call_llm",
            return_value=self._mock_response("hello"),
        ) as llm:
            run_oneshot(instructions="be brief", user_input="say hi")

        assert llm.call_args.kwargs["reasoning_config"] is None

    def test_disabled_reasoning_config_forwards(self):
        with patch(
            "agent.oneshot.call_llm",
            return_value=self._mock_response("hello"),
        ) as llm:
            run_oneshot(
                instructions="be brief", user_input="say hi",
                reasoning_config={"enabled": False},
            )

        assert llm.call_args.kwargs["reasoning_config"] == {"enabled": False}
