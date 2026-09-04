"""Tests for agent/provider/router.py — SmartModelRouter."""
from __future__ import annotations

from typing import Any, Dict, Sequence
from unittest.mock import MagicMock, patch

import pytest

from agent.provider.router import (
    RoutingRule,
    SmartModelRouter,
    RouterResult,
    classify_task,
    TASK_CATEGORIES,
    _has_code_block,
    _has_diff_block,
    _has_image_attachment,
    _parse_price,
    _CODE_HINTS,
    _ANALYSIS_HINTS,
)


# =========================================================================
# classify_task tests
# =========================================================================


class TestClassifyTask:
    def test_empty_messages_returns_chat(self):
        assert classify_task([]) == "chat"

    def test_plain_chat_returns_chat(self):
        assert classify_task([{"role": "user", "content": "Hello!"}]) == "chat"

    def test_vision_image_url_attachment(self):
        msgs = [
            {"role": "user", "content": [{"type": "image_url", "image_url": "https://example.com/img.png"}]}
        ]
        assert classify_task(msgs) == "vision"

    def test_vision_image_type(self):
        msgs = [
            {"role": "user", "content": [{"type": "image", "data": b"fake"}]}
        ]
        assert classify_task(msgs) == "vision"

    def test_vision_media_reference(self):
        msgs = [{"role": "user", "content": "Look at this MEDIA: screenshot.png"}]
        assert classify_task(msgs) == "vision"

    def test_vision_image_reference(self):
        msgs = [{"role": "user", "content": "[IMAGE: diagram.png]"}]
        assert classify_task(msgs) == "vision"

    def test_code_block_detected(self):
        msgs = [{"role": "user", "content": "Here's some code:\n```python\nx = 1\n```"}]
        assert classify_task(msgs) == "code"

    def test_code_hint_implement(self):
        msgs = [{"role": "user", "content": "Implement a sorting function"}]
        assert classify_task(msgs) == "code"

    def test_code_hint_write_code(self):
        msgs = [{"role": "user", "content": "Write code for an API endpoint"}]
        assert classify_task(msgs) == "code"

    def test_code_hint_fix_bug(self):
        msgs = [{"role": "user", "content": "Fix bug in the login flow"}]
        assert classify_task(msgs) == "code"

    def test_code_hint_refactor(self):
        msgs = [{"role": "user", "content": "Refactor the database layer"}]
        assert classify_task(msgs) == "code"

    def test_code_hint_create_function(self):
        msgs = [{"role": "user", "content": "Create a function that parses CSV"}]
        assert classify_task(msgs) == "code"

    def test_analysis_hint_analyse(self):
        msgs = [{"role": "user", "content": "Analyse this dataset"}]
        assert classify_task(msgs) == "analysis"

    def test_analysis_hint_summarize(self):
        msgs = [{"role": "user", "content": "Summarize the key points"}]
        assert classify_task(msgs) == "analysis"

    def test_analysis_hint_compare(self):
        msgs = [{"role": "user", "content": "Compare these two approaches"}]
        assert classify_task(msgs) == "analysis"

    def test_analysis_hint_explain(self):
        msgs = [{"role": "user", "content": "Explain how this works"}]
        assert classify_task(msgs) == "analysis"

    def test_vision_takes_priority_over_code(self):
        """Vision attachments should be prioritized even with code keywords."""
        msgs = [
            {"role": "user", "content": "Write code based on this image"},
            {"role": "user", "content": [{"type": "image_url", "image_url": "http://img.url"}]},
        ]
        assert classify_task(msgs) == "vision"

    def test_latest_user_message_used(self):
        """Only the latest user message is used for code/analysis classification."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What's the weather?"},
        ]
        assert classify_task(msgs) == "chat"

    def test_diff_block_detected(self):
        msgs = [{"role": "user", "content": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"}]
        assert classify_task(msgs) == "code"

    def test_structured_content_text_only(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Implement a sorting function"}]}]
        assert classify_task(msgs) == "code"


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelpers:
    def test_has_code_block(self):
        assert _has_code_block("Some text ```python\nx=1\n``` more") is True
        assert _has_code_block("No code here") is False

    def test_has_diff_block(self):
        assert _has_diff_block("--- a/foo.py\n+++ b/foo.py") is True
        assert _has_diff_block("No diff") is False

    def test_has_image_attachment(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": "u"}]}]
        assert _has_image_attachment(msgs) is True
        assert _has_image_attachment([{"role": "user", "content": "text"}]) is False

    def test_parse_price(self):
        assert _parse_price(0.5) == 0.5
        assert _parse_price("0.15") == 0.15
        assert _parse_price(10) == 10.0
        assert _parse_price(None) == 0.0
        assert _parse_price("invalid") == 0.0

    def test_code_hints_implement(self):
        assert _CODE_HINTS.search("Implement a feature") is not None

    def test_code_hints_write_code(self):
        assert _CODE_HINTS.search("Write code") is not None
        assert _CODE_HINTS.search("writing code") is not None

    def test_code_hints_refactor(self):
        assert _CODE_HINTS.search("Refactor the module") is not None
        assert _CODE_HINTS.search("refactoring the module") is not None

    def test_analysis_hints_analyse(self):
        assert _ANALYSIS_HINTS.search("Analyse this data") is not None
        assert _ANALYSIS_HINTS.search("analysing the data") is not None

    def test_analysis_hints_summarize(self):
        assert _ANALYSIS_HINTS.search("Summarize the report") is not None
        assert _ANALYSIS_HINTS.search("summarizing is not") is not None


# =========================================================================
# SmartModelRouter tests
# =========================================================================


class TestSmartModelRouter:
    def test_disabled_router_returns_none(self):
        router = SmartModelRouter(rules=[], available_providers=[], enabled=False)
        assert router.route([]) is None

    def test_no_rules_returns_none(self):
        router = SmartModelRouter(rules=[], available_providers=["deepseek"], enabled=True)
        assert router.route([]) is None

    def test_from_config_basic(self):
        cfg = {
            "model_router": {
                "enabled": True,
                "rules": [
                    {"task": "chat", "provider": "deepseek"},
                ],
            }
        }
        router = SmartModelRouter.from_config(cfg, ["deepseek", "openai"])
        assert router.enabled is True
        assert len(router.rules) == 1
        assert router.rules[0].task == "chat"
        assert router.rules[0].provider == "deepseek"

    def test_from_config_unknown_task_warns(self):
        cfg = {
            "model_router": {
                "enabled": True,
                "rules": [
                    {"task": "unknown_task", "provider": "deepseek"},
                    {"task": "chat", "provider": "openai"},
                ],
            }
        }
        router = SmartModelRouter.from_config(cfg, ["deepseek", "openai"])
        # Only the valid rule should be kept
        assert len(router.rules) == 1
        assert router.rules[0].task == "chat"

    def test_from_config_empty_provider_skipped(self):
        cfg = {
            "model_router": {
                "enabled": True,
                "rules": [
                    {"task": "chat", "provider": ""},
                ],
            }
        }
        router = SmartModelRouter.from_config(cfg, ["deepseek"])
        assert len(router.rules) == 0

    def test_from_config_disabled_by_default(self):
        cfg = {}
        router = SmartModelRouter.from_config(cfg, ["deepseek"])
        assert router.enabled is False

    def test_from_config_non_dict_config(self):
        """Should handle non-dict config gracefully."""
        router = SmartModelRouter.from_config("invalid", ["deepseek"])
        assert router.enabled is False
        assert len(router.rules) == 0

    def test_from_config_rules_not_list(self):
        """Should handle non-list rules gracefully."""
        cfg = {"model_router": {"enabled": True, "rules": "not_a_list"}}
        router = SmartModelRouter.from_config(cfg, ["deepseek"])
        assert len(router.rules) == 0

    def test_from_config_rule_not_dict(self):
        """Should skip non-dict rules gracefully."""
        cfg = {"model_router": {"enabled": True, "rules": ["invalid", {"task": "chat", "provider": "openai"}]}}
        router = SmartModelRouter.from_config(cfg, ["openai"])
        assert len(router.rules) == 1

    @patch("agent.provider.router.get_model_capabilities")
    @patch("agent.provider.router.list_provider_models")
    def test_route_basic(self, mock_list, mock_caps):
        mock_list.return_value = ["deepseek-v4-flash", "deepseek-v4-pro"]
        mock_caps.side_effect = lambda p, m: MagicMock(
            supports_tools=True,
            supports_vision=(m == "deepseek-v4-pro"),
            supports_reasoning=False,
            context_window=200000,
            max_output_tokens=8192,
        )

        router = SmartModelRouter(
            rules=[RoutingRule(task="chat", provider="deepseek")],
            available_providers=["deepseek"],
            enabled=True,
        )

        result = router.route([{"role": "user", "content": "Hello!"}])
        assert result is not None
        assert result.task == "chat"
        assert result.provider == "deepseek"

    @patch("agent.provider.router.get_model_capabilities")
    @patch("agent.provider.router.list_provider_models")
    def test_route_with_min_context(self, mock_list, mock_caps):
        mock_list.return_value = ["small-model", "big-model"]
        mock_caps.side_effect = lambda p, m: MagicMock(
            supports_tools=True,
            supports_vision=False,
            supports_reasoning=False,
            context_window=50000 if m == "small-model" else 200000,
            max_output_tokens=8192,
        )

        router = SmartModelRouter(
            rules=[RoutingRule(task="chat", provider="deepseek", min_context=100000)],
            available_providers=["deepseek"],
            enabled=True,
        )

        result = router.route([{"role": "user", "content": "Hello!"}])
        # Only big-model meets the min_context requirement
        assert result is not None
        assert result.model == "big-model"

    def test_bool_conversion(self):
        r = RouterResult(provider="p", model="m", task="chat", rule=MagicMock(), price=0.0)
        assert bool(r) is True


# =========================================================================
# Integration smoke tests
# =========================================================================


class TestIntegration:
    @patch("agent.provider.router.get_model_capabilities")
    @patch("agent.provider.router.list_provider_models")
    def test_end_to_end_routing(self, mock_list, mock_caps):
        """Full routing pipeline: classify → filter → rank → return."""
        mock_list.return_value = ["cheap-model", "expensive-model"]
        mock_caps.side_effect = lambda p, m: MagicMock(
            supports_tools=True,
            supports_vision=False,
            supports_reasoning=False,
            context_window=128000,
            max_output_tokens=8192,
        )

        with patch("agent.provider.router._safe_model_price") as mock_price:
            mock_price.side_effect = lambda p, m: 0.01 if m == "cheap-model" else 0.50

            router = SmartModelRouter(
                rules=[RoutingRule(task="chat", provider="deepseek")],
                available_providers=["deepseek"],
                enabled=True,
            )

            result = router.route([{"role": "user", "content": "Hello world"}])
            assert result is not None
            # Should pick the cheapest model
            assert result.model == "cheap-model"

    def test_classify_chat_no_hints(self):
        """Plain text with no code/analysis keywords should classify as chat."""
        msg = "What is the capital of France?"
        assert classify_task([{"role": "user", "content": msg}]) == "chat"

    def test_classify_analysis_explicit(self):
        msg = "Analyse the quarterly report data"
        assert classify_task([{"role": "user", "content": msg}]) == "analysis"
