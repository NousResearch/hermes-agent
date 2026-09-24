"""Tests for proactive vision-tool-message downgrade (issue #41072).

When a provider supports vision in user messages but rejects list-type
tool message content (e.g. Xiaomi MiMo's 400 "text is not set"),
``_tool_result_content_for_active_model`` should proactively downgrade
to a text summary instead of waiting for a reactive 400 recovery.

The fix adds ``supports_vision_tool_messages`` to ``ProviderProfile``
and checks it in ``_tool_result_content_for_active_model``.

Re-verified 2026-09-25: ``api.xiaomimimo.com`` accepts list-type tool
content containing ``image_url`` parts — both ``mimo-v2.5`` and
``mimo-v2.6-flash``, including image-only lists (the exact shape that
used to 400). The #41072 error no longer reproduces, so the xiaomi
profile veto was lifted and the two downgrade tests below became
pass-through assertions. The reactive ``_no_list_tool_content_models``
recovery (``_try_strip_image_parts_from_tool_messages``) still
downgrades if an endpoint regresses, so a flipped flag degrades to a
text summary rather than a broken turn.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(provider="openrouter", model="gpt-4o"):
    """Create a minimal AIAgent mock with provider/model attributes."""
    from run_agent import AIAgent
    agent = MagicMock(spec=AIAgent)
    agent.provider = provider
    agent.model = model
    agent._no_list_tool_content_models = set()

    def _real_content_has_image_parts(content):
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    agent._content_has_image_parts = _real_content_has_image_parts
    agent._model_supports_vision = lambda: AIAgent._model_supports_vision(agent)
    agent._provider_supports_vision_tool_messages = lambda: AIAgent._provider_supports_vision_tool_messages(agent)
    agent._tool_result_content_for_active_model = (
        lambda name, result: AIAgent._tool_result_content_for_active_model(agent, name, result)
    )
    return agent


def _multimodal_result(text="screenshot", image_url="data:image/png;base64,AAAA"):
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
        "text_summary": text,
    }


# ---------------------------------------------------------------------------
# _provider_supports_vision_tool_messages
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# _tool_result_content_for_active_model — proactive downgrade
# ---------------------------------------------------------------------------


class TestToolResultContentProactiveDowngrade:
    def test_xiaomi_keeps_list_content(self):
        """Xiaomi: profile veto lifted (re-verified 2026-09-25) → multimodal list preserved."""
        agent = _make_agent("xiaomi", "mimo-v2.6-flash")
        result = _multimodal_result(text="screenshot captured")

        with patch.object(agent, "_model_supports_vision", return_value=True):
            content = agent._tool_result_content_for_active_model("browser_screenshot", result)

        assert isinstance(content, list)
        assert any(p.get("type") == "image_url" for p in content if isinstance(p, dict))

    def test_openrouter_xiaomi_route_keeps_list_content(self):
        """OpenRouter→xiaomi route: the target profile no longer vetoes list-type tool content."""
        agent = _make_agent("openrouter", "xiaomi/mimo-v2.6-flash")
        result = _multimodal_result(text="aggregated screenshot captured")

        with patch.object(agent, "_model_supports_vision", return_value=True):
            content = agent._tool_result_content_for_active_model("browser_screenshot", result)

        assert isinstance(content, list)
        assert any(p.get("type") == "image_url" for p in content if isinstance(p, dict))

    def test_xiaomi_non_multimodal_passes_through(self):
        """Non-multimodal results should pass through unchanged."""
        agent = _make_agent("xiaomi", "mimo-v2.5")
        result = "plain text result"

        content = agent._tool_result_content_for_active_model("some_tool", result)

        assert content == "plain text result"

    def test_openrouter_vision_keeps_list_content(self):
        """OpenRouter with vision: list content preserved."""
        agent = _make_agent("openrouter", "gpt-4o")
        result = _multimodal_result()

        with patch.object(agent, "_model_supports_vision", return_value=True):
            content = agent._tool_result_content_for_active_model("browser_screenshot", result)

        assert isinstance(content, list)
        assert any(p.get("type") == "image_url" for p in content if isinstance(p, dict))




    def test_reactive_cache_still_works(self):
        """In-session cache (_no_list_tool_content_models) still triggers."""
        agent = _make_agent("openrouter", "some-model")
        agent._no_list_tool_content_models = {("openrouter", "some-model")}
        result = _multimodal_result(text="cached downgrade")

        with patch.object(agent, "_model_supports_vision", return_value=True):
            content = agent._tool_result_content_for_active_model("browser_screenshot", result)

        assert isinstance(content, str)
        assert "cached downgrade" in content


# ---------------------------------------------------------------------------
# ProviderProfile.supports_vision_tool_messages field
# ---------------------------------------------------------------------------



