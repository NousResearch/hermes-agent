"""Tests for agent.title_generation — shared title-generation helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.title_generation import (
    async_generate_title_from_message,
    generate_title_from_exchange,
    generate_title_from_message,
)


class TestGenerateTitleFromExchange:
    def test_returns_title_on_success(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Debugging Python Import Errors"

        with patch("agent.title_generation.call_llm", return_value=mock_response):
            title = generate_title_from_exchange("help me fix this import", "Sure, let me check...")
            assert title == "Debugging Python Import Errors"

    def test_strips_quotes(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '"Setting Up Docker Environment"'

        with patch("agent.title_generation.call_llm", return_value=mock_response):
            title = generate_title_from_exchange("how do I set up docker", "First install...")
            assert title == "Setting Up Docker Environment"

    def test_strips_title_prefix(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Title: Kubernetes Pod Debugging"

        with patch("agent.title_generation.call_llm", return_value=mock_response):
            title = generate_title_from_exchange("my pod keeps crashing", "Let me look...")
            assert title == "Kubernetes Pod Debugging"

    def test_truncates_long_titles(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A" * 100

        with patch("agent.title_generation.call_llm", return_value=mock_response):
            title = generate_title_from_exchange("question", "answer")
            assert len(title) == 80
            assert title.endswith("...")

    def test_returns_none_on_empty_response(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        with patch("agent.title_generation.call_llm", return_value=mock_response):
            assert generate_title_from_exchange("question", "answer") is None

    def test_returns_none_on_exception(self):
        with patch("agent.title_generation.call_llm", side_effect=RuntimeError("no provider")):
            assert generate_title_from_exchange("question", "answer") is None

    def test_truncates_long_messages(self):
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Short Title"
            return resp

        with patch("agent.title_generation.call_llm", side_effect=mock_call_llm):
            generate_title_from_exchange("x" * 1000, "y" * 1000)

        user_content = captured_kwargs["messages"][1]["content"]
        assert len(user_content) < 1100


class TestGenerateTitleFromMessage:
    def test_uses_message_prompt(self):
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Discord Deployment Help"
            return resp

        with patch("agent.title_generation.call_llm", side_effect=mock_call_llm):
            title = generate_title_from_message("help me deploy this Discord bot")

        assert title == "Discord Deployment Help"
        assert captured_kwargs["messages"][1]["content"] == "help me deploy this Discord bot"

    def test_returns_none_for_empty_message(self):
        assert generate_title_from_message("") is None

    @pytest.mark.asyncio
    async def test_async_uses_to_thread(self):
        with patch(
            "agent.title_generation.asyncio.to_thread",
            new=AsyncMock(return_value="Discord Gateway Startup"),
        ) as mock_to_thread:
            title = await async_generate_title_from_message(
                "please help me debug the discord gateway startup",
                timeout=15.0,
            )

        assert title == "Discord Gateway Startup"
        mock_to_thread.assert_awaited_once_with(
            generate_title_from_message,
            "please help me debug the discord gateway startup",
            15.0,
        )

    @pytest.mark.asyncio
    async def test_async_returns_none_on_exception(self):
        with patch("agent.title_generation.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("no provider"))):
            title = await async_generate_title_from_message("hello", timeout=15.0)

        assert title is None
