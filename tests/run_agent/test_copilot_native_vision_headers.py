from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_copilot_agent():
    with patch("run_agent.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        agent = AIAgent(
            api_key="gh-token",
            base_url="https://api.githubcopilot.com",
            provider="copilot",
            model="gpt-5.4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    return agent


def test_request_client_adds_copilot_vision_header_for_native_image_payload():
    agent = _make_copilot_agent()
    built_kwargs = []

    def fake_create(kwargs, *, reason, shared):
        built_kwargs.append(dict(kwargs))
        return MagicMock()

    api_kwargs = {
        "model": "gpt-5.4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
    }

    agent.client = object()
    with patch.object(agent, "_is_openai_client_closed", return_value=False), patch.object(
        agent, "_create_openai_client", side_effect=fake_create
    ):
        agent._create_request_openai_client(reason="test", api_kwargs=api_kwargs)

    headers = built_kwargs[-1]["default_headers"]
    assert headers["Copilot-Vision-Request"] == "true"


def test_request_client_leaves_copilot_text_requests_without_vision_header():
    agent = _make_copilot_agent()
    built_kwargs = []

    def fake_create(kwargs, *, reason, shared):
        built_kwargs.append(dict(kwargs))
        return MagicMock()

    api_kwargs = {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]}

    agent.client = object()
    with patch.object(agent, "_is_openai_client_closed", return_value=False), patch.object(
        agent, "_create_openai_client", side_effect=fake_create
    ):
        agent._create_request_openai_client(reason="test", api_kwargs=api_kwargs)

    headers = built_kwargs[-1]["default_headers"]
    assert "Copilot-Vision-Request" not in headers


def test_request_client_does_not_add_vision_header_after_non_vision_fallback():
    agent = _make_copilot_agent()
    built_kwargs = []

    def fake_create(kwargs, *, reason, shared):
        built_kwargs.append(dict(kwargs))
        return MagicMock()

    # This is the shape after _prepare_messages_for_non_vision_model has
    # replaced image parts with text, so Copilot should not get the vision route.
    api_kwargs = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "user", "content": "[user image: a dog]\n\nWhat is in this image?"}
        ],
    }

    agent.client = object()
    with patch.object(agent, "_is_openai_client_closed", return_value=False), patch.object(
        agent, "_create_openai_client", side_effect=fake_create
    ):
        agent._create_request_openai_client(reason="test", api_kwargs=api_kwargs)

    headers = built_kwargs[-1]["default_headers"]
    assert "Copilot-Vision-Request" not in headers
