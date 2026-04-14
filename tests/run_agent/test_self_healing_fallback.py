from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


def test_probe_driven_fallback_uses_only_healthy_routes_first():
    fbs = [
        {"provider": "openrouter", "model": "openai/gpt-oss-20b:free"},
        {"provider": "google", "model": "gemini-2.0-flash"},
    ]
    agent = _make_agent(fallback_model=fbs)

    with patch("agent.fallback_probe.probe_fallback_chain") as mock_probe, patch(
        "agent.auxiliary_client.resolve_provider_client"
    ) as mock_rpc:
        mock_probe.return_value = {
            "ok": False,
            "passed": 1,
            "failed": 1,
            "results": [
                {
                    "provider": "openrouter",
                    "cli_provider": "openrouter",
                    "model": "openai/gpt-oss-20b:free",
                    "ok": False,
                    "classification": "rate_limit",
                },
                {
                    "provider": "google",
                    "cli_provider": "gemini",
                    "model": "gemini-2.0-flash",
                    "ok": True,
                    "classification": "ok",
                },
            ],
        }
        mock_rpc.return_value = (_mock_client(base_url="https://generativelanguage.googleapis.com/v1beta/openai"), "gemini-2.0-flash")

        assert agent._try_activate_fallback() is True

    assert agent.provider == "google"
    assert agent.model == "gemini-2.0-flash"
    assert agent._fallback_probe_summary["passed"] == 1


def test_probe_runs_once_then_reuses_health_ordering():
    fbs = [
        {"provider": "openrouter", "model": "openai/gpt-oss-20b:free"},
        {"provider": "google", "model": "gemini-2.0-flash"},
    ]
    agent = _make_agent(fallback_model=fbs)

    with patch("agent.fallback_probe.probe_fallback_chain") as mock_probe, patch(
        "agent.auxiliary_client.resolve_provider_client"
    ) as mock_rpc:
        mock_probe.return_value = {
            "ok": True,
            "passed": 2,
            "failed": 0,
            "results": [
                {
                    "provider": "google",
                    "cli_provider": "gemini",
                    "model": "gemini-2.0-flash",
                    "ok": True,
                    "classification": "ok",
                },
                {
                    "provider": "openrouter",
                    "cli_provider": "openrouter",
                    "model": "openai/gpt-oss-20b:free",
                    "ok": True,
                    "classification": "ok",
                },
            ],
        }
        mock_rpc.side_effect = [
            (_mock_client(base_url="https://generativelanguage.googleapis.com/v1beta/openai"), "gemini-2.0-flash"),
            (_mock_client(), "openai/gpt-oss-20b:free"),
        ]

        assert agent._try_activate_fallback() is True
        assert agent._try_activate_fallback() is True

    mock_probe.assert_called_once()
    assert agent._fallback_index == 2


def test_probe_failure_keeps_original_chain_order():
    fbs = [
        {"provider": "openrouter", "model": "openai/gpt-oss-20b:free"},
        {"provider": "google", "model": "gemini-2.0-flash"},
    ]
    agent = _make_agent(fallback_model=fbs)

    with patch("agent.fallback_probe.probe_fallback_chain", side_effect=RuntimeError("boom")), patch(
        "agent.auxiliary_client.resolve_provider_client"
    ) as mock_rpc:
        mock_rpc.return_value = (_mock_client(), "openai/gpt-oss-20b:free")

        assert agent._try_activate_fallback() is True

    assert agent.provider == "openrouter"
    assert agent.model == "openai/gpt-oss-20b:free"
