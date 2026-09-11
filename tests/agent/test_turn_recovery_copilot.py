from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from agent.auxiliary_client import call_llm
from agent.turn_recovery import _refresh_credentials_after_401
from agent.turn_retry_state import TurnRetryState


def test_copilot_enterprise_403_refreshes_credentials_and_retries():
    agent = SimpleNamespace(
        provider="copilot",
        api_mode="codex_responses",
        base_url="https://api.enterprise.githubcopilot.com",
        _try_refresh_copilot_client_credentials=Mock(return_value=True),
        _buffer_vprint=Mock(),
    )
    retry = TurnRetryState()

    recovered = _refresh_credentials_after_401(
        agent,
        RuntimeError("HTTP 403: forbidden"),
        retry,
        status_code=403,
    )

    assert recovered is True
    assert retry.copilot_auth_retry_attempted is True
    agent._try_refresh_copilot_client_credentials.assert_called_once_with()
    agent._buffer_vprint.assert_called_once_with(
        "🔐 Copilot credentials refreshed after auth failure. Retrying request..."
    )

    assert _refresh_credentials_after_401(
        agent,
        RuntimeError("HTTP 403: forbidden"),
        retry,
        status_code=403,
    ) is False
    agent._try_refresh_copilot_client_credentials.assert_called_once_with()


class _AuxAuth403(Exception):
    status_code = 403


class _DummyResponse:
    def __init__(self, content):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = None


def test_auto_routed_enterprise_copilot_403_refreshes_and_retries():
    stale_client = MagicMock()
    stale_client.base_url = "https://api.enterprise.githubcopilot.com"
    stale_client.chat.completions.create.side_effect = _AuxAuth403("HTTP 403: forbidden")

    fresh_client = MagicMock()
    fresh_client.base_url = "https://api.enterprise.githubcopilot.com"
    fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-auto-copilot")

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            side_effect=[(stale_client, "gpt-5.6-sol"), (fresh_client, "gpt-5.6-sol")],
        ) as get_client,
        patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as refresh,
        patch("agent.auxiliary_client._evict_cached_clients") as evict,
    ):
        response = call_llm(
            task="title_generation",
            messages=[{"role": "user", "content": "hi"}],
            main_runtime={"provider": "copilot", "model": "gpt-5.6-sol"},
        )

    assert response.choices[0].message.content == "fresh-auto-copilot"
    refresh.assert_called_once_with("copilot")
    evict.assert_called_once_with("auto")
    assert get_client.call_args_list[0].args[0] == "auto"
    assert get_client.call_args_list[1].args[0] == "copilot"
    assert stale_client.chat.completions.create.call_count == 1
    assert fresh_client.chat.completions.create.call_count == 1


def test_non_copilot_403_is_not_treated_as_refreshable_auth_failure():
    agent = SimpleNamespace(
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1",
    )

    recovered = _refresh_credentials_after_401(
        agent,
        RuntimeError("HTTP 403: forbidden"),
        TurnRetryState(),
        status_code=403,
    )

    assert recovered is False
