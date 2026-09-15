"""Auxiliary ``provider: auto`` must walk ``fallback_providers`` after a candidate quota error.

Regression for #106367: a walk-eligible rate-limit/quota error on fallback[0]
used to escape and abort the walker, so fallback[1] never ran.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm


@pytest.fixture(autouse=True)
def _clear_aux_unhealthy_cache():
    import agent.auxiliary_client as aux
    aux._aux_unhealthy_until.clear()
    aux._aux_unhealthy_logged_at.clear()
    yield
    aux._aux_unhealthy_until.clear()
    aux._aux_unhealthy_logged_at.clear()


class GoUsageLimitError(Exception):
    """Mirrors the Go/OpenCode weekly-quota wall that arrives as 429."""

    status_code = 429

    def __init__(self, message="You have reached your weekly usage limit"):
        super().__init__(message)


class _PrimaryUsageLimit(Exception):
    status_code = 429

    def __init__(self, message="usage_limit_reached"):
        super().__init__(message)


class _DummyResponse:
    def __init__(self, text="ok"):
        self.choices = [MagicMock(message=MagicMock(content=text))]


def _client(base_url, *, create=None, async_mode=False):
    client = MagicMock()
    client.base_url = base_url
    client.HERMES_SKIP_ASYNC_WRAP = True
    raise_it = isinstance(create, BaseException)
    if async_mode:
        client.chat.completions.create = (
            AsyncMock(side_effect=create) if raise_it else AsyncMock(return_value=create)
        )
    elif raise_it:
        client.chat.completions.create.side_effect = create
    else:
        client.chat.completions.create.return_value = create
    return client


def _chain_entries():
    return [
        {"provider": "openrouter", "model": "openrouter/quota-model"},
        {"provider": "nous", "model": "nous/ok-model"},
    ]


@contextmanager
def _auto_main_chain(primary, clients_by_provider, *, discovery=None):
    """``provider: auto`` primary + ``fallback_providers`` [B, C] via get_fallback_chain."""

    def _resolve_entry(entry):
        provider = str(entry.get("provider") or "")
        model = str(entry.get("model") or "") or None
        client = clients_by_provider[provider]
        return client, model

    disc = discovery if discovery is not None else (None, None, "")
    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "primary-model", None, None, None)), \
         patch("agent.auxiliary_client._get_cached_client",
               return_value=(primary, "primary-model")), \
         patch("agent.auxiliary_client._try_configured_fallback_chain",
               return_value=(None, None, "")), \
         patch("hermes_cli.fallback_config.get_fallback_chain",
               return_value=_chain_entries()), \
         patch("agent.auxiliary_client._resolve_fallback_entry",
               side_effect=_resolve_entry), \
         patch("agent.auxiliary_client._try_payment_fallback",
               return_value=disc) as mock_discovery, \
         patch("agent.auxiliary_client._to_async_client",
               side_effect=lambda client, model, **kw: (client, model)):
        yield mock_discovery


class TestAuxAutoMainFallbackMultiHop:
    """Configured main fallback_providers must continue after a walk-eligible candidate error."""

    def test_aux_auto_main_fallback_chain_continues_after_candidate_quota(self):
        """Sync: A 429 → B weekly-quota 429 → C succeeds; B and C each attempted once."""
        primary = _client(
            "https://api.openai.com/v1",
            create=_PrimaryUsageLimit(),
        )
        client_b = _client(
            "https://openrouter.ai/api/v1",
            create=GoUsageLimitError(),
        )
        client_c = _client(
            "https://inference-api.nousresearch.com/v1",
            create=_DummyResponse("from-fallback-c"),
        )

        with _auto_main_chain(primary, {"openrouter": client_b, "nous": client_c}):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "from-fallback-c"
        assert client_b.chat.completions.create.call_count == 1
        assert client_c.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_aux_auto_main_fallback_chain_continues_after_candidate_quota_async(self):
        """Async mirror of the multi-hop success path."""
        primary = _client(
            "https://api.openai.com/v1",
            create=_PrimaryUsageLimit(),
            async_mode=True,
        )
        client_b = _client(
            "https://openrouter.ai/api/v1",
            create=GoUsageLimitError(),
            async_mode=True,
        )
        client_c = _client(
            "https://inference-api.nousresearch.com/v1",
            create=_DummyResponse("from-fallback-c-async"),
            async_mode=True,
        )

        with _auto_main_chain(primary, {"openrouter": client_b, "nous": client_c}):
            result = await async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "from-fallback-c-async"
        assert client_b.chat.completions.create.call_count == 1
        assert client_c.chat.completions.create.call_count == 1

    def test_aux_auto_main_fallback_chain_exhausts_without_discovery(self):
        """T4: A/B/C all 429 — each configured candidate once; no fourth discovery provider."""
        primary = _client(
            "https://api.openai.com/v1",
            create=_PrimaryUsageLimit(),
        )
        client_b = _client(
            "https://openrouter.ai/api/v1",
            create=GoUsageLimitError(),
        )
        client_c = _client(
            "https://inference-api.nousresearch.com/v1",
            create=GoUsageLimitError("weekly usage limit on C"),
        )
        discovery = _client(
            "https://payg.example/v1",
            create=_DummyResponse("should-not-use-discovery"),
        )

        with _auto_main_chain(
            primary,
            {"openrouter": client_b, "nous": client_c},
            discovery=(discovery, "payg-model", "openrouter"),
        ) as mock_discovery:
            with pytest.raises(_PrimaryUsageLimit, match="usage_limit_reached"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                )

        assert client_b.chat.completions.create.call_count == 1
        assert client_c.chat.completions.create.call_count == 1
        assert mock_discovery.call_count == 0
        assert discovery.chat.completions.create.call_count == 0

    def test_non_walk_eligible_candidate_error_still_raises(self):
        """CONTROL: candidate ValueError is not walk-eligible and still raises."""
        primary = _client(
            "https://api.openai.com/v1",
            create=_PrimaryUsageLimit(),
        )
        broken = _client(
            "https://openrouter.ai/api/v1",
            create=ValueError("malformed response"),
        )
        client_c = _client(
            "https://inference-api.nousresearch.com/v1",
            create=_DummyResponse("must-not-run"),
        )

        with _auto_main_chain(primary, {"openrouter": broken, "nous": client_c}):
            with pytest.raises(ValueError, match="malformed response"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                )

        assert broken.chat.completions.create.call_count == 1
        assert client_c.chat.completions.create.call_count == 0
