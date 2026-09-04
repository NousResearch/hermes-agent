"""Async conversion must preserve a callable API-key provider.

Regression context: a sync client built from a per-request credential source
(``key_cmd``, Azure Entra) stores a callable in ``_api_key_provider`` and
leaves ``api_key`` empty until the first request. ``_to_async_client`` used to
pass the still-empty ``sync_client.api_key`` to ``AsyncOpenAI``, so every async
auxiliary leg (vision, titles, compression) sent a placeholder key and 401ed
while the identical sync call succeeded. The invariant under test:

    converting a sync client to async must preserve a callable key provider —
    the async client must mint (not inherit) its key per request.
"""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and clear module caches (mirrors sibling tests)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _sync_client(base_url="https://gateway.invalid/v1", **kwargs):
    from openai import OpenAI

    return OpenAI(api_key=kwargs.pop("api_key", "sk-static"), base_url=base_url)


class TestAsyncConversionPreservesCallableKeyProvider:
    """The invariant: _to_async_client must not drop a callable key source."""

    def test_callable_provider_survives_conversion(self):
        from agent.auxiliary_client import _to_async_client

        mints = {"count": 0}

        def provider():
            mints["count"] += 1
            return f"minted-{mints['count']}"

        sync_client = _sync_client(api_key=provider)
        # OpenAI SDK 2.x contract: callable api_key -> _api_key_provider set,
        # .api_key empty until first request.
        assert sync_client._api_key_provider is not None
        assert sync_client.api_key == ""

        async_client, _model = _to_async_client(sync_client, "test-model")

        assert async_client._api_key_provider is not None, (
            "_to_async_client dropped the callable key provider — async "
            "auxiliary calls would 401 with a placeholder key"
        )
        # The async client must MINT its key per request (SDK awaits the
        # provider in _refresh_api_key), not inherit the empty string.
        asyncio.run(async_client._refresh_api_key())
        assert async_client.api_key == "minted-1"
        # Repeated requests re-mint — the token source stays live.
        asyncio.run(async_client._refresh_api_key())
        assert async_client.api_key == "minted-2"

    def test_awaitable_provider_survives_conversion(self):
        from agent.auxiliary_client import _to_async_client

        async def provider():
            return "awaited-token"

        sync_client = _sync_client(api_key=provider)
        async_client, _model = _to_async_client(sync_client, "test-model")
        assert async_client._api_key_provider is not None
        asyncio.run(async_client._refresh_api_key())
        assert async_client.api_key == "awaited-token"

    def test_static_key_client_unchanged(self):
        """No behavior change for the common static-key case."""
        from agent.auxiliary_client import _to_async_client

        sync_client = _sync_client(api_key="sk-static")
        async_client, _model = _to_async_client(sync_client, "test-model")
        assert async_client.api_key == "sk-static"
        assert async_client._api_key_provider is None

    def test_key_cmd_resolved_client_keeps_provider_through_async(
        self, tmp_path, monkeypatch
    ):
        """End-to-end shape of the reported bug: a key_cmd named custom
        provider resolves on the sync path and 401s on the async path.
        The conversion must carry the minting callable across."""
        import yaml

        config_path = tmp_path / ".hermes" / "config.yaml"
        config_path.write_text(yaml.dump({
            "model": {"default": "test-model"},
            "providers": {
                "gateway": {
                    "base_url": "https://gateway.invalid/v1",
                    "api_mode": "chat_completions",
                    "key_cmd": "printf minted-token",
                },
            },
        }))

        from agent.auxiliary_client import resolve_provider_client

        sync_client, _model = resolve_provider_client(
            "gateway", "test-model", async_mode=False
        )
        assert sync_client is not None
        assert sync_client._api_key_provider is not None, (
            "precondition: key_cmd must resolve to a callable on the sync path"
        )

        async_client, _model = resolve_provider_client(
            "gateway", "test-model", async_mode=True
        )
        assert async_client is not None
        assert async_client._api_key_provider is not None, (
            "key_cmd provider was dropped by async conversion — async "
            "auxiliary calls will 401"
        )
        asyncio.run(async_client._refresh_api_key())
        assert async_client.api_key == "minted-token"