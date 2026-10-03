"""Issue #124597 incident 4: explicit ``provider: custom`` with no endpoint
must fail loudly instead of silently substituting a paid API-key provider.

Six auxiliary tasks configured as ``provider: custom`` with no custom endpoint
fell back to a paid lane with only a debug log line. Every other explicit
branch (openrouter/nous/codex/xai-oauth) returns (None, None) with a WARNING
when its credentials are missing; the custom branch fell through to
``_resolve_api_key_provider`` and returned an unrelated client as if the
requested custom endpoint had succeeded.
"""
import pytest
from unittest.mock import MagicMock, patch

from agent.auxiliary_client import resolve_provider_client


def test_explicit_custom_with_no_endpoint_does_not_use_paid_lane(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import agent.auxiliary_client as ac

    fake_paid_client = MagicMock(name="paid_lane_client")
    with patch.object(ac, "_try_custom_endpoint", return_value=(None, None)), \
         patch.object(ac, "_resolve_api_key_provider",
                      return_value=(fake_paid_client, "paid-model")):
        client, _model = resolve_provider_client("custom", model="test-model")
    assert client is None, (
        "explicit provider=custom with no endpoint must return (None, None) "
        "instead of silently substituting a paid API-key provider"
    )


def test_explicit_ollama_alias_with_no_endpoint_does_not_use_paid_lane(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import agent.auxiliary_client as ac

    fake_paid_client = MagicMock(name="paid_lane_client")
    with patch.object(ac, "_try_custom_endpoint", return_value=(None, None)), \
         patch.object(ac, "_resolve_api_key_provider",
                      return_value=(fake_paid_client, "paid-model")):
        client, _model = resolve_provider_client("ollama", model="test-model")
    assert client is None, (
        "explicit local-server alias with no endpoint must return (None, None) "
        "instead of silently substituting a paid API-key provider"
    )


# Spellings the guard missed before the #124897 review: ``local`` normalizes to ``custom``
# via hermes_cli.auth._PROVIDER_ALIASES but was absent from _LOCAL_SERVER_ALIASES, and the
# empty ``custom:`` suffix normalizes to ``custom`` while keeping its raw pre-alias spelling.
@pytest.mark.parametrize("spelling", ["custom", "CUSTOM", "  custom  ", "custom:", "local", "ollama", "vllm", "llama.cpp"])
def test_explicit_custom_spellings_do_not_use_paid_lane(monkeypatch, spelling):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import agent.auxiliary_client as ac

    fake_paid_client = MagicMock(name="paid_lane_client")
    with patch.object(ac, "_try_custom_endpoint", return_value=(None, None)), \
         patch.object(ac, "_resolve_api_key_provider",
                      return_value=(fake_paid_client, "paid-model")):
        client, _model = resolve_provider_client(spelling, model="test-model")
    assert client is None, (
        f"explicit {spelling!r} with no endpoint must return (None, None) "
        "instead of silently substituting a paid API-key provider"
    )


@pytest.mark.parametrize("spelling", ["local", "ollama"])
def test_local_alias_borrows_neither_openai_key_nor_bare_host(monkeypatch, spelling):
    """SECURITY: a local-server alias never sends OPENAI_API_KEY to its own host, and gets /v1."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-that-must-not-be-sent")
    import agent.auxiliary_client as ac

    created = {}

    def _fake_create(**kwargs):
        created.update(kwargs)
        return MagicMock(name="custom_client")

    with patch.object(ac, "_create_openai_client", side_effect=_fake_create):
        client, _model = resolve_provider_client(
            spelling, model="test-model", explicit_base_url="http://localhost:11434",
        )
    assert client is not None
    assert created["api_key"] == "no-key-required", "local-server alias borrowed OPENAI_API_KEY"
    assert created["base_url"].endswith("/v1"), f"missing /v1 tail: {created['base_url']}"
