"""Explicit ``provider: ollama`` (and the sibling local-server aliases ``vllm`` / ``llamacpp``)
must route through the ``custom`` branch in the auxiliary client, as it already does in
``hermes_cli.auth``: a lane pointing at a local server with an empty ``api_key`` builds a
client with the ``no-key-required`` placeholder instead of raising
``Provider 'ollama' is set in config.yaml but no API key was found`` (issue #106010).

Second half of the same report: these servers expose the OpenAI surface under ``/v1``, so
a bare ``host:port`` base_url gains the tail — but only for the alias group; a literal
``provider: custom`` base_url is used verbatim and an existing path is never doubled.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _keyless_environment(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    # The key ladder inherits the main config's key on a host match; the placeholder
    # assertions below need that rung empty.
    with patch("agent.auxiliary_client._read_main_api_key_if_same_host", return_value=None):
        yield


def _client_attr(client, attr: str) -> str:
    for chain in ((attr,), ("_real_client", attr), ("_client", attr)):
        obj = client
        try:
            for name in chain:
                obj = getattr(obj, name)
            return str(obj)
        except AttributeError:
            continue
    return ""


@pytest.mark.parametrize(
    "api_key, env_openai_key, expected_key",
    [
        ("", "", "no-key-required"),
        ("test-key-123", "", "test-key-123"),
        # SECURITY: a keyless local-server lane must not borrow OPENAI_API_KEY (or the main
        # key on a host match) — that would send an OpenAI secret to the lane's base_url.
        ("", "sk-USER-OPENAI-SECRET", "no-key-required"),
    ],
)
def test_ollama_lane_with_bare_local_host_resolves_custom_client_under_v1(
    monkeypatch, api_key, env_openai_key, expected_key,
):
    """The reporter's lane, through the real task-config path: keyless gets the placeholder,
    an explicit key is sent verbatim, and both post to /v1."""
    from agent import auxiliary_client as ac

    if env_openai_key:
        monkeypatch.setenv("OPENAI_API_KEY", env_openai_key)
        monkeypatch.setattr(ac, "_read_main_api_key_if_same_host", lambda _base: "sk-MAIN-SECRET")
    lane = {"provider": "ollama", "model": "qwen3.8:27b", "base_url": "http://127.0.0.1:11434", "api_key": api_key}
    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: dict(lane) if task == "title_generation" else {})
    provider, model, base_url, key, _mode = ac._resolve_task_provider_model("title_generation")

    client, _model = ac.resolve_provider_client(provider, model=model, explicit_base_url=base_url, explicit_api_key=key)
    assert client is not None, "provider: ollama + local base_url must build a client via the custom branch"
    assert _client_attr(client, "base_url").rstrip("/") == "http://127.0.0.1:11434/v1"
    assert _client_attr(client, "api_key") == expected_key


def test_v1_tail_applies_only_to_bare_alias_hosts_and_never_doubles():
    from agent.auxiliary_client import resolve_provider_client

    custom, _ = resolve_provider_client(
        "custom", model="m", explicit_base_url="http://127.0.0.1:11434", explicit_api_key="test-key-123",
    )
    assert _client_attr(custom, "base_url").rstrip("/") == "http://127.0.0.1:11434"

    with_path, _ = resolve_provider_client(
        "vllm", model="m", explicit_base_url="http://127.0.0.1:8000/v1", explicit_api_key="test-key-123",
    )
    assert _client_attr(with_path, "base_url").rstrip("/") == "http://127.0.0.1:8000/v1"
    assert _client_attr(with_path, "api_key") == "test-key-123"
