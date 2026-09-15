"""Explicit ``provider: ollama`` (and sibling local-server aliases) must route through
the ``custom`` branch like they do in the main runtime, so an aux lane pointing at a
local Ollama/vLLM/llama.cpp server with an empty ``api_key`` builds a client with the
``no-key-required`` placeholder instead of dead-ending in the unknown-provider arm.

The bug (issue #106010): ``agent/auxiliary_client.py`` kept its own copy of the
provider alias table without the "local server aliases" group that
``hermes_cli.auth._PROVIDER_ALIASES`` has. An explicit ``provider: ollama`` lane with
``base_url`` + empty ``api_key`` kept the ollama identity, matched no registry entry,
and every call raised ``RuntimeError: Provider 'ollama' is set in config.yaml but no
API key was found`` — while the same endpoint worked as ``provider: custom``.

Secondary fix covered here: the OpenAI-compatible surface of these servers lives
under ``/v1``, so a bare host base_url (``http://127.0.0.1:11434``) must gain the
``/v1`` tail or the first request 404s on the server's native API.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _no_same_host_main_key():
    # The key ladder falls back to the main config's key when hosts match; the
    # placeholder assertions below need that rung to stay empty.
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


def test_ollama_bare_local_base_url_gets_v1_tail_and_placeholder_key(monkeypatch):
    """provider: ollama + loopback base_url + empty api_key must resolve a client
    (placeholder key, /v1 base) — not return None and raise downstream (#106010)."""
    from agent.auxiliary_client import resolve_provider_client

    client, _model = resolve_provider_client(
        "ollama",
        model="llama3.2",
        explicit_base_url="http://127.0.0.1:11434",
        explicit_api_key=None,
    )
    assert client is not None, (
        "explicit provider: ollama with a local base_url must build a client via the "
        "custom branch instead of dead-ending in the unknown-provider arm"
    )
    assert _client_attr(client, "base_url").rstrip("/") == "http://127.0.0.1:11434/v1"
    assert _client_attr(client, "api_key") == "no-key-required"


def test_ollama_explicit_api_key_passthrough_with_v1_tail(monkeypatch):
    """An explicit api_key must be honored verbatim (not swapped for the placeholder),
    with the /v1 tail still appended to a bare local base_url."""
    from agent.auxiliary_client import resolve_provider_client

    monkeypatch.setenv("HERMES_TEST_FAKE_KEY", "sk-test-local")
    client, _model = resolve_provider_client(
        "ollama",
        model="llama3.2",
        explicit_base_url="http://localhost:11434",
        explicit_api_key=os.environ["HERMES_TEST_FAKE_KEY"],
    )
    assert client is not None
    assert _client_attr(client, "base_url").rstrip("/") == "http://localhost:11434/v1"
    assert _client_attr(client, "api_key") == os.environ["HERMES_TEST_FAKE_KEY"]


def test_ollama_base_url_with_path_not_rewritten(monkeypatch):
    """A base_url that already carries a path (e.g. .../v1) must be kept as-is —
    never double-appended."""
    from agent.auxiliary_client import resolve_provider_client

    monkeypatch.setenv("HERMES_TEST_FAKE_KEY", "sk-test-local")
    client, _model = resolve_provider_client(
        "ollama",
        model="llama3.2",
        explicit_base_url="http://127.0.0.1:11434/v1",
        explicit_api_key=os.environ["HERMES_TEST_FAKE_KEY"],
    )
    assert client is not None
    assert _client_attr(client, "base_url").rstrip("/") == "http://127.0.0.1:11434/v1"


def test_bare_custom_loopback_base_url_kept_as_is(monkeypatch):
    """Only the local-server aliases get the /v1 tail: a literal ``provider: custom``
    with a bare loopback base keeps its URL verbatim (existing custom semantics)."""
    from agent.auxiliary_client import resolve_provider_client

    monkeypatch.setenv("HERMES_TEST_FAKE_KEY", "sk-test-local")
    client, _model = resolve_provider_client(
        "custom",
        model="llama3.2",
        explicit_base_url="http://127.0.0.1:11434",
        explicit_api_key=os.environ["HERMES_TEST_FAKE_KEY"],
    )
    assert client is not None
    assert _client_attr(client, "base_url").rstrip("/") == "http://127.0.0.1:11434"


def test_vllm_local_server_alias_routes_the_same_way():
    """Sibling local-server aliases (vllm, llama.cpp) share the custom routing and
    the /v1 tail."""
    from agent.auxiliary_client import resolve_provider_client

    client, _model = resolve_provider_client(
        "vllm",
        model="Qwen3-32B",
        explicit_base_url="http://127.0.0.1:8000",
        explicit_api_key=None,
    )
    assert client is not None
    assert _client_attr(client, "base_url").rstrip("/") == "http://127.0.0.1:8000/v1"
    assert _client_attr(client, "api_key") == "no-key-required"
