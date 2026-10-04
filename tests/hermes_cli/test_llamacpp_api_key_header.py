"""Managed llama.cpp authentication must send X-Api-Key alongside Bearer (#132799).

llama.cpp 0.4.x authenticates ``X-Api-Key`` and answers a Bearer-only request with
401 "Authentication header is not provided", which killed every auxiliary slot
(compression, title generation, MCP) routed at the local runtime. All managed-router
callers — supervisor management API, endpoint GETs, and the primary/auxiliary OpenAI
clients — therefore carry both headers, which keeps pre-0.4 builds (Authorization
only) working unchanged.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# run_agent's module level runs the interrupted-update-pull recovery probe before the
# test-time home I/O guard is installed (the same reason every tests/agent client test
# imports it eagerly); agent_runtime_helpers._ra() would otherwise import it lazily
# mid-test and trip the guard against the worktree's shared git dir.
import run_agent  # noqa: F401
from agent import agent_runtime_helpers, auxiliary_client
from hermes_cli.local_runtime import endpoint as lr_endpoint

# Synthetic marker values (assembled, not secrets): they only need to differ from the
# managed key so override assertions can tell the two apart.
_EXPLICIT_KEY = "user" + "-" + "configured"


def _managed(monkeypatch, base="http://127.0.0.1:18434/v1", key="t" * 24):
    monkeypatch.setattr(lr_endpoint, "_state_endpoint", lambda: {"base_url": base, "api_key": key})
    return lr_endpoint


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b"{}"


def _lower(req):
    return {k.lower(): v for k, v in req.headers.items()}


def test_managed_auth_header_pair_carries_both_schemes():
    from hermes_cli.local_runtime.endpoint import _managed_auth_headers

    headers = _managed_auth_headers("t" * 24)
    assert headers["Authorization"].startswith("Bearer ")
    assert headers["X-Api-Key"] == "t" * 24


def test_managed_auth_header_pair_empty_key_still_bearer():
    from hermes_cli.local_runtime.endpoint import _managed_auth_headers

    headers = _managed_auth_headers("")
    assert headers["Authorization"] == "Bearer "
    assert "X-Api-Key" not in headers


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:18434/v1",
    "http://127.0.0.1:18434",
    "http://127.0.0.1:18434/",
])
def test_twin_header_scoped_to_managed_root(monkeypatch, base_url):
    endpoint = _managed(monkeypatch)
    assert endpoint.llamacpp_auth_headers(base_url, "t" * 24) == {"X-Api-Key": "t" * 24}


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:11434/v1",   # a different local server (Ollama)
    "https://api.example.com/v1",  # a remote OpenAI-compatible gateway
])
def test_no_twin_header_off_the_managed_root(monkeypatch, base_url):
    endpoint = _managed(monkeypatch)
    assert endpoint.llamacpp_auth_headers(base_url, "t" * 24) == {}


@pytest.mark.parametrize("key", ["", "   ", "no-key-required", None, lambda: "token"])
def test_no_twin_header_without_usable_key(monkeypatch, key):
    endpoint = _managed(monkeypatch)
    assert endpoint.llamacpp_auth_headers("http://127.0.0.1:18434/v1", key) == {}


def test_no_twin_header_without_managed_server(monkeypatch):
    endpoint = _managed(monkeypatch)
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: None)
    assert endpoint.llamacpp_auth_headers("http://127.0.0.1:18434/v1", "t" * 24) == {}


def test_managed_get_json_sends_both_schemes(monkeypatch):
    endpoint = _managed(monkeypatch)
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured.update(_lower(req))
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    endpoint.managed_get_json("http://127.0.0.1:18434", "t" * 24, "/props?model=x", 1.0)
    assert captured["authorization"].startswith("Bearer ")
    assert captured["x-api-key"] == "t" * 24


def test_supervisor_requests_carry_both_schemes(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "_stable_api_key", lambda: "t" * 24)
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured.update(_lower(req))
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    sup = supervisor.LlamaServerSupervisor(tmp_path / "bin", tmp_path, port=59998)
    with sup._open("/models"):
        pass
    assert captured["authorization"].startswith("Bearer ")
    assert captured["x-api-key"] == "t" * 24


def test_aux_client_injects_twin_header(monkeypatch):
    _managed(monkeypatch)

    with patch.object(auxiliary_client, "OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        auxiliary_client._create_openai_client(
            api_key="t" * 24, base_url="http://127.0.0.1:18434/v1")
    headers = mock_openai.call_args.kwargs.get("default_headers") or {}
    assert headers.get("X-Api-Key") == "t" * 24


def test_aux_client_skips_twin_header_off_managed_root(monkeypatch):
    _managed(monkeypatch)

    with patch.object(auxiliary_client, "OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        auxiliary_client._create_openai_client(
            api_key="t" * 24, base_url="http://127.0.0.1:11434/v1")
    headers = mock_openai.call_args.kwargs.get("default_headers") or {}
    assert "X-Api-Key" not in headers


def test_aux_client_explicit_headers_win_over_twin(monkeypatch):
    _managed(monkeypatch)

    with patch.object(auxiliary_client, "OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        auxiliary_client._create_openai_client(
            api_key="t" * 24, base_url="http://127.0.0.1:18434/v1",
            default_headers={"X-Api-Key": _EXPLICIT_KEY})
    assert mock_openai.call_args.kwargs["default_headers"]["X-Api-Key"] == _EXPLICIT_KEY


def _chokepoint_agent():
    """Minimal agent stand-in for ``create_openai_client`` (provider profile lookups miss,
    no keepalive transport, no log context) — the chokepoint is otherwise self-contained."""
    return SimpleNamespace(
        provider="custom",
        model="test/model",
        _client_log_context=lambda: "",
        _build_keepalive_http_client=lambda base_url, verify=None: None,
    )


def test_primary_client_injects_twin_header(monkeypatch):
    _managed(monkeypatch)

    with patch("agent.process_bootstrap.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        agent_runtime_helpers.create_openai_client(
            _chokepoint_agent(),
            {"api_key": "t" * 24, "base_url": "http://127.0.0.1:18434/v1"},
            reason="test", shared=False)
    matching = [
        c for c in mock_openai.call_args_list
        if c.kwargs.get("base_url") == "http://127.0.0.1:18434/v1"
    ]
    assert matching, "OpenAI was never constructed with the managed base_url"
    assert all(
        (c.kwargs.get("default_headers") or {}).get("X-Api-Key") == "t" * 24
        for c in matching), (
        "the primary client chokepoint must reinstall the X-Api-Key twin on every "
        "rebuild from bare {api_key, base_url} kwargs (#132799)"
    )


def test_primary_client_skips_twin_header_off_managed_root(monkeypatch):
    _managed(monkeypatch)

    with patch("agent.process_bootstrap.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        agent_runtime_helpers.create_openai_client(
            _chokepoint_agent(),
            {"api_key": "t" * 24, "base_url": "http://127.0.0.1:11434/v1"},
            reason="test", shared=False)
    matching = [
        c for c in mock_openai.call_args_list
        if c.kwargs.get("base_url") == "http://127.0.0.1:11434/v1"
    ]
    assert matching
    assert all(
        "X-Api-Key" not in (c.kwargs.get("default_headers") or {})
        for c in matching)
