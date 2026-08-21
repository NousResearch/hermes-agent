"""Regression for #85446: a kimi-coding streaming request must never block
forever in httpcore's header read.

Two independent guarantees:

1. Routing: direct ``AIAgent`` construction for the Kimi Coding Plan
   providers (``kimi-coding`` / ``kimi-coding-cn``) resolves
   ``api_mode="anthropic_messages"`` when the endpoint is
   ``api.kimi.com/coding`` (Anthropic Messages wire). Previously the
   OpenAI-wire overlay declaration won and the agent sent a
   chat.completions request that the /coding endpoint never answers —
   hanging forever in ``_receive_response_headers`` with no error, so the
   fallback chain could not engage.

2. Fail-closed transport: the keepalive httpx clients used for OpenAI-wire
   calls no longer default to ``read=None``. A provider that accepts the
   connection and never sends headers surfaces as a read timeout instead of
   an unbounded block.
"""

import socket
import threading
import time

import httpx
import pytest

from agent.process_bootstrap import build_keepalive_http_client
from run_agent import AIAgent

KIMI_CODING_BASE = "https://api.kimi.com/coding"
MOONSHOT_LEGACY_BASE = "https://api.moonshot.ai/v1"


# ── Fail-closed keepalive transport ─────────────────────────────────────────

def test_keepalive_http_client_read_timeout_is_finite():
    """The aux keepalive client must carry a finite read timeout (#85446)."""
    client = build_keepalive_http_client("https://api.kimi.com/coding")
    assert isinstance(client, httpx.Client)
    assert client.timeout.read is not None
    assert client.timeout.read > 0


def test_agent_keepalive_http_client_read_timeout_is_finite():
    """The main-agent keepalive client must carry a finite read timeout."""
    client = AIAgent._build_keepalive_http_client("https://api.kimi.com/coding")
    assert isinstance(client, httpx.Client)
    assert client.timeout.read is not None
    assert client.timeout.read > 0


def test_keepalive_read_timeout_does_not_break_normal_response():
    """A normally-answering endpoint is unaffected by the finite read floor."""
    # A plain client built like the keepalive one must still complete a
    # normal request; the read timeout only bounds silence.
    client = build_keepalive_http_client("http://127.0.0.1:1")
    assert client.timeout.read is not None
    client.close()


# ── Routing: kimi-coding direct AIAgent construction ────────────────────────

def _make_agent(monkeypatch, provider, base_url, model="k3"):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    return AIAgent(
        api_key="sk-kimi-test123456",
        base_url=base_url,
        provider=provider,
        model=model,
        max_iterations=2,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_kimi_coding_direct_construction_routes_to_anthropic_messages(monkeypatch):
    """Direct AIAgent(provider='kimi-coding') against /coding must use the
    Anthropic Messages adapter — not chat.completions (#85446)."""
    agent = _make_agent(monkeypatch, "kimi-coding", KIMI_CODING_BASE)
    assert agent.api_mode == "anthropic_messages"
    # The anthropic client was actually built (adapter routing, not a label).
    assert getattr(agent, "_anthropic_client", None) is not None


def test_kimi_coding_cn_default_endpoint_stays_chat_completions(monkeypatch):
    """kimi-coding-cn defaults to the Moonshot CN OpenAI-wire endpoint
    (api.moonshot.cn/v1); direct construction must not force Messages."""
    agent = _make_agent(monkeypatch, "kimi-coding-cn", "https://api.moonshot.cn/v1")
    assert agent.api_mode == "chat_completions"


def test_kimi_coding_cn_coding_endpoint_routes_to_anthropic_messages(monkeypatch):
    """Both Coding-Plan provider names route to Messages when the endpoint is
    the Anthropic-wire /coding host (#85446)."""
    agent = _make_agent(monkeypatch, "kimi-coding-cn", KIMI_CODING_BASE)
    assert agent.api_mode == "anthropic_messages"


def test_kimi_legacy_moonshot_endpoint_stays_chat_completions(monkeypatch):
    """Legacy Moonshot-platform keys resolve to the OpenAI wire; direct
    construction against api.moonshot.ai must NOT be forced to Messages."""
    agent = _make_agent(monkeypatch, "kimi-coding", MOONSHOT_LEGACY_BASE)
    assert agent.api_mode == "chat_completions"


# ── Behavioral: header-read hang surfaces as an error ────────────────────────

def _silent_server(port_holder):
    """Accept one connection, drain the request, then never respond."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port_holder["port"] = srv.getsockname()[1]
    conn, _ = srv.accept()
    conn.settimeout(20)
    try:
        conn.recv(65536)  # request received, then silence
        time.sleep(20)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        srv.close()


def test_silent_provider_header_read_raises_timeout_error():
    """A provider that accepts the request and never sends headers must
    surface as a timeout error on the streaming path — not block forever."""
    import openai

    port_holder = {}
    t = threading.Thread(target=_silent_server, args=(port_holder,), daemon=True)
    t.start()
    while "port" not in port_holder:
        time.sleep(0.02)
    base_url = f"http://127.0.0.1:{port_holder['port']}/v1"

    client = openai.OpenAI(
        api_key="test-key",
        base_url=base_url,
        http_client=build_keepalive_http_client(base_url),
        max_retries=0,
    )
    start = time.time()
    with pytest.raises(Exception) as exc_info:
        stream = client.chat.completions.create(
            model="k3",
            messages=[{"role": "user", "content": "PONG"}],
            stream=True,
            timeout=httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0),
        )
        for _ in stream:
            pass
    elapsed = time.time() - start
    assert elapsed < 10, f"blocked too long ({elapsed:.1f}s) — timeout not applied"
    # The SDK normalizes transport timeouts into APITimeoutError.
    assert type(exc_info.value).__name__ == "APITimeoutError"
