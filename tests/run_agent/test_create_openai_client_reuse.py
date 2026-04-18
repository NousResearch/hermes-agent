"""Regression guardrail: sequential _create_openai_client calls must not
share a closed transport across invocations.

This is the behavioral twin of test_create_openai_client_kwargs_isolation.py.
That test pins "don't mutate input kwargs" at the syntactic level — it catches
#10933 specifically because the bug mutated ``client_kwargs`` in place. This
test pins the user-visible invariant at the behavioral level: no matter HOW a
future keepalive / transport reimplementation plumbs sockets in, the Nth call
to ``_create_openai_client`` must not hand back a client wrapping a
now-closed httpx transport from an earlier call.

AlexKucera's Discord report (2026-04-16): after ``hermes update`` pulled
#10933, the first chat on a session worked, every subsequent chat failed
with ``APIConnectionError('Connection error.')`` whose cause was
``RuntimeError: Cannot send a request, as the client has been closed``.
That is the exact scenario this test reproduces at object level without a
network, so it runs in CI on every PR.
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent():
    return AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _make_fake_openai_factory(constructed):
    """Return a fake ``OpenAI`` class that records every constructed instance
    along with whatever ``http_client`` it was handed (or ``None`` if the
    caller did not inject one).

    The fake also forwards ``.close()`` calls down to the http_client if one
    is present, mirroring what the real OpenAI SDK does during teardown and
    what would expose the #10933 bug.
    """

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self._http_client = kwargs.get("http_client")
            self._closed = False
            constructed.append(self)

        def close(self):
            self._closed = True
            hc = self._http_client
            if hc is not None and hasattr(hc, "close"):
                try:
                    hc.close()
                except Exception:
                    pass

    return _FakeOpenAI


def test_second_create_does_not_wrap_closed_transport_from_first():
    """Back-to-back _create_openai_client calls on the same _client_kwargs
    must not hand call N a closed http_client from call N-1.

    The bug class: call 1 injects an httpx.Client into self._client_kwargs,
    client 1 closes (SDK teardown), its http_client closes with it, call 2
    reads the SAME now-closed http_client from self._client_kwargs and wraps
    it. Every request through client 2 then fails.
    """
    agent = _make_agent()
    constructed: list = []
    fake_openai = _make_fake_openai_factory(constructed)

    # Seed a baseline kwargs dict resembling real runtime state.
    agent._client_kwargs = {
        "api_key": "test-key-value",
        "base_url": "https://api.example.com/v1",
    }

    with patch("run_agent.OpenAI", fake_openai):
        # Call 1 — what _replace_primary_openai_client does at init/rebuild.
        client_a = agent._create_openai_client(
            agent._client_kwargs, reason="initial", shared=True
        )
        # Simulate the SDK teardown that follows a rebuild: the old client's
        # close() is invoked, which closes its underlying http_client if one
        # was injected. This is exactly what _replace_primary_openai_client
        # does via _close_openai_client after a successful rebuild.
        client_a.close()

        # Call 2 — the rebuild path. This is where #10933 crashed on the
        # next real request.
        client_b = agent._create_openai_client(
            agent._client_kwargs, reason="rebuild", shared=True
        )

    assert len(constructed) == 2, f"expected 2 OpenAI constructions, got {len(constructed)}"
    assert constructed[0] is client_a
    assert constructed[1] is client_b

    hc_a = constructed[0]._http_client
    hc_b = constructed[1]._http_client

    # If the implementation does not inject http_client at all, we're safely
    # past the bug class — nothing to share, nothing to close. That's fine.
    if hc_a is None and hc_b is None:
        return

    # If ANY http_client is injected, the two calls MUST NOT share the same
    # object, because call 1's object was closed between calls.
    if hc_a is not None and hc_b is not None:
        assert hc_a is not hc_b, (
            "Regression of #10933: _create_openai_client handed the same "
            "http_client to two sequential constructions. After the first "
            "client is closed (normal SDK teardown on rebuild), the second "
            "wraps a closed transport and every subsequent chat raises "
            "'Cannot send a request, as the client has been closed'."
        )

    # And whatever http_client the LATEST call handed out must not be closed
    # already. This catches implementations that cache the injected client on
    # ``self`` (under any attribute name) and rebuild the SDK client around
    # it even after the previous SDK close closed the cached transport.
    if hc_b is not None:
        is_closed_attr = getattr(hc_b, "is_closed", None)
        if is_closed_attr is not None:
            assert not is_closed_attr, (
                "Regression of #10933: second _create_openai_client returned "
                "a client whose http_client is already closed. New chats on "
                "this session will fail with 'Cannot send a request, as the "
                "client has been closed'."
            )


def test_replace_primary_openai_client_survives_repeated_rebuilds():
    """Full rebuild path: exercise _replace_primary_openai_client three times
    back-to-back and confirm every resulting ``self.client`` is a fresh,
    usable construction rather than a wrapper around a previously-closed
    transport.

    _replace_primary_openai_client is the real rebuild entrypoint — it is
    what runs on 401 credential refresh, pool rotation, and model switch.
    If a future keepalive tweak stores state on ``self`` between calls,
    this test is what notices.
    """
    agent = _make_agent()
    constructed: list = []
    fake_openai = _make_fake_openai_factory(constructed)

    agent._client_kwargs = {
        "api_key": "test-key-value",
        "base_url": "https://api.example.com/v1",
    }

    with patch("run_agent.OpenAI", fake_openai):
        # Seed the initial client so _replace has something to tear down.
        agent.client = agent._create_openai_client(
            agent._client_kwargs, reason="seed", shared=True
        )
        # Three rebuilds in a row. Each one must install a fresh live client.
        for label in ("rebuild_1", "rebuild_2", "rebuild_3"):
            ok = agent._replace_primary_openai_client(reason=label)
            assert ok, f"rebuild {label} returned False"
            cur = agent.client
            assert not cur._closed, (
                f"after rebuild {label}, self.client is already closed — "
                "this breaks the very next chat turn"
            )
            hc = cur._http_client
            if hc is not None:
                is_closed_attr = getattr(hc, "is_closed", None)
                if is_closed_attr is not None:
                    assert not is_closed_attr, (
                        f"after rebuild {label}, self.client.http_client is "
                        "closed — reproduces #10933 (AlexKucera report, "
                        "Discord 2026-04-16)"
                    )

    # All four constructions (seed + 3 rebuilds) should be distinct objects.
    # If two are the same, the rebuild is cacheing the SDK client across
    # teardown, which also reproduces the bug class.
    assert len({id(c) for c in constructed}) == len(constructed), (
        "Some _create_openai_client calls returned the same object across "
        "a teardown — rebuild is not producing fresh clients"
    )


def _stub_proxies(monkeypatch, proxies):
    """Stub urllib's proxy discovery so tests stay hermetic."""
    import urllib.request

    monkeypatch.setattr(urllib.request, "getproxies", lambda: dict(proxies))
    for name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def _has_keepalive(transport):
    if transport is None:
        return False
    pool = getattr(transport, "_pool", None)
    return bool(getattr(pool, "_socket_options", None))


def test_create_openai_client_skips_injection_when_https_proxy_set(monkeypatch):
    agent = _make_agent()
    constructed: list = []
    fake_openai = _make_fake_openai_factory(constructed)
    _stub_proxies(monkeypatch, {"https": "http://127.0.0.1:7897"})

    with patch("run_agent.OpenAI", fake_openai):
        agent._create_openai_client(
            {
                "api_key": "test-key-value",
                "base_url": "https://api.example.com/v1",
            },
            reason="proxy_env_https",
            shared=False,
        )

    assert len(constructed) == 1
    assert "http_client" not in constructed[0]._kwargs
    assert constructed[0]._http_client is None, (
        "Proxy-configured clients must delegate to the SDK default transport "
        "so httpx can apply trust_env proxy mounts."
    )


def test_create_openai_client_skips_injection_for_each_proxy_key(monkeypatch):
    agent = _make_agent()

    for key in ("http", "https", "all"):
        constructed: list = []
        fake_openai = _make_fake_openai_factory(constructed)
        _stub_proxies(monkeypatch, {key: "http://127.0.0.1:7897"})

        with patch("run_agent.OpenAI", fake_openai):
            agent._create_openai_client(
                {
                    "api_key": "test-key-value",
                    "base_url": "https://api.example.com/v1",
                },
                reason=f"proxy_{key}",
                shared=False,
            )

        assert constructed[0]._http_client is None


def test_create_openai_client_keepalive_direct_path_when_no_proxy(monkeypatch):
    agent = _make_agent()
    constructed: list = []
    fake_openai = _make_fake_openai_factory(constructed)
    _stub_proxies(monkeypatch, {})

    with patch("run_agent.OpenAI", fake_openai):
        agent._create_openai_client(
            {
                "api_key": "test-key-value",
                "base_url": "https://api.example.com/v1",
            },
            reason="no_proxy_keepalive",
            shared=False,
        )

    http_client = constructed[0]._http_client
    assert http_client is not None
    assert _has_keepalive(http_client._transport), (
        "Direct-connect path lost keepalive socket options."
    )


def test_has_proxy_configured_matches_urllib_getproxies(monkeypatch):
    _stub_proxies(monkeypatch, {})
    assert AIAgent._has_proxy_configured() is False

    _stub_proxies(monkeypatch, {"https": "http://10.0.0.1:8080"})
    assert AIAgent._has_proxy_configured() is True

    _stub_proxies(monkeypatch, {})
    assert AIAgent._has_proxy_configured() is False

    _stub_proxies(monkeypatch, {"no": "localhost"})
    assert AIAgent._has_proxy_configured() is False


def test_create_openai_client_skips_injection_for_system_proxy(monkeypatch):
    agent = _make_agent()
    constructed: list = []
    fake_openai = _make_fake_openai_factory(constructed)
    _stub_proxies(monkeypatch, {"https": "http://10.0.0.1:8080"})

    with patch("run_agent.OpenAI", fake_openai):
        agent._create_openai_client(
            {
                "api_key": "test-key-value",
                "base_url": "https://api.example.com/v1",
            },
            reason="system_proxy",
            shared=False,
        )

    assert constructed[0]._http_client is None, (
        "System-level proxy config should also suppress keepalive transport "
        "injection."
    )


def _fake_httpx_client_with_mock_sockets(default_sock, mount_sock):
    from types import SimpleNamespace

    def _conn(sock):
        stream = SimpleNamespace(_sock=sock)
        return SimpleNamespace(_network_stream=stream)

    def _transport(sock):
        pool = SimpleNamespace(_connections=[_conn(sock)])
        return SimpleNamespace(_pool=pool)

    class _URLPattern:
        def __init__(self, pattern):
            self._pattern = pattern

    mounts = {_URLPattern("https://"): _transport(mount_sock)}
    return SimpleNamespace(
        _transport=_transport(default_sock),
        _mounts=mounts,
    )


def test_iter_client_sockets_covers_mounts():
    default_sock = MagicMock(name="default_sock")
    mount_sock = MagicMock(name="mount_sock")
    http_client = _fake_httpx_client_with_mock_sockets(default_sock, mount_sock)

    seen = list(AIAgent._iter_client_sockets(http_client))
    assert default_sock in seen
    assert mount_sock in seen, (
        "Proxy-mount sockets must be included in client socket iteration."
    )


def test_force_close_tcp_sockets_closes_mount_sockets():
    default_sock = MagicMock(name="default_sock")
    mount_sock = MagicMock(name="mount_sock")
    http_client = _fake_httpx_client_with_mock_sockets(default_sock, mount_sock)
    fake_openai = MagicMock(_client=http_client)

    closed = AIAgent._force_close_tcp_sockets(fake_openai)

    assert closed == 2
    default_sock.shutdown.assert_called_once()
    default_sock.close.assert_called_once()
    mount_sock.shutdown.assert_called_once()
    mount_sock.close.assert_called_once()
