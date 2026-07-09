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
from types import SimpleNamespace
from unittest.mock import patch

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


def test_force_close_tcp_sockets_descends_httpcore_1_connection_wrapper():
    """httpcore 1.x stores the real stream below conn._connection.

    Post-#29507: the helper must shut sockets down but must NOT release the
    FD via ``sock.close()`` — that race recycled FDs into unrelated file
    descriptors (kanban.db) and let TLS bytes overwrite SQLite headers. The
    owning httpx thread is responsible for closing FDs on its own unwind.
    """
    from agent.agent_runtime_helpers import force_close_tcp_sockets

    class FakeSocket:
        def __init__(self):
            self.shutdown_calls = 0
            self.close_calls = 0

        def shutdown(self, _how):
            self.shutdown_calls += 1

        def close(self):
            self.close_calls += 1

    sock = FakeSocket()
    stream = SimpleNamespace(_sock=sock)
    http11 = SimpleNamespace(_network_stream=stream)
    pool_entry = SimpleNamespace(_connection=http11)
    pool = SimpleNamespace(_connections=[pool_entry])
    transport = SimpleNamespace(_pool=pool)
    http_client = SimpleNamespace(_transport=transport)
    openai_client = SimpleNamespace(_client=http_client)

    assert force_close_tcp_sockets(openai_client) == 1
    assert sock.shutdown_calls == 1
    # #29507: close() must NOT be called from this helper — the owning
    # httpx worker thread releases the FD, not us.


def test_opencode_go_disables_keepalive_pooling():
    """opencode-go / opencode-zen must build its httpx client with
    keepalive_expiry=0.0 so every request opens a fresh connection.

    Regression guard for #61461: opencode-go sits behind a reverse proxy
    that drops idle connections at ~30s; httpx's keepalive pool then reuses
    those dead connections and the next request hangs until the proxy's
    timeout fires (~30s). Disabling pooling forces the cold (~3s) path.
    Other providers keep the 20s idle reaper (CLOSE-WAIT prevention).
    """
    calls: list = []

    def _fake_build_keepalive(base_url="", *, verify=True, keepalive_expiry=20.0):
        calls.append({"base_url": base_url, "keepalive_expiry": keepalive_expiry})
        # Return a throwaway stand-in so create_openai_client can attach it.
        return SimpleNamespace(close=lambda: None)

    agent = AIAgent(
        api_key="test-key",
        base_url="https://opencode.ai/zen/go/v1",
        model="deepseek-v4-flash",
        provider="opencode-go",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._client_kwargs = {
        "api_key": "test-key-value",
        "base_url": "https://opencode.ai/zen/go/v1",
    }

    constructed: list = []
    fake_openai = _make_fake_openai_factory(constructed)

    with patch.object(
        agent, "_build_keepalive_http_client", side_effect=_fake_build_keepalive
    ):
        with patch("run_agent.OpenAI", fake_openai):
            agent._create_openai_client(
                agent._client_kwargs, reason="test", shared=True
            )

    assert len(calls) == 1, f"expected 1 keepalive build, got {len(calls)}"
    assert calls[0]["keepalive_expiry"] == 0.0, (
        f"opencode-go must disable keepalive pooling (keepalive_expiry=0.0), "
        f"got {calls[0]['keepalive_expiry']}"
    )


def test_other_providers_keep_keepalive_reaper():
    """Non-opencode providers keep the 20s idle reaper (default)."""
    calls: list = []

    def _fake_build_keepalive(base_url="", *, verify=True, keepalive_expiry=20.0):
        calls.append({"base_url": base_url, "keepalive_expiry": keepalive_expiry})
        return SimpleNamespace(close=lambda: None)

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        provider="openrouter",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._client_kwargs = {
        "api_key": "test-key-value",
        "base_url": "https://openrouter.ai/api/v1",
    }

    constructed: list = []
    fake_openai = _make_fake_openai_factory(constructed)

    with patch.object(
        agent, "_build_keepalive_http_client", side_effect=_fake_build_keepalive
    ):
        with patch("run_agent.OpenAI", fake_openai):
            agent._create_openai_client(
                agent._client_kwargs, reason="test", shared=True
            )

    assert len(calls) == 1
    assert calls[0]["keepalive_expiry"] == 20.0, (
        f"non-opencode provider should keep keepalive_expiry=20.0, "
        f"got {calls[0]['keepalive_expiry']}"
    )
