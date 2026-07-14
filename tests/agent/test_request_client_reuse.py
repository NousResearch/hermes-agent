"""Per-request OpenAI wire client reuse across sequential LLM calls.

Building a fresh ``openai.OpenAI`` client per LLM call costs ~19-35ms (new
httpx pool, TCP+TLS handshake), so ``_create_request_openai_client`` caches
ONE reusable wire client on the agent, keyed by the effective client kwargs:

- identical kwargs → same client object handed back (the reuse win);
- kwargs change (credential rotation, provider failover) → evict + rebuild;
- cross-thread abort (#29507) poisons the slot → the owner-thread close does
  a real close and the next create rebuilds;
- non-reuse close reasons (error cleanups, stale/interrupt kills, retry
  cleanups) discard — only a request that produced a response reports a
  reuse reason (request_complete / stream_request_complete);
- vision-header copilot variant is a distinct kwargs key;
- teardown (release_clients / close) really closes the cached client, or
  detaches it to the in-flight worker's own close when checked out (#29507);
- MoA facade and Mock passthroughs never enter the cache.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


class _StubClient:
    """Minimal non-Mock client: _is_openai_client_closed reads ``is_closed``."""

    def __init__(self):
        self.is_closed = False

    def close(self):
        self.is_closed = True


def _make_agent(provider="openai", base_url="https://api.openai.com/v1", model="gpt-5.4"):
    with patch("run_agent.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        agent = AIAgent(
            api_key="sk-test",
            base_url=base_url,
            provider=provider,
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    # Non-Mock shared client so the Mock passthrough branch doesn't trigger.
    agent.client = _StubClient()
    return agent


class _Harness:
    """Patch the wire-client build/close/socket seams and record calls."""

    def __init__(self, agent):
        self.agent = agent
        self.built = []   # (kwargs_copy, reason)
        self.closed = []  # (client, reason)
        self._patchers = []

    def __enter__(self):
        def _fake_create(kwargs, *, reason, shared):
            # Only record per-request wire clients; teardown tests can also
            # trigger a shared-client rebuild via _ensure_primary_openai_client.
            if not shared:
                self.built.append((dict(kwargs), reason))
            return _StubClient()

        def _fake_close(client, *, reason, shared):
            self.closed.append((client, reason))

        self._patchers = [
            patch.object(self.agent, "_create_openai_client", side_effect=_fake_create),
            patch.object(self.agent, "_close_openai_client", side_effect=_fake_close),
            patch.object(self.agent, "_force_close_tcp_sockets", return_value=0),
        ]
        for p in self._patchers:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patchers:
            p.stop()

    def closed_clients(self):
        return [client for client, _reason in self.closed]


def test_reuse_on_identical_kwargs_same_object():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="chat_completion_request")
        # The agent's outer loop owns retries — the SDK loop must stay off.
        assert h.built[-1][0]["max_retries"] == 0
        agent._close_request_openai_client(a, reason="request_complete")
        assert h.closed == []  # kept for reuse, not really closed

        b = agent._create_request_openai_client(reason="chat_completion_request")
        assert b is a
        assert len(h.built) == 1


def test_reuse_after_streaming_clean_finish():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="chat_completion_stream_request")
        agent._close_request_openai_client(a, reason="stream_request_complete")
        b = agent._create_request_openai_client(reason="chat_completion_stream_request")
        assert b is a
        assert h.closed == []


def test_rebuild_on_client_kwargs_change():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        agent._close_request_openai_client(a, reason="request_complete")

        # Credential rotation / provider failover mutate _client_kwargs.
        agent._client_kwargs["api_key"] = "sk-rotated"
        b = agent._create_request_openai_client(reason="r")
        assert b is not a
        # The stale cached client was really closed on eviction.
        assert a in h.closed_clients()
        assert h.built[-1][0]["api_key"] == "sk-rotated"

        # And the rotated client is itself reusable.
        agent._close_request_openai_client(b, reason="request_complete")
        c = agent._create_request_openai_client(reason="r")
        assert c is b


def test_rebuild_after_cross_thread_abort_poisons_cache():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        # Stranger-thread abort (#29507): stale-call detector / interrupt loop
        # shutdown(SHUT_RDWR) the pool's sockets. This client must never be
        # reused even though the worker's own finally reports a clean reason.
        agent._abort_request_openai_client(a, reason="stale_call_kill")
        agent._close_request_openai_client(a, reason="request_complete")
        assert a in h.closed_clients()  # poisoned → real close, not kept

        b = agent._create_request_openai_client(reason="r")
        assert b is not a


def test_kill_reason_close_discards_cached_client():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        agent._close_request_openai_client(a, reason="stream_retry_cleanup")
        assert a in h.closed_clients()

        b = agent._create_request_openai_client(reason="r")
        assert b is not a


def test_externally_closed_cached_client_rebuilds():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        agent._close_request_openai_client(a, reason="request_complete")
        a.is_closed = True  # e.g. closed behind our back

        b = agent._create_request_openai_client(reason="r")
        assert b is not a
        assert len(h.built) == 2


def test_copilot_vision_variant_gets_own_client():
    agent = _make_agent(provider="copilot", base_url="https://api.githubcopilot.com")
    text_kwargs = {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]}
    image_kwargs = {
        "model": "gpt-5.4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
    }
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r", api_kwargs=text_kwargs)
        agent._close_request_openai_client(a, reason="request_complete")

        # Vision request: different default_headers → must not reuse the
        # text-variant client.
        b = agent._create_request_openai_client(reason="r", api_kwargs=image_kwargs)
        assert b is not a
        assert h.built[-1][0]["default_headers"]["Copilot-Vision-Request"] == "true"
        agent._close_request_openai_client(b, reason="request_complete")

        # Consecutive vision requests reuse the vision-variant client.
        c = agent._create_request_openai_client(reason="r", api_kwargs=image_kwargs)
        assert c is b


def test_release_clients_closes_cached_request_client():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        agent._close_request_openai_client(a, reason="request_complete")

        agent.release_clients()
        assert (a, "cache_evict") in h.closed

        # Next create rebuilds instead of handing back the closed client.
        b = agent._create_request_openai_client(reason="r")
        assert b is not a


def test_agent_close_closes_cached_request_client():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        agent._close_request_openai_client(a, reason="request_complete")

        agent.close()
        assert (a, "agent_close") in h.closed

        # Idempotent: a second teardown must not double-close.
        before = len(h.closed)
        agent._close_cached_request_openai_client(reason="agent_close")
        assert len(h.closed) == before


def test_teardown_while_checked_out_defers_close_to_worker():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        # Checked out by an in-flight worker (workers can outlive turns):
        # teardown must not client.close() from a stranger thread (#29507) —
        # it aborts the sockets and detaches the slot instead.
        agent._close_cached_request_openai_client(reason="agent_close")
        assert a not in h.closed_clients()

        # The worker's own finally sees an untracked client → real close on
        # the owning thread, even with a clean-finish reason.
        agent._close_request_openai_client(a, reason="request_complete")
        assert a in h.closed_clients()

        b = agent._create_request_openai_client(reason="r")
        assert b is not a


def test_concurrent_checkout_gets_untracked_client():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        # Slot checked out by the first (still in-flight) call: a concurrent
        # call gets a fresh client with the old per-request lifecycle.
        b = agent._create_request_openai_client(reason="r")
        assert b is not a

        agent._close_request_openai_client(b, reason="request_complete")
        assert b in h.closed_clients()  # untracked → really closed

        agent._close_request_openai_client(a, reason="request_complete")
        assert a not in h.closed_clients()  # cached → kept

        c = agent._create_request_openai_client(reason="r")
        assert c is a


def test_moa_passthrough_unaffected():
    agent = _make_agent()
    agent.provider = "moa"
    facade = agent.client
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        assert a is facade
        assert h.built == []  # facade handed back, no wire client built

        # Close behaves exactly as before the cache existed.
        agent._close_request_openai_client(facade, reason="request_complete")
        assert facade in h.closed_clients()


def test_mock_client_passthrough_unaffected():
    agent = _make_agent()
    agent.client = MagicMock()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        assert a is agent.client
        assert h.built == []
