"""Per-request Anthropic wire client reuse across sequential LLM calls.

Mirrors ``tests/agent/test_request_client_reuse.py`` (the OpenAI-wire cache)
for the Anthropic request-local client. Before this cache existed,
``_create_request_anthropic_client`` built a fresh ``anthropic.Anthropic``
(and its httpx pool) on every single LLM call and ``_close_request_anthropic_client``
always fully closed it — no reuse across a turn's sequential tool-loop calls,
unlike the OpenAI-wire path.

- identical cache key (credentials, base URL, timeout, 1M-beta flag) → same
  client object handed back (the reuse win);
- key changes (credential rotation, base URL change) → evict + rebuild;
- cross-thread abort poisons the slot → the owner-thread close does a real
  close and the next create rebuilds;
- non-reuse close reasons (error cleanups, stale/interrupt kills) discard —
  only request_complete / stream_request_complete reuse;
- teardown (release_clients / close) really closes the cached client, or
  detaches it to the in-flight worker's own close when checked out.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


class _StubClient:
    """Minimal non-Mock client: _is_openai_client_closed reads ``is_closed``."""

    def __init__(self):
        self.is_closed = False

    def close(self):
        self.is_closed = True


def _make_agent(provider="anthropic", base_url="https://api.anthropic.com", model="claude-sonnet-5"):
    agent = AIAgent.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.api_mode = "anthropic_messages"
    agent._anthropic_api_key = "sk-ant-test"
    agent._anthropic_base_url = base_url
    agent._oauth_1m_beta_disabled = False
    # Real credential-refresh reaches auth/network state we don't need here;
    # the cache logic under test is agnostic to it.
    agent._try_refresh_anthropic_client_credentials = MagicMock(return_value=False)
    return agent


class _Harness:
    """Patch the Anthropic client build/socket seams and record calls."""

    def __init__(self, agent):
        self.agent = agent
        self.built = []  # reason
        self._patchers = []

    def __enter__(self):
        def _fake_build(*a, **k):
            self.built.append(k.get("drop_context_1m_beta"))
            return _StubClient()

        self._patchers = [
            patch("agent.anthropic_adapter.build_anthropic_client", side_effect=_fake_build),
            patch.object(self.agent, "_force_close_tcp_sockets", return_value=0),
        ]
        for p in self._patchers:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patchers:
            p.stop()


def test_reuse_on_identical_key_same_object():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_anthropic_client(reason="chat_completion_request")
        agent._close_request_anthropic_client(a, reason="request_complete")
        assert not a.is_closed  # kept for reuse, not really closed

        b = agent._create_request_anthropic_client(reason="chat_completion_request")
        assert b is a
        assert len(h.built) == 1


def test_rebuild_on_credential_rotation():
    agent = _make_agent()
    with _Harness(agent):
        a = agent._create_request_anthropic_client(reason="r")
        agent._close_request_anthropic_client(a, reason="request_complete")

        agent._anthropic_api_key = "sk-ant-rotated"
        b = agent._create_request_anthropic_client(reason="r")
        assert b is not a
        assert a.is_closed  # stale slot really closed on eviction

        agent._close_request_anthropic_client(b, reason="request_complete")
        c = agent._create_request_anthropic_client(reason="r")
        assert c is b


def test_non_reuse_reason_discards_client():
    agent = _make_agent()
    with _Harness(agent):
        a = agent._create_request_anthropic_client(reason="r")
        agent._close_request_anthropic_client(a, reason="request_error_cleanup")
        assert a.is_closed

        b = agent._create_request_anthropic_client(reason="r")
        assert b is not a


def test_cross_thread_abort_poisons_slot():
    agent = _make_agent()
    with _Harness(agent):
        a = agent._create_request_anthropic_client(reason="r")
        agent._abort_request_anthropic_client(a, reason="interrupt")
        # Owner thread's close now sees the poisoned slot and really closes.
        agent._close_request_anthropic_client(a, reason="request_complete")
        assert a.is_closed

        b = agent._create_request_anthropic_client(reason="r")
        assert b is not a


def test_concurrent_call_gets_untracked_client():
    agent = _make_agent()
    with _Harness(agent):
        a = agent._create_request_anthropic_client(reason="r")
        # Slot still checked out (in_use=True) — a second concurrent call
        # must not share it.
        b = agent._create_request_anthropic_client(reason="r")
        assert b is not a

        # Finishing the untracked one does a real close, not a slot release.
        agent._close_request_anthropic_client(b, reason="request_complete")
        assert b.is_closed
        # The tracked slot is unaffected and still reusable.
        agent._close_request_anthropic_client(a, reason="request_complete")
        c = agent._create_request_anthropic_client(reason="r")
        assert c is a


def test_agent_close_closes_cached_request_client():
    agent = _make_agent()
    with _Harness(agent):
        a = agent._create_request_anthropic_client(reason="r")
        agent._close_request_anthropic_client(a, reason="request_complete")
        assert not a.is_closed

        agent._close_cached_request_anthropic_client(reason="agent_close")
        assert a.is_closed

        # Idempotent: a second teardown must not error or double-act.
        agent._close_cached_request_anthropic_client(reason="agent_close")
