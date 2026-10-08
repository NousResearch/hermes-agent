"""api_server keeps one memory provider per session across requests (#120116)."""

from types import SimpleNamespace

import pytest

from gateway.platforms.api_server_memory_sessions import ApiServerMemorySessions


class _Manager:
    def __init__(self):
        self.shut_down = False

    def flush_pending(self, timeout=None):
        return True

    def shutdown_all(self):
        self.shut_down = True


def _agent(manager, session_id="sess-1"):
    return SimpleNamespace(_memory_manager=manager, session_id=session_id)


def test_manager_survives_across_requests_and_is_exclusive_per_home(monkeypatch):
    registry = ApiServerMemorySessions(max_size=8, idle_ttl_secs=3600.0)
    home = ["home-a"]
    monkeypatch.setattr(ApiServerMemorySessions, "_owner_home", staticmethod(lambda: (home[0], None)))
    # Shutdowns run inline so the assertion below does not race a daemon thread.
    monkeypatch.setattr(registry, "_shutdown_async", registry._shutdown)

    assert registry.checkout("sess-1") is None  # first request builds its own
    manager = _Manager()
    registry.checkin(_agent(manager))
    assert registry.checkout("sess-1") is manager  # next request on the session gets it back
    assert registry.checkout("sess-1") is None  # ...exclusively: a concurrent request builds a new one
    loser = _Manager()
    registry.checkin(_agent(manager))
    registry.checkin(_agent(loser))  # the concurrent request checks in behind the winner
    assert manager.shut_down and not loser.shut_down
    # Same session id under another profile home is a different key (#120116 x multiplex).
    home[0] = "home-b"
    assert registry.checkout("sess-1") is None
    home[0] = "home-a"
    assert registry.checkout("sess-1") is loser


@pytest.mark.asyncio
async def test_a_parked_manager_never_serves_another_session_key(monkeypatch):
    """Two channels opening with the same words derive one chat-completions session id; the
    provider initialised under one ``X-Hermes-Session-Key`` must not recall/retain for the other."""
    from aiohttp.test_utils import TestClient, TestServer

    from tests.gateway.test_api_server import _create_app, _make_adapter, _patch_create_agent_runtime

    built = []

    class _Agent:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]
            self._gateway_session_key = kwargs.get("gateway_session_key")
            # AIAgent adopts a handed-back manager as-is; otherwise it initialises one for its own key.
            self._memory_manager = kwargs.get("memory_manager") or SimpleNamespace(
                scope=self._gateway_session_key, flush_pending=lambda timeout=None: True,
                shutdown_all=lambda: None)
            self.session_prompt_tokens = self.session_completion_tokens = self.session_total_tokens = 0
            built.append(self)

        def run_conversation(self, **kwargs):
            return {"final_response": "ok", "messages": [], "api_calls": 1}

    _patch_create_agent_runtime(monkeypatch, {}, _Agent)
    adapter = _make_adapter(api_key="sk-test-key-0123456789")
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        for key in ("webui:dm:alice", "webui:dm:bob", "webui:dm:alice"):
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-test-key-0123456789", "X-Hermes-Session-Key": key},
                json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]})
            assert resp.status == 200
    alice, bob, alice_again = built
    assert alice.session_id == bob.session_id == alice_again.session_id
    assert bob._memory_manager.scope == "webui:dm:bob"
    assert alice_again._memory_manager is alice._memory_manager
