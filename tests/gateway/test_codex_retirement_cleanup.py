"""A failed writer retirement must not discard independent soft-eviction work."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.codex_lifecycle import codex_retirements, wait_for_codex_writer
from agent.transports.codex_app_server_session import CodexAppServerSession
from gateway.run import GatewayRunner
from run_agent import AIAgent


@pytest.mark.parametrize('route', ['soft', 'commit', 'pressure'])
def test_failed_retirement_preserves_cleanup_and_fences_successor(tmp_path, monkeypatch, route):
    events = []
    exited = threading.Event()

    class Client:
        def close(self):
            events.append('retire')
            if not exited.is_set():
                raise TimeoutError('synthetic writer still alive')

    session = CodexAppServerSession(codex_home=str(tmp_path), resume_thread_id='synthetic-thread')
    session._client = Client()
    agent = AIAgent.__new__(AIAgent)
    agent._codex_session = session
    agent._session_messages = [{'role': 'user', 'content': 'synthetic memory'}]
    transcript = list(agent._session_messages)
    agent._db_flush_scan_prefix = list(transcript)
    agent._memory_manager = object()
    agent.commit_memory_session = Mock(side_effect=lambda messages: events.append(('memory', list(messages))))
    agent.shutdown_memory_provider = Mock()
    agent._close_task_resources = Mock()
    review, child = Mock(), Mock()
    agent._active_children_lock = threading.Lock()
    agent._active_children = {review, child}
    agent._background_review_agent = review
    agent.client = object()
    agent._retire_shared_openai_client = Mock(side_effect=lambda *a, **kw: events.append('shared-client'))
    agent._close_cached_request_openai_client = Mock()
    agent._close_cached_request_anthropic_client = Mock()
    runner = GatewayRunner.__new__(GatewayRunner)
    owners = codex_retirements(runner)
    owners.track(agent)
    monkeypatch.setattr('hermes_cli.mem_trim.trim_memory', lambda **_: None)

    if route == 'soft':
        runner._release_evicted_agent_soft(agent)
    elif route == 'commit':
        runner._commit_then_release_soft(agent, 'synthetic-lane')
    else:
        # A second healthy owner must exit before the first owner's memory extraction.
        peer_session = Mock(writer_key=(str(tmp_path), 'peer-thread'))
        peer_session.close.side_effect = lambda: events.append('peer-retire')
        peer = SimpleNamespace(_codex_session=peer_session, release_clients=Mock())
        owners.track(peer)
        plan = [('synthetic-lane', agent), ('peer-lane', peer)]
        runner._release_pressure_batch(plan)
        assert not plan
        peer.release_clients.assert_called_once()
        assert peer._codex_session is None
        assert events.index('peer-retire') < events.index(('memory', transcript))

    # Use the real AIAgent soft-release implementation: only the failing transport is injected.
    assert events[0] == 'retire'
    if route == 'soft':
        agent.commit_memory_session.assert_not_called()
    else:
        agent.commit_memory_session.assert_called_once_with(transcript)
        assert events.index(('memory', transcript)) < events.index('shared-client')
    agent._retire_shared_openai_client.assert_called_once()
    agent._close_cached_request_openai_client.assert_called_once_with(reason='cache_evict')
    agent._close_cached_request_anthropic_client.assert_called_once_with(reason='cache_evict')
    child.release_clients.assert_called_once()
    review.release_clients.assert_not_called()
    review.close.assert_not_called()
    assert agent._active_children == {review}
    agent.shutdown_memory_provider.assert_not_called()
    agent._close_task_resources.assert_not_called()
    assert agent.client is None
    assert agent._session_messages == [] and agent._db_flush_scan_prefix is None
    assert agent._codex_session is session

    successor_client = Mock()
    successor_client.request.return_value = {'thread': {'id': 'synthetic-thread'}}
    factory = Mock(return_value=successor_client)
    successor_session = CodexAppServerSession(codex_home=str(tmp_path), resume_thread_id='synthetic-thread',
                                              client_factory=factory)
    successor = SimpleNamespace(_codex_session=successor_session, _codex_retirements=owners)
    try:
        with pytest.raises(TimeoutError, match='writer still alive'):
            wait_for_codex_writer(successor)
        factory.assert_not_called()
        assert owners._owners[id(agent)][1] is session
        exited.set()
        wait_for_codex_writer(successor)
        assert agent._codex_session is None and not owners._owners
        assert successor_session.ensure_started() == 'synthetic-thread'
        factory.assert_called_once()
        owners.release(agent)  # idempotent; does not close the new writer
        successor_client.close.assert_not_called()
    finally:
        exited.set()
        session.close()
        successor_session.close()
