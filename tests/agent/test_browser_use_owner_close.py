"""Hard agent close releases turn identities, not just session identity."""
from unittest.mock import Mock


def test_hard_close_releases_every_owner_independently(monkeypatch):
    from agent.client_lifecycle import ClientLifecycleMixin
    from tools import browser_use_cli
    import run_agent
    from tools.process_registry import process_registry
    monkeypatch.setattr(process_registry, 'list_sessions', lambda: [])
    monkeypatch.setattr(run_agent, 'cleanup_vm', lambda task: None)
    monkeypatch.setattr(run_agent, 'cleanup_browser', lambda task: None)
    release = Mock(side_effect=lambda owner: (_ for _ in ()).throw(OSError('retry')) if owner == 'bad' else None)
    monkeypatch.setattr(browser_use_cli, 'release_browser_use_owner', release, raising=False)
    agent = ClientLifecycleMixin()
    agent._process_owner_task_ids = {'cron:run-A', 'turn-B', 'bad'}
    agent._close_task_resources('session-A')
    assert {c.args[0] for c in release.call_args_list} == {'cron:run-A', 'turn-B', 'bad', 'session-A'}
