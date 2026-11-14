"""The real agent facade enforces the host budget before entering the provider turn loop."""

import threading
import json
import subprocess
import sys

from run_agent import AIAgent
from tests.agent.test_cross_process_turn_lease import _agent_with_db, _DB


def test_real_facade_limits_two_executing_conversations_across_entry_surfaces(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("max_concurrent_sessions: 2\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.relay_cwd.resolve_relay_scope_cwds", lambda *args: ("", ""))
    entered = {key: threading.Event() for key in ("desktop", "telegram")}
    release = threading.Event()
    executed, results = [], []

    def turn(agent, *args, **kwargs):
        executed.append(agent.session_id)
        entered[agent.session_id].set()
        assert release.wait(15)
        return {"final_response": "synthetic done", "messages": []}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", turn)
    agents = [_agent_with_db(_DB(session_exists=False), session_id=key, platform=key)
              for key in ("desktop", "telegram", "cli")]
    workers = [threading.Thread(target=lambda agent=agent: results.append(AIAgent.run_conversation(agent, "synthetic")))
               for agent in agents[:2]]
    try:
        for worker in workers:
            worker.start()
        assert all(event.wait(15) for event in entered.values())
        refused = AIAgent.run_conversation(agents[2], "synthetic")
        assert refused["failed"] is True and refused["error_code"] == "MAX_CONCURRENT_SESSIONS"
        assert sorted(executed) == ["desktop", "telegram"]
    finally:
        release.set()
        for worker in workers:
            worker.join(15)
    assert all(not worker.is_alive() for worker in workers)
    assert len(results) == 2


def test_native_frozen_worker_borrows_exact_parent_reservation_until_provider_settles(tmp_path):
    from hermes_cli.execution_budget import execution_budget_snapshot, try_acquire_execution_permit
    from tests.waiting import wait_until

    (tmp_path / "config.yaml").write_text("max_concurrent_sessions: 1\n", encoding="utf-8")
    parent, refusal = try_acquire_execution_permit(profile_home=tmp_path, execution_id="managed", surface="managed")
    assert refusal is None
    ready = tmp_path / "ready"
    script = """
import json, os, sys
from pathlib import Path
from agent.managed_worker import validate_bootstrap, bind_worker_policy
vouch = json.loads(sys.stdin.readline())
home = vouch['profile_home']
os.environ['HERMES_HOME'] = home
frame = {'version': 1, 'home': home, 'scope': {'profile_id': home, 'session_id': 'managed',
    'execution_id': 'work', 'generation': 1, 'pid': os.getpid(), 'birth': 0, 'secret': 'synthetic', 'epoch': 1},
    'policy': {'config_json': '{"max_concurrent_sessions":99}'}, 'api_key': None, 'text': 'synthetic',
    'route': 'route', 'user_id': 'synthetic', 'chat_id': 'synthetic', 'turn_author': None,
    'safe_mode': False, 'ignore_user_config': False, 'execution_permit': vouch}
bind_worker_policy(validate_bootstrap(frame))
from tests.agent.test_cross_process_turn_lease import _agent_with_db, _DB
from run_agent import AIAgent
import agent.conversation_loop, agent.relay_cwd
worker = _agent_with_db(_DB(session_exists=False), session_id='managed', platform='cli')
worker._execution_permit_vouch = frame['execution_permit']
agent.relay_cwd.resolve_relay_scope_cwds = lambda *a: ('', '')
def provider(*args, **kwargs):
    Path(sys.argv[1]).write_text('entered')
    sys.stdin.readline()
    return {'final_response': 'done', 'messages': []}
agent.conversation_loop.run_conversation = provider
result = AIAgent.run_conversation(worker, 'synthetic')
assert result['final_response'] == 'done', result
"""
    child = subprocess.Popen([sys.executable, "-c", script, str(ready)], text=True,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        child.stdin.write(json.dumps(parent.vouch()) + "\n")
        child.stdin.flush()
        wait_until(lambda: ready.exists() or child.poll() is not None, timeout=20)
        if child.poll() is not None:
            stdout, stderr = child.communicate()
            raise AssertionError(stderr or stdout)
        assert ready.exists(), "native worker did not reach its guarded provider phase"
        parent.release()
        assert len(execution_budget_snapshot(profile_home=tmp_path)[1]) == 1
        extra, refusal = try_acquire_execution_permit(profile_home=tmp_path, execution_id="extra", surface="desktop")
        assert extra is None and refusal.reason == "MAX_CONCURRENT_SESSIONS"
        stdout, stderr = child.communicate("finish\n", timeout=20)
        assert child.returncode == 0, stderr
        assert execution_budget_snapshot(profile_home=tmp_path)[1] == []
    finally:
        parent.release()
        if child.poll() is None:
            child.kill()
            child.communicate()
