"""Managed Kanban recovery / replay-hold contract (design §5 and §12):
a failed managed attempt's replacement worker is bounded by the routing
policy (one safe alternate), uncertain-outcome attempts are never blindly
replayed, and operator reconciliation is the only path back to a replay.

Real sqlite board, real ``dispatch_once`` ticks, real
``_record_task_failure`` — nothing here mocks away the dispatcher path.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _requirements(**overrides):
    base = {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": "t_fallback_containment", "attempt_id": "1", "slot_id": "",
        "task_class": "cross-component", "required_capabilities": [],
        "input_tokens": 1000, "reserve_tokens": 8192, "reasoning": "high",
        "provenance": {"frozen_sha": "deadbeef", "verified_by": "test",
                       "complete": True, "contributors": []},
    }
    base.update(overrides)
    return base


def _policy():
    return {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": "openai-gpt5", "route_revision": 1, "provider": "openai",
            "model": "gpt-5", "endpoint": "https://api.openai.com/v1", "maker": "openai",
            "model_family": "gpt-5", "status": "approved", "allowed_roles": ["builder"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {"builder": {"deep": ["openai-gpt5"], "shallow": ["openai-gpt5"]}},
    }


@pytest.fixture
def routing_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _recovery_policy():
    policy = _policy()
    policy["routes"].append({
        **policy["routes"][0],
        "route_id": "anthropic-alternate", "provider": "anthropic",
        "model": "claude-alternate", "endpoint": "https://api.anthropic.com",
        "maker": "anthropic", "model_family": "claude",
    })
    policy["rankings"]["builder"]["deep"] = ["openai-gpt5", "anthropic-alternate"]
    policy["rankings"]["builder"]["shallow"] = ["openai-gpt5", "anthropic-alternate"]
    return policy


def _fail_current_run(conn, task):
    from hermes_cli import kanban_db_dispatch as kbd

    kbd._record_task_failure(
        conn, task.id, "fixture provider failure", outcome="crashed",
        failure_limit=10, release_claim=True, end_run=True,
        expected_run_id=task.current_run_id,
    )


def _health_payload(home, receipt_id, *, status, replay_safe):
    import time
    from agent.model_selection_store import get_receipt

    route = get_receipt(home, receipt_id)["selected"]
    now = int(time.time())
    return {
        "status": status, "replay_safe": replay_safe, "target_profile": "alice",
        "route_revision": route["route_revision"], "endpoint": route["endpoint"],
        "observed_at": now, "retry_after": now + 600 if status != "healthy" else 0,
    }


def test_managed_recovery_allows_one_safe_alternate_then_holds(
    routing_home, all_assignees_spawnable,
):
    """A known pre-output provider refusal may select one alternate route, but
    a third automatic worker attempt in the same failure episode is blocked."""
    from agent.model_selection_store import activate_policy, append_outcome, publish_policy
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.init_db()
    policy = _recovery_policy()
    record = publish_policy(routing_home, policy, approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", record["revision"])
    spawned = []

    def _spawn(task, workspace):
        spawned.append(task.model_override)
        return None

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="bounded replacement", assignee="alice", routing_role="builder",
            routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192},
            max_retries=10,
        )
        kbd.dispatch_once(conn, spawn_fn=_spawn)
        first = kb.get_task(conn, tid)
        append_outcome(routing_home, first.routing_receipt_id, "routing_started", {})
        append_outcome(
            routing_home, first.routing_receipt_id, "routing_health",
            _health_payload(
                routing_home, first.routing_receipt_id,
                status="denied_model", replay_safe=True,
            ),
        )
        _fail_current_run(conn, first)

        kbd.dispatch_once(conn, spawn_fn=_spawn)
        second = kb.get_task(conn, tid)
        append_outcome(routing_home, second.routing_receipt_id, "routing_started", {})
        append_outcome(
            routing_home, second.routing_receipt_id, "routing_health",
            _health_payload(
                routing_home, second.routing_receipt_id,
                status="denied_model", replay_safe=True,
            ),
        )
        _fail_current_run(conn, second)

        final = kbd.dispatch_once(conn, spawn_fn=_spawn)
        task = kb.get_task(conn, tid)

    assert spawned == ["gpt-5", "claude-alternate"]
    assert final.spawned == []
    assert task.status == "blocked"
    assert "alternate" in (task.last_failure_error or "").lower()


def test_managed_recovery_holds_when_prior_contact_is_uncertain(
    routing_home, all_assignees_spawnable,
):
    """A request that may have produced output or tool effects is never
    automatically replayed by a replacement worker."""
    from agent.model_selection_store import activate_policy, append_outcome, publish_policy
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.init_db()
    policy = _recovery_policy()
    record = publish_policy(routing_home, policy, approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", record["revision"])
    spawned = []

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="uncertain replay", assignee="alice", routing_role="builder",
            routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192},
            max_retries=10,
        )
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override))
        first = kb.get_task(conn, tid)
        append_outcome(routing_home, first.routing_receipt_id, "routing_started", {})
        append_outcome(
            routing_home, first.routing_receipt_id, "routing_health",
            _health_payload(
                routing_home, first.routing_receipt_id,
                status="healthy", replay_safe=False,
            ),
        )
        _fail_current_run(conn, first)

        result = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override))
        task = kb.get_task(conn, tid)

    assert spawned == ["gpt-5"]
    assert result.spawned == []
    assert task.status == "blocked"
    assert "uncertain" in (task.last_failure_error or "").lower()


def test_operator_reconciliation_allows_one_explicit_replay(
    routing_home, all_assignees_spawnable,
):
    """A replay hold is recoverable only through an auditable operator
    disposition; unblocking alone is not treated as reconciliation."""
    from agent.model_selection_store import (
        activate_policy, append_outcome, authorize_replay, publish_policy,
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.init_db()
    policy = _recovery_policy()
    record = publish_policy(routing_home, policy, approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", record["revision"])
    spawned = []

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="reconciled replay", assignee="alice", routing_role="builder",
            routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192},
            max_retries=10,
        )
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override))
        first = kb.get_task(conn, tid)
        append_outcome(routing_home, first.routing_receipt_id, "routing_started", {})
        append_outcome(
            routing_home, first.routing_receipt_id, "routing_health",
            _health_payload(
                routing_home, first.routing_receipt_id,
                status="healthy", replay_safe=False,
            ),
        )
        _fail_current_run(conn, first)
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override))
        assert kb.get_task(conn, tid).status == "blocked"

        authorize_replay(
            routing_home, first.routing_receipt_id,
            reason="verified no external effect", approval_ref="operator:test-reconcile",
        )
        assert kb.unblock_task(conn, tid)
        result = kbd.dispatch_once(
            conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override),
        )

    assert result.spawned
    assert spawned == ["gpt-5", "gpt-5"]


@pytest.mark.parametrize("history", ["effect_then_refusal", "refusal_then_interruption"])
def test_recovery_retains_whole_attempt_uncertainty(
    routing_home, all_assignees_spawnable, history,
):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from openai import OpenAI, RateLimitError
    from agent.managed_route_health import observe_request
    from agent.model_selection_store import activate_policy, append_outcome, publish_policy
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_dispatch as kbd

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            success = history == "effect_then_refusal" and len(requests) == 1
            payload = ({"choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                        "tool_calls": [{"id": "effect", "type": "function", "function": {
                            "name": "write_effect", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}
                       if success else {"error": {"message": "quota", "type": "rate_limit_error"}})
            body = json.dumps(payload).encode()
            self.send_response(200 if success else 429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    policy = _recovery_policy()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    policy["routes"][0]["endpoint"] = endpoint
    publish_policy(routing_home, policy, approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", 1)
    kb.init_db()
    spawned = []
    try:
        with kbc.connect() as conn, OpenAI(api_key="fixture", base_url=endpoint, max_retries=0) as client:
            tid = kb.create_task(conn, title="whole attempt replay", assignee="alice", routing_role="builder",
                                 routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192}, max_retries=10)
            spawn = lambda task, workspace: spawned.append(task.model_override)
            kbd.dispatch_once(conn, spawn_fn=spawn)
            first = kb.get_task(conn, tid)
            receipt = first.routing_receipt_id
            append_outcome(routing_home, receipt, "routing_started", {})
            if history == "effect_then_refusal":
                with observe_request(routing_home, receipt):
                    response = client.chat.completions.create(model="gpt-5", messages=[{"role": "user", "content": "act"}])
                assert response.choices[0].message.tool_calls[0].function.name == "write_effect"
                (routing_home / "effect.txt").write_text("external effect completed")
            with pytest.raises(RateLimitError), observe_request(routing_home, receipt):
                client.chat.completions.create(model="gpt-5", messages=[{"role": "user", "content": "continue"}])
            if history == "refusal_then_interruption":
                with pytest.raises(KeyboardInterrupt), observe_request(routing_home, receipt):
                    raise KeyboardInterrupt("worker died after request contact began")
            _fail_current_run(conn, first)
            result = kbd.dispatch_once(conn, spawn_fn=spawn)
            task = kb.get_task(conn, tid)
            assert result.spawned == []
            assert spawned == ["gpt-5"]
            assert task.status == "blocked"
            assert "uncertain" in task.last_failure_error.lower()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("history", ["effect_then_refusal", "refusal_then_interruption", "refusal"])
def test_actual_agent_attempt_drives_dispatch_replay_hold(
    routing_home, all_assignees_spawnable, history,
):
    import json
    import os
    import subprocess
    import sys
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from agent.model_selection_store import activate_policy, list_outcomes, publish_policy
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_dispatch as kbd

    requests = []
    accepted = threading.Event()
    release = threading.Event()
    effect = routing_home / "effect.txt"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith("/chat/completions"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
                return
            requests.append(request)
            if history == "refusal_then_interruption" and len(requests) == 2:
                accepted.set()
                release.wait(timeout=60)
                return
            success = history == "effect_then_refusal" and len(requests) == 1
            payload = ({"id": "fixture", "choices": [{"index": 0, "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "effect", "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps({
                    "path": str(effect), "content": "effect from actual tool loop"})}}]},
                "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
                if success else {"error": {"message": "Insufficient credits", "type": "insufficient_quota"}})
            body = json.dumps(payload).encode()
            self.send_response(200 if success else 402)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    policy = _recovery_policy()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    policy["routes"][0]["endpoint"] = endpoint
    publish_policy(routing_home, policy, approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", 1)
    kb.init_db()
    spawned = []
    process = None
    code = '''
import sys
from run_agent import AIAgent
from agent.managed_route_runtime import enforce_worker_route
home, receipt, endpoint, history = sys.argv[1:]
agent = AIAgent(provider="openai", model="gpt-5", base_url=endpoint, api_key="fixture-key",
    api_mode="chat_completions", reasoning_config={"enabled": True, "effort": "medium"}, max_iterations=4,
    request_overrides={"reasoning_effort": "medium"},
    enabled_toolsets=["file"], quiet_mode=True, skip_memory=True, skip_context_files=True,
    save_trajectories=False)
agent._disable_streaming = True
agent._managed_routing_home = home
agent._managed_routing_receipt_id = receipt
enforce_worker_route(home, receipt, actual_provider="openai", actual_model="gpt-5",
    actual_endpoint=endpoint, actual_reasoning="medium")
try:
    print(agent.run_conversation("Perform the harmless file effect, then finish."), flush=True)
    if history == "refusal_then_interruption":
        agent.run_conversation("Try the next request in the same attempt.")
finally:
    agent.close()
'''
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="actual worker replay", assignee="alice", routing_role="builder",
                                 routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192}, max_retries=10)
            spawn = lambda task, workspace: spawned.append(task.model_override)
            kbd.dispatch_once(conn, spawn_fn=spawn)
            first = kb.get_task(conn, tid)
            env = {key: value for key, value in os.environ.items() if not key.startswith("HERMES_KANBAN_")}
            env.update(HERMES_HOME=str(routing_home), TERMINAL_CWD=str(routing_home))
            with (routing_home / "worker.log").open("w+") as output:
                process = subprocess.Popen([sys.executable, "-c", code, str(routing_home),
                    first.routing_receipt_id, endpoint, history], env=env, stdout=output, stderr=subprocess.STDOUT)
                if history == "refusal_then_interruption":
                    assert accepted.wait(timeout=60), "worker never reached second accepted request"
                    process.kill()
                process.wait(timeout=60)
                output.seek(0)
                worker_log = output.read()
            if history != "refusal_then_interruption":
                assert process.returncode == 0, worker_log
            events = list_outcomes(routing_home, first.routing_receipt_id)
            starts = [event for event in events if event["kind"] == "routing_request_started"]
            health = [event for event in events if event["kind"] == "routing_health"]
            assert starts and health, worker_log
            if history == "effect_then_refusal":
                assert effect.exists(), worker_log
                assert effect.read_text() == "effect from actual tool loop", worker_log
                assert any(message.get("role") == "tool" for message in requests[-1]["messages"])
            elif history == "refusal_then_interruption":
                assert len(starts) > len(health)
                assert len(requests) == 2
            _fail_current_run(conn, first)
            result = kbd.dispatch_once(conn, spawn_fn=spawn)
            if history == "refusal":
                assert result.spawned
                assert spawned == ["gpt-5", "claude-alternate"]
            else:
                assert result.spawned == []
                assert spawned == ["gpt-5"]
                assert kb.get_task(conn, tid).status == "blocked"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()
