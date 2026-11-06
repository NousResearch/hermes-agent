from types import SimpleNamespace

from agent.turn_lifecycle_hooks import begin_agent_turn_hooks, finish_agent_turn_hooks


def test_turn_hooks_send_only_lifecycle_metadata_and_verified_project(monkeypatch):
    from gateway.session_context import scoped_desktop_project

    calls = []

    def invoke(hook_name, **payload):
        calls.append((hook_name, payload))
        return [{"action": "allow"}] if hook_name == "pre_agent_turn" else []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke)
    agent = SimpleNamespace(platform="desktop")
    project = {"project_id": "p_123", "primary_path": "/trusted/project"}

    with scoped_desktop_project(project):
        started, message = begin_agent_turn_hooks(
            agent, session_id="s1", turn_id="t1", task_id="task1"
        )
        finish_agent_turn_hooks(
            agent,
            session_id="s1",
            turn_id="t1",
            task_id="task1",
            outcome="success",
        )

    assert started is True
    assert message is None
    assert [name for name, _ in calls] == ["pre_agent_turn", "post_agent_turn"]
    assert calls[0][1] == {
        "session_id": "s1",
        "turn_id": "t1",
        "task_id": "task1",
        "platform": "desktop",
        "model": "",
        "desktop_project": project,
        "workspace_candidate": {"source": "desktop_project_root", "path": "/trusted/project"},
    }
    assert calls[1][1] == {
        "session_id": "s1",
        "turn_id": "t1",
        "task_id": "task1",
        "platform": "desktop",
        "desktop_project": project,
        "outcome": "success",
        "error_type": None,
    }
    assert all("prompt" not in payload and "response" not in payload for _, payload in calls)


def test_workspace_candidate_uses_session_cwd_for_gateway_and_never_falls_back_for_desktop(monkeypatch):
    from agent.runtime_cwd import reset_session_cwd, set_session_cwd
    from agent.turn_lifecycle_hooks import _workspace_candidate
    from gateway import session_context
    from gateway.session_context import scoped_desktop_project

    monkeypatch.setattr(session_context, "get_session_env", lambda *_args: "")
    token = set_session_cwd("/trusted/gateway-workspace")
    try:
        with scoped_desktop_project(None):
            assert _workspace_candidate(SimpleNamespace(platform="telegram")) == {
                "source": "session_context_cwd",
                "path": "/trusted/gateway-workspace",
            }
            assert _workspace_candidate(SimpleNamespace(platform="desktop")) is None
    finally:
        reset_session_cwd(token)


def test_turn_admission_hook_blocks_when_policy_refuses_or_fails(monkeypatch):
    agent = SimpleNamespace(platform="desktop")

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [{"action": "block", "message": "project gate refused"}],
    )
    started, message = begin_agent_turn_hooks(
        agent, session_id="s1", turn_id="t1", task_id="task1"
    )
    assert started is False
    assert message == "project gate refused"

    def broken_dispatch(*_args, **_kwargs):
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", broken_dispatch)
    started, message = begin_agent_turn_hooks(
        agent, session_id="s1", turn_id="t2", task_id="task2"
    )
    assert started is False
    assert message == "Turn admission hook failed (RuntimeError); the turn was not started."

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: None)
    started, message = begin_agent_turn_hooks(
        agent, session_id="s1", turn_id="t3", task_id="task3"
    )
    assert started is False
    assert message == "Turn admission hook dispatch returned an invalid result; the turn was not started."


def test_turn_facade_refusal_stops_before_conversation_execution(monkeypatch):
    from agent import conversation_loop
    from agent import turn_facade_lease
    from agent.turn_facade import TurnFacadeMixin
    from gateway.session_context import scoped_desktop_project

    class Lease:
        def stop_refresher(self):
            pass

        def join_threads(self):
            pass

        def clear_interrupt(self):
            pass

        def release(self):
            pass

    lease = Lease()
    monkeypatch.setattr(
        turn_facade_lease,
        "admit_durable_turn_lease",
        lambda *_args, **_kwargs: SimpleNamespace(
            early_result=None, lease=lease, conversation_history=[]
        ),
    )
    monkeypatch.setattr(
        "agent.background_review.cancel_background_review_for_live_turn",
        lambda *_args, **_kwargs: None,
    )
    conversation_calls = []
    monkeypatch.setattr(
        conversation_loop,
        "run_conversation",
        lambda *_args, **_kwargs: conversation_calls.append("called"),
    )
    hook_payloads = []

    def invoke(hook_name, **payload):
        hook_payloads.append((hook_name, payload))
        return [{"action": "block", "message": "Host Project gate refused"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke)
    agent = SimpleNamespace(session_id="s1", platform="desktop", _persist_disabled=False)
    project = {"project_id": "p_123", "primary_path": "/trusted/project"}

    with scoped_desktop_project(project):
        result = TurnFacadeMixin.run_conversation(agent, "private prompt", conversation_history=[])

    assert result["failed"] is True
    assert result["final_response"] == "Host Project gate refused"
    assert conversation_calls == []
    assert [name for name, _ in hook_payloads] == ["pre_agent_turn"]
    assert hook_payloads[0][1]["desktop_project"] == project


def test_settlement_interrupt_does_not_skip_core_turn_cleanup(monkeypatch):
    from contextlib import nullcontext

    from agent import conversation_loop, relay_runtime, turn_facade_lease
    from agent import portal_tags
    from agent import relay_cwd
    from agent import review_idle_queue
    from agent.turn_facade import TurnFacadeMixin

    events = []

    class Lease:
        def stop_refresher(self):
            events.append("stop_refresher")

        def join_threads(self):
            events.append("join_threads")

        def clear_interrupt(self):
            events.append("clear_interrupt")

        def release(self):
            events.append("lease_release")

        def start(self):
            events.append("lease_start")

    lease = Lease()
    relay_turn = SimpleNamespace(relay_enabled=False)
    relay_lease = object()
    coordinator = SimpleNamespace(
        acquire_conversation=lambda **_kwargs: relay_lease,
        begin_turn=lambda *_args, **_kwargs: relay_turn,
        finish_logical_calls=lambda *_args, **_kwargs: None,
        end_turn=lambda *_args, **_kwargs: events.append("relay_end"),
        release_conversation=lambda *_args, **_kwargs: events.append("relay_release"),
    )
    monkeypatch.setattr(relay_runtime, "SESSION_COORDINATOR", coordinator)
    monkeypatch.setattr(relay_cwd, "resolve_relay_scope_cwds", lambda *_args: ("", ""))
    monkeypatch.setattr(
        turn_facade_lease,
        "admit_durable_turn_lease",
        lambda *_args, **_kwargs: SimpleNamespace(
            early_result=None, lease=lease, conversation_history=[]
        ),
    )
    monkeypatch.setattr(
        "agent.background_review.cancel_background_review_for_live_turn",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(conversation_loop, "run_conversation", lambda *_args, **_kwargs: {
        "failed": False, "final_response": "done"
    })
    monkeypatch.setattr(portal_tags, "set_conversation_context", lambda *_args: object())
    monkeypatch.setattr(portal_tags, "set_affinity_scope", lambda *_args: object())
    monkeypatch.setattr("agent.aux_accounting.set_accounting_context", lambda *_args: object())
    monkeypatch.setattr("agent.subagent_lifecycle.bind_subagent_parent", lambda *_args: nullcontext())
    monkeypatch.setattr("agent.auxiliary_client.scoped_runtime_main", lambda *_args: nullcontext())
    monkeypatch.setattr("agent.interrupt_scope.track_in_interrupt_scope", lambda *_args: nullcontext())
    monkeypatch.setattr(review_idle_queue.QUEUE, "note_turn_started", lambda: None)
    monkeypatch.setattr(review_idle_queue.QUEUE, "note_turn_finished", lambda: None)

    def invoke(hook_name, **_payload):
        if hook_name == "pre_agent_turn":
            return [{"action": "allow"}]
        raise KeyboardInterrupt

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke)
    agent = SimpleNamespace(
        session_id="s1", platform="desktop", _persist_disabled=False,
        _conversation_root_id=lambda: "root", _reset_activity_labels_after_turn=lambda: None,
    )

    try:
        TurnFacadeMixin.run_conversation(agent, "private prompt", conversation_history=[])
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("the plugin interruption should propagate after core cleanup")

    assert events == [
        "lease_start", "stop_refresher", "relay_end", "relay_release",
        "stop_refresher", "join_threads", "clear_interrupt", "lease_release",
    ]
