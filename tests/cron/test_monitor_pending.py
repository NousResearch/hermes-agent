"""Recovery relationships, including crash and competing-attempt boundaries."""
import importlib
import pytest


@pytest.fixture
def pending(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.monitor_pending
    return importlib.reload(cron.monitor_pending)


def test_bounded_retry_retains_observation_and_fences_stale_owners(pending):
    job = {"id": "fixture", "monitor_script": "fixture.py", "prompt": "summarize"}
    first = pending.begin(job, "event A")
    with pytest.raises(pending.RecoveryRequired):
        pending.resume(job)  # A crash is not proof that the first attempt did nothing.
    with pytest.raises(pending.RecoveryRequired):
        pending.begin(job, "event B")
    pending.settle(job, first["token"], safe_failure=True)
    second = pending.resume(job)
    assert second["output"] == "event A" and second["token"] != first["token"]
    with pytest.raises(pending.RecoveryRequired):
        pending.settle(job, first["token"], response="stale")
    pending.settle(job, second["token"], safe_failure=True)
    with pytest.raises(pending.RecoveryRequired):
        pending.resume(job)
    assert pending.inspect(job)["output"] == "event A"
    assert pending.inspect(job)["attempts"] == pending.MAX_ATTEMPTS


def test_ready_response_survives_unknown_delivery_and_ack_suppresses_duplicate(pending):
    job = {"id": "fixture", "monitor_script": "fixture.py", "deliver": "local"}
    attempt = pending.begin(job, "event A")
    pending.settle(job, attempt["token"], response="saved summary", completed=True)
    with pytest.raises(pending.RecoveryRequired):
        pending.resume(job)  # Includes restart after a partial/unknown delivery.
    assert pending.inspect(job)["response"] == "saved summary"
    with pytest.raises(pending.RecoveryRequired):
        pending.resume(dict(job, deliver="another-target"))
    pending.acknowledge(job, attempt["token"])
    assert pending.begin(job, "event A") is None
    next_attempt = pending.begin(job, "event B")
    assert next_attempt["output"] == "event B"
    with pytest.raises(pending.RecoveryRequired):
        pending.acknowledge(job, attempt["token"])


def test_external_runtime_and_tool_middleware_cannot_claim_pre_effect_failure(pending):
    from types import SimpleNamespace
    for mode in ("codex_app_server", "chat_completions", "acp"):
        job = {"id": mode, "monitor_commit_policy": "safe_retry"}
        row = pending.begin(job, "event")
        job["_monitor_pending_commit"] = {"token": row["token"]}
        attempt = pending.Attempt(job)
        calls = []
        agent = SimpleNamespace(api_mode=mode, base_url="acp://fixture" if mode == "acp" else "",
                                _execute_tool_calls=lambda: calls.append("middleware"))
        attempt.attach(agent)
        attempt.started = True
        if mode == "chat_completions":
            agent._execute_tool_calls()
            assert calls == ["middleware"]
        attempt.result = {"failed": True, "turn_exit_reason": "api_error"}
        attempt.finish({})
        assert pending.inspect(job)["state"] == "unknown"
        with pytest.raises(pending.RecoveryRequired):
            pending.resume(job)


def test_real_agent_failure_and_retry_preserve_the_same_observation(monkeypatch):
    """Actual agent + local HTTP provider; no model service or delivery is contacted."""
    from tests.agent.test_empty_tool_name_loop_dampening import agent_env, _text_resp
    context = agent_env.__wrapped__()
    agent, handler = next(context)
    try:
        import cron.monitor_pending as store
        job = {"id": "native-agent-fixture", "monitor_commit_policy": "safe_retry"}
        record = store.begin(job, "event A")
        job["_monitor_pending_commit"] = {"token": record["token"]}
        attempt = store.Attempt(job)
        attempt.attach(agent)
        attempt.started = True
        original = handler.do_POST

        def rejected(request):
            request.rfile.read(int(request.headers.get("Content-Length", 0)))
            body = b'{"error":{"message":"fixture denied","type":"authentication_error"}}'
            request.send_response(401)
            request.send_header("Content-Type", "application/json")
            request.send_header("Content-Length", str(len(body)))
            request.end_headers()
            request.wfile.write(body)

        monkeypatch.setattr(handler, "do_POST", rejected)
        attempt.result = agent.run_conversation("Summarize event A", conversation_history=[], task_id="native-failure")
        assert attempt.result.get("failed") is True
        attempt.finish({})
        record = store.resume(job)
        assert record["output"] == "event A" and record["attempts"] == 2
        monkeypatch.setattr(handler, "do_POST", original)
        handler.response_queue.append(_text_resp("Summary A"))
        job["_monitor_pending_commit"] = {"token": record["token"]}
        recovered = store.Attempt(job)
        recovered.attach(agent)
        recovered.started = True
        recovered.result = agent.run_conversation("Summarize event A", conversation_history=[], task_id="native-recovery")
        recovered.response = recovered.result.get("final_response")
        recovered.finish({})
        saved = store.inspect(job)
        assert saved["state"] == "ready" and saved["response"] == "Summary A"
    finally:
        context.close()
