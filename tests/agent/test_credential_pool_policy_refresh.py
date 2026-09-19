"""External pool policy takes effect only when a cached session admits its next turn."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest


def _home(path, prefix):
    path.mkdir()
    (path / "config.yaml").write_text("model:\n  provider: openrouter\n  default: openai/gpt-4o-mini\n")
    (path / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"openrouter": [
        {"id": f"{prefix}-{i}", "label": f"private-label-{i}", "priority": i,
         "source": "manual", "auth_type": "api_key", "access_token": f"test-secret-{prefix}-{i}",
         "base_url": "https://openrouter.ai/api/v1"} for i in range(2)]}}))


def _agent(monkeypatch, handler, **overrides):
    from agent.client_lifecycle import ClientLifecycleMixin
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    monkeypatch.setattr(ClientLifecycleMixin, "_build_keepalive_http_client", staticmethod(
        lambda *a, **kw: httpx.Client(transport=httpx.MockTransport(handler))))
    runtime = resolve_runtime_provider(requested="openrouter", target_model="openai/gpt-4o-mini", **overrides)
    return AIAgent(model="openai/gpt-4o-mini", provider=runtime["provider"],
                   api_key=runtime["api_key"], base_url=runtime["base_url"],
                   api_mode=runtime["api_mode"], credential_pool=runtime.get("credential_pool"),
                   enabled_toolsets=[], skip_context_files=True, skip_memory=True,
                   skip_background_review=True, save_trajectories=False, quiet_mode=True)


def _reply(request):
    payload = {"id": "reply", "object": "chat.completion", "created": 1,
        "model": "openai/gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": "done"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}}
    if json.loads(request.content).get("stream"):
        payload["object"] = "chat.completion.chunk"
        payload["choices"][0]["delta"] = payload["choices"][0].pop("message")
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n")
    return httpx.Response(200, json=payload)


@pytest.mark.parametrize("publication", ["external", "shared_pool", "external_remove", "shared_remove"])
def test_external_policy_waits_for_next_turn_and_stays_profile_scoped(tmp_path, monkeypatch, caplog, publication):
    from agent.credential_pool import load_pool
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    first, second = tmp_path / "a", tmp_path / "b"
    _home(first, "a")
    _home(second, "b")
    entered, release = threading.Event(), threading.Event()
    seen = []

    def transport(request):
        seen.append(request.headers["authorization"])
        if len(seen) == 1:
            entered.set()
            assert release.wait(20)
        return _reply(request)

    monkeypatch.setenv("HERMES_HOME", str(first))
    agent = _agent(monkeypatch, transport)
    old_client = agent.client
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(agent.run_conversation, "first")
        try:
            assert entered.wait(20)
            publisher = agent._credential_pool if publication.startswith("shared") else load_pool("openrouter")
            if publication.endswith("remove"):
                publisher.remove_index(1)
            else:
                publisher.move_entry("a-1", 0)
            # Routine bookkeeping from a stale process must not undo the operator's preference.
            agent._credential_pool._persist()
            assert agent.client is old_client and agent.api_key == "test-secret-a-0"
        finally:
            release.set()
        history = active.result(timeout=30)["messages"]
    prompt = agent._cached_system_prompt
    token = set_hermes_home_override(second)
    try:
        other = _agent(monkeypatch, transport)
        other.run_conversation("other profile")
        assert other.api_key == "test-secret-b-0"
        other.close()
    finally:
        reset_hermes_home_override(token)
    with caplog.at_level("INFO", logger="agent.credential_pool_policy"):
        agent.run_conversation("next", conversation_history=history)
    assert agent.api_key == "test-secret-a-1"
    assert seen == ["Bearer test-secret-a-0", "Bearer test-secret-b-0", "Bearer test-secret-a-1"]
    assert agent._cached_system_prompt == prompt
    assert agent.model == "openai/gpt-4o-mini" and agent.provider == "openrouter"
    policy_logs = "\n".join(r.message for r in caplog.records if r.name == "agent.credential_pool_policy")
    assert "source=pool" in policy_logs and "generation=1" in policy_logs
    assert all(secret not in policy_logs for secret in ("test-secret", "private-label", "a-1"))
    agent.close()


@pytest.mark.parametrize("guard", ["torn_pool", "unreadable_pool", "torn_config", "missing_config",
                                   "invalid_strategy", "invalid_pool", "explicit_key", "lease", "client_failure",
                                   "round_robin", "status_only", "strategy"])
def test_policy_refresh_preserves_last_good_and_explicit_intent(tmp_path, monkeypatch, guard, caplog):
    from agent.credential_pool import load_pool
    from agent.client_lifecycle import ClientLifecycleMixin

    home = tmp_path / "home"
    _home(home, "a")
    monkeypatch.setenv("HERMES_HOME", str(home))
    if guard == "round_robin":
        with (home / "config.yaml").open("a") as f:
            f.write("credential_pool_strategies:\n  openrouter: round_robin\n")
    overrides = {"explicit_api_key": "manual-pinned-key"} if guard == "explicit_key" else {}
    agent = _agent(monkeypatch, _reply, **overrides)
    result = agent.run_conversation("first")
    old_key, old_client = agent.api_key, agent.client
    pool = agent._credential_pool
    lease = pool.acquire_lease(agent._credential_pool_entry_id) if guard == "lease" else None
    if guard == "round_robin":
        load_pool("openrouter").select()
    elif guard == "status_only":
        pool._persist()
    elif guard == "strategy":
        pool._persist()
        with (home / "config.yaml").open("a") as f:
            f.write("credential_pool_strategies:\n  openrouter: least_used\n")
    else:
        load_pool("openrouter").move_entry("a-1", 0)
    auth = (home / "auth.json").read_text()
    config = (home / "config.yaml").read_text()
    if guard == "torn_pool":
        (home / "auth.json").write_text('{"credential_pool":')
    elif guard == "invalid_pool":
        (home / "auth.json").write_text('{"credential_pool": {"openrouter": [{"id": "partial", "priority": "x"}]}}')
    elif guard == "unreadable_pool":
        (home / "auth.json").unlink()
        (home / "auth.json").mkdir()
    elif guard == "torn_config":
        (home / "config.yaml").write_text("credential_pool_strategies: [")
    elif guard == "missing_config":
        (home / "config.yaml").unlink()
    elif guard == "invalid_strategy":
        (home / "config.yaml").write_text(config + "credential_pool_strategies: []\n")
    elif guard == "client_failure":
        def broken_transport(*args, **kwargs):
            raise ValueError("test-secret-client-build")
        monkeypatch.setattr(ClientLifecycleMixin, "_build_keepalive_http_client", staticmethod(broken_transport))
    damaged_auth = (home / "auth.json").read_text() if guard in {"torn_pool", "invalid_pool"} else None
    with caplog.at_level("INFO", logger="agent.credential_pool_policy"):
        result = agent.run_conversation("next", conversation_history=result["messages"])
    if guard == "strategy":
        assert agent.api_key == "test-secret-a-1" and agent.client is not old_client
    else:
        assert agent.api_key == old_key and agent.client is old_client
    if damaged_auth is not None:
        assert (home / "auth.json").read_text() == damaged_auth
    assert result["final_response"] == "done"
    assert agent.model == "openai/gpt-4o-mini" and agent.provider == "openrouter"
    logs = "\n".join(r.message for r in caplog.records if r.name == "agent.credential_pool_policy")
    assert all(value not in logs for value in ("test-secret", "manual-pinned-key", "private-label"))
    if guard == "explicit_key":
        assert agent._credential_pool is None
    elif guard in {"round_robin", "status_only", "strategy"}:
        assert json.loads((home / "auth.json").read_text()).get("credential_pool_generations", {}) == {}
    else:
        if guard == "unreadable_pool":
            (home / "auth.json").rmdir()
        (home / "auth.json").write_text(auth)
        (home / "config.yaml").write_text(config)
        if lease:
            pool.release_lease(lease)
        monkeypatch.setattr(ClientLifecycleMixin, "_build_keepalive_http_client", staticmethod(
            lambda *a, **kw: httpx.Client(transport=httpx.MockTransport(_reply))))
        agent.run_conversation("recovered", conversation_history=result["messages"])
        assert agent.api_key == "test-secret-a-1"
    agent.close()
