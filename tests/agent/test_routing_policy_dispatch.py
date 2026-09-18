"""Routing-policy outbound and recovery invariants."""

from types import SimpleNamespace

import pytest


class _RecordingCreate:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[])


def test_chat_dispatch_denial_after_request_mutation_sends_nothing(monkeypatch):
    """The last chat-completions seam rejects the mutated model before create()."""
    from agent import chat_completion_helpers as helpers
    from hermes_cli.routing_policy import RoutingPolicyError

    create = _RecordingCreate()
    client = SimpleNamespace(chat=SimpleNamespace(completions=create))
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        client=client,
    )
    monkeypatch.setattr(helpers, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["glm-*"]},
    })

    with pytest.raises(RoutingPolicyError):
        helpers._dispatch_nonstreaming_api_request(
            agent, {"model": "openrouter:z-ai/glm-5.2"}, make_client=lambda *_a, **_kw: client,
        )

    assert create.calls == []


def test_codex_responses_denial_sends_nothing(monkeypatch):
    """Responses transport is guarded separately from chat-completions."""
    from agent.codex_runtime import run_codex_stream
    from hermes_cli.routing_policy import RoutingPolicyError

    create = _RecordingCreate()
    client = SimpleNamespace(responses=create, base_url="https://openrouter.ai/api/v1")
    agent = SimpleNamespace(
        provider="openrouter", base_url="https://openrouter.ai/api/v1", model="z-ai/glm-5.2",
    )
    monkeypatch.setattr("hermes_cli.routing_policy.current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["z-ai/*"]},
    })

    with pytest.raises(RoutingPolicyError):
        run_codex_stream(agent, {"model": "z-ai/glm-5.2"}, client=client)

    assert create.calls == []


def test_policy_error_is_terminal_to_auxiliary_recovery_ladder():
    """A denial is not a recoverable provider failure and cannot advance a fallback."""
    from agent.auxiliary_client import _drive_ladder, _rung
    from hermes_cli.routing_policy import RoutingPolicyError

    attempted = []

    def ladder():
        _response, _error = yield from _rung(
            ("primary",), lambda _exc: True,
        )
        attempted.append("fallback")
        yield ("fallback",)

    def perform(step):
        if step == ("primary",):
            raise RoutingPolicyError("denied")
        attempted.append("send")

    with pytest.raises(RoutingPolicyError):
        _drive_ladder(ladder(), perform)

    assert attempted == []


def test_denied_cron_route_is_not_saved(monkeypatch):
    """A rejected cron route fails before the jobs-file write seam."""
    import cron.jobs as jobs
    from hermes_cli.routing_policy import RoutingPolicyError

    writes = []
    monkeypatch.setattr(jobs, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["z-ai/*"]},
    })
    monkeypatch.setattr(jobs, "save_jobs", lambda records: writes.append(records))

    with pytest.raises(RoutingPolicyError):
        jobs.create_job(prompt="p", schedule="every 1h", provider="openrouter", model="z-ai/glm-5.2")

    assert writes == []



def test_denied_cron_route_update_is_not_saved(monkeypatch):
    """Editing a legacy cron route rejects before the replacement record is saved."""
    import cron.jobs as jobs
    from hermes_cli.routing_policy import RoutingPolicyError

    writes = []
    monkeypatch.setattr(jobs, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["z-ai/*"]},
    })
    monkeypatch.setattr(jobs, "save_jobs", lambda records: writes.append(records))

    def invoke_apply(_job_id, apply):
        return apply([{"id": "j1"}], 0, {
            "id": "j1", "name": "legacy", "prompt": "p", "schedule": {"kind": "interval"},
            "enabled": True, "state": "scheduled", "next_run_at": "later", "repeat": {"completed": 0},
        })

    monkeypatch.setattr(jobs, "_with_job", invoke_apply)
    with pytest.raises(RoutingPolicyError):
        jobs.update_job("j1", {"model": "z-ai/glm-5.2", "provider": "openrouter"})

    assert writes == []


def test_denied_persisted_session_route_is_not_written(monkeypatch):
    """Session model mutation checks policy before it touches SQLite."""
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError

    db = object.__new__(SessionDB)
    writes = []
    monkeypatch.setattr("hermes_cli.routing_policy.current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["z-ai/*"]},
    })
    monkeypatch.setattr(db, "flush_token_counts", lambda: None)
    monkeypatch.setattr(db, "_write_model_config_patch", lambda *_a, **_kw: writes.append(True))

    with pytest.raises(RoutingPolicyError):
        db.update_session_model("s1", "z-ai/glm-5.2", provider="openrouter")

    assert writes == []


def test_denied_host_config_is_canonicalized_before_matching():
    """Operator host spelling and case cannot weaken the deny rule."""
    from hermes_cli.routing_policy import RoutingPolicyError, check_route

    with pytest.raises(RoutingPolicyError, match="base-url host"):
        check_route(
            {"enabled": True, "deny": {"base_url_hosts": ["HTTPS://API.Z.AI/v1/"]}},
            provider="openrouter", model="allowed", base_url="https://api.z.ai/v1",
        )


def test_auxiliary_primary_denial_blocks_sync_async_and_stream_sends(monkeypatch):
    """Resolved auxiliary wire routes are checked before every primary create call."""
    import asyncio
    from agent import auxiliary_client as auxiliary
    from hermes_cli.routing_policy import RoutingPolicyError

    class SyncCreate:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[])

    class AsyncCreate:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[])

    sync_create, async_create = SyncCreate(), AsyncCreate()
    sync_client = SimpleNamespace(
        base_url="https://denied.example/v1",
        chat=SimpleNamespace(completions=sync_create),
    )
    async_client = SimpleNamespace(
        base_url="https://denied.example/v1",
        chat=SimpleNamespace(completions=async_create),
    )
    monkeypatch.setattr(auxiliary, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"base_url_hosts": ["denied.example"]},
    }, raising=False)

    for client, async_mode, stream in ((sync_client, False, False), (sync_client, False, True), (async_client, True, False)):
        req = auxiliary._PreparedAuxRequest(
            client, "allowed-model", {"model": "allowed-model", "messages": []},
            "openrouter", "openrouter", "allowed-model", "https://requested.example/v1",
            None, None, 30.0, {}, "https://requested.example/v1",
        )
        monkeypatch.setattr(auxiliary, "_plan_aux_call", lambda *a, _req=req, **kw: (_req, {}, {}))
        with pytest.raises(RoutingPolicyError):
            if async_mode:
                asyncio.run(auxiliary._async_call_llm_impl(messages=[]))
            else:
                auxiliary._call_llm_impl(messages=[], stream=stream)

    assert sync_create.calls == []
    assert async_create.calls == []


def test_iteration_summary_denial_blocks_direct_create(monkeypatch):
    """The iteration-limit direct summary call checks its request client's wire route."""
    from agent import chat_completion_helpers as helpers
    from hermes_cli.routing_policy import RoutingPolicyError

    create = _RecordingCreate()
    client = SimpleNamespace(base_url="https://denied.example/v1", chat=SimpleNamespace(completions=create))
    agent = SimpleNamespace(
        provider="openrouter", model="allowed", base_url="https://requested.example/v1",
        _build_api_kwargs=lambda _messages: {"model": "allowed", "messages": []},
        _ensure_primary_openai_client=lambda **_kwargs: client,
    )
    monkeypatch.setattr(helpers, "sanitize_outbound_kwargs", lambda *_args: None)
    monkeypatch.setattr(helpers, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"base_url_hosts": ["denied.example"]},
    })

    attempt = helpers._chat_summary_attempt(agent, [], "summary-request")
    with pytest.raises(RoutingPolicyError):
        attempt(0)

    assert create.calls == []
