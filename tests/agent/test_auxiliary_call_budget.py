"""Opt-in limits exercise real auxiliary retry, recovery and SDK dispatch paths."""
import asyncio
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from agent import auxiliary_client as aux
from agent.auxiliary_call_budget import AuxiliaryCallBudget, AuxiliaryBudgetExceeded, auxiliary_call_budget
from tests.agent.test_connected_assistance import completion


@pytest.fixture
def route(monkeypatch):
    clients = []
    attempts = []
    def install(handler):
        def request(request):
            attempts.append(request)
            return handler(request)
        client = aux._create_openai_client(api_key="fixture", base_url="https://fixture.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(request)))
        clients.append(client)
        monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda *a, **kw:
            ("custom", "fixture", "https://fixture.test/v1", "fixture", None))
        monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (client, "fixture"))
        monkeypatch.setattr(aux, "_TRANSIENT_RETRY_BACKOFF_BASE", 0)
        return client, attempts
    yield install
    for client in clients:
        client.close()
    aux._reset_aux_unhealthy_cache()


def ok(_):
    return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 1,
        "model": "fixture", "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": "ok"}}]})


def call():
    return aux.call_llm(messages=[{"role": "user", "content": "fixture"}], timeout=60)


def test_unscoped_retries_unchanged_and_exception_restores_scope(route):
    count = []
    def handler(request):
        count.append(1)
        if len(count) < 3:
            raise httpx.ConnectError("connection reset by peer", request=request)
        return ok(request)
    _, attempts = route(handler)
    with auxiliary_call_budget(AuxiliaryCallBudget(160, 1, clock=lambda: 100)):
        with pytest.raises(AuxiliaryBudgetExceeded):
            call()
    assert len(attempts) == 1
    assert call().choices[0].message.content == "ok"
    assert len(attempts) == 3  # a fresh unscoped call still retries


def test_remaining_timeout_recomputed_after_retry(route):
    clock = [100]
    def handler(request):
        if clock[0] == 100:
            clock[0] = 150
            raise httpx.ConnectError("connection reset by peer", request=request)
        return ok(request)
    _, attempts = route(handler)
    budget = AuxiliaryCallBudget(160, 2, clock=lambda: clock[0])
    with auxiliary_call_budget(budget):
        call()
    assert [r.extensions["timeout"]["read"] for r in attempts] == [60, 10]


def test_nested_scopes_restore_outer_budget_and_threads_do_not_share(route):
    _, attempts = route(ok)
    outer = AuxiliaryCallBudget(160, 1, clock=lambda: 100)
    with auxiliary_call_budget(outer):
        with auxiliary_call_budget(AuxiliaryCallBudget(160, 1, clock=lambda: 100)):
            call()
        call()
        with pytest.raises(AuxiliaryBudgetExceeded):
            call()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(call).result(timeout=5).choices[0].message.content == "ok"
    assert call().choices[0].message.content == "ok"
    assert len(attempts) == 4


@pytest.mark.asyncio
async def test_concurrent_contexts_keep_independent_budgets(route):
    _, attempts = route(ok)
    ready = asyncio.Event()
    async def scoped(limit):
        with auxiliary_call_budget(AuxiliaryCallBudget(160, limit, clock=lambda: 100)):
            await ready.wait()
            try:
                call()
                return "ok"
            except AuxiliaryBudgetExceeded:
                return "blocked"
    tasks = [asyncio.create_task(scoped(0)), asyncio.create_task(scoped(1))]
    ready.set()
    assert await asyncio.gather(*tasks) == ["blocked", "ok"]
    assert len(attempts) == 1
    call()
    assert len(attempts) == 2


def test_queue_expiry_never_starts_request_or_leaks_permit(route, monkeypatch):
    _, attempts = route(ok)
    clock = [100]
    class Queue:
        released = False
        def acquire(self, *, timeout):
            assert timeout <= 0.1
            clock[0] = 161
            return False
        def release(self):
            self.released = True
    queue = Queue()
    monkeypatch.setattr(aux, "_acquire_sync_aux_semaphore", lambda _: queue)
    with auxiliary_call_budget(AuxiliaryCallBudget(160, 2, clock=lambda: clock[0])):
        with pytest.raises(AuxiliaryBudgetExceeded):
            call()
    assert attempts == [] and not queue.released


def test_policy_rechecked_after_auxiliary_hook_before_dispatch(route, monkeypatch):
    from agent import auxiliary_hooks
    _, attempts = route(ok)
    paused = []
    def check():
        if paused:
            raise PermissionError("paused")
    monkeypatch.setattr(auxiliary_hooks, "_has_hook", lambda _: True)
    monkeypatch.setattr(auxiliary_hooks._AuxCallHooks, "pre", lambda self: paused.append(True))
    with auxiliary_call_budget(AuxiliaryCallBudget(160, 2, check, clock=lambda: 100)):
        with pytest.raises(PermissionError, match="paused"):
            call()
    assert attempts == []


@pytest.mark.parametrize("allowance, expected", [(1, ["primary"]), (2, ["primary", "fallback"])])
def test_real_recovery_fallback_cannot_bypass_count(route, monkeypatch, allowance, expected):
    seen = []
    def handler(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(402, json={"error": {"message": "insufficient credits"}})
    client, _ = route(handler)
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (client, "primary"))
    monkeypatch.setattr(aux, "_try_configured_fallback_chain", lambda *a, **kw:
        (client, "fallback", "fallback_chain[0](custom)"))
    monkeypatch.setattr(aux, "_try_main_fallback_chain", lambda *a, **kw: (None, None, ""))
    monkeypatch.setattr(aux, "_try_payment_fallback", lambda *a, **kw: (None, None, ""))
    monkeypatch.setattr(aux, "_try_main_agent_model_fallback", lambda *a, **kw: (None, None, ""))
    with auxiliary_call_budget(AuxiliaryCallBudget(160, allowance, clock=lambda: 100)):
        with pytest.raises(Exception):
            call()
    assert seen == expected


def test_native_anthropic_stream_fallback_counts_each_physical_request(monkeypatch):
    from agent import anthropic_adapter
    attempts = []
    @contextmanager
    def stream(**kwargs):
        attempts.append(("stream", kwargs.get("timeout")))
        error = RuntimeError("streaming is not supported")
        error.status_code = 400
        raise error
        yield
    def create(**kwargs):
        attempts.append(("create", kwargs.get("timeout")))
        return SimpleNamespace(content=[], usage=None, stop_reason="end_turn")
    sdk = SimpleNamespace(messages=SimpleNamespace(stream=stream, create=create))
    client = aux.AnthropicAuxiliaryClient(sdk, "fixture", "fixture", "https://fixture.test")
    monkeypatch.setattr(anthropic_adapter, "_is_stream_unavailable_error", lambda _: True)
    with auxiliary_call_budget(AuxiliaryCallBudget(160, 1, clock=lambda: 100)):
        with pytest.raises(AuxiliaryBudgetExceeded):
            aux._create_with_progress(client, {"model": "fixture", "messages": [], "timeout": 60})
    assert attempts == [("stream", 60)]


def test_bounded_calls_retain_provider_work_on_owning_thread(monkeypatch):
    monkeypatch.setattr(aux, "_aux_interrupt_protected", lambda: True)
    monkeypatch.setattr(aux, "_capture_aux_cancel_check", lambda: lambda: False)
    owner = threading.current_thread()
    with auxiliary_call_budget(AuxiliaryCallBudget(160, 2, clock=lambda: 100)):
        observed = aux._run_protected_sync_provider_call(lambda _: threading.current_thread(), {})
    assert observed is owner


def test_unbounded_bedrock_adapter_is_refused_only_for_scoped_calls(monkeypatch):
    from agent import bedrock_adapter
    attempts = []
    def converse(**kwargs):
        attempts.append(kwargs)
        return completion({"findings": []})
    monkeypatch.setattr(bedrock_adapter, "call_converse", converse)
    client = aux.BedrockAuxiliaryClient("us-east-1", "fixture")
    with auxiliary_call_budget(AuxiliaryCallBudget(160, 2, clock=lambda: 100)):
        with pytest.raises(ValueError, match="provider_budget_unsupported"):
            aux._create_with_progress(client, {"model": "fixture", "messages": [], "timeout": 60})
    assert attempts == []
    aux._create_with_progress(client, {"model": "fixture", "messages": [], "timeout": 60})
    assert len(attempts) == 1
