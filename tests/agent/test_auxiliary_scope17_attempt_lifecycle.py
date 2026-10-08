"""Offline attempt lifecycle regressions for Codex auxiliary calls."""
import threading
import time
from types import SimpleNamespace

import pytest
import agent.auxiliary_client as aux


def _response(text="ok"):
    return SimpleNamespace(
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])],
        usage=None,
    )



def test_codex_timeout_isolated_from_concurrent_cached_request(monkeypatch):
    release = threading.Event()
    started = threading.Event()
    leaves = []

    class Leaf:
        api_key = "test"
        base_url = "https://chatgpt.com/backend-api/codex"

        def __init__(self):
            self.closed = False
            self.responses = SimpleNamespace(create=self.create)
            leaves.append(self)

        def create(self, **kwargs):
            if self is leaves[0]:
                started.set()
                release.wait(2)
            if self.closed:
                raise RuntimeError("shared transport closed")
            return _response()

        def close(self):
            self.closed = True

    monkeypatch.setattr(aux, "_client_cache_key", lambda *a, **kw: ("same-route",))
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *a, **kw: (aux.CodexAuxiliaryClient(Leaf(), "gpt-5.6-sol"), "gpt-5.6-sol"))
    monkeypatch.setattr(aux, "_evict_cached_client_instance", lambda *a: None)
    first, _ = aux._get_cached_client("openai-codex", "gpt-5.6-sol")
    second, _ = aux._get_cached_client("openai-codex", "gpt-5.6-sol")
    assert first is not second
    outcome = {}

    def slow():
        try:
            first.chat.completions.create(messages=[{"role": "user", "content": "slow"}], timeout=0.05)
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=slow)
    worker.start()
    try:
        assert started.wait(2)
        time.sleep(0.15)
        assert second.chat.completions.create(
            messages=[{"role": "user", "content": "fast"}], timeout=1,
        ).choices[0].message.content == "ok"
    finally:
        release.set()
        worker.join(2)
    assert isinstance(outcome.get("error"), TimeoutError)
    assert leaves[0].closed and not leaves[1].closed



def test_uncached_codex_transport_closes_when_request_owner_releases_it(monkeypatch):
    import gc

    closed = []

    class Leaf:
        api_key = "test"
        base_url = "https://chatgpt.com/backend-api/codex"

        def close(self):
            closed.append(True)

    monkeypatch.setattr(aux, "_client_cache_key", lambda *a, **kw: ("same-route",))
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *a, **kw: (aux.CodexAuxiliaryClient(Leaf(), "m"), "m"))
    client, _ = aux._get_cached_client("openai-codex", "m")
    assert closed == []
    del client
    gc.collect()
    assert closed == [True]



def test_completed_responses_object_cannot_escape_expired_deadline():
    leaf = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=lambda **kw: (time.sleep(0.08), _response())[1]),
        close=lambda: None,
    )
    with pytest.raises(TimeoutError):
        aux._CodexCompletionsAdapter(leaf, "gpt-5.6-sol").create(
            messages=[{"role": "user", "content": "x"}], timeout=0.02)



def test_codex_timeout_retry_rebuilds_client_before_second_attempt(monkeypatch):
    first = aux.CodexAuxiliaryClient(
        SimpleNamespace(api_key="test", base_url="https://chatgpt.com/backend-api/codex"), "m")
    second = aux.CodexAuxiliaryClient(
        SimpleNamespace(api_key="test", base_url="https://chatgpt.com/backend-api/codex"), "m")
    req = aux._PreparedAuxRequest(first, "m", {"model": "m", "messages": []},
                                  "openai-codex", "openai-codex", "m", None, None,
                                  "codex_responses", 30, {}, str(first.base_url))
    monkeypatch.setattr(aux, "_plan_aux_call", lambda *a, **kw: (req, {"main_runtime": None}, {}))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (second, "m"))
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: 1)
    monkeypatch.setattr(aux.time, "sleep", lambda *_: None)
    seen = []

    def send(client, kwargs, **opts):
        seen.append(client)
        if client is first:
            raise TimeoutError("Codex auxiliary Responses stream produced no output within 1s (no-progress timeout)")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(aux, "_relay_sync_completion", send)
    result = aux._call_llm_impl(task="compression", messages=[])
    assert result.choices[0].message.content == "ok"
    assert seen == [first, second]



def test_async_codex_timeout_retry_rebuilds_client(monkeypatch):
    import asyncio

    def wrapped():
        leaf = SimpleNamespace(api_key="test", base_url="https://chatgpt.com/backend-api/codex")
        return aux.AsyncCodexAuxiliaryClient(aux.CodexAuxiliaryClient(leaf, "m"))

    first, second = wrapped(), wrapped()
    req = aux._PreparedAuxRequest(first, "m", {"model": "m", "messages": []},
                                  "openai-codex", "openai-codex", "m", None, None,
                                  "codex_responses", 30, {}, str(first.base_url))
    monkeypatch.setattr(aux, "_plan_aux_call", lambda *a, **kw: (req, {"main_runtime": None}, {}))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (second, "m"))
    seen = []

    async def send(client, kwargs, **opts):
        seen.append(client)
        if client is first:
            raise TimeoutError("Codex auxiliary Responses stream produced no output within 1s (no-progress timeout)")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(aux, "_relay_async_completion", send)
    result = asyncio.run(aux._async_call_llm_impl(task="compression", messages=[]))
    assert result.choices[0].message.content == "ok"
    assert seen == [first, second]



@pytest.mark.parametrize("phase", ["acquire", "consume"])
def test_late_completion_respects_explicit_cancel(monkeypatch, phase):
    cancelled = threading.Event()

    def acquire(**kwargs):
        if phase == "acquire":
            cancelled.set()
            return _response()
        return iter(())

    def consume(*args, **kwargs):
        cancelled.set()
        return _response()

    monkeypatch.setattr("agent.codex_runtime._consume_codex_event_stream", consume)
    leaf = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=acquire), close=lambda: None,
    )
    with aux.aux_interrupt_protection(cancel_event=cancelled):
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            aux._CodexCompletionsAdapter(leaf, "m").create(messages=[], timeout=30)
def test_consumed_completion_cannot_escape_watchdog_timeout(monkeypatch):
    guards = []
    original = aux._CodexStreamGuard

    def make_guard(*args, **kwargs):
        guard = original(*args, **kwargs)
        guards.append(guard)
        return guard

    def consume(*args, **kwargs):
        guards[0].timed_out.set()
        return _response()

    monkeypatch.setattr(aux, "_CodexStreamGuard", make_guard)
    monkeypatch.setattr("agent.codex_runtime._consume_codex_event_stream", consume)
    leaf = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=lambda **kw: iter(())), close=lambda: None,
    )
    with pytest.raises(TimeoutError):
        aux._CodexCompletionsAdapter(leaf, "m").create(messages=[], timeout=30)
