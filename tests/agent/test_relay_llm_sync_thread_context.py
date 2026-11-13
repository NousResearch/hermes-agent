"""Thread-context contract for synchronous Relay completion (#131375 / PR #131377).

Unlike ``tests/agent/test_relay_llm.py`` this module has **no** module-level
``pytest.importorskip("nemo_relay")``: it drives ``relay_llm.execute`` through a
stubbed ``_ManagedAttempt`` only, so it runs where the Relay runtime is not
installable — an always-on regression guard for environments in which the
existing Relay suite skips. Verified on macOS with ``nemo_relay`` absent
(``import nemo_relay`` -> ModuleNotFoundError) while ``from agent import
relay_llm`` imports cleanly.

Contributed by @gindinconsulting (Daniel Gindin), follow-up to
https://github.com/NousResearch/hermes-agent/pull/131377#issuecomment-6687064144
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent import relay_llm


class StubAttempt:
    """Stand-in for ``_ManagedAttempt``, recording what ``execute()`` did with it.

    ``runtime.relay.llm.execute`` is the executor Relay normally uses to drive
    the managed physical callback; calling it here mirrors real managed
    execution rather than returning a canned value.
    """

    def __init__(self) -> None:
        self.pending_request: dict | None = None
        self.managed_calls = 0
        self.managed_via_executor = False
        self.callback_requests: list[dict] = []
        self.result_saw: object = None
        self.result_defer: bool | None = None
        self.runtime = SimpleNamespace(relay=SimpleNamespace(llm=SimpleNamespace(execute=self._executor)))

    def _executor(self, managed_callback, *args, **kwargs):
        self.managed_via_executor = True
        return managed_callback(self.pending_request)

    async def run_managed(self, executor, managed_callback, *args, **kwargs):
        self.managed_calls += 1
        return {"managed": executor(managed_callback, *args, **kwargs)}

    def invoke(self, provider_callback, request):
        self.callback_requests.append(request)
        return provider_callback(request)

    def result(self, managed, defer_logical_completion=False):
        self.result_saw, self.result_defer = managed, defer_logical_completion
        return managed

    def resolve_failure(self, exc, defer_logical_completion=False):
        raise exc


@pytest.fixture()
def stubbed_resolve(monkeypatch):
    """Route ``_ManagedAttempt.resolve`` to a stub, keeping the request it was given."""

    def _install(attempt: StubAttempt) -> StubAttempt:
        def resolve(session_id, request, metadata, *, name, model_name):
            attempt.pending_request = request
            return attempt

        monkeypatch.setattr(relay_llm._ManagedAttempt, "resolve", staticmethod(resolve))
        return attempt

    return _install


def _provider_callback(request: dict) -> dict:
    return {"via": "provider-callback", "echo": request["payload"]}


def test_sync_execute_off_event_loop_keeps_managed_attempt(stubbed_resolve):
    """A sync caller with no running loop must still use managed Relay execution."""
    attempt = stubbed_resolve(StubAttempt())

    out = relay_llm.execute(
        {"payload": "request"},
        _provider_callback,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        defer_logical_completion=True,
    )

    assert attempt.managed_calls == 1, "off-loop sync callers must keep managed execution"
    assert attempt.managed_via_executor, "the managed callback must run through Relay's executor"
    assert attempt.callback_requests == [{"payload": "request"}], "managed callback must receive the request"
    assert attempt.result_saw == {"managed": {"via": "provider-callback", "echo": "request"}}
    assert attempt.result_defer is True, "defer_logical_completion must reach attempt.result()"
    assert out == {"managed": {"via": "provider-callback", "echo": "request"}}


def test_sync_execute_on_event_loop_thread_falls_back_to_callback(stubbed_resolve):
    """On an event-loop thread a sync caller cannot drive the coroutine, so it
    must take the unmanaged callback instead of raising — and must not build a
    coroutine it then drops (the leaked-coroutine warning in #148172)."""
    attempt = stubbed_resolve(StubAttempt())

    async def invoke_on_loop():
        return relay_llm.execute(
            {"payload": "request"},
            _provider_callback,
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
        )

    out = asyncio.run(invoke_on_loop())

    assert out == {"via": "provider-callback", "echo": "request"}
    assert attempt.managed_calls == 0, "managed execution must not start on an event-loop thread"
    assert attempt.callback_requests == [], "the managed callback must not have been wrapped"
