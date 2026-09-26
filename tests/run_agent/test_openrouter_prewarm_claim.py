"""#24651: the OpenRouter metadata pre-warm must be claimed atomically.

``_finalize_routing`` read ``_openrouter_prewarm_done.is_set()`` and then called ``.set()`` as
two separate steps.  Concurrent agent inits (gateway sessions, the batch runner, in-process
subagents) could all observe the Event clear and each start an ``openrouter-prewarm`` thread —
the very thread leak the process-level Event was added to prevent.

These run the real ``_finalize_routing`` from several threads; only the metadata fetch itself is
stubbed, so the claim is exercised on the production code path.
"""

import threading
import time
from types import SimpleNamespace

import agent.agent_init as agent_init
import run_agent


_WORKERS = 8
_CLAIM_WINDOW = 0.1


class _SlowClaimEvent(threading.Event):
    """An Event whose ``set()`` takes ``_CLAIM_WINDOW`` seconds to land.

    Every read stays truthful — ``is_set()`` is untouched — so this only widens the window
    between "the claimant checked" and "the claim is visible", the gap the guard has to cover.
    Widening it cannot *create* a race: with an atomic claim the delay is serialized behind the
    lock and every later caller reads an Event that is already set.
    """

    def set(self) -> None:
        time.sleep(_CLAIM_WINDOW)
        super().set()


def _openrouter_agent():
    """The minimum an agent needs to reach the pre-warm block of ``_finalize_routing``."""
    return SimpleNamespace(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4",
        api_mode="chat_completions",
        _transport_cache={},
        _get_transport=lambda: None,
        _is_openrouter_url=lambda: True,
        _is_azure_openai_url=lambda: False,
        _is_direct_openai_url=lambda: False,
        _provider_model_requires_responses_api=lambda *a, **k: False,
    )


def _race_finalize_routing(monkeypatch, event):
    """Drive ``_WORKERS`` concurrent ``_finalize_routing`` calls; return (prewarm calls, errors)."""
    started, errors = [], []
    record_lock = threading.Lock()

    def _record_prewarm():
        with record_lock:
            started.append(threading.current_thread().name)

    monkeypatch.setattr(agent_init, "fetch_model_metadata", _record_prewarm)
    monkeypatch.setattr(run_agent, "_openrouter_prewarm_done", event)

    barrier = threading.Barrier(_WORKERS)

    def _init_one():
        try:
            barrier.wait(timeout=30)
            agent_init._finalize_routing(_openrouter_agent(), "chat_completions", None)
        except Exception as exc:  # surfaced by the caller's assertion
            with record_lock:
                errors.append(exc)

    workers = [threading.Thread(target=_init_one, name=f"init-{i}") for i in range(_WORKERS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    for thread in threading.enumerate():  # let every prewarm thread that DID start be counted
        if thread.name == "openrouter-prewarm":
            thread.join(timeout=30)
    return started, errors


def test_concurrent_init_starts_exactly_one_openrouter_prewarm(monkeypatch):
    """Racing inits claim the pre-warm once; the losers must not spawn their own thread."""
    event = _SlowClaimEvent()

    started, errors = _race_finalize_routing(monkeypatch, event)

    assert errors == []
    assert len(started) == 1, f"{len(started)} prewarm threads started, expected exactly 1"
    assert event.is_set() is True


def test_prewarm_is_skipped_once_the_event_is_already_set(monkeypatch):
    """The Event stays the readable "already warmed" flag: a set Event means no spawn at all."""
    event = _SlowClaimEvent()
    event.set()

    started, errors = _race_finalize_routing(monkeypatch, event)

    assert errors == []
    assert started == []
