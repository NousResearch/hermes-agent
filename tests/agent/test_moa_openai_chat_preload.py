"""MoA must import ``openai.resources.chat`` before the reference thread pool.

Each worker's first ``client.chat`` access lazily imports that package. On
Python 3.14, concurrent first-imports abort the loser with
``_frozen_importlib._DeadlockError: deadlock detected by
_ModuleLock('openai.resources.chat')`` — one reference then fails with
``[failed: deadlock detected by _ModuleLock('openai.resources.chat')]``
while the others continue (observed on a live /moa turn).
"""

from __future__ import annotations

from types import SimpleNamespace

from agent import moa_loop


def _response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def test_run_references_parallel_preloads_openai_chat_prior_to_workers(monkeypatch):
    """The calling thread must import the SDK chat resources before workers run."""
    order: list[str] = []

    def fake_preload() -> bool:
        order.append("preload")
        return True

    def fake_call_llm(**kwargs):
        order.append("worker")
        return _response(f"resp-{kwargs['provider']}")

    monkeypatch.setattr(moa_loop, "get_transport", lambda *_a, **_k: None)
    monkeypatch.setattr(moa_loop, "_preload_openai_chat_resources", fake_preload)
    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)

    out = moa_loop._run_references_parallel(
        [
            {"provider": "p1", "model": "ok"},
            {"provider": "p2", "model": "ok"},
        ],
        [{"role": "user", "content": "hi"}],
    )

    assert [label for label, _, _ in out] == ["p1:ok", "p2:ok"]
    assert order, "preload helper was never called"
    assert order[0] == "preload"
    assert order.count("worker") == 2


def test_preload_openai_chat_resources_is_best_effort(monkeypatch):
    """A missing OpenAI SDK must not raise into the fan-out."""
    import builtins

    real_import = builtins.__import__

    def boom(name, *args, **kwargs):
        if name.startswith("openai.resources"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert moa_loop._preload_openai_chat_resources() is False
