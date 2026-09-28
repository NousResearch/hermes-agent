"""Native turn completion must flush Full events queued while the sender idles."""

import json
import threading

import pytest

from gateway import run_turn_runner
from tests.gateway import test_run_progress_topics as progress


def park_progress_sender_idle(monkeypatch):
    """Park only the real sender's empty-queue wait until native completion cancels it."""
    idle = threading.Event()
    sender = {}
    native_sender = run_turn_runner.TurnRunner.send_progress_messages
    native_asyncio = run_turn_runner.asyncio

    async def observe_sender(self):
        sender["task"] = native_asyncio.current_task()
        return await native_sender(self)

    class AsyncioView:
        def __getattr__(self, name):
            return getattr(native_asyncio, name)

        async def sleep(self, delay, *args, **kwargs):
            if delay == 0.3 and native_asyncio.current_task() is sender.get("task"):
                idle.set()
                await native_asyncio.Event().wait()
            else:
                await native_asyncio.sleep(delay, *args, **kwargs)

    monkeypatch.setattr(run_turn_runner.TurnRunner, "send_progress_messages", observe_sender)
    monkeypatch.setattr(run_turn_runner, "asyncio", AsyncioView())
    return idle


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "verbose"])
async def test_native_completion_during_idle_flushes_only_full(monkeypatch, tmp_path, mode):
    idle = park_progress_sender_idle(monkeypatch)
    arguments = {"question": "final nonsecret payload " * 400}

    class Agent(progress.FakeAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            assert idle.wait(3), "sender did not enter the native empty-queue wait"
            assert self.tool_progress_callback is not None
            self.tool_progress_callback("tool.started", "final_tool", "preview", arguments)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter, result = await progress._run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="idle-completion",
        config_data={"display": {"tool_progress": mode}},
    )
    assert result["final_response"] == "done"
    if mode == "full":
        assert len(adapter.sent) > 1
        text = "".join(post["content"] for post in adapter.sent)
        header, _, body = text.partition("\n")
        assert header.endswith("final_tool")
        assert json.loads(body) == arguments
    else:
        assert adapter.sent == []  # Existing non-Full cancellation behavior is preserved.


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["current", "interrupted", "stale"])
async def test_idle_completion_short_entry_respects_owner(monkeypatch, tmp_path, state):
    idle = park_progress_sender_idle(monkeypatch)
    owner = {}
    native_make_runner = progress._make_runner

    def make_runner(adapter):
        runner = native_make_runner(adapter)
        native_run = runner._run_agent

        async def tracked_run(*args, **kwargs):
            key = kwargs["session_key"]
            generation = runner._begin_session_run_generation(key)
            owner.update(runner=runner, key=key, generation=generation)
            return await native_run(*args, **kwargs, run_generation=generation)

        runner._run_agent = tracked_run
        return runner

    monkeypatch.setattr(progress, "_make_runner", make_runner)
    arguments = {"question": "short final payload"}

    class Agent(progress.FakeAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            assert idle.wait(3), "sender did not enter the native empty-queue wait"
            runner, key, generation = owner["runner"], owner["key"], owner["generation"]
            assert runner._is_session_run_current(key, generation)
            assert self.tool_progress_callback is not None
            self.tool_progress_callback("tool.started", "final_tool", "preview", arguments)
            if state == "interrupted":
                self.is_interrupted = True
            elif state == "stale":
                runner._invalidate_session_run_generation(key)
                assert not runner._is_session_run_current(key, generation)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter, result = await progress._run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="idle-short-owner",
        config_data={"display": {"tool_progress": "full"}},
    )
    assert result["final_response"] == "done"
    assert adapter.edits == []
    if state == "current":
        assert len(adapter.sent) == 1
        header, _, body = adapter.sent[0]["content"].partition("\n")
        assert header.endswith("final_tool")
        assert json.loads(body) == arguments
    else:
        assert adapter.sent == []
