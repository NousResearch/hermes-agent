"""ContextSessionStore: one agent per context, history continuity, cancellation."""

from __future__ import annotations

import threading

from plugins.platforms.a2a.sessions import ContextSessionStore


class _InterruptibleAgent:
    """Fake that mimics AIAgent's interrupt semantics.

    ``interrupt()`` sets a sticky ``_interrupt_requested`` flag; a turn that
    sees the flag set at its start returns interrupted (as AIAgent's loop does);
    ``clear_interrupt()`` resets it.
    """

    def __init__(self) -> None:
        self._interrupt_requested = False
        self.stream_delta_callback = None
        self.reasoning_callback = None
        self.tool_progress_callback = None
        self.step_callback = None
        self.thinking_callback = None
        self.runs: list[str] = []

    def run_conversation(
        self, *, user_message, conversation_history=None, task_id=None, **kw
    ):
        if self._interrupt_requested:
            return {"final_response": None, "interrupted": True}
        self.runs.append(user_message)
        msgs = list(conversation_history or [])
        msgs += [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"echo: {user_message}"},
        ]
        return {"final_response": f"echo: {user_message}", "messages": msgs}

    def interrupt(self, message=None):
        self._interrupt_requested = True

    def clear_interrupt(self):
        self._interrupt_requested = False


def test_same_context_reuses_one_agent(fakes):
    created = []

    def factory():
        agent = fakes.FakeAgent()
        created.append(agent)
        return agent

    store = ContextSessionStore(agent_factory=factory)
    first = store.get_or_create("ctx-1")
    again = store.get_or_create("ctx-1")
    other = store.get_or_create("ctx-2")

    assert first is again
    assert other is not first
    assert len(created) == 2


def test_run_turn_appends_to_history(fakes):
    store = ContextSessionStore(agent_factory=fakes.FakeAgent)
    session = store.get_or_create("ctx-1")

    first = session.run_turn("hello", task_id="t1")
    assert first["final_response"] == "echo: hello"
    assert session.history[-1] == {"role": "assistant", "content": "echo: hello"}

    # Second turn sees the prior history.
    session.run_turn("again", task_id="t2")
    assert session.agent.runs == ["hello", "again"]
    assert len(session.history) == 4


def test_cancel_sets_event_and_interrupts(fakes):
    agent = fakes.FakeAgent()
    store = ContextSessionStore(agent_factory=lambda: agent)
    session = store.get_or_create("ctx-1")

    session.cancel()

    assert session.cancel_event.is_set()
    assert agent.interrupted is True


def test_lru_eviction_caps_session_count(fakes):
    store = ContextSessionStore(agent_factory=fakes.FakeAgent, max_sessions=2)
    a = store.get_or_create("a")
    store.get_or_create("b")
    # Touch "a" so "b" becomes the least-recently-used entry.
    store.get_or_create("a")
    c = store.get_or_create("c")  # over cap -> evicts LRU ("b")

    assert store.get("b") is None
    assert store.get("a") is a
    assert store.get("c") is c


def test_cancel_after_turn_does_not_poison_next_turn():
    """A cancel that arrives with no turn running must not abort the next turn.

    AIAgent.interrupt() sets a sticky flag that is only cleared by a turn that
    runs to completion. If cancel() fires while idle (e.g. just after a turn
    finished, or on a reused context), the next turn must start from a clean
    slate instead of inheriting the stale interrupt and aborting immediately.
    """
    agent = _InterruptibleAgent()
    session = ContextSessionStore(agent_factory=lambda: agent).get_or_create("ctx")

    session.cancel()  # no turn running; sets the sticky interrupt flag
    assert agent._interrupt_requested is True

    result = session.run_turn("hello", task_id="t-new")
    assert result.get("interrupted") is not True
    assert result["final_response"] == "echo: hello"


def test_cancel_of_non_running_task_does_not_interrupt_agent():
    """Task-scoped cancel: cancelling a task that is not the running one must
    not interrupt the agent (which would kill an unrelated in-flight turn on the
    same context). The targeted task is instead skipped if it later starts."""
    agent = _InterruptibleAgent()
    session = ContextSessionStore(agent_factory=lambda: agent).get_or_create("ctx")

    session.cancel(task_id="queued-task")  # nothing running on this context
    assert agent._interrupt_requested is False

    result = session.run_turn("hi", task_id="queued-task")
    assert result.get("interrupted") is True
    assert agent.runs == []  # the cancelled task never actually ran


def test_run_turn_applies_passed_callbacks_under_the_turn():
    """Per-turn callbacks must be the ones active during this turn's run, and be
    cleared afterwards — so concurrent turns can't cross-wire the shared agent."""
    agent = _InterruptibleAgent()
    seen = []
    agent.run_conversation = (  # type: ignore[method-assign]
        lambda **kw: (
            agent.stream_delta_callback("d"),
            {"final_response": "ok", "messages": []},
        )[1]
    )

    def cb(text):
        seen.append(text)

    session = ContextSessionStore(agent_factory=lambda: agent).get_or_create("ctx")
    session.run_turn("go", task_id="t1", callbacks={"stream_delta_callback": cb})

    assert seen == ["d"]  # the passed callback fired during the turn
    assert agent.stream_delta_callback is None  # cleared after the turn


def test_lru_does_not_evict_an_in_flight_session(fakes):
    """An in-flight turn's session must survive eviction; an idle LRU session is
    dropped instead. Evicting a busy session silently forks its history into a
    fresh empty agent and orphans the running worker thread."""
    release = threading.Event()
    started = threading.Event()

    class BlockingAgent(_InterruptibleAgent):
        def run_conversation(
            self, *, user_message, conversation_history=None, task_id=None, **kw
        ):
            started.set()
            release.wait(5)
            return {"final_response": "done", "messages": []}

    blocking = BlockingAgent()
    agents = iter([blocking, fakes.FakeAgent(), fakes.FakeAgent()])
    store = ContextSessionStore(agent_factory=lambda: next(agents), max_sessions=2)

    busy = store.get_or_create("busy")
    store.get_or_create("idle")  # idle, least-recently-used after "busy" runs

    worker = threading.Thread(target=lambda: busy.run_turn("x", task_id="t-busy"))
    worker.start()
    assert started.wait(5)  # turn is now in flight -> "busy" is active

    try:
        third = store.get_or_create("third")  # over cap -> must evict an idle one
        assert store.get("busy") is busy  # in-flight session preserved
        assert store.get("idle") is None  # idle LRU evicted instead
        assert store.get("third") is third
    finally:
        release.set()
        worker.join(5)


def test_run_turn_scopes_resources_to_context_and_denies_remote_approval():
    from tools import terminal_tool

    seen = {}

    class ApprovalAgent(_InterruptibleAgent):
        def run_conversation(self, *, task_id=None, **kwargs):
            callback = terminal_tool._get_approval_callback()
            seen["task_id"] = task_id
            seen["decision"] = callback("rm -rf /tmp/x", "destructive")
            return {"final_response": "denied", "messages": []}

    sentinel = lambda *_args, **_kwargs: "once"
    terminal_tool.set_approval_callback(sentinel)
    try:
        session = ContextSessionStore(agent_factory=ApprovalAgent).get_or_create(
            "ctx-stable"
        )
        session.run_turn("go", task_id="a2a-task-ephemeral")
        assert terminal_tool._get_approval_callback() is sentinel
    finally:
        terminal_tool.set_approval_callback(None)

    assert seen == {"task_id": "ctx-stable", "decision": "deny"}


def test_real_agent_factory_honors_platform_and_global_tool_config(monkeypatch):
    import hermes_cli.config as config_module
    import hermes_cli.runtime_provider as runtime_module
    import hermes_cli.tools_config as tools_config_module
    import run_agent

    captured = {}
    config = {
        "model": {"default": "model-x", "provider": "provider-x"},
        "platform_toolsets": {"a2a": ["file", "no_mcp"]},
        "agent": {"disabled_toolsets": ["terminal"]},
    }

    class CapturingAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def resolve_tools(user_config, platform):
        assert user_config is config
        assert platform == "a2a"
        return {"file"}

    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(tools_config_module, "_get_platform_tools", resolve_tools)
    monkeypatch.setattr(
        runtime_module,
        "resolve_runtime_provider",
        lambda **_kwargs: {"provider": "provider-x"},
    )
    monkeypatch.setattr(run_agent, "AIAgent", CapturingAgent)

    ContextSessionStore()._make_agent("ctx-configured")

    assert captured["enabled_toolsets"] == ["file"]
    assert captured["disabled_toolsets"] == ["terminal"]
    assert captured["platform"] == "a2a"
    assert captured["session_id"] == "ctx-configured"


def test_session_lease_blocks_eviction_and_eviction_closes_agent():
    class ClosableAgent(_InterruptibleAgent):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    agents = []

    def factory():
        agent = ClosableAgent()
        agents.append(agent)
        return agent

    store = ContextSessionStore(agent_factory=factory, max_sessions=1)
    leased = store.acquire("leased")
    transient = store.get_or_create("transient")

    assert store.get("leased") is leased
    assert store.get("transient") is None
    assert transient.agent.closed is True

    store.release(leased)
    replacement = store.get_or_create("replacement")
    assert store.get("leased") is None
    assert leased.agent.closed is True
    assert store.get("replacement") is replacement


def test_store_close_releases_all_retained_agents():
    class ClosableAgent(_InterruptibleAgent):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    agents = [ClosableAgent(), ClosableAgent()]
    iterator = iter(agents)
    store = ContextSessionStore(agent_factory=lambda: next(iterator))
    store.get_or_create("one")
    store.get_or_create("two")

    store.close()

    assert all(agent.closed for agent in agents)


def test_releasing_a_lease_repairs_temporary_session_cap_overshoot():
    store = ContextSessionStore(agent_factory=_InterruptibleAgent, max_sessions=1)
    first = store.acquire("first")
    second = store.acquire("second")
    assert store.size() == 2

    store.release(first)
    assert store.size() == 1
    assert store.get("first") is None
    assert store.get("second") is second

    store.release(second)


def test_begin_close_prevents_queued_turn_from_starting():
    started = threading.Event()
    release = threading.Event()

    class BlockingAgent(_InterruptibleAgent):
        def run_conversation(self, **_kwargs):
            self.runs.append("run")
            started.set()
            release.wait(5)
            return {"final_response": None, "interrupted": True, "messages": []}

        def interrupt(self, message=None):
            super().interrupt(message)
            release.set()

    session = ContextSessionStore(agent_factory=BlockingAgent).get_or_create(
        "ctx-shutdown"
    )
    results = []
    first = threading.Thread(
        target=lambda: results.append(session.run_turn("one", task_id="one"))
    )
    second = threading.Thread(
        target=lambda: results.append(session.run_turn("two", task_id="two"))
    )
    first.start()
    assert started.wait(5)
    second.start()

    session.begin_close()
    first.join(5)
    second.join(5)

    assert session.agent.runs == ["run"]
    assert len(results) == 2
    assert all(result.get("interrupted") for result in results)
