"""Tests for DeepAgents middleware — lifecycle, error handling, memory sync."""

from threading import Thread
from unittest.mock import MagicMock

import pytest

from agent.deep_agents_middleware import _HermesInterruptSignal, _HermesMiddleware


@pytest.fixture
def agent_stub():
    agent = MagicMock()
    agent.skip_memory = False
    return agent


# ---------------------------------------------------------------------------
# _HermesInterruptSignal
# ---------------------------------------------------------------------------


class TestHermesInterruptSignal:
    """Tests for the thread-safe interrupt signal."""

    def test_is_set_returns_false_by_default(self):
        signal = _HermesInterruptSignal()
        assert signal.is_set() is False

    def test_set_makes_is_return_true(self):
        signal = _HermesInterruptSignal()
        signal.set()
        assert signal.is_set() is True

    def test_clear_makes_is_set_return_false(self):
        signal = _HermesInterruptSignal()
        signal.set()
        assert signal.is_set() is True
        signal.clear()
        assert signal.is_set() is False

    def test_clear_on_already_cleared_no_crash(self):
        signal = _HermesInterruptSignal()
        signal.clear()

    def test_set_multiple_times_no_error(self):
        signal = _HermesInterruptSignal()
        signal.set()
        signal.set()
        signal.set()
        assert signal.is_set() is True

    def test_clear_after_set_then_set_again(self):
        signal = _HermesInterruptSignal()
        signal.set()
        signal.clear()
        signal.set()
        assert signal.is_set() is True

    def test_thread_safety_basic_write_then_read(self):
        """Verify basic thread-safety: one thread sets, another reads."""
        signal = _HermesInterruptSignal()

        def writer():
            import time

            time.sleep(0.05)
            signal.set()

        t = Thread(target=writer)
        t.start()
        t.join()

        assert signal.is_set() is True

    def test_concurrent_set_and_clear(self):
        """Stress test: many threads interleave set/clear."""
        signal = _HermesInterruptSignal()
        errors = []

        def toggle(n):
            try:
                for _ in range(100):
                    if n % 2 == 0:
                        signal.set()
                    else:
                        signal.clear()
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=toggle, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert signal.is_set() in (True, False)


# ---------------------------------------------------------------------------
# _HermesMiddleware — interrupt API
# ---------------------------------------------------------------------------


class TestMiddlewareInterruptAPI:
    def test_constructs_with_interrupt_signal(self, agent_stub):
        mw = _HermesMiddleware(agent_stub)
        assert mw.interrupt_signal is not None
        assert isinstance(mw.interrupt_signal, _HermesInterruptSignal)
        assert mw.interrupt_signal.is_set() is False

    def test_request_interrupt_sets_signal(self, agent_stub):
        mw = _HermesMiddleware(agent_stub)
        mw.request_interrupt()
        assert mw.interrupt_signal.is_set() is True

    def test_clear_interrupt_clears_signal(self, agent_stub):
        mw = _HermesMiddleware(agent_stub)
        mw.request_interrupt()
        mw.clear_interrupt()
        assert mw.interrupt_signal.is_set() is False

    def test_request_interrupt_propagates_is_set(self, agent_stub):
        mw = _HermesMiddleware(agent_stub)
        mw.request_interrupt()
        assert mw.interrupt_signal.is_set() is True

    def test_interrupt_signal_is_same_object(self, agent_stub):
        mw = _HermesMiddleware(agent_stub)
        assert mw.interrupt_signal is mw.interrupt_signal


# ---------------------------------------------------------------------------
# _HermesMiddleware — before_agent
# ---------------------------------------------------------------------------


class TestMiddlewareBeforeAgent:
    @pytest.mark.asyncio
    async def test_skip_memory_true_skips_memory_call(self, agent_stub):
        """When skip_memory=True, before_agent does NOT call get_memory_context."""
        agent_stub.skip_memory = True

        mw = _HermesMiddleware(agent_stub)
        state = {}

        result = mw.before_agent(state)

        agent_stub.get_memory_context.assert_not_called()
        assert "memory" not in result

    @pytest.mark.asyncio
    async def test_skip_memory_false_calls_get_memory_context(self, agent_stub):
        """When skip_memory=False, before_agent calls agent.get_memory_context()."""
        agent_stub.skip_memory = False
        agent_stub.get_memory_context.return_value = "cached recall"

        mw = _HermesMiddleware(agent_stub)
        state = {}

        result = mw.before_agent(state)

        assert result is state
        agent_stub.get_memory_context.assert_called_once()
        assert state["memory"] == {"context": "cached recall"}

    @pytest.mark.asyncio
    async def test_before_agent_none_memory_uses_default(self, agent_stub):
        """When get_memory_context() returns None, no memory key is added to state."""
        agent_stub.skip_memory = False
        agent_stub.get_memory_context.return_value = None

        mw = _HermesMiddleware(agent_stub)
        state = {}

        result = mw.before_agent(state)
        assert result is state
        assert "memory" not in state

    @pytest.mark.asyncio
    async def test_before_agent_nonempty_memory_sets_context(self, agent_stub):
        """When get_memory_context() returns a string, memory.context is set."""
        agent_stub.skip_memory = False
        agent_stub.get_memory_context.return_value = "relevant memory fragments"

        mw = _HermesMiddleware(agent_stub)
        state = {}

        result = mw.before_agent(state)
        assert result["memory"]["context"] == "relevant memory fragments"

    @pytest.mark.asyncio
    async def test_before_agent_preserves_existing_memory(self, agent_stub):
        """Existing state['memory'] keys are preserved, only 'context' added."""
        agent_stub.skip_memory = False
        agent_stub.get_memory_context.return_value = "new context"

        mw = _HermesMiddleware(agent_stub)
        state = {"memory": {"existing_key": "value"}}

        result = mw.before_agent(state)
        assert result["memory"]["existing_key"] == "value"
        assert result["memory"]["context"] == "new context"

    @pytest.mark.asyncio
    async def test_before_agent_preserves_other_state_keys(self, agent_stub):
        """Other state keys are untouched when memory is added."""
        agent_stub.skip_memory = False

        mw = _HermesMiddleware(agent_stub)
        state = {
            "messages": [],
            "__pregel_tasks": [],
            "extra": 42,
        }

        result = mw.before_agent(state)
        assert result["messages"] == []
        assert result["__pregel_tasks"] == []
        assert result["extra"] == 42

    @pytest.mark.asyncio
    async def test_before_agent_handles_exception_gracefully(self, agent_stub):
        """If get_memory_context raises, the error is logged but before_agent does not crash."""
        agent_stub.skip_memory = False
        agent_stub.get_memory_context.side_effect = RuntimeError("mem0 down")

        mw = _HermesMiddleware(agent_stub)
        state = {}

        result = mw.before_agent(state)

        assert result is state
        assert "memory" not in state

    @pytest.mark.asyncio
    async def test_before_agent_returns_state_same_object(self, agent_stub):
        """before_agent returns the same state dict reference."""
        agent_stub.skip_memory = True

        mw = _HermesMiddleware(agent_stub)
        state = {"foo": "bar"}

        result = mw.before_agent(state)
        assert result is state


# ---------------------------------------------------------------------------
# _HermesMiddleware — after_agent
# ---------------------------------------------------------------------------


class TestMiddlewareAfterAgent:
    @pytest.mark.asyncio
    async def test_skip_memory_true_skips_save(self, agent_stub):
        """When skip_memory=True, after_agent does NOT call save_memory."""
        agent_stub.skip_memory = True

        mw = _HermesMiddleware(agent_stub)
        state = {"_messages": []}

        result = mw.after_agent(state)

        agent_stub.save_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_memory_false_calls_save_memory(self, agent_stub):
        """When skip_memory=False and there are messages, save_memory is called."""
        agent_stub.skip_memory = False

        mw = _HermesMiddleware(agent_stub)
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        state = {"_messages": msgs}

        result = mw.after_agent(state)

        agent_stub.save_memory.assert_called_once_with(msgs)

    @pytest.mark.asyncio
    async def test_after_agent_empty_messages_skips_save(self, agent_stub):
        """When _messages is empty, save_memory is NOT called even if skip_memory=False."""
        agent_stub.skip_memory = False

        mw = _HermesMiddleware(agent_stub)
        state = {"_messages": []}

        result = mw.after_agent(state)

        agent_stub.save_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_after_agent_missing_messages_key_skips_save(self, agent_stub):
        """When _messages key is absent, save_memory is NOT called."""
        agent_stub.skip_memory = False

        mw = _HermesMiddleware(agent_stub)
        state = {}

        result = mw.after_agent(state)

        agent_stub.save_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_after_agent_clears_interrupt(self, agent_stub):
        """after_agent always clears the interrupt signal."""
        agent_stub.skip_memory = True

        mw = _HermesMiddleware(agent_stub)
        mw.request_interrupt()

        mw.after_agent({"_messages": []})

        assert mw.interrupt_signal.is_set() is False

    @pytest.mark.asyncio
    async def test_after_agent_clears_interrupt_even_with_memory(self, agent_stub):
        """Interrupt is cleared regardless of whether memory path is taken."""
        agent_stub.skip_memory = False

        mw = _HermesMiddleware(agent_stub)
        mw.request_interrupt()

        msgs = [{"role": "user", "content": "test"}]
        mw.after_agent({"_messages": msgs})

        assert mw.interrupt_signal.is_set() is False

    @pytest.mark.asyncio
    async def test_after_agent_handles_save_exception_gracefully(self, agent_stub):
        """If save_memory raises, after_agent does not crash and still clears interrupt."""
        agent_stub.skip_memory = False
        agent_stub.save_memory.side_effect = RuntimeError("sqlite error")

        mw = _HermesMiddleware(agent_stub)
        mw.request_interrupt()

        msgs = [{"role": "user", "content": "test"}]
        result = mw.after_agent({"_messages": msgs})

        assert "_messages" in result
        assert mw.interrupt_signal.is_set() is False

    @pytest.mark.asyncio
    async def test_after_agent_returns_state_unchanged(self, agent_stub):
        """after_agent returns the state dict with no structural changes."""
        agent_stub.skip_memory = True

        mw = _HermesMiddleware(agent_stub)
        state = {"_messages": [], "__pregel_unpacked": True}

        result = mw.after_agent(state)

        assert result is state
        assert result["__pregel_unpacked"] is True


# ---------------------------------------------------------------------------
# _HermesMiddleware — end-to-end lifecycle
# ---------------------------------------------------------------------------


class TestMiddlewareLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle_calls_memory_roundtrip(self, agent_stub):
        """before_agent calls get_memory_context, after_agent calls save_memory."""
        agent_stub.skip_memory = False
        agent_stub.get_memory_context.return_value = "recall data"

        mw = _HermesMiddleware(agent_stub)
        msgs = [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        state = {"_messages": msgs}
        mw.before_agent(state)
        mw.after_agent(state)

        agent_stub.get_memory_context.assert_called_once()
        agent_stub.save_memory.assert_called_once_with(msgs)

    @pytest.mark.asyncio
    async def test_full_lifecycle_context_passed_through(self, agent_stub):
        """before_agent puts memory context into state['memory']['context']."""
        agent_stub.get_memory_context.return_value = "context A"

        mw = _HermesMiddleware(agent_stub)
        state = {"_messages": [{"role": "user", "content": "hi"}]}

        mw.before_agent(state)

        assert state["memory"]["context"] == "context A"

    @pytest.mark.asyncio
    async def test_lifecycle_interrupt_set_between_hooks(self, agent_stub):
        """Interrupt set before before_agent persists until after_agent clears it."""
        agent_stub.skip_memory = True

        mw = _HermesMiddleware(agent_stub)
        mw.request_interrupt()

        before_state = mw.before_agent({"_messages": []})

        assert mw.interrupt_signal.is_set() is True

        state = mw.after_agent(before_state)
        assert mw.interrupt_signal.is_set() is False

    @pytest.mark.asyncio
    async def test_lifecycle_interrupt_cleared_on_first_turn(self, agent_stub):
        """Even with no interrupt set, after_agent runs without error."""
        agent_stub.skip_memory = True

        mw = _HermesMiddleware(agent_stub)

        before_state = mw.before_agent({"_messages": []})
        after_state = mw.after_agent(before_state)
        assert after_state is before_state

    @pytest.mark.asyncio
    async def test_lifecycle_skip_memory_both_hooks(self, agent_stub):
        """When skip_memory=True, neither get_memory_context nor save_memory is called."""
        agent_stub.skip_memory = True

        mw = _HermesMiddleware(agent_stub)
        state = {"_messages": [{"role": "user", "content": "hi"}]}

        mw.before_agent(state)
        mw.after_agent(state)

        agent_stub.get_memory_context.assert_not_called()
        agent_stub.save_memory.assert_not_called()
