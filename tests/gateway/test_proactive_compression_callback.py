"""Gateway post-delivery proactive compression tests."""

import asyncio
from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class CallbackAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id="m1")


class FakeSessionStore:
    def __init__(self):
        self.saved = 0
        self.updated = []

    def _save(self):
        self.saved += 1

    def update_session(self, session_key, **kwargs):
        self.updated.append((session_key, kwargs))


class FakeAgent:
    compression_enabled = True
    _proactive_compression_enabled = True

    def __init__(self):
        self.session_id = "sess-old"
        self.status_callback = lambda *_args, **_kwargs: None
        self.compressed = []
        self.persisted = []

    def _compress_context(self, messages, system_message, *, approx_tokens=None, task_id="default", focus_topic=None):
        assert self.status_callback is None
        self.compressed.append(
            {
                "messages": messages,
                "system_message": system_message,
                "approx_tokens": approx_tokens,
                "task_id": task_id,
                "focus_topic": focus_topic,
            }
        )
        self.session_id = "sess-new"
        return [{"role": "assistant", "content": "summary"}], "new system"

    def _persist_session(self, messages, conversation_history=None):
        self.persisted.append((messages, conversation_history))


def _make_runner(agent):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._running_agents = {"session-key": agent}
    runner.session_store = FakeSessionStore()
    runner._is_session_run_current = lambda session_key, generation: True

    async def _run_in_executor_with_context(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    runner._run_in_executor_with_context = _run_in_executor_with_context
    return runner


def test_registers_post_delivery_compression_and_updates_session_entry():
    async def run():
        adapter = CallbackAdapter()
        agent = FakeAgent()
        runner = _make_runner(agent)
        session_entry = SimpleNamespace(session_id="sess-old")
        agent_result = {
            "session_id": "sess-old",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            "proactive_compression": {
                "enabled": True,
                "should_compress": True,
                "current_tokens": 560,
                "projected_tokens": 760,
                "threshold_tokens": 750,
            },
        }

        registered = runner._register_proactive_compression_callback(
            adapter=adapter,
            session_key="session-key",
            run_generation=3,
            session_entry=session_entry,
            agent_result=agent_result,
        )

        assert registered is True
        callback = adapter.pop_post_delivery_callback("session-key", generation=3)
        assert callable(callback)

        await callback()

        assert agent.compressed == [
            {
                "messages": agent_result["messages"],
                "system_message": None,
                "approx_tokens": 560,
                "task_id": "sess-old",
                "focus_topic": "next turn reserve",
            }
        ]
        assert agent.persisted == [([{"role": "assistant", "content": "summary"}], None)]
        assert session_entry.session_id == "sess-new"
        assert runner.session_store.saved == 1
        assert runner.session_store.updated == [("session-key", {"last_prompt_tokens": 560})]

    asyncio.run(run())


def test_does_not_register_when_pressure_says_no_compression():
    adapter = CallbackAdapter()
    runner = _make_runner(FakeAgent())
    session_entry = SimpleNamespace(session_id="sess-old")

    registered = runner._register_proactive_compression_callback(
        adapter=adapter,
        session_key="session-key",
        run_generation=1,
        session_entry=session_entry,
        agent_result={
            "messages": [{"role": "user", "content": "hello"}],
            "proactive_compression": {"enabled": True, "should_compress": False},
        },
    )

    assert registered is False
    assert adapter.pop_post_delivery_callback("session-key", generation=1) is None
