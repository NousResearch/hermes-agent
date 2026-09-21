"""Real adapter lifecycle, threaded model hooks, and two isolated profile stores."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from plugins.groupchat.policy import GroupchatAddon
from plugins.groupchat import coordination_runtime as work
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("separate_tasks", [False, True])
async def test_real_gateway_lifecycle_hands_back_once_without_redirect(tmp_path, monkeypatch, separate_tasks):
    peers, resumes, notices = [], [], []
    finished = []
    original = ("Lena, list cafés in Klagenfurt. Felix, list pizzerias in Villach."
                if separate_tasks else "edit the video")
    completed = asyncio.Event()
    begin_barrier, claim_barrier, yield_barrier = (asyncio.Barrier(2) for _ in range(3))
    monkeypatch.setattr("hermes_cli.plugins.invoke_middleware",
                        lambda name, adapter: [GroupchatAddon(adapter)])

    for name in ("a", "b"):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({"groupchat": {
            "enabled": True, "platforms": ["matrix"],
            "relevance": {"enabled": False}, "pingpong_guard": {"enabled": False},
            "coordination": {"enabled": True, "fallback_delay_seconds": 10}}}))
        monkeypatch.setattr("plugins.groupchat.policy.get_hermes_home", lambda home=home: home)
        adapter = MatrixAdapter(PlatformConfig(extra={"user_id": name}))
        adapter._client = SimpleNamespace(mxid=name)
        adapter.on_processing_start = AsyncMock()
        adapter.on_processing_complete = AsyncMock()
        adapter._start_typing_refresh = lambda *args: None
        adapter.conversation_middleware()
        adapter._session_store = SimpleNamespace(peek_session_id=lambda key, name=name: "session-" + name)
        peers.append(adapter)

    async def publish(sender, chat_id, payload):
        for peer in peers:
            if peer is not sender:
                await peer.conversation_middleware().signal(chat_id, sender.conversation_user_id(), payload)
        return True

    async def run(adapter, event):
        name = adapter.conversation_user_id()
        resumed = bool(event.metadata.get("groupchat_work_resume"))
        ids = {"session_id": "session-" + name, "turn_id": "resume" if resumed else "initial"}
        context = await asyncio.to_thread(work.pre_llm_call, **ids)
        assert "groupchat_work" in context["context"]
        assert "independent tasks" in context["context"]
        if resumed:
            resumes.append(name)
            assert "Original request:\nedit the video" in event.text
            result = json.loads(await asyncio.to_thread(work.work_tool, {"action": "yield"}))
            assert "error" in result
            completed.set()
            return None
        await begin_barrier.wait()
        result = json.loads(await asyncio.to_thread(work.work_tool, {"action": "work"}))
        assert result["success"]
        await claim_barrier.wait()
        # Drain the native signals queued by the tool handler before the next tool result.
        runtime = adapter.conversation_middleware().handlers[0].coordination
        key = next(iter(runtime.bindings))
        async def claims_arrived():
            while len([p for p in runtime.rounds[key].participants.values() if p["state"] == "working"]) < 2:
                await asyncio.sleep(.01)
        await asyncio.wait_for(claims_arrived(), 3)
        result = await asyncio.to_thread(work.transform_tool_result, '{"success":true}', **ids)
        notices.append(json.loads(result)["groupchat_coordination"])
        assert "continue that task in parallel" in notices[-1]
        assert await asyncio.to_thread(work.transform_tool_result, '{}', **ids) is None
        if separate_tasks:
            assert await asyncio.to_thread(work.pre_tool_call, "terminal", **ids) is None
            assert await asyncio.to_thread(work.transform_llm_output, **ids) is None
            await yield_barrier.wait()
            finished.append(name)
            if len(finished) == 2:
                completed.set()
            return None
        result = json.loads(await asyncio.to_thread(work.work_tool, {"action": "yield", "note": "keep the audio"}))
        assert result["success"]
        assert (await asyncio.to_thread(work.pre_tool_call, "terminal", **ids))["action"] == "block"
        assert await asyncio.to_thread(work.transform_llm_output, **ids) == "[SILENT]"
        await yield_barrier.wait()
        return None

    for adapter in peers:
        adapter.conversation_send_signal = lambda chat_id, payload, adapter=adapter: publish(adapter, chat_id, payload)
        adapter._message_handler = lambda event, adapter=adapter: run(adapter, event)
    try:
        for adapter in peers:
            event = MessageEvent(text=original, message_id="$request", source=SessionSource(
                platform=adapter.platform, chat_id="!room", chat_type="group", user_id="human"))
            await adapter.handle_message(event)
        await asyncio.wait_for(completed.wait(), 8)
        await asyncio.sleep(.1)
        assert len(resumes) == (0 if separate_tasks else 1) and len(notices) == 2
        assert (tmp_path / "a/groupchat/coordination.sqlite3").exists()
        assert (tmp_path / "b/groupchat/coordination.sqlite3").exists()
        rounds = peers[0].conversation_middleware().handlers[0].coordination.rounds
        round = next(iter(rounds.values()))
        first = min(round.participants, key=lambda actor: (round.participants[actor]["started"], actor))
        if separate_tasks:
            assert sorted(finished) == ["a", "b"]
            assert all(p["state"] == "completed" for p in round.participants.values())
        else:
            assert resumes == [first]
        assert work._binding.get() is None
    finally:
        for adapter in peers:
            await adapter.conversation_middleware().close()
            tasks = list(adapter._background_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


def test_native_registration_resolves_tool_and_hooks_without_cross_profile_state(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from plugins.groupchat import register
    from tools.registry import registry
    from toolsets import resolve_toolset

    manager = PluginManager()
    register(PluginContext(PluginManifest(name="groupchat"), manager))
    assert "groupchat_work" in resolve_toolset("groupchat")
    entry = registry.get_entry("groupchat_work")
    assert entry is not None
    for name, enabled in (("a", True), ("b", False), ("a", True)):
        home = tmp_path / name
        home.mkdir(exist_ok=True)
        (home / "config.yaml").write_text(json.dumps({"groupchat": {
            "enabled": True, "coordination": {"enabled": enabled}}}))
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert work.enabled() is enabled
        assert "error" in json.loads(entry.handler({"action": "yield"}))
        assert manager.invoke_hook("pre_llm_call", session_id=name, turn_id="t") == []


@pytest.mark.anyio
async def test_stopped_handoff_recovers_once_but_never_after_session_reset(tmp_path):
    import time
    from plugins.groupchat.coordination import WorkRound, WorkStore

    for reset in (False, True):
        home = tmp_path / str(reset)
        adapter = MatrixAdapter(PlatformConfig(extra={"user_id": "a"}))
        adapter._client = SimpleNamespace(mxid="a")
        adapter._session_store = SimpleNamespace(peek_session_id=lambda key: "new" if reset else "session-a")
        adapter.conversation_send_signal = AsyncMock(return_value=True)
        accepted = []
        async def admit(event):
            accepted.append(event)
            event._gateway_accepted = True
        adapter.handle_message = admit
        key = json.dumps(["matrix", "!room", "$request"])
        round = WorkRound("$request", "a")
        for index, name in enumerate(("a", "b")):
            round.observe(name, dict(message_id="$request", state="yielded", run=name,
                seq=1, started=time.time() - 300 + index, stopped=True, note=""), time.time() - 200)
        store = WorkStore(home / "groupchat/coordination.sqlite3")
        store.save(key, round)
        source = SessionSource(platform=adapter.platform, chat_id="!room", chat_type="group", user_id="human")
        store.save_origin(key, dict(message_id="$request", text="edit the video", source=source.to_dict(),
                                   session_key="session-key", session_id="session-a"))
        runtime = work.CoordinationRuntime(adapter, home, {"fallback_delay_seconds": 10})
        async def drained():
            while runtime.tasks:
                await asyncio.sleep(.01)
        await asyncio.wait_for(drained(), 3)
        assert len(accepted) == (0 if reset else 1)
        assert store.pending_origins() == []
        await runtime.close()
        again = work.CoordinationRuntime(adapter, home, {"fallback_delay_seconds": 10})
        await asyncio.gather(*tuple(again.tasks))
        assert len(accepted) == (0 if reset else 1)
        await again.close()
