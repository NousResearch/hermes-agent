"""Wire-contract probe: real poller, SQLite, adapter lifecycle; fake model/transport.

Run from the repo with a clean environment and a temporary HERMES_HOME:
  .venv/bin/python evals/heartbeat_idle_wire.py
Pass --base-poller /tmp/run_goals_base.py to compare the old poller. No network.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.config import GatewayConfig, Platform, PlatformConfig  # noqa: E402
from gateway.platforms.base import BasePlatformAdapter, SendResult  # noqa: E402
from gateway.platforms.event import MessageEvent  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402
from gateway.session import SessionSource, SessionStore, build_session_key  # noqa: E402
from hermes_cli import heartbeat  # noqa: E402


class WireAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.wire.append(content)
        return SendResult(success=True, message_id=str(len(self.wire)))


async def main(base_poller):
    assert os.environ.get("HERMES_HOME"), "Use a temporary HERMES_HOME"
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._run_in_executor_with_context = asyncio.to_thread
    adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    runner._adapter_for_source = lambda source: adapter
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm")
    key = build_session_key(source)
    watch = {key: (source, "wire-session")}
    if base_poller:
        scope = {"__name__": "heartbeat_base"}
        exec(compile(Path(base_poller).read_text(encoding="utf-8"), base_poller, "exec"), scope)
        runner._heartbeat_poll_once = scope["GatewayGoalsMixin"]._heartbeat_poll_once.__get__(runner)
    await runner._warm_goals_session_db("wire-contract")
    clock = SimpleNamespace(now=1000.0)
    received = []
    release = asyncio.Event()

    async def handler(event):
        event._heartbeat_execution_started = True  # fake agent execution boundary
        received.append(event.text)
        await release.wait()
        return "wire reply"

    async def drain():
        while adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))

    def snapshot():
        return {"turns": len(received), "queue_depth": runner._queue_depth(key, adapter=adapter),
                "fire_count": heartbeat.HeartbeatManager("wire-session").state.fire_count}

    adapter.set_message_handler(handler)
    await adapter.connect()
    with patch.object(heartbeat, "time", SimpleNamespace(time=lambda: clock.now)):
        heartbeat.HeartbeatManager("wire-session").set("check status", 60)
        for _ in range(15):
            clock.now += 60
            await runner._heartbeat_poll_once(watch)
            await asyncio.sleep(0)
        print(json.dumps({"phase": "15 minute polls, first turn held", **snapshot()}))
        release.set()
        await adapter.handle_message(MessageEvent(text="real-user-wire-input", source=source))
        await drain()
        print(json.dumps({"phase": "user wake drain", "wire_sends": len(adapter.wire), **snapshot()}))
        if not base_poller:
            assert len(received) == 2 and len(adapter.wire) == 2
            assert snapshot()["queue_depth"] == 0
            # A genuinely idle 15-minute gap coalesces to one, even with repeated polls.
            heartbeat.HeartbeatManager("wire-session").set("check status", 60)
            before = len(received)
            clock.now += 900
            for _ in range(5):
                await runner._heartbeat_poll_once(watch)
                await drain()
            assert len(received) - before == 1
            assert heartbeat.HeartbeatManager("wire-session").state.fire_count == 1
            print(json.dumps({"phase": "idle 15-minute gap + 5 same-time polls", "new_turns": 1,
                              "queue_depth": snapshot()["queue_depth"], "fire_count": 1}))
            # A pinned route must not enter the pre-claim executor recovery gap.
            # The callback represents competing traffic changing the last active topic.
            recovery = []
            adapter._topic_recovery_fn = lambda source: recovery.append(source) or "foreign-topic"
            clock.now += 60
            await runner._heartbeat_poll_once(watch)
            assert key in adapter._active_sessions and recovery == []
            await drain()
            print(json.dumps({"phase": "pinned route", "recovery_calls": len(recovery)}))
            # Admission can be cancelled before the fake agent boundary is reached.
            clock.now += 60
            before = heartbeat.HeartbeatManager("wire-session").state.to_json()
            turns_before = len(received)
            await runner._heartbeat_poll_once(watch)
            await adapter.cancel_session_processing(key)
            await drain()
            assert len(received) == turns_before
            assert heartbeat.HeartbeatManager("wire-session").state.to_json() == before
            print(json.dumps({"phase": "cancelled admission", "claim_refunded": True}))
            runner.config = GatewayConfig()
            runner.session_store = SessionStore(
                sessions_dir=Path(os.environ["HERMES_HOME"]) / "sessions", config=runner.config)
            runner.adapters = {Platform.TELEGRAM: adapter}
            runner._profile_adapters = {}
            runner._recover_telegram_topic_thread_id = adapter._topic_recovery_fn
            event = MessageEvent(text="pinned heartbeat", source=source,
                                 metadata={"gateway_session_key": key})
            resolved = await runner._hmwa_resolve_session(event, source)
            assert resolved is not None and resolved[2] == key and recovery == []
            print(json.dumps({"phase": "runner pinned route", "session_key": resolved[2],
                              "recovery_calls": len(recovery)}))
            # Route mismatch is a real adapter rejection, not a fabricated return value.
            adapter._session_key_profile = lambda source: "different-profile"
            route_sid = resolved[1].session_id
            heartbeat.HeartbeatManager(route_sid).set("check status", 60)
            clock.now += 60
            before = heartbeat.HeartbeatManager(route_sid).state.to_json()
            await runner._heartbeat_poll_once(watch)
            assert watch[key][1] == route_sid
            assert heartbeat.HeartbeatManager(route_sid).state.to_json() == before
            assert snapshot()["queue_depth"] == 0
            print(json.dumps({"phase": "rejected route", "claim_refunded": True}))
    await adapter.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-poller")
    asyncio.run(main(parser.parse_args().base_poller))
