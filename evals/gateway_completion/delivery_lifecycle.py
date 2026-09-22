import os, sys, asyncio, json, tempfile, types
from pathlib import Path

ROOT = os.environ.get("HERMES_EVAL_REPO", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, ROOT)
HOME = Path(os.environ["HERMES_HOME"])
HOME.mkdir(parents=True, exist_ok=True)
(HOME / "config.yaml").write_text(
    "display:\n  background_process_notifications: concise\n"
)
from gateway.run import GatewayRunner
from gateway.config import GatewayConfig, Platform, PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from gateway.platforms.base import SendResult
from gateway.session_context import set_session_vars, clear_session_vars
from tools.terminal_tool_background import spawn_background_process
from tools.process_registry import process_registry as pr
import gateway.run_notifications as rn

print("SOURCE", rn.__file__, "HOME", HOME)


async def main():
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig())
    runner.adapters[Platform.DISCORD] = adapter
    turns = []
    sends = []

    async def agent_boundary(event, *args, **kwargs):
        turns.append({
            "text": event.text,
            "internal": event.internal,
            "chat": event.source.chat_id,
        })
        return "FAKE_MODEL_BOUNDARY reached"

    async def send_boundary(chat_id, content, *args, **kwargs):
        sends.append({"chat": chat_id, "text": content})
        return SendResult(success=True, message_id=str(len(sends)))

    async def typing_boundary(*a, **kw):
        pass

    runner._handle_message_with_agent = agent_boundary
    adapter.send = send_boundary
    adapter.send_typing = typing_boundary
    adapter.set_message_handler(runner._handle_message)
    runner._running = True
    idle = asyncio.create_task(runner._async_delegation_watcher(interval=0.05))
    key = "agent:main:discord:dm:123"

    def spawn(label, notify=False, patterns=None):
        tokens = set_session_vars(
            platform="discord",
            chat_id="123",
            chat_type="dm",
            session_key=key,
            user_id="456",
        )
        try:
            result = json.loads(
                spawn_background_process(
                    command=f'sleep 0.3; printf "{label}\\n"',
                    env=types.SimpleNamespace(env=None),
                    env_type="local",
                    effective_task_id=label,
                    task_id=label,
                    session_key=key,
                    workdir=str(HOME),
                    cwd=str(HOME),
                    effective_pty=False,
                    notify_on_complete=notify,
                    watch_patterns=patterns,
                    approval_note=None,
                    pty_disabled_reason=None,
                )
            )
        finally:
            clear_session_vars(tokens)
        assert "session_id" in result, result
        return result

    async def settle():
        for _ in range(200):
            if not adapter._background_tasks:
                return
            await asyncio.sleep(0.025)
        raise RuntimeError("adapter tasks did not settle")

    # Real spawn, reader thread, durable process output and actual post-turn scheduling.
    result = spawn("NOTIFY_OK", notify=True)
    await runner._hmwa_post_turn_hooks({}, {}, "")
    await asyncio.sleep(5.5)
    await settle()
    print("NOTIFY_TRUE", json.dumps({"spawn": result, "turns": turns, "sends": sends}))
    assert len(turns) == 1 and turns[0]["internal"] and "NOTIFY_OK" in turns[0]["text"]
    turns.clear()
    sends.clear()
    # Pattern arrives after post-turn drain while idle watcher remains running.
    result = spawn("READY_AUDIT", patterns=["READY_AUDIT"])
    await runner._hmwa_post_turn_hooks({}, {}, "")
    await asyncio.sleep(1)
    print(
        "WATCH_IDLE",
        json.dumps({
            "spawn": result,
            "turns": turns,
            "queue_types": [e["type"] for e in list(pr.completion_queue.queue)],
        }),
    )
    assert len(turns) == 1
    assert not any(e["type"] == "watch_match" for e in list(pr.completion_queue.queue))
    # Same hook that a subsequent user turn executes, with no synthetic queue fabrication.
    await runner._hmwa_post_turn_hooks({}, {}, "")
    await settle()
    print("WATCH_AFTER_NEXT_POST_TURN", json.dumps({"turns": turns, "sends": sends}))
    assert len(turns) == 1 and "READY_AUDIT" in turns[0]["text"]
    turns.clear()
    sends.clear()
    # Raw legacy/restored watcher deliberately only sends a notice without a model turn.
    result = spawn("RAW_NOTICE")
    await runner._run_process_watcher({
        "session_id": result["session_id"],
        "check_interval": 0.05,
        "session_key": key,
        "platform": "discord",
        "chat_id": "123",
        "notify_on_complete": False,
    })
    await settle()
    print(
        "RAW_NOTIFICATION_ONLY",
        json.dumps({"turns": turns, "sends": sends, "spawn": result}),
    )
    assert not turns and len(sends) == 1
    # Real async executor/ledger/queue; only child model replaced by a local subprocess.
    from tools.async_delegation import dispatch_async_delegation, get_durable_delegation
    import subprocess

    def local_child():
        output = subprocess.check_output(
            ["/bin/sh", "-c", "printf CHILD_LOCAL_DONE"], text=True
        )
        (HOME / "child-artifact.txt").write_text(output)
        return {"summary": output, "status": "completed", "api_calls": 0}

    async def dispatch_case(route):
        h = dispatch_async_delegation(
            goal="local lifecycle",
            context=None,
            toolsets=None,
            role="leaf",
            model=None,
            session_key=route,
            runner=local_child,
        )
        for _ in range(100):
            row = get_durable_delegation(h["delegation_id"])
            if row and row.get("delivery_attempts", 0) > 0:
                break
            await asyncio.sleep(0.05)
        await settle()
        return h, get_durable_delegation(h["delegation_id"])

    turns.clear()
    sends.clear()
    h, row = await dispatch_case(key)
    print(
        "ASYNC_DISCORD_IDLE",
        json.dumps({
            "handle": h,
            "delivery_state": row["delivery_state"],
            "attempts": row["delivery_attempts"],
            "turns": turns,
        }),
    )
    assert len(turns) == 1 and row["delivery_state"] == "delivered"
    turns.clear()
    sends.clear()
    h, row = await dispatch_case("raw_session_without_api_adapter")
    print(
        "ASYNC_RAW_NO_TRANSPORT",
        json.dumps({
            "handle": h,
            "delivery_state": row["delivery_state"],
            "attempts": row["delivery_attempts"],
            "queue_size": pr.completion_queue.qsize(),
            "turns": turns,
        }),
    )
    assert (
        not turns
        and row["delivery_state"] == "pending"
        and row["delivery_attempts"] == 0
        and not pr.completion_queue.empty()
    )
    # Two real adapters: private completion must never enter the primary bot.
    secondary = DiscordAdapter(PlatformConfig())
    secondary_turns = []

    async def secondary_boundary(event):
        secondary_turns.append(event.source.profile)
        return None

    secondary.set_message_handler(secondary_boundary)
    secondary.send_typing = typing_boundary
    runner._profile_adapters = {"research": {Platform.DISCORD: secondary}}
    turns.clear()
    evt = {"session_key": "agent:research:discord:dm:123", "type": "completion"}
    assert await runner._inject_watch_notification("private completion", evt) is True
    for task in list(secondary._background_tasks):
        await task
    assert secondary_turns == ["research"] and not turns
    runner._profile_adapters = {}
    assert await runner._inject_watch_notification("private completion", evt) is False
    assert not turns
    print("MULTIPLEX", json.dumps({"secondary": secondary_turns, "primary": turns}))
    runner._running = False
    idle.cancel()
    await asyncio.gather(idle, return_exceptions=True)
    await runner._cancel_process_completion_batch_tasks()
    print("PASS all lifecycle assertions")


asyncio.run(main())
